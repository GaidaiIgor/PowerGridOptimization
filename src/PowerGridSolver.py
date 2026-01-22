from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial

import numpy as np
from numpy import random
from pyscipopt import Model, sin, cos, quicksum
from pyscipopt.recipes.nonlinear import set_nonlinear_objective

from src import utils
from src.PowerFlowACSolution import PowerFlowACSolution
from src.PowerGridProblem import PowerGridProblem, PowerGridSolution, PowerNetwork
from src.Sampler import ExactSampler
from src.VariationalQuantumProgram import VariationalQuantumProgram


class PowerGridSolver(ABC):
    """ Base class for power grid network solvers. """

    @abstractmethod
    def solve(self, problem: PowerGridProblem, *args, **kwargs) -> PowerGridSolution:
        """ Solves a given power grid optimization network and returns its solution. """
        pass


@dataclass
class HybridSolver(PowerGridSolver):
    """ Optimizes binary variables on a quantum computer, continuous variables classically. """
    vqp: VariationalQuantumProgram

    def solve(self, problem: PowerGridProblem, penalty_mult: float = 10) -> PowerGridSolution:
        seed = 0
        rng = random.default_rng(seed)

        initial_angles = rng.uniform(-np.pi, np.pi, len(self.vqp.circuit.parameters))
        cost_function = partial(problem.evaluate, penalty_mult=penalty_mult)
        result = self.vqp.optimize_parameters(cost_function, initial_angles)
        assert result.success, f"Angle optimization failed: {result.message}"

        best_sample = min(problem.optimize_power.cache.items(), key=lambda pair: pair[1].total)
        solution = PowerGridSolution(best_sample[0], best_sample[1].x, best_sample[1].fun)
        solution.extra["opt_result"] = best_sample[1]

        exact_sampler = ExactSampler()
        solution.extra["final_probs"] = exact_sampler.get_sample_probabilities(self.vqp.circuit, result.x)
        solution.extra["cost_expectation"] = utils.get_cost_expectation(cost_function, solution.extra["final_probs"])
        solution.extra["num_jobs"] = result.nfev
        return solution


class ClassicalACSolver:
    """ Uses SCIP library to solve the AC network classically. """

    @staticmethod
    def build_model(network: PowerNetwork) -> Model:
        """ Builds model based on network description. Adds variables to the network graph. """
        model = Model("PowerFlowAC")
        cost_terms = []
        for node_label, node_data in network.graph.nodes(data=True):
            node_data["v"] = model.addVar(lb=node_data["voltage_range"][0], ub=node_data["voltage_range"][1], name=f"v_{node_label}")
            node_data["d"] = model.addVar(lb=node_data["angle_range"][0], ub=node_data["angle_range"][1], name=f"d_{node_label}")
            node_data["u"] = []
            node_data["p"] = []
            node_data["q"] = []
            for i, gen in enumerate(node_data["generators"]):
                u = model.addVar(vtype="B", name=f"u_{node_label}_{i}")
                p = model.addVar(lb=0, ub=gen.power_range[1], name=f"p_{node_label}_{i}")
                q = model.addVar(lb=0, ub=gen.reactive_power_range[1], name=f"q_{node_label}_{i}")
                model.addCons(p >= gen.power_range[0] * u, name=f"p_min_{node_label}_{i}")
                model.addCons(p <= gen.power_range[1] * u, name=f"p_max_{node_label}_{i}")
                model.addCons(q >= gen.reactive_power_range[0] * u, name=f"q_min_{node_label}_{i}")
                model.addCons(q <= gen.reactive_power_range[1] * u, name=f"q_max_{node_label}_{i}")
                cost_terms.append(gen.cost_terms[0] * p * p + gen.cost_terms[1] * p + gen.cost_terms[2] * u)
                node_data["u"].append(u)
                node_data["p"].append(p)
                node_data["q"].append(q)

        for node_label, node_data in network.graph.nodes(data=True):
            if node_data["node_ind"] == 0:
                model.addCons(node_data["d"] == 0, name="fixed angle")
            real_flows = []
            imag_flows = []
            for _, neighbor_label, line_data in network.graph.edges(node_label, data=True):
                delta = node_data["d"] - network.graph.nodes[neighbor_label]["d"]
                alpha = line_data["admittance"].real
                beta = line_data["admittance"].imag
                v_i = node_data["v"]
                v_j = network.graph.nodes[neighbor_label]["v"]
                real_flow = alpha * v_i * v_i - v_i * v_j * (alpha * cos(delta) + beta * sin(delta))
                imag_flow = -beta * v_i * v_i + v_i * v_j * (beta * cos(delta) - alpha * sin(delta))
                model.addCons(real_flow * real_flow + imag_flow * imag_flow <= line_data["capacity"] ** 2, name=f"capacity_{node_label}_{neighbor_label}")
                real_flows.append(real_flow)
                imag_flows.append(imag_flow)
            model.addCons(quicksum(node_data["p"]) - node_data["load"].real - quicksum(real_flows) >= 0, name=f"net_power_real_{node_label}_{neighbor_label}")
            model.addCons(quicksum(node_data["q"]) - node_data["load"].imag - quicksum(imag_flows) >= 0, name=f"net_power_imag_{node_label}_{neighbor_label}")

        set_nonlinear_objective(model, quicksum(cost_terms), sense="minimize")
        return model

    @staticmethod
    def write_solution(network: PowerNetwork, model: Model):
        """ Writes solution from optimized model into the network. """
        network.graph.graph["num_sols"] = model.getNSols()
        var_names_single = ["v", "d"]
        var_names_list = ["u", "p", "q"]
        for node_label, node_data in network.graph.nodes(data=True):
            for name in var_names_single:
                node_data[name] = model.getVal(node_data[name])
            for name in var_names_list:
                for i in range(len(node_data[name])):
                    node_data[name][i] = model.getVal(node_data[name][i])

    @staticmethod
    def solve(network: PowerNetwork) -> PowerNetwork:
        """ Solves the AC optimization problem and returns an updated network instance with the optimal values. """
        model = ClassicalACSolver.build_model(network)
        model.optimize()
        ClassicalACSolver.write_solution(network, model)
        return network
