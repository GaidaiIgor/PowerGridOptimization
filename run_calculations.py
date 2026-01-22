import time

import numpy as np
from networkx import Graph

from src.CircuitLayer import AllToAllEntangler, ZXMixer
from src.PowerGridSolver import HybridSolver, ClassicalACSolver
from src.PowerGridProblem import GeneratorCommitmentProblem, Generator, PowerNetwork
from src.Sampler import ExactSampler
from src.VariationalQuantumProgram import VariationalQuantumProgram


def get_generator_commitment_problem() -> GeneratorCommitmentProblem:
    generators = np.array([Generator((15, 20), (0, 0), (0, 1, 10)),
                           Generator((0, 10), (0, 0), (1, 0, 1))])
    load = 10

    # generators = np.array([Generator((100, 600), (0.002, 10, 500)),
    #                        Generator((100, 400), (0.0025, 8, 300)),
    #                        Generator((50, 200), (0.005, 6, 100))])
    # load = 170
    #
    # generators = np.array([Generator((150, 455), (0.00048, 16.19, 1000)),
    #                        Generator((150, 455), (0.00031, 17.26, 970)),
    #                        Generator((20, 130), (0.002, 16.6, 700)),
    #                        Generator((20, 130), (0.00211, 16.5, 680)),
    #                        Generator((25, 162), (0.00398, 19.7, 450)),
    #                        Generator((20, 80), (0.00712, 22.26, 370)),
    #                        Generator((25, 85), (0.00079, 27.74, 480)),
    #                        Generator((10, 55), (0.00413, 25.92, 660)),
    #                        Generator((10, 55), (0.00222, 27.27, 665)),
    #                        Generator((10, 55), (0.00173, 27.79, 670))
    #                        ])
    # load = 700

    problem = GeneratorCommitmentProblem(generators, load)
    return problem


def get_power_flow_ac_problem() -> PowerNetwork:
    voltage_range = (0, 10)
    angle_range = (-np.pi, np.pi)
    graph = Graph()

    graph.add_node(0, generators=[Generator((0, 100), (0, 100), (0, 1, 1))], load=0, voltage_range=voltage_range, angle_range=angle_range)
    graph.add_node(1, generators=[], load=10, voltage_range=voltage_range, angle_range=angle_range)
    graph.add_edge(0, 1, capacity=100, admittance=1+1j)

    # graph.add_node(0, generators=[Generator((0, 30), (0, 0), (0, 10, 1))], load=0, voltage_range=voltage_range, angle_range=angle_range)
    # graph.add_node(1, generators=[Generator((0, 10), (0, 0), (0, 20, 1))], load=10, voltage_range=voltage_range, angle_range=angle_range)
    # graph.add_node(2, generators=[], load=10, voltage_range=voltage_range, angle_range=angle_range)
    # graph.add_edge(0, 1, capacity=10, admittance=1)
    # graph.add_edge(0, 2, capacity=5, admittance=1)
    # graph.add_edge(1, 2, capacity=10, admittance=1)

    return PowerNetwork(graph)


def get_variational_quantum_program(num_qubits: int) -> VariationalQuantumProgram:
    entangler = AllToAllEntangler(num_qubits)
    mixer = ZXMixer(num_qubits)
    num_layers = 1

    sampler = ExactSampler()
    # sampler = MySamplerV2(StatevectorSampler(default_shots=1000))
    # sampler = IonQSampler("simulator", 1000, None)
    # sampler = IonQSampler("qpu.forte-enterprise-1", 1000, None)

    return VariationalQuantumProgram(num_layers, [entangler, mixer], sampler)


def main():
    # network = get_generator_commitment_problem()
    network = get_power_flow_ac_problem()

    solver = ClassicalACSolver()

    # num_gen = len(network.generators)
    # vqp = get_variational_quantum_program(num_gen)
    # solver = HybridSolver(vqp)

    network = solver.solve(network)
    network.print_solution()

    # print(f"Optimized probabilities: {solution.extra["final_probs"]}")
    # print(f"Optimized expectation: {solution.extra["cost_expectation"]}")
    # print(f"Number of jobs: {solution.extra["num_jobs"]}")
    #
    # print("=== Best sample ===")
    # print(f"Classical optimization successful: {solution.extra["opt_result"].success}")
    # print(f"Generators selected: {solution.generator_statuses}")
    # print(f"Optimized classical variables: {solution.grid_parameters}")
    # print(f"Optimized cost: {solution.cost}")
    # print(f"Penalty: {solution.extra["opt_result"].penalty}")


if __name__ == "__main__":
    t1 = time.perf_counter()
    main()
    t2 = time.perf_counter()
    print(f"Elapsed time {t2 - t1} seconds")
