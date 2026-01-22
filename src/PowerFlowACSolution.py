from dataclasses import dataclass

from networkx import Graph


@dataclass
class PowerFlowACSolution:
    """ Solution to a power flow AC network. Given graph should be annotated with the values of the variables. """
    graph: Graph
