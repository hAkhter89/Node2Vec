import networkx as nx


class Node2VecWalker:
    def __init__(self, graph: nx.Graph, p: float, q: float):
        self.G = graph
        self.p = p
        self.q = q

    def get_unnormalizes_weights(self, t, v):
        return 0
