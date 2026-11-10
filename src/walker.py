import networkx as nx


class Node2VecWalker:
    def __init__(self, graph: nx.Graph, p: float, q: float):
        self.G = graph
        self.p = p
        self.q = q

    def get_unnormalized_weights(self, t, v):
        """
        Calculates alpha(t,x) * w(v,x) for all neighbors x of v.
        t: previous node
        v: current node
        """
        unnormalized_probs = []
        neighbors = list(self.G.neighbors(v))

        for x in neighbors:
            # Default weight to 1.0 if the graph is unweighted
            weight = self.G[v][x].get("weight", 1.0)

            if x == t:
                # Distance 0: Returning to the previous node
                unnormalized_probs.append(weight / self.p)
            elif self.G.has_edge(x, t):
                # Distance 1: Next node is connected to previous node (homophily)
                unnormalized_probs.append(weight)
            else:
                # Distance 2: Next node is NOT connected to previous node (exploration)
                unnormalized_probs.append(weight / self.q)

        return neighbors, unnormalized_probs
