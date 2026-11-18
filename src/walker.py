"""
node2vec biased random walks.

After moving t -> v, the walk picks the next node x (a neighbour of v) with
probability proportional to alpha(t, x) * w(v, x):
    alpha = 1/p  if x == t                     (go back)
    alpha = 1    if x is also a neighbour of t  (stay close to t)
    alpha = 1/q  otherwise                      (move away from t)

Because this depends on the previous node, we build one alias table for every
directed edge (t -> v) so that each step of a walk is O(1).

The functions with @njit are compiled by numba, otherwise the walks on
BlogCatalog are way too slow in plain python.
"""
import numpy as np
from numba import njit, prange


# ---------------- alias method ----------------

@njit
def alias_setup(probs, prob, alias):
    """Vose's alias method. Fills prob[] and alias[] for the distribution probs."""
    K = len(probs)
    scaled = probs * K
    small = np.empty(K, dtype=np.int64)   # used as stacks
    large = np.empty(K, dtype=np.int64)
    ns, nl = 0, 0
    for i in range(K):
        if scaled[i] < 1.0:
            small[ns] = i
            ns += 1
        else:
            large[nl] = i
            nl += 1

    while ns > 0 and nl > 0:
        ns -= 1
        s = small[ns]
        nl -= 1
        l = large[nl]
        prob[s] = scaled[s]
        alias[s] = l
        scaled[l] = scaled[l] - (1.0 - scaled[s])   # l gives away some of its mass
        if scaled[l] < 1.0:
            small[ns] = l
            ns += 1
        else:
            large[nl] = l
            nl += 1

    # whatever is left has probability 1 (rounding errors)
    while nl > 0:
        nl -= 1
        prob[large[nl]] = 1.0
        alias[large[nl]] = large[nl]
    while ns > 0:
        ns -= 1
        prob[small[ns]] = 1.0
        alias[small[ns]] = small[ns]


@njit
def alias_draw(prob, alias, start, K):
    """Sample from the table stored at prob/alias[start : start + K]."""
    i = np.random.randint(0, K)
    if np.random.random() < prob[start + i]:
        return i
    return alias[start + i]


# ---------------- transition probabilities ----------------

@njit
def get_weights(indptr, indices, weights, t, v, p, q, out):
    """alpha(t, x) * w(v, x) for every neighbour x of v (written into out).
    Both neighbour lists are sorted so we can check "x is a neighbour of t"
    by moving a pointer j through t's list (like merging two sorted lists)."""
    j = indptr[t]
    k = 0
    for pos in range(indptr[v], indptr[v + 1]):
        x = indices[pos]
        while j < indptr[t + 1] and indices[j] < x:
            j += 1
        if x == t:
            out[k] = weights[pos] / p
        elif j < indptr[t + 1] and indices[j] == x:
            out[k] = weights[pos]
        else:
            out[k] = weights[pos] / q
        k += 1


@njit
def build_node_tables(indptr, weights, prob, alias):
    """First step of a walk has no previous node -> normal table per node."""
    for v in range(len(indptr) - 1):
        a, b = indptr[v], indptr[v + 1]
        if b > a:
            w = weights[a:b]
            alias_setup(w / w.sum(), prob[a:b], alias[a:b])


@njit(parallel=True)
def build_edge_tables(indptr, indices, weights, p, q, offsets, prob, alias):
    """One table per directed edge e = (t -> v), stored at prob/alias[offsets[e]:offsets[e+1]].
    Edge e is position e in the CSR indices array (row t, column v)."""
    n = len(indptr) - 1
    for t in prange(n):
        for e in range(indptr[t], indptr[t + 1]):
            v = indices[e]
            w = np.empty(indptr[v + 1] - indptr[v])
            get_weights(indptr, indices, weights, t, v, p, q, w)
            alias_setup(w / w.sum(), prob[offsets[e]:offsets[e + 1]], alias[offsets[e]:offsets[e + 1]])


# ---------------- walks ----------------

@njit
def set_seed(seed):
    np.random.seed(seed)


@njit
def walk(indptr, indices, node_prob, node_alias, offsets, edge_prob, edge_alias,
         start_nodes, walk_length, second_order):
    walks = np.full((len(start_nodes), walk_length), -1, dtype=np.int32)  # -1 = walk got stuck
    for w in range(len(start_nodes)):
        cur = start_nodes[w]
        walks[w, 0] = cur
        if indptr[cur + 1] == indptr[cur]:     # isolated node
            continue
        # first step (1st order)
        k = alias_draw(node_prob, node_alias, indptr[cur], indptr[cur + 1] - indptr[cur])
        e = indptr[cur] + k          # id of the edge we just used
        cur = indices[e]
        walks[w, 1] = cur
        for step in range(2, walk_length):
            deg = indptr[cur + 1] - indptr[cur]
            if second_order:
                k = alias_draw(edge_prob, edge_alias, offsets[e], deg)
            else:
                k = alias_draw(node_prob, node_alias, indptr[cur], deg)
            # picking neighbour k of cur means walking along edge indptr[cur] + k,
            # which is also the index of the table we need for the next step
            e = indptr[cur] + k
            cur = indices[e]
            walks[w, step] = cur
    return walks


def node2vec_walks(A, p=1.0, q=1.0, num_walks=10, walk_length=80, seed=0):
    """r = num_walks walks from every node. Returns an int array (num_walks * n, walk_length)."""
    indptr = A.indptr.astype(np.int64)
    indices = A.indices.astype(np.int32)
    weights = A.data.astype(np.float64)
    n = A.shape[0]

    node_prob = np.empty(len(indices))
    node_alias = np.empty(len(indices), dtype=np.int32)
    build_node_tables(indptr, weights, node_prob, node_alias)

    # p = q = 1 is DeepWalk: all alphas are 1, so the edge tables would be the
    # same as the node tables. Skip them (they take ~3GB for BlogCatalog).
    second_order = not (p == 1 and q == 1)
    if second_order:
        sizes = np.diff(indptr)[indices]                 # table of edge t->v has deg(v) entries
        offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        edge_prob = np.empty(offsets[-1], dtype=np.float32)
        edge_alias = np.empty(offsets[-1], dtype=np.int32)
        build_edge_tables(indptr, indices, weights, p, q, offsets, edge_prob, edge_alias)
    else:
        offsets = np.zeros(1, dtype=np.int64)
        edge_prob = np.zeros(1, dtype=np.float32)
        edge_alias = np.zeros(1, dtype=np.int32)

    # every node starts num_walks walks, in a random order each round
    rng = np.random.default_rng(seed)
    start_nodes = np.concatenate([rng.permutation(n) for _ in range(num_walks)])
    set_seed(seed)
    return walk(indptr, indices, node_prob, node_alias, offsets, edge_prob, edge_alias,
                start_nodes, walk_length, second_order)
