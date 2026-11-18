import numpy as np
import scipy.io as sio
import scipy.sparse as sp

DATASETS = {
    "blogcatalog": "data/blogcatalog.mat",
    "ppi": "data/Homo_sapiens.mat",
}


def load_dataset(name):
    """Returns the adjacency matrix (scipy CSR) and the 0/1 label matrix (nodes x labels).

    In CSR format the neighbours of node v are A.indices[A.indptr[v]:A.indptr[v+1]]
    and the edge weights are in A.data at the same positions.
    """
    mat = sio.loadmat(DATASETS[name])

    A = sp.csr_matrix(mat["network"], dtype=np.float64)
    A = A.maximum(A.T)      # make sure the graph is undirected
    A.setdiag(0)            # remove self loops (PPI has some)
    A.eliminate_zeros()
    A.sort_indices()        # sorted neighbour lists, the walker needs this

    labels = sp.csr_matrix(mat["group"]).toarray().astype(np.int8)
    return A, labels
