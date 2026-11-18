import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier


def spectral_embedding(A, dim=128):
    """Baseline: eigenvectors of the normalized Laplacian L = I - D^-1/2 A D^-1/2
    with the smallest eigenvalues (= largest eigenvalues of D^-1/2 A D^-1/2)."""
    deg = np.asarray(A.sum(1)).ravel()
    d = np.zeros_like(deg)
    d[deg > 0] = 1 / np.sqrt(deg[deg > 0])
    S = sp.diags(d) @ A @ sp.diags(d)
    v0 = np.random.default_rng(0).random(A.shape[0])
    vals, vecs = eigsh(S, k=dim, which="LA", v0=v0)
    return vecs


def node_classification(emb, labels, train_ratio, repeats=5):
    """One-vs-rest logistic regression. A test node with k labels gets its k
    highest scoring labels (same protocol as DeepWalk / node2vec papers).
    Returns mean Micro-F1 and Macro-F1 over `repeats` random splits."""
    micro, macro = [], []
    n = len(labels)
    for r in range(repeats):
        perm = np.random.default_rng(r).permutation(n)
        n_train = int(train_ratio * n)
        train, test = perm[:n_train], perm[n_train:]

        clf = OneVsRestClassifier(LogisticRegression(solver="liblinear"), n_jobs=-1)
        clf.fit(emb[train], labels[train])
        scores = clf.predict_proba(emb[test])

        pred = np.zeros_like(labels[test])
        k = labels[test].sum(1)
        top = np.argsort(-scores, axis=1)
        for i in range(len(test)):
            pred[i, top[i, :k[i]]] = 1

        micro.append(f1_score(labels[test], pred, average="micro", zero_division=0))
        macro.append(f1_score(labels[test], pred, average="macro", zero_division=0))
    return np.mean(micro), np.mean(macro), np.std(macro)
