"""
Baseline experiment: multi-label node classification (Table 2 / Figure 4 of the paper).
Compares Spectral clustering, DeepWalk (p = q = 1) and node2vec.

    python main.py --dataset ppi
    python main.py --dataset blogcatalog
    python main.py --dataset ppi --p 0.5 --q 2      (other p, q for node2vec)
"""
import argparse
import json
import os
import time
import warnings

import matplotlib.pyplot as plt

from src.evaluate import node_classification, spectral_embedding
from src.graph import load_dataset
from src.skipgram import train_skipgram
from src.walker import node2vec_walks

warnings.filterwarnings("ignore")   # sklearn complains when a label has no positives in a split

# Macro-F1 at 50% labelled from Table 2 of the paper, and the (p, q) they used
PAPER = {
    "blogcatalog": {"Spectral": 0.0405, "DeepWalk": 0.2110, "node2vec": 0.2581, "pq": (0.25, 0.25)},
    "ppi": {"Spectral": 0.0681, "DeepWalk": 0.1768, "node2vec": 0.1791, "pq": (4, 1)},
}
RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]


def node2vec(A, p, q):
    t = time.time()
    walks = node2vec_walks(A, p=p, q=q, num_walks=10, walk_length=80)
    print(f"  walks: {walks.shape} in {time.time() - t:.1f}s")
    t = time.time()
    emb = train_skipgram(walks, A.shape[0], dim=128, window=10, num_neg=5, epochs=1)
    print(f"  skip-gram training: {time.time() - t:.1f}s")
    return emb


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="ppi", choices=["ppi", "blogcatalog"])
parser.add_argument("--p", type=float, default=None)
parser.add_argument("--q", type=float, default=None)
parser.add_argument("--repeats", type=int, default=5)
args = parser.parse_args()

A, labels = load_dataset(args.dataset)
print(f"{args.dataset}: {A.shape[0]} nodes, {A.nnz // 2} edges, {labels.shape[1]} labels")
p, q = PAPER[args.dataset]["pq"]
if args.p is not None:
    p, q = args.p, args.q

embeddings = {}
print("Spectral")
embeddings["Spectral"] = spectral_embedding(A, dim=128)
print("DeepWalk (p=1, q=1)")
embeddings["DeepWalk"] = node2vec(A, 1, 1)
print(f"node2vec (p={p}, q={q})")
embeddings["node2vec"] = node2vec(A, p, q)

results = {}
for name, emb in embeddings.items():
    results[name] = {}
    for ratio in RATIOS:
        micro, macro, macro_std = node_classification(emb, labels, ratio, repeats=args.repeats)
        results[name][ratio] = {"micro": micro, "macro": macro, "macro_std": macro_std}

print(f"\nResults at 50% labelled ({args.repeats} random splits)")
print(f"{'method':10s} {'Micro-F1':>9s} {'Macro-F1':>9s} {'paper Macro-F1':>15s}")
for name in results:
    r = results[name][0.5]
    print(f"{name:10s} {r['micro']:9.4f} {r['macro']:9.4f} {PAPER[args.dataset][name]:15.4f}")
gain = results["node2vec"][0.5]["macro"] / results["DeepWalk"][0.5]["macro"] - 1
print(f"node2vec gain over DeepWalk (Macro-F1): {100 * gain:+.1f}%")

os.makedirs("results", exist_ok=True)
with open(f"results/{args.dataset}.json", "w") as f:
    json.dump({"p": p, "q": q, "results": results}, f, indent=2)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for name, marker in [("Spectral", "s"), ("DeepWalk", "o"), ("node2vec", "^")]:
    label = name if name != "node2vec" else f"node2vec (p={p}, q={q})"
    axes[0].plot(RATIOS, [results[name][r]["micro"] for r in RATIOS], marker=marker, label=label)
    axes[1].plot(RATIOS, [results[name][r]["macro"] for r in RATIOS], marker=marker, label=label)
for ax, metric in zip(axes, ["Micro-F1", "Macro-F1"]):
    ax.set_xlabel("fraction of labelled nodes")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.3)
axes[1].legend()
fig.suptitle(args.dataset)
fig.tight_layout()
fig.savefig(f"results/{args.dataset}.png", dpi=150)
print(f"saved results/{args.dataset}.json and results/{args.dataset}.png")
