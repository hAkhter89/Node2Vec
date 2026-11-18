# node2vec

Reimplementation of *node2vec: Scalable Feature Learning for Networks* (Grover & Leskovec, 2016).

Muhammad Hamza Bin Tariq (mb09640), Hassan Akhter (ha10609)

## Files
- `src/graph.py` – loads the datasets (`.mat` files) as a sparse adjacency matrix + labels
- `src/walker.py` – alias tables and the 2nd-order biased random walks (p, q)
- `src/skipgram.py` – skip-gram with negative sampling in PyTorch
- `src/evaluate.py` – spectral clustering baseline and node classification (Micro/Macro-F1)
- `main.py` – runs Spectral, DeepWalk (p=q=1) and node2vec on a dataset and saves the results

## Setup
```
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Run
```
python main.py --dataset ppi
python main.py --dataset blogcatalog
```
Results (json + plot) are saved in `results/`.

Settings are the ones from the paper: d = 128, r = 10 walks per node, walk length l = 80, window k = 10, 5 negative samples, 1 epoch.
