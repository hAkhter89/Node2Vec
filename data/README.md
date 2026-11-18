# Datasets

| File | Dataset | Nodes | Edges | Labels |
|---|---|---|---|---|
| `blogcatalog.mat` | BlogCatalog (bloggers, labels = interests) | 10,312 | 333,983 | 39 |
| `Homo_sapiens.mat` | PPI (proteins, labels = biological states) | 3,890 | 37,845 (after removing self loops) | 50 |

Sources:
- BlogCatalog: original file from Tang & Liu (`leitang.net/code/social-dimension/data/blogcatalog.mat`). The site is down, we got it from the Internet Archive.
- PPI: http://snap.stanford.edu/node2vec/Homo_sapiens.mat

Each file has a sparse `network` matrix (adjacency) and a sparse `group` matrix (node x label, 0/1).
