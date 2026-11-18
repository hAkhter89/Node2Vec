"""
Skip-gram with negative sampling (word2vec) trained on the random walks.

For a pair (u, c) where c is within `window` steps of u in a walk, and K random
negative nodes n_1..n_K:
    loss = -log sigmoid(f(u).g(c)) - sum_k log sigmoid(-f(u).g(n_k))
f = embeddings we keep, g = context embeddings (only used for training).
Negatives are sampled with probability ~ count(node)^0.75 like in word2vec.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGram(nn.Module):
    def __init__(self, num_nodes, dim):
        super().__init__()
        self.in_emb = nn.Embedding(num_nodes, dim, sparse=True)    # f
        self.out_emb = nn.Embedding(num_nodes, dim, sparse=True)   # g
        nn.init.uniform_(self.in_emb.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.out_emb.weight)

    def forward(self, u, c, neg):
        fu = self.in_emb(u)                                  # (B, d)
        pos = (fu * self.out_emb(c)).sum(1)                  # (B,)
        negs = torch.bmm(self.out_emb(neg), fu.unsqueeze(2)).squeeze(2)   # (B, K)
        loss = -F.logsigmoid(pos) - F.logsigmoid(-negs).sum(1)
        return loss.mean()


def make_pairs(walks, window):
    """All (center, context) pairs that are at most `window` apart in a batch of walks."""
    L = walks.shape[1]
    centers, contexts = [], []
    for o in range(1, window + 1):
        a = walks[:, :L - o].reshape(-1)
        b = walks[:, o:].reshape(-1)
        centers += [a, b]       # context on the right and on the left
        contexts += [b, a]
    centers, contexts = torch.cat(centers), torch.cat(contexts)
    keep = (centers >= 0) & (contexts >= 0)     # drop the -1 padding
    return centers[keep], contexts[keep]


def train_skipgram(walks, num_nodes, dim=128, window=10, num_neg=5, epochs=1, lr=0.01, batch_walks=64, seed=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    walks = torch.as_tensor(walks, dtype=torch.long, device=device)

    counts = torch.bincount(walks[walks >= 0], minlength=num_nodes).float()
    noise = counts ** 0.75
    noise = noise / noise.sum()

    model = SkipGram(num_nodes, dim).to(device)
    # SparseAdam only updates the rows used in the batch.
    # (plain SGD with large batches is a problem for hub nodes that appear many times in a batch)
    opt = torch.optim.SparseAdam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        losses = []
        perm = torch.randperm(len(walks), device=device)
        for i in range(0, len(walks), batch_walks):
            u, c = make_pairs(walks[perm[i:i + batch_walks]], window)
            neg = torch.multinomial(noise, len(u) * num_neg, replacement=True).view(-1, num_neg)
            opt.zero_grad()
            loss = model(u, c, neg)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"  epoch {epoch + 1}: loss {np.mean(losses):.4f}")

    return model.in_emb.weight.detach().cpu().numpy()
