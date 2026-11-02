"""Chapter 7 -- two-tower retrieval (PyTorch) + an IVF approximate-nearest-neighbour demo."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import common
import dense_recs
from recsys import candidates

torch.set_num_threads(1)
torch.manual_seed(0)
np.random.seed(0)

N_HIST = 20
TEMP = 0.05
BATCH = 512
EPOCHS = 10


class Towers(nn.Module):
    def __init__(self, n_users, n_items, n_cats, n_brands, n_buckets):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, 32)
        self.item_emb = nn.Embedding(n_items + 1, 32, padding_idx=n_items)  # shared; last row = padding
        self.cat_emb = nn.Embedding(n_cats, 8)
        self.brand_emb = nn.Embedding(n_brands, 8)
        self.price_emb = nn.Embedding(n_buckets, 8)
        self.pad = n_items
        self.user_mlp = nn.Sequential(nn.Linear(32 + 32 + n_cats, 64), nn.ReLU(), nn.Linear(64, 32))
        self.item_mlp = nn.Sequential(nn.Linear(32 + 8 + 8 + 8, 64), nn.ReLU(), nn.Linear(64, 32))

    def user_vec(self, uid, hist_items, aff):
        mask = (hist_items != self.pad).float().unsqueeze(-1)
        h = self.item_emb(hist_items)
        mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        x = torch.cat([self.user_emb(uid), mean, aff], dim=-1)
        return F.normalize(self.user_mlp(x), dim=-1)

    def item_vec(self, iid, cat, brand, bucket):
        x = torch.cat([self.item_emb(iid), self.cat_emb(cat), self.brand_emb(brand), self.price_emb(bucket)], dim=-1)
        return F.normalize(self.item_mlp(x), dim=-1)


def kmeans(x: np.ndarray, k: int, iters: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    cent = x[rng.choice(len(x), k, replace=False)].copy()
    for _ in range(iters):
        assign = np.argmax(x @ cent.T - 0.5 * (cent**2).sum(1), axis=1)  # nearest centroid (euclidean)
        for c in range(k):
            pts = x[assign == c]
            if len(pts):
                cent[c] = pts.mean(0)
    assign = np.argmax(x @ cent.T - 0.5 * (cent**2).sum(1), axis=1)
    return cent, assign


def main() -> None:
    hist = common.history()
    users = common.eval_users()
    items = hist.items.reset_index(drop=True)  # every item listed at cutoff
    item_ids = items["item_id"].to_numpy()
    item_pos = pd.Index(item_ids)
    all_users = hist.users["user_id"].to_numpy()
    user_pos = pd.Index(all_users)
    cats = hist.categories.sort_values("category_id").reset_index(drop=True)
    n_cats = len(cats)
    cat_pos = pd.Index(cats["category_id"])

    # ---- features
    brands = pd.Index(sorted(items["brand"].unique()))
    ratio = (items["price"] / items["median_price"]).to_numpy()
    edges = np.quantile(ratio, [0.2, 0.4, 0.6, 0.8])
    bucket = np.searchsorted(edges, ratio)
    t_cat = torch.tensor(cat_pos.get_indexer(items["category_id"]))
    t_brand = torch.tensor(brands.get_indexer(items["brand"]))
    t_bucket = torch.tensor(bucket)
    t_item = torch.arange(len(items))

    aff = candidates.category_affinity(hist).pivot_table(index="user_id", columns="category_id", values="share", fill_value=0.0)
    aff = aff.reindex(index=all_users, columns=cats["category_id"], fill_value=0.0).fillna(0.0)
    t_aff = torch.tensor(aff.to_numpy(), dtype=torch.float32)

    pos = hist.interactions().sort_values("ts")
    pos = pos[pos["item_id"].isin(item_pos) & pos["user_id"].isin(user_pos)]
    last = pos.groupby("user_id")["item_id"].apply(lambda s: list(dict.fromkeys(s[::-1]))[:N_HIST])
    hist_mat = np.full((len(all_users), N_HIST), len(items), dtype=np.int64)
    for uid, lst in last.items():
        ix = item_pos.get_indexer(lst)
        hist_mat[user_pos.get_loc(uid), : len(ix)] = ix
    t_hist = torch.tensor(hist_mat)

    pairs = pos.groupby(["user_id", "item_id"], as_index=False)["w"].sum()
    p_u = torch.tensor(user_pos.get_indexer(pairs["user_id"]))
    p_i = torch.tensor(item_pos.get_indexer(pairs["item_id"]))

    common.section("Setup")
    print(f"users {len(all_users)}  items {len(items)}  categories {n_cats}  brands {len(brands)}  price buckets 5")
    print(f"training pairs (distinct user-item positives before cutoff): {len(pairs)}")
    print(f"user history window: last {N_HIST} items; temperature {TEMP}; batch {BATCH}; epochs {EPOCHS}")

    model = Towers(len(all_users), len(items), n_cats, len(brands), 5)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(0)

    common.section("Training (in-batch softmax)")
    losses = []
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(p_u), generator=g)
        total, nb = 0.0, 0
        for s in range(0, len(perm), BATCH):
            b = perm[s : s + BATCH]
            u, i = p_u[b], p_i[b]
            h = t_hist[u].clone()
            h[h == i[:, None]] = model.pad  # hide the target item from its own history (no leakage)
            uv = model.user_vec(u, h, t_aff[u])
            iv = model.item_vec(i, t_cat[i], t_brand[i], t_bucket[i])
            logits = uv @ iv.T / TEMP
            same = i[:, None] == i[None, :]
            logits = logits.masked_fill(same & ~torch.eye(len(b), dtype=torch.bool), -1e9)  # duplicate items are not negatives
            loss = F.cross_entropy(logits, torch.arange(len(b)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        losses.append(total / nb)
        print(f"epoch {ep:>2d}  in-batch softmax loss {losses[-1]:.4f}")

    # ---- embeddings
    model.eval()
    with torch.no_grad():
        iv_all = model.item_vec(t_item, t_cat, t_brand, t_bucket).numpy()
        u_rows = user_pos.get_indexer(users)
        uv_eval = model.user_vec(torch.tensor(u_rows), t_hist[u_rows], t_aff[u_rows]).numpy()

    avail_mask = items["in_stock"].to_numpy()
    avail_ids = item_ids[avail_mask]
    V = iv_all[avail_mask].astype(np.float32)
    has_hist = np.isin(users, last.index.to_numpy())
    scores = uv_eval @ V.T
    scores[~has_hist] = 0.0  # untrained user-id embedding: no opinion -> popularity padding
    # cosine scores can be negative; shift so "has an opinion" == positive
    scores = np.where(has_hist[:, None], scores - scores.min(axis=1, keepdims=True) + 1e-6, 0.0)

    recs, st = dense_recs.scores_to_recs(scores, users, avail_ids)
    res = common.evaluate(recs, "two_tower")
    common.section("Evaluation: brute-force top 50 over in-stock items")
    print(f"users without history (padded with 7-day popularity): {st['users_without_model_scores']} of {st['users']}")
    common.print_table([res])

    # ---- ANN (IVF) demo
    common.section("Approximate nearest neighbours: IVF with 32 k-means clusters")
    cent, assign = kmeans(V, 32, 20)
    sizes = np.bincount(assign, minlength=32)
    print(f"cluster sizes: min {sizes.min()}  median {int(np.median(sizes))}  max {sizes.max()}")
    q = uv_eval[has_hist][:400]
    members = [np.flatnonzero(assign == c) for c in range(32)]

    t0 = time.perf_counter()
    exact = [np.argsort(-(V @ qi))[:20] for qi in q]
    brute_ms = (time.perf_counter() - t0) / len(q) * 1000
    print(f"brute force: {brute_ms:.3f} ms/query over {len(V)} items  ({len(q)} queries)")
    ann = []
    for n_probe in (1, 2, 4, 8, 16):
        t0 = time.perf_counter()
        rec_hits, scanned = [], []
        for qi, ex in zip(q, exact):
            probe = np.argsort(-(cent @ qi))[:n_probe]
            cand = np.concatenate([members[c] for c in probe])
            top = cand[np.argsort(-(V[cand] @ qi))[:20]]
            rec_hits.append(len(set(top) & set(ex)) / 20)
            scanned.append(len(cand))
        ms = (time.perf_counter() - t0) / len(q) * 1000
        row = {"n_probe": n_probe, "recall_vs_brute_force@20": float(np.mean(rec_hits)),
               "ms_per_query": ms, "items_scanned": float(np.mean(scanned)),
               "share_scanned": float(np.mean(scanned) / len(V))}
        ann.append(row)
        print(f"probe {n_probe:>2d} clusters  recall@20 vs brute force {row['recall_vs_brute_force@20']:.3f}  "
              f"{ms:.3f} ms/query  scanned {row['share_scanned']:.1%} of items")

    # ---- PCA of item vectors
    Vc = V - V.mean(0)
    _, sv, vt = np.linalg.svd(Vc, full_matrices=False)
    proj = Vc @ vt[:2].T
    var = (sv**2) / (sv**2).sum()
    cat_of = items.set_index("item_id").loc[avail_ids, "category_id"].to_numpy()
    cat_name = cats.set_index("category_id")["name"].to_dict()
    parent = cats.set_index("category_id")["parent_name"].to_dict()
    print(f"\nPCA: first two components explain {var[:2].sum():.1%} of item-vector variance")

    common.save_results("ch07_two_tower", {
        "results": [res],
        "padding": st,
        "loss": losses,
        "ann": ann,
        "brute_force_ms_per_query": brute_ms,
        "n_items_indexed": int(len(V)),
        "pca": {"x": proj[:, 0], "y": proj[:, 1], "category_id": cat_of,
                "category": [cat_name[c] for c in cat_of], "parent": [parent[c] for c in cat_of],
                "explained_variance": var[:2]},
    })


if __name__ == "__main__":
    main()
