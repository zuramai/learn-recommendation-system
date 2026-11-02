"""Chapter 2: ranking metrics by hand on a 10-item list, then a random recommender through `evaluate`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

import common
from common import section
from recsys.evaluate import GAIN

res: dict = {}

# A hand-made ranked list: label 0 = not interesting, 1 = click, 2 = add to cart, 3 = order.
ranked = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
rel = [0, 3, 0, 1, 0, 0, 2, 0, 0, 1]
future_positives = {"B": 3, "D": 1, "G": 2, "J": 1, "Z": 2}  # Z was never recommended
n_pos = len(future_positives)
K = 10

section("The ranked list (what the user later did with each item)")
print(f"{'pos':>3} {'item':<5}{'label':>6}{'gain':>6}{'1/log2(pos+2)':>15}{'discounted gain':>17}")
disc, gains, dcg_terms = [], [], []
for i, (it, r) in enumerate(zip(ranked, rel)):
    g = GAIN.get(r, 0.0)
    d = 1 / np.log2(i + 2)
    gains.append(g)
    disc.append(d)
    dcg_terms.append(g * d)
    print(f"{i + 1:>3} {it:<5}{r:>6}{g:>6.1f}{d:>15.4f}{g * d:>17.4f}")
res["discounts"] = disc

section("Precision and recall at each cut-off")
hits = np.array([r > 0 for r in rel])
print(f"future positives in total: {n_pos} (one, 'Z', is never recommended)")
print(f"{'k':>3}{'hits':>6}{'precision@k':>13}{'recall@k':>10}")
pr = []
for k in range(1, K + 1):
    h = int(hits[:k].sum())
    pr.append({"k": k, "hits": h, "precision": h / k, "recall": h / n_pos})
    print(f"{k:>3}{h:>6}{h / k:>13.3f}{h / n_pos:>10.3f}")
res["precision_recall"] = pr

section("Average precision")
ap_terms = []
for k in range(K):
    if hits[k]:
        p = hits[: k + 1].sum() / (k + 1)
        ap_terms.append(p)
        print(f"hit at pos {k + 1}: precision@{k + 1} = {p:.4f}")
ap = sum(ap_terms) / n_pos
print(f"AP = sum / {n_pos} future positives = {sum(ap_terms):.4f} / {n_pos} = {ap:.4f}")
res["ap"] = ap

section("DCG, IDCG and NDCG")
dcg = sum(dcg_terms)
ideal_labels = sorted(future_positives.values(), reverse=True)[:K]
print(f"ideal order of the 5 future positives (labels): {ideal_labels}")
idcg_terms = [GAIN[l] / np.log2(i + 2) for i, l in enumerate(ideal_labels)]
for i, (l, t) in enumerate(zip(ideal_labels, idcg_terms)):
    print(f"  ideal pos {i + 1}: label {l}, gain {GAIN[l]:.1f} / log2({i + 2}) = {t:.4f}")
idcg = sum(idcg_terms)
print(f"DCG  = {dcg:.4f}")
print(f"IDCG = {idcg:.4f}")
print(f"NDCG = DCG / IDCG = {dcg / idcg:.4f}")
res.update(dcg=dcg, idcg=idcg, ndcg=dcg / idcg, idcg_terms=idcg_terms)

section("Same positives, but the order item B pushed to the last slot")
alt = [0, 1, 0, 0, 2, 0, 0, 1, 0, 3]
alt_dcg = sum(GAIN.get(r, 0.0) / np.log2(i + 2) for i, r in enumerate(alt))
print(f"DCG = {alt_dcg:.4f}   NDCG = {alt_dcg / idcg:.4f}   (recall@10 is unchanged: {hits.sum() / n_pos:.3f})")
res["ndcg_swapped"] = alt_dcg / idcg

section("Random recommender through common.evaluate (same code every chapter uses)")
rng = np.random.default_rng(0)
avail = common.available_items()["item_id"].to_numpy()
users = common.eval_users()
rows = [pd.DataFrame({"user_id": u, "item_id": rng.choice(avail, common.TOP_N, replace=False)}) for u in users]
recs = pd.concat(rows, ignore_index=True)
r = common.evaluate(recs, "random")
common.print_table([r])
print(f"\nevaluated users: {len(users)}, available items: {len(avail)}")
res["random"] = r
res["n_eval_users"] = len(users)
res["n_available_items"] = len(avail)
common.save_results("ch02_metrics", res)
