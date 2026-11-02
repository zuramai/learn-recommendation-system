"""Offline ranking metrics.

recall@K       share of the user's future positives (click/cart/order) that appear in the top K
order_recall@K same, counting only items the user actually ordered (closest to revenue)
ndcg@K         graded and position-aware: an order at slot 1 is worth far more than a click at slot 20

All are averaged over users who had at least one positive in the label window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GAIN = {1: 1.0, 2: 3.0, 3: 7.0}  # same as ranker label_gain


def metrics(recs: pd.DataFrame, labels: pd.DataFrame, k: int) -> dict[str, float]:
    """`recs`: user_id, item_id in display order (top K per user). `labels`: user_id, item_id, label."""
    recs = recs.groupby("user_id").head(k)[["user_id", "item_id"]].copy()
    recs["pos"] = recs.groupby("user_id").cumcount()
    hits = labels.merge(recs, on=["user_id", "item_id"], how="left")

    per_user = hits.groupby("user_id").agg(n_pos=("label", "size"), n_hit=("pos", "count"))
    recall = (per_user["n_hit"] / per_user["n_pos"]).mean()

    orders = hits[hits["label"] == 3]
    ou = orders.groupby("user_id").agg(n=("label", "size"), h=("pos", "count"))
    order_recall = (ou["h"] / ou["n"]).mean()

    found = hits.dropna(subset=["pos"])
    dcg = (found["label"].map(GAIN) / np.log2(found["pos"] + 2)).groupby(found["user_id"]).sum()
    ideal_gains = labels.assign(g=labels["label"].map(GAIN)).sort_values(["user_id", "g"], ascending=[True, False])
    ideal_gains = ideal_gains.groupby("user_id").head(k)
    ideal_gains["pos"] = ideal_gains.groupby("user_id").cumcount()
    idcg = (ideal_gains["g"] / np.log2(ideal_gains["pos"] + 2)).groupby(ideal_gains["user_id"]).sum()
    ndcg = (dcg.reindex(idcg.index, fill_value=0.0) / idcg).mean()

    return {f"recall@{k}": recall, f"order_recall@{k}": order_recall, f"ndcg@{k}": ndcg}


def candidate_recall(cands: pd.DataFrame, labels: pd.DataFrame) -> float:
    """Upper bound for any ranker: share of future positives present in the candidate set."""
    m = labels.merge(cands[["user_id", "item_id"]].assign(found=1), on=["user_id", "item_id"], how="left")
    return float(m.groupby("user_id")["found"].apply(lambda s: s.notna().mean()).mean())
