"""Turn a dense (users x available items) score matrix into top-50 recommendations.

Shared by the matrix-based chapters (5, 6, 7). Users whose model scores are all
zero (no usable history) -- or who have fewer than 50 positively scored items --
are padded with 7-day popularity, so every evaluated user always gets 50 recs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common
from recsys import candidates


def pop7_vector(item_ids: np.ndarray) -> np.ndarray:
    """7-day popularity of each item id (0 when unseen), scaled to [0, 1]."""
    pop = candidates.item_popularity(common.history(), 7).reindex(item_ids).fillna(0.0).to_numpy()
    return pop / max(pop.max(), 1e-9)


def scores_to_recs(scores: np.ndarray, users: np.ndarray, item_ids: np.ndarray, n: int = common.TOP_N):
    """scores: (len(users), len(item_ids)); only score > 0 counts as 'the model has an opinion'.

    Returns (recs DataFrame, stats dict) where stats describes how much padding was needed.
    """
    scores = np.where(np.isfinite(scores), scores, 0.0)
    flag = scores > 0
    row_max = np.maximum(scores.max(axis=1, keepdims=True), 1e-12)
    pop = pop7_vector(item_ids)[None, :]
    key = flag * 2.0 + np.where(flag, scores / row_max, 0.0) + np.where(flag, 0.0, pop)
    top = np.argsort(-key, axis=1, kind="stable")[:, :n]
    n_model = np.minimum(flag.sum(axis=1), n)
    recs = pd.DataFrame({"user_id": np.repeat(users, n), "item_id": item_ids[top].ravel()})
    stats = {
        "users": int(len(users)),
        "users_without_model_scores": int((n_model == 0).sum()),
        "users_padded_partially": int(((n_model > 0) & (n_model < n)).sum()),
        "share_of_slots_from_popularity_padding": float(1 - n_model.sum() / (n * len(users))),
    }
    return recs, stats
