"""Candidate+feature frames shared by chapters 9-12 (built once, cached in build/).

train frame: cutoff = CUTOFF - 7d, labels from the following 7 days
eval  frame: cutoff = CUTOFF,      labels = common.labels()
This mirrors recsys.pipeline.train_and_evaluate(tables, END).
"""

from __future__ import annotations

import pickle
import sys
from functools import cache
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402
from recsys import pipeline  # noqa: E402

_CACHE = common.BOOK / "build" / "frames.pkl"


@cache
def frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train_frame, eval_frame), each with a `label` column."""
    if _CACHE.exists():
        with _CACHE.open("rb") as f:
            return pickle.load(f)
    tables = common.load_tables()
    train, _ = pipeline.build_frame(tables, common.CUTOFF - pd.Timedelta(days=7))
    ev, _ = pipeline.build_frame(tables, common.CUTOFF)
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE.open("wb") as f:
        pickle.dump((train, ev), f)
    return train, ev


def top_by(frame: pd.DataFrame, score) -> pd.DataFrame:
    """Top-50 per user by `score` (a Series aligned to frame.index)."""
    return common.top_n(frame.assign(score=score), "score")
