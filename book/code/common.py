"""Shared data + evaluation for every companion script, so results are comparable across chapters.

Every algorithm in the book sees the same simulated marketplace (seed 7), the
same history (everything before CUTOFF = day 53) and is judged on the same
labels (what users did in the 7 days after CUTOFF).
"""

from __future__ import annotations

import json
import pickle
import sys
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys import data, evaluate as rs_eval, schema
from recsys.simulate import SimConfig, simulate

BOOK = Path(__file__).resolve().parent.parent
RESULTS = BOOK / "results"
CACHE = BOOK / "build" / "sim.pkl"

END = pd.Timestamp(SimConfig().start_date) + pd.Timedelta(days=60)
CUTOFF = END - pd.Timedelta(days=7)
K = 20
TOP_N = 50  # every method returns its top 50 per user


@cache
def load_tables() -> dict[str, pd.DataFrame]:
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    tables = simulate(SimConfig())
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps(tables))
    return tables


@cache
def history() -> data.History:
    return data.History.at(load_tables(), CUTOFF)


@cache
def labels() -> pd.DataFrame:
    return data.labels(load_tables(), CUTOFF, 7)


def eval_users() -> np.ndarray:
    return np.sort(labels()["user_id"].unique())


def available_items() -> pd.DataFrame:
    """Items listed and in stock at CUTOFF (the only things we may recommend)."""
    items = history().items
    return items[items["in_stock"]].reset_index(drop=True)


def interaction_matrix(hist: data.History) -> tuple[sp.csr_matrix, pd.Index, pd.Index]:
    """User x item matrix of summed EVENT_WEIGHT (click 1, cart 3, order 5)."""
    pos = hist.interactions()
    user_index = pd.Index(np.sort(pos["user_id"].unique()), name="user_id")
    item_index = pd.Index(np.sort(pos["item_id"].unique()), name="item_id")
    agg = pos.groupby(["user_id", "item_id"])["w"].sum().reset_index()
    m = sp.csr_matrix(
        (agg["w"].to_numpy(), (user_index.get_indexer(agg["user_id"]), item_index.get_indexer(agg["item_id"]))),
        shape=(len(user_index), len(item_index)),
    )
    return m, user_index, item_index


def top_n(scores: pd.DataFrame, score_col: str = "score", n: int = TOP_N) -> pd.DataFrame:
    """(user_id, item_id, score) rows -> top n per user, in rank order."""
    s = scores.sort_values(["user_id", score_col], ascending=[True, False], kind="stable")
    return s.groupby("user_id").head(n)[["user_id", "item_id"]].reset_index(drop=True)


def evaluate(recs: pd.DataFrame, method: str) -> dict[str, float]:
    """`recs`: user_id, item_id in rank order (top 50 per user). Same metrics for every chapter."""
    lab = labels()
    out = {"method": method, **rs_eval.metrics(recs, lab, K)}
    for k in (5, 10, 50):
        out[f"recall@{k}"] = rs_eval.metrics(recs, lab, k)[f"recall@{k}"]
    top = recs.groupby("user_id").head(K)
    out[f"coverage@{K}"] = top["item_id"].nunique() / len(available_items())
    return out


def save_results(chapter: str, results: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{chapter}.json"
    path.write_text(json.dumps(results, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    raise TypeError(type(o))


METRIC_COLS = ["recall@5", "recall@10", "recall@20", "recall@50", "order_recall@20", "ndcg@20", "coverage@20"]


def print_table(results: list[dict] | dict) -> None:
    """Aligned metrics table; this stdout is what the book quotes."""
    rows = results if isinstance(results, list) else [results]
    df = pd.DataFrame(rows).set_index("method")
    cols = [c for c in METRIC_COLS if c in df.columns] + [c for c in df.columns if c not in METRIC_COLS]
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        print(df[cols].round(4).to_string())
    sys.stdout.flush()


def section(title: str) -> None:
    print(f"\n=== {title} ===")


__all__ = [
    "BOOK", "CUTOFF", "END", "K", "TOP_N", "available_items", "eval_users", "evaluate", "history",
    "interaction_matrix", "labels", "load_tables", "print_table", "save_results", "schema", "section", "top_n",
]  # fmt: skip
