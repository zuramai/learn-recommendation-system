"""Stage 2 — learning-to-rank with LightGBM (LambdaMART).

Each user is a "query group". LambdaMART learns to order items *within* a group
so that higher-label items (order > cart > click > nothing) come first; it
directly optimises NDCG rather than per-row accuracy.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import pandas as pd

from . import schema

PARAMS = dict(
    objective="lambdarank",
    learning_rate=0.05,
    num_leaves=31,
    min_data_in_leaf=50,
    bagging_fraction=0.8,
    bagging_freq=1,
    feature_fraction=0.8,
    # gain for label 0..3 — an order is worth much more than a click
    label_gain=[0, 1, 3, 7],
    lambdarank_truncation_level=20,
    verbose=-1,
    seed=0,
)
NUM_ROUNDS = 300


def train(frame: pd.DataFrame, feature_names: list[str] = schema.FEATURE_NAMES, seed: int = 0) -> lgb.Booster:
    """`frame` = features + `label`. Groups without any positive teach nothing; drop them."""
    has_pos = frame.groupby("user_id")["label"].transform("max") > 0
    frame = frame[has_pos].sort_values("user_id", kind="stable")
    groups = frame.groupby("user_id", sort=False).size().to_numpy()
    ds = lgb.Dataset(frame[feature_names], label=frame["label"], group=groups)
    return lgb.train({**PARAMS, "seed": seed}, ds, num_boost_round=NUM_ROUNDS)


def score(model: lgb.Booster, frame: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict(frame[model.feature_name()]), index=frame.index, name="score")


def importance(model: lgb.Booster) -> pd.Series:
    s = pd.Series(model.feature_importance(importance_type="gain"), index=model.feature_name())
    return (s / s.sum()).sort_values(ascending=False)


def save(model: lgb.Booster, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))


def load(path: Path) -> lgb.Booster:
    if not path.exists():
        raise SystemExit(f"No model at {path} — run `uv run recsys train` first.")
    return lgb.Booster(model_file=str(path))
