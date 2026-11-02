"""Chapter 3: popularity baselines, from random to 'popular in the categories I like'."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

import common
from common import TOP_N, section
from recsys import candidates

HALF_LIFE = 7.0
hist = common.history()
avail = common.available_items()
users = common.eval_users()
item_ids = avail["item_id"].to_numpy()
pos = hist.interactions()


def per_item(values: pd.Series) -> np.ndarray:
    return values.reindex(item_ids).fillna(0.0).to_numpy()


def recs_from_scores(S: np.ndarray, name: str) -> pd.DataFrame:
    """S: (n_users, n_items) or (n_items,) scores -> top-50 rows per user. Ties broken by item order (stable)."""
    if S.ndim == 1:
        S = np.broadcast_to(S, (len(users), len(item_ids)))
    top = np.argsort(-S, axis=1, kind="stable")[:, :TOP_N]
    return pd.DataFrame({"user_id": np.repeat(users, TOP_N), "item_id": item_ids[top].ravel()})


section("Building the baselines")
scores: dict[str, np.ndarray] = {}

rng = np.random.default_rng(0)
scores["random"] = rng.random((len(users), len(item_ids)))

# book:start ch03_scores
pop_all = per_item(pos.groupby("item_id")["w"].sum())
scores["pop_all_time"] = pop_all

pop7 = per_item(candidates.item_popularity(hist, 7))
scores["pop_7d"] = pop7

w_decay = pos["w"] * np.power(0.5, pos["age_days"] / HALF_LIFE)
pop_decay = per_item(w_decay.groupby(pos["item_id"]).sum())
scores["pop_decay"] = pop_decay
# book:end ch03_scores

# Bayesian average: shrink each item's rating toward the global mean; m = mean rating_count.
items = avail.set_index("item_id")
# book:start ch03_bayes
v, R = items["rating_count"].to_numpy(float), items["rating_avg"].to_numpy(float)
m = v.mean()
C = float((R * v).sum() / v.sum())
bayes = v / (v + m) * R + m / (v + m) * C
# book:end ch03_bayes
scores["bayesian_rating"] = bayes
print(f"m (mean rating_count) = {m:.1f}, C (global mean rating) = {C:.3f}")
ex = pd.DataFrame({"rating_avg": R, "rating_count": v, "bayes": bayes}, index=item_ids)
print("\nSmoothing examples (few ratings are pulled toward C, many ratings are trusted):")
print(ex.sort_values("rating_count").iloc[[0, 10, 1000, -1]].round(3).to_string())

# Popular in my categories: affinity share (user x category) times per-category normalised popularity.
aff = candidates.category_affinity(hist)
cats = np.sort(hist.categories["category_id"].unique())
cat_idx = {c: i for i, c in enumerate(cats)}
A = np.zeros((len(users), len(cats)))
uidx = {u: i for i, u in enumerate(users)}
a = aff[aff["user_id"].isin(uidx)]
A[a["user_id"].map(uidx), a["category_id"].map(cat_idx)] = a["share"].to_numpy()
item_cat = avail["category_id"].map(cat_idx).to_numpy()
P = np.zeros((len(cats), len(item_ids)))
cat_max = pd.Series(pop7).groupby(item_cat).transform("max").to_numpy().clip(min=1e-9)
P[item_cat, np.arange(len(item_ids))] = pop7 / cat_max
# Users with no history get an all-zero row -> the tiny pop_7d term makes them fall back to global trending.
# book:start ch03_personal
scores["pop_in_my_categories"] = A @ P + 1e-6 * pop7 / max(pop7.max(), 1e-9)
# book:end ch03_personal
n_cold = int((A.sum(axis=1) == 0).sum())
print(f"\neval users with no category signal (fall back to pop_7d): {n_cold} of {len(users)}")

section("Results (top 50 per user, same labels and metrics for everyone)")
rows, recs_by = [], {}
for name, S in scores.items():
    recs = recs_from_scores(S, name)
    assert recs.groupby("user_id").size().eq(TOP_N).all()
    recs_by[name] = recs
    rows.append(common.evaluate(recs, name))
common.print_table(rows)

section("Top 5 items of the global popularity methods")
cols = ["item_id", "title", "price"]
for name in ("pop_all_time", "pop_7d", "pop_decay"):
    idx = np.argsort(-scores[name], kind="stable")[:5]
    print(f"\n{name}:")
    print(avail.iloc[idx][cols].to_string(index=False))

section("Decay weights (half-life 7 days)")
ages = [0, 1, 3, 7, 14, 21, 30, 53]
for d in ages:
    print(f"age {d:>2} days -> weight {0.5 ** (d / HALF_LIFE):.4f}")

common.save_results("ch03_popularity", {
    "results": rows,
    "half_life_days": HALF_LIFE,
    "bayes_m": m, "bayes_C": C,
    "decay_curve": {"age_days": list(range(0, 61)), "weight": [0.5 ** (d / HALF_LIFE) for d in range(61)]},
    "n_eval_users_without_category_signal": n_cold,
})
