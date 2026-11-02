"""Stage 1 — candidate generation (a.k.a. retrieval / recall).

Goal: from thousands of items, cheaply pick ~100 per user that *might* be
relevant. Precision doesn't matter here; recall does: anything missed here can
never be recommended. Each retriever covers a different reason to like an item:

  recent     "you looked at this"          — personal, time-decayed history
  covisit    "people who viewed X also viewed Y" — item-item collaborative filtering
  catpop     "popular in categories you like"   — uses clicks AND search history
  global     "trending right now"                — fallback for new/cold users

Each retriever's score is kept as a feature so the ranker learns how much to
trust each source.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import History

HALF_LIFE_DAYS = 7.0  # interest decays: an action 7 days ago counts half
N_RECENT = 20
N_SEEDS = 10  # recent items used as covisit seeds
N_COVISIT = 40
N_TOP_CATEGORIES = 3
N_PER_CATEGORY = 20
N_GLOBAL = 30
COVISIT_WINDOW_DAYS = 30
POPULARITY_WINDOW_DAYS = 7


def _decay(age_days: pd.Series) -> pd.Series:
    return np.power(0.5, age_days / HALF_LIFE_DAYS)


def _top_per_user(df: pd.DataFrame, score: str, n: int) -> pd.DataFrame:
    df = df.sort_values(["user_id", score], ascending=[True, False])
    return df.groupby("user_id").head(n)


def recent(hist: History) -> pd.DataFrame:
    pos = hist.interactions()
    pos["s"] = pos["w"] * _decay(pos["age_days"])
    out = pos.groupby(["user_id", "item_id"], as_index=False)["s"].sum()
    return _top_per_user(out.rename(columns={"s": "cand_recent_score"}), "cand_recent_score", N_RECENT)


def covisitation_matrix(hist: History) -> pd.DataFrame:
    """(item_a, item_b, weight): normalised co-occurrence of positive actions within a session."""
    pos = hist.interactions()
    pos = pos[pos["age_days"] < COVISIT_WINDOW_DAYS]
    per_session = pos.groupby(["session_id", "item_id"], as_index=False)["w"].max()
    pairs = per_session.merge(per_session, on="session_id", suffixes=("_a", "_b"))
    pairs = pairs[pairs["item_id_a"] != pairs["item_id_b"]]
    m = pairs.groupby(["item_id_a", "item_id_b"], as_index=False)["w_b"].sum()
    # Normalise per source item, so very popular seeds don't dominate everything.
    m["w"] = m["w_b"] / m.groupby("item_id_a")["w_b"].transform("sum")
    return m[["item_id_a", "item_id_b", "w"]]


def covisit(hist: History, seeds: pd.DataFrame) -> pd.DataFrame:
    top_seeds = _top_per_user(seeds, "cand_recent_score", N_SEEDS)
    m = covisitation_matrix(hist)
    j = top_seeds.merge(m, left_on="item_id", right_on="item_id_a")
    j["s"] = j["cand_recent_score"] * j["w"]
    out = j.groupby(["user_id", "item_id_b"], as_index=False)["s"].sum()
    out = out.rename(columns={"item_id_b": "item_id", "s": "cand_covisit_score"})
    return _top_per_user(out, "cand_covisit_score", N_COVISIT)


def item_popularity(hist: History, window_days: float) -> pd.Series:
    pos = hist.interactions()
    return pos[pos["age_days"] < window_days].groupby("item_id")["w"].sum()


def category_affinity(hist: History) -> pd.DataFrame:
    """User x category interest from clicks/carts/orders AND searches (both decayed), as a share."""
    pos = hist.interactions().merge(hist.items[["item_id", "category_id"]], on="item_id")
    pos["s"] = pos["w"] * _decay(pos["age_days"])
    se = hist.searched_categories()
    se["s"] = 2.0 * _decay(se["age_days"])  # a search ~ two clicks of intent
    both = pd.concat([pos[["user_id", "category_id", "s"]], se[["user_id", "category_id", "s"]]])
    aff = both.groupby(["user_id", "category_id"], as_index=False)["s"].sum()
    aff["share"] = aff["s"] / aff.groupby("user_id")["s"].transform("sum")
    return aff


def catpop(hist: History, available: pd.DataFrame) -> pd.DataFrame:
    aff = category_affinity(hist)
    aff = aff.sort_values(["user_id", "share"], ascending=[True, False]).groupby("user_id").head(N_TOP_CATEGORIES)
    pop = item_popularity(hist, POPULARITY_WINDOW_DAYS).rename("pop")
    items = available[["item_id", "category_id"]].join(pop, on="item_id").fillna({"pop": 0.0})
    items = items.sort_values("pop", ascending=False).groupby("category_id").head(N_PER_CATEGORY)
    items["pop_norm"] = items["pop"] / items.groupby("category_id")["pop"].transform("max").clip(lower=1e-9)
    j = aff.merge(items, on="category_id")
    j["cand_catpop_score"] = j["share"] * (0.1 + j["pop_norm"])
    return j[["user_id", "item_id", "cand_catpop_score"]]


def global_trending(hist: History, available: pd.DataFrame) -> pd.DataFrame:
    pop = item_popularity(hist, 3.0)
    pop = pop[pop.index.isin(available["item_id"])].nlargest(N_GLOBAL)
    return pd.DataFrame({"item_id": pop.index, "cand_global_pop_rank": np.arange(len(pop))})


def generate(hist: History, user_ids: np.ndarray) -> pd.DataFrame:
    """Union of all retrievers -> one row per (user_id, item_id) with per-source scores."""
    available = hist.items[hist.items["in_stock"]]
    rec = recent(hist)
    sources = [rec, covisit(hist, rec), catpop(hist, available)]
    users = pd.DataFrame({"user_id": user_ids})
    trending = users.merge(global_trending(hist, available), how="cross")
    sources.append(trending)

    cands = users.merge(sources[0], on="user_id")
    for s in sources[1:]:
        cands = cands.merge(s, on=["user_id", "item_id"], how="outer")
    cands = cands[cands["user_id"].isin(user_ids) & cands["item_id"].isin(available["item_id"])]

    # Negative feedback is a hard filter: never re-show an item the user hid.
    hidden = hist.events.loc[hist.events["event_type"] == "hide", ["user_id", "item_id"]].drop_duplicates()
    cands = cands.merge(hidden.assign(_hidden=True), on=["user_id", "item_id"], how="left")
    cands = cands[cands["_hidden"].isna()].drop(columns="_hidden")

    score_cols = ["cand_recent_score", "cand_covisit_score", "cand_catpop_score"]
    cands["cand_n_sources"] = cands[score_cols].notna().sum(axis=1) + cands["cand_global_pop_rank"].notna()
    cands[score_cols] = cands[score_cols].fillna(0.0)
    cands["cand_global_pop_rank"] = cands["cand_global_pop_rank"].fillna(999)
    return cands.reset_index(drop=True)
