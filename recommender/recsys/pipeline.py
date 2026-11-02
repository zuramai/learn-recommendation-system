"""End-to-end: time split -> candidates -> features -> train -> evaluate -> serve.

Time-based split (never random — random splits let the model peek at the future):

  start ..................... end-14d ... end-7d .... end
  |---- history for TRAIN rows ---|- labels -|
  |------------- history for EVAL rows -----|- labels -|
  |------------------- history for SERVING (recommend) ---|-> now

The model is trained on "what did users do in [end-14d, end-7d) given everything
before end-14d" and evaluated on "[end-7d, end) given everything before end-7d".
`end` is the end of the dataset: the CSV lab's last simulated day, or "now" for
the live database.
"""

from __future__ import annotations

import pandas as pd

from . import candidates, data, evaluate, features, ranker, rerank, schema

LABEL_DAYS = 7
K = 20

# Why an item was retrieved, in plain words (shown in the CLI and the web app's "Why this?").
SOURCES = {
    "recent": "you viewed it",
    "covisit": "viewed together with your items",
    "category_popular": "popular in a category you like",
    "trending": "trending now",
    "same_category": "popular in the same category",
}
_SOURCE_COLS = {"cand_recent_score": "recent", "cand_covisit_score": "covisit", "cand_catpop_score": "category_popular"}


def metric_key(method: str, metric: str) -> str:
    """'lightgbm + rerank', 'ndcg@20' -> 'lightgbm_rerank/ndcg@20' (the gRPC ModelInfo metric keys)."""
    return f"{method.lower().replace(' + ', '_').replace(' ', '_')}/{metric}"


def build_frame(tables: dict[str, pd.DataFrame], cutoff: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Candidates + features as of `cutoff`, joined with labels from the following LABEL_DAYS."""
    labels = data.labels(tables, cutoff, LABEL_DAYS)
    hist = data.History.at(tables, cutoff)
    cands = candidates.generate(hist, labels["user_id"].unique())
    frame = features.build(cands, hist)
    frame = frame.merge(labels, on=["user_id", "item_id"], how="left").fillna({"label": 0})
    return frame, labels


def _top(frame: pd.DataFrame, score: pd.Series) -> pd.DataFrame:
    return frame.assign(score=score).sort_values(["user_id", "score"], ascending=[True, False]).groupby("user_id").head(K)


def heuristic_score(frame: pd.DataFrame) -> pd.Series:
    """Hand-tuned blend of retriever scores (each normalised per user). What you'd ship before ML."""
    s = pd.Series(0.0, index=frame.index)
    for col in ["cand_recent_score", "cand_covisit_score", "cand_catpop_score"]:
        s += frame[col] / frame.groupby("user_id")[col].transform("max").clip(lower=1e-9)
    return s


def train_and_evaluate(tables: dict[str, pd.DataFrame], end: pd.Timestamp, ablation: bool = False):
    train_cutoff = end - pd.Timedelta(days=2 * LABEL_DAYS)
    eval_cutoff = end - pd.Timedelta(days=LABEL_DAYS)
    train_frame, _ = build_frame(tables, train_cutoff)
    eval_frame, eval_labels = build_frame(tables, eval_cutoff)
    model = ranker.train(train_frame)

    eval_scored = eval_frame.assign(score=ranker.score(model, eval_frame))
    popular = eval_frame[eval_frame["cand_global_pop_rank"] < 999]
    results = {
        "global popular": evaluate.metrics(_top(popular, -popular["cand_global_pop_rank"]), eval_labels, K),
        "heuristic blend": evaluate.metrics(_top(eval_frame, heuristic_score(eval_frame)), eval_labels, K),
        "lightgbm ranker": evaluate.metrics(_top(eval_frame, eval_scored["score"]), eval_labels, K),
        "lightgbm + rerank": evaluate.metrics(rerank.rerank(eval_scored, K), eval_labels, K),
    }
    report = {
        "train_rows": len(train_frame),
        "eval_rows": len(eval_frame),
        "eval_users": eval_labels["user_id"].nunique(),
        "candidate_recall": evaluate.candidate_recall(eval_frame, eval_labels),
        "results": pd.DataFrame(results).T,
        "importance": ranker.importance(model),
    }
    if ablation:
        report["ablation"] = group_ablation(train_frame, eval_frame, eval_labels, report["results"].loc["lightgbm ranker"])
    return model, report


def group_ablation(train_frame, eval_frame, eval_labels, full: pd.Series) -> pd.DataFrame:
    """Retrain without each feature group: how much does each group of variables matter?

    Also retrains the *full* model with other random seeds: differences smaller than that
    seed noise are not evidence of anything.
    """
    def run(names: list[str], seed: int = 0) -> dict[str, float]:
        m = ranker.train(train_frame, names, seed)
        return evaluate.metrics(_top(eval_frame, ranker.score(m, eval_frame)), eval_labels, K)

    rows = {}
    for group in dict.fromkeys(f.group for f in schema.FEATURES):
        res = run([f.name for f in schema.FEATURES if f.group != group])
        rows[f"without {group}"] = {k: v - full[k] for k, v in res.items()}
    out = pd.DataFrame(rows).T.sort_values(f"ndcg@{K}")
    seeds = pd.DataFrame([run(schema.FEATURE_NAMES, s) for s in (1, 2, 3)] + [full.to_dict()])
    out.loc["seed noise (±1 std)"] = seeds.std()
    return out


def recommend(
    tables: dict[str, pd.DataFrame], now: pd.Timestamp, model, user_id: int, k: int = 10
) -> tuple[pd.DataFrame, int]:
    """Serve: cutoff = `now`, score this user's candidates, apply business rules.

    Returns (top-k rows after re-ranking, number of candidate rows scored).
    """
    hist = data.History.at(tables, now)
    cands = candidates.generate(hist, [user_id])
    frame = features.build(cands, hist)
    scored = frame.assign(score=ranker.score(model, frame))
    return rerank.rerank(scored, k), len(frame)


def source_of(frame: pd.DataFrame) -> pd.Series:
    """Per row: the retriever with the largest score, or 'trending' when only global popularity found it."""
    scores = frame[list(_SOURCE_COLS)]
    best = scores.idxmax(axis=1).map(_SOURCE_COLS)
    return best.where(scores.max(axis=1) > 0, "trending")


def similar_items(tables: dict[str, pd.DataFrame], now: pd.Timestamp, item_id: int, k: int) -> pd.DataFrame:
    """Product page "similar items": co-visited items first, then popular items of the same category."""
    hist = data.History.at(tables, now)
    empty = pd.DataFrame({"item_id": pd.Series(dtype="int64"), "score": pd.Series(dtype="float64"), "source": pd.Series(dtype="object")})
    if item_id not in set(hist.items["item_id"]):
        return empty
    in_stock = hist.items.loc[hist.items["in_stock"], ["item_id", "category_id"]]
    m = candidates.covisitation_matrix(hist)
    co = m[(m["item_id_a"] == item_id) & (m["item_id_b"] != item_id) & m["item_id_b"].isin(in_stock["item_id"])]
    co = co.sort_values("w", ascending=False).head(k)
    out = pd.DataFrame({"item_id": co["item_id_b"].to_numpy(), "score": co["w"].to_numpy(), "source": "covisit"})

    category = hist.items.loc[hist.items["item_id"] == item_id, "category_id"].iloc[0]
    pop = candidates.item_popularity(hist, 7)
    same = in_stock[(in_stock["category_id"] == category) & (in_stock["item_id"] != item_id) & ~in_stock["item_id"].isin(out["item_id"])]
    same = same.assign(score=same["item_id"].map(pop).fillna(0.0)).sort_values("score", ascending=False, kind="stable")
    fill = pd.DataFrame({"item_id": same["item_id"].to_numpy(), "score": same["score"].to_numpy(), "source": "same_category"})
    out = pd.concat([out, fill.head(k - len(out))], ignore_index=True) if len(out) < k else out
    return out.astype({"item_id": "int64", "score": "float64"}).reset_index(drop=True) if len(out) else empty
