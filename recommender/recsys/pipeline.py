"""End-to-end: time split -> candidates -> features -> train -> evaluate -> serve.

Time-based split (never random — random splits let the model peek at the future):

  day 0 ........................ 46 ....... 53 ....... 60
  |---- history for TRAIN rows ---|- labels -|
  |------------- history for EVAL rows -----|- labels -|
  |------------------- history for SERVING (recommend) ---|-> now

The model is trained on "what did users do in days 46-53 given everything up
to 46" and evaluated on "days 53-60 given everything up to 53".
"""

from __future__ import annotations

import pandas as pd

from . import candidates, data, evaluate, features, ranker, rerank, schema

TRAIN_CUTOFF_DAY = 46
EVAL_CUTOFF_DAY = 53
LABEL_DAYS = 7
K = 20


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


def train_and_evaluate(tables: dict[str, pd.DataFrame], meta: data.Meta, ablation: bool = False):
    train_frame, _ = build_frame(tables, meta.day(TRAIN_CUTOFF_DAY))
    eval_frame, eval_labels = build_frame(tables, meta.day(EVAL_CUTOFF_DAY))
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


def recommend(tables: dict[str, pd.DataFrame], meta: data.Meta, model, user_id: int, k: int = 10) -> pd.DataFrame:
    """Serve: cutoff = end of data ("now"), score this user's candidates, apply business rules."""
    hist = data.History.at(tables, meta.day(meta.n_days))
    cands = candidates.generate(hist, [user_id])
    frame = features.build(cands, hist)
    scored = frame.assign(score=ranker.score(model, frame))
    return rerank.rerank(scored, k)
