"""Chapter 10 - pointwise vs LambdaMART vs hand-tuned blend, importance, SHAP, ablation."""

import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402
import ltr_data  # noqa: E402
from recsys import evaluate as rs_eval, pipeline, ranker, schema  # noqa: E402


def main() -> None:
    tables = common.load_tables()
    train, ev = ltr_data.frames()
    labels = common.labels()
    names = schema.FEATURE_NAMES

    common.section("Three ways to order the same candidates")
    # 1. pointwise: predict P(label > 0) per row, ignore the list structure
    pw = lgb.train(
        {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50,
         "bagging_fraction": 0.8, "bagging_freq": 1, "feature_fraction": 0.8, "verbose": -1, "seed": 0},
        lgb.Dataset(train[names], label=(train["label"] > 0).astype(int)),
        num_boost_round=ranker.NUM_ROUNDS,
    )  # fmt: skip
    pw_score = pd.Series(pw.predict(ev[names]), index=ev.index)
    # 2. LambdaMART (the repo's ranker)
    model = ranker.train(train)
    lm_score = ranker.score(model, ev)
    # 3. hand-tuned blend
    blend = pipeline.heuristic_score(ev)

    res = [
        common.evaluate(ltr_data.top_by(ev, blend), "heuristic blend"),
        common.evaluate(ltr_data.top_by(ev, pw_score), "pointwise LightGBM"),
        common.evaluate(ltr_data.top_by(ev, lm_score), "LambdaMART"),
    ]
    common.print_table(res)
    print(f"\ncandidate recall ceiling: {rs_eval.candidate_recall(ev, labels):.4f}")

    common.section("Seed check: pointwise vs LambdaMART over 4 seeds (ndcg@20)")
    seed_rows = []
    for seed in range(4):
        pw_s = lgb.train({**pw.params, "seed": seed}, lgb.Dataset(train[names], label=(train["label"] > 0).astype(int)),
                         num_boost_round=ranker.NUM_ROUNDS)  # fmt: skip
        lm_s = ranker.train(train, seed=seed)
        seed_rows.append({
            "seed": seed,
            "pointwise": common.evaluate(ltr_data.top_by(ev, pd.Series(pw_s.predict(ev[names]), index=ev.index)), "p")["ndcg@20"],
            "lambdamart": common.evaluate(ltr_data.top_by(ev, ranker.score(lm_s, ev)), "l")["ndcg@20"],
        })  # fmt: skip
    seeds = pd.DataFrame(seed_rows).set_index("seed")
    print(seeds.round(4).to_string())
    print("mean:", seeds.mean().round(4).to_dict(), " std:", seeds.std().round(4).to_dict())
    print("At this data size pointwise is marginally ahead (about one seed-std): with ~100 candidates per user and few")
    print("positives, the list-aware objective has little extra to exploit. Both beat the heuristic blend by a wide margin.")

    common.section("Feature importance (share of total gain), top 15")
    imp = ranker.importance(model)
    desc = {f.name: f for f in schema.FEATURES}
    imp_rows = [{"feature": n, "group": desc[n].group, "gain_share": float(v)} for n, v in imp.items()]
    for r in imp_rows[:15]:
        print(f"  {r['feature']:<24} {r['group']:<14} {r['gain_share']:.3f}")
    by_group = pd.DataFrame(imp_rows).groupby("group")["gain_share"].sum().sort_values(ascending=False)
    print("\nGain share by feature group:")
    print(by_group.round(3).to_string())

    common.section("SHAP breakdown: first recommendation for user 42")
    recs, n_cands = pipeline.recommend(tables, common.END, model, 42, 10)
    contrib = ranker.contributions(model, recs, top=8)
    first = recs.iloc[0]
    print(f"candidates scored: {n_cands}; top item_id={int(first['item_id'])} score={first['score']:.3f}")
    bias = float(model.predict(recs[names].iloc[:1], pred_contrib=True)[0, -1])
    shap_rows = []
    for f, v, c in contrib[0]:
        print(f"  {f:<24} value={v:>10.3f}  contribution={c:+.3f}   {desc[f].description}")
        shap_rows.append({"feature": f, "value": v, "contribution": c})
    print(f"  (bias / base score: {bias:+.3f}; contributions + bias = raw score)")
    full = model.predict(recs[names].iloc[:1], pred_contrib=True)[0]
    print(f"  sum of all contributions + bias = {full.sum():.3f}  (model score {first['score']:.3f})")
    print("\nTop 10 recommended item ids for user 42:", [int(i) for i in recs["item_id"]])

    common.section("Ablation: retrain without one feature group (delta vs full model)")
    full_metrics = pd.Series(rs_eval.metrics(ltr_data.top_by(ev, lm_score).groupby("user_id").head(20), labels, 20))
    abl = pipeline.group_ablation(train, ev, labels, full_metrics)
    with pd.option_context("display.width", 140):
        print(abl.round(4).to_string())
    print("\nThe last row is the std over 4 seeds of the *same* model: deltas smaller than it are noise.")

    common.save_results("ch10_ltr", {
        "results": res,
        "seed_check": seeds.reset_index().to_dict(orient="records"),
        "candidate_recall": rs_eval.candidate_recall(ev, labels),
        "importance": imp_rows,
        "group_gain_share": by_group.to_dict(),
        "shap_user42": {"item_id": int(first["item_id"]), "score": float(first["score"]), "bias": bias,
                        "contributions": shap_rows},
        "ablation": {k: {m: float(x) for m, x in row.items()} for k, row in abl.iterrows()},
        "full_ndcg@20": float(full_metrics["ndcg@20"]),
    })


if __name__ == "__main__":
    main()
