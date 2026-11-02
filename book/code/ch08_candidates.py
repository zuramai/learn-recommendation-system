"""Chapter 8 - multi-source candidate generation: what each retriever finds, and the union."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402
from recsys import candidates, evaluate as rs_eval, pipeline  # noqa: E402

KS = [5, 10, 20, 30, 50, 75, 100]


def main() -> None:
    hist = common.history()
    users = common.eval_users()
    labels = common.labels()
    avail = hist.items[hist.items["in_stock"]]
    ok = set(avail["item_id"])

    recent = candidates.recent(hist)
    sources = {
        "recent": (recent, "cand_recent_score"),
        "covisit": (candidates.covisit(hist, recent), "cand_covisit_score"),
        "catpop": (candidates.catpop(hist, avail), "cand_catpop_score"),
    }
    trending = pd.DataFrame({"user_id": users}).merge(candidates.global_trending(hist, avail), how="cross")
    trending["neg_rank"] = -trending["cand_global_pop_rank"]
    sources["global_trending"] = (trending, "neg_rank")

    per_source: dict[str, pd.DataFrame] = {}
    for name, (df, col) in sources.items():
        df = df[df["user_id"].isin(users) & df["item_id"].isin(ok)]
        per_source[name] = df.rename(columns={col: "score"})[["user_id", "item_id", "score"]]

    union = candidates.generate(hist, users)
    union["heuristic"] = pipeline.heuristic_score(union)

    common.section("Retrievers one by one (eval users: %d, label pairs: %d)" % (len(users), len(labels)))
    pairs = labels[["user_id", "item_id"]]
    covered = {}
    for name, df in per_source.items():
        covered[name] = set(zip(df["user_id"], df["item_id"]))
    label_pairs = set(zip(pairs["user_id"], pairs["item_id"]))
    rows = []
    for name, df in per_source.items():
        others = set().union(*(covered[o] for o in covered if o != name))
        found = label_pairs & covered[name]
        rows.append({
            "source": name,
            "cands_per_user": len(df) / len(users),
            "users_with_cands": df["user_id"].nunique() / len(users),
            "recall": rs_eval.candidate_recall(df, labels),
            "pair_recall": len(found) / len(label_pairs),
            "unique_pair_recall": len(found - others) / len(label_pairs),
        })
    union_pairs = set(zip(union["user_id"], union["item_id"]))
    rows.append({
        "source": "UNION (generate)",
        "cands_per_user": len(union) / len(users),
        "users_with_cands": union["user_id"].nunique() / len(users),
        "recall": rs_eval.candidate_recall(union, labels),
        "pair_recall": len(label_pairs & union_pairs) / len(label_pairs),
        "unique_pair_recall": float("nan"),
    })
    table = pd.DataFrame(rows).set_index("source")
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        print(table.round(4).to_string())
    print("\nrecall = mean over users of (their future positives that are in the set);")
    print("pair_recall = same, pooled over all (user, item) positives; unique = found by this source only.")

    common.section("Recall vs list length K (candidates ranked by their own score)")
    curve = {}
    for name, df in per_source.items():
        curve[name] = [rs_eval.metrics(common.top_n(df, "score", 200), labels, k)[f"recall@{k}"] for k in KS]
    curve["union (heuristic blend)"] = [
        rs_eval.metrics(common.top_n(union, "heuristic", 200), labels, k)[f"recall@{k}"] for k in KS
    ]
    curve["union ceiling (all candidates)"] = [rs_eval.candidate_recall(union, labels)] * len(KS)
    print(pd.DataFrame(curve, index=pd.Index(KS, name="K")).round(4).to_string())

    common.section("Union: how many sources proposed each candidate")
    n_src = union["cand_n_sources"].astype(int).value_counts().sort_index()
    lab = union.merge(labels, on=["user_id", "item_id"], how="left")
    hit_rate = lab.assign(hit=lab["label"].notna()).groupby(lab["cand_n_sources"].astype(int))["hit"].mean()
    out = pd.DataFrame({"candidates": n_src, "positive_rate": hit_rate})
    print(out.round(4).to_string())

    common.section("Union ranked by the heuristic blend")
    res = common.evaluate(common.top_n(union, "heuristic"), "union + heuristic blend")
    common.print_table([res])

    common.save_results("ch08_candidates", {
        "eval_users": len(users),
        "label_pairs": len(label_pairs),
        "sources": table.reset_index().to_dict(orient="records"),
        "ks": KS,
        "recall_vs_k": curve,
        "n_sources": {int(k): {"candidates": int(v), "positive_rate": float(hit_rate[k])} for k, v in n_src.items()},
        "results": [res],
    })


if __name__ == "__main__":
    main()
