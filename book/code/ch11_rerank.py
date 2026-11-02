"""Chapter 11 - MMR diversity re-ranking and the repo's rule-based caps, on top of LambdaMART scores."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402
import ltr_data  # noqa: E402
from recsys import ranker, rerank  # noqa: E402

LAMBDAS = [1.0, 0.9, 0.7, 0.5, 0.3]


def similarity(cat: np.ndarray, parent: np.ndarray) -> np.ndarray:
    """1 if same category, 0.5 if same parent category, else 0."""
    same_cat = cat[:, None] == cat[None, :]
    same_par = parent[:, None] == parent[None, :]
    return np.where(same_cat, 1.0, np.where(same_par, 0.5, 0.0))


def mmr(scored: pd.DataFrame, parent_of: dict[int, str], lam: float, n: int = common.TOP_N) -> pd.DataFrame:
    """Greedy MMR per user: pick argmax  lam * rel - (1 - lam) * max_sim_to_selected."""
    rows = []
    for uid, g in scored.groupby("user_id", sort=True):
        s = g["score"].to_numpy()
        rel = (s - s.min()) / max(s.max() - s.min(), 1e-9)
        cat = g["category_id"].to_numpy()
        par = np.array([parent_of[c] for c in cat])
        sim = similarity(cat, par)
        chosen: list[int] = []
        max_sim = np.zeros(len(g))
        avail = np.ones(len(g), dtype=bool)
        for _ in range(min(n, len(g))):
            mmr_score = np.where(avail, lam * rel - (1 - lam) * max_sim, -np.inf)
            j = int(np.argmax(mmr_score))
            chosen.append(j)
            avail[j] = False
            max_sim = np.maximum(max_sim, sim[j])
        sel = g.iloc[chosen]
        rows.append(sel[["user_id", "item_id", "category_id"]])
    return pd.concat(rows, ignore_index=True)


def diversity(recs: pd.DataFrame, cat_of: pd.Series) -> float:
    top = recs.groupby("user_id").head(common.K)
    return float(top["item_id"].map(cat_of).groupby(top["user_id"]).nunique().mean())


def main() -> None:
    train, ev = ltr_data.frames()
    model = ranker.train(train)
    scored = ev.assign(score=ranker.score(model, ev))
    cats = common.history().categories
    parent_of = dict(zip(cats["category_id"], cats["parent_name"]))
    cat_of = ev.drop_duplicates("item_id").set_index("item_id")["category_id"]

    common.section("MMR:  pick argmax  lambda * relevance - (1 - lambda) * max similarity to already chosen")
    rows = []
    for lam in LAMBDAS:
        recs = mmr(scored, parent_of, lam)
        res = common.evaluate(recs, f"MMR lambda={lam}")
        res["distinct_categories@20"] = diversity(recs, cat_of)
        rows.append(res)

    common.section("Repo rule-based re-ranker (rerank.rerank: bought-durable filter + caps 8/category, 4/shop)")
    ruled = rerank.rerank(scored, common.TOP_N)
    res = common.evaluate(ruled, "rerank.rerank caps")
    res["distinct_categories@20"] = diversity(ruled, cat_of)
    rows.append(res)

    only_filter = scored[~((scored["ui_orders"] > 0) & (scored["item_is_consumable"] == 0))]
    res = common.evaluate(ltr_data.top_by(only_filter, only_filter["score"]), "already-bought filter only")
    res["distinct_categories@20"] = diversity(ltr_data.top_by(only_filter, only_filter["score"]), cat_of)
    rows.append(res)

    common.section("Results (same eval users and labels as every chapter)")
    common.print_table(rows)
    base = rows[0]
    print(f"\nLambdaMART alone: ndcg@20 {base['ndcg@20']:.4f}, {base['distinct_categories@20']:.2f} distinct categories in the top 20")
    for r in rows[1:len(LAMBDAS)]:
        print(f"  {r['method']:<18} ndcg change {r['ndcg@20'] - base['ndcg@20']:+.4f}, "
              f"distinct categories {r['distinct_categories@20'] - base['distinct_categories@20']:+.2f}")

    common.save_results("ch11_rerank", {"lambdas": LAMBDAS, "results": rows})


if __name__ == "__main__":
    main()
