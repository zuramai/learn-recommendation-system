"""Chapter 13: where does one Recommend call spend its time?

Times each stage of `pipeline.recommend` on the simulated marketplace (same code
path as the gRPC service, cutoff = END), median of 7 runs per stage, for three
users with different amounts of history. Runs offline: no database needed.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402
from recsys import candidates, data, features, ranker, rerank  # noqa: E402
from recsys.simulate import SimConfig  # noqa: E402

REPEATS = 7


def timed(fn, *args):
    times, out = [], None
    for _ in range(REPEATS):
        t = time.perf_counter()
        out = fn(*args)
        times.append((time.perf_counter() - t) * 1000)
    return out, float(np.median(times))



def main() -> None:
    tables = common.load_tables()
    from recsys import pipeline

    train_frame, _ = pipeline.build_frame(tables, common.CUTOFF - (common.END - common.CUTOFF))
    model = ranker.train(train_frame)

    pos = data.History.at(tables, common.END).interactions()
    counts = pos.groupby("user_id").size().sort_values()
    users = {"light": int(counts.index[len(counts) // 10]), "median": int(counts.index[len(counts) // 2]),
             "heavy": int(counts.index[-10])}  # fmt: skip

    common.section("Stage timings for one request (median of 7 runs, milliseconds)")
    print(f"catalogue: {len(tables['items']):,} items, {len(tables['events']):,} events, cutoff = END")
    rows = []
    for label, uid in users.items():
        hist, t_hist = timed(data.History.at, tables, common.END)
        _, t_covisit = timed(candidates.covisitation_matrix, hist)
        _, t_items = timed(features.item_features, hist)
        cands, t_gen = timed(candidates.generate, hist, [uid])
        frame, t_feat = timed(features.build, cands, hist)
        scored, t_score = timed(lambda f: f.assign(score=ranker.score(model, f)), frame)
        _, t_rerank = timed(rerank.rerank, scored, 20)
        _, t_shap = timed(ranker.contributions, model, scored.head(20))
        total = t_hist + t_gen + t_feat + t_score + t_rerank + t_shap
        rows.append({"user": label, "user_id": uid, "interactions": int(counts[uid]), "candidates": len(frame),
                     "history_at_ms": t_hist, "generate_ms": t_gen, "covisit_matrix_ms": t_covisit,
                     "features_build_ms": t_feat, "item_features_ms": t_items, "score_ms": t_score,
                     "rerank_ms": t_rerank, "shap_ms": t_shap, "total_ms": total})  # fmt: skip

    header = f"{'user':<7}{'inter.':>7}{'cands':>7}{'History.at':>12}{'generate':>10}{'features':>10}{'score':>8}{'rerank':>8}{'shap':>7}{'total':>8}"
    print(header)
    for r in rows:
        print(f"{r['user']:<7}{r['interactions']:>7}{r['candidates']:>7}{r['history_at_ms']:>12.0f}{r['generate_ms']:>10.0f}"
              f"{r['features_build_ms']:>10.0f}{r['score_ms']:>8.0f}{r['rerank_ms']:>8.0f}{r['shap_ms']:>7.0f}{r['total_ms']:>8.0f}")  # fmt: skip

    common.section("Inside features.build: every table is aggregated for ALL users/items, then joined")
    hist = data.History.at(tables, common.END)
    parts = {}
    for fn in (features.item_features, features.user_features, features.user_item_features,
               features.user_category_features, features.user_shop_features, features.user_price_level):
        _, parts[fn.__name__] = timed(fn, hist)
    item_level = parts["item_features"]
    user_level = sum(v for k, v in parts.items() if k != "item_features")
    for name, ms in parts.items():
        scope = "item-level (same for every user)" if name == "item_features" else "all users' events"
        print(f"{name:<24}{ms:>6.0f} ms   {scope}")
    print(f"covisitation_matrix     {rows[0]['covisit_matrix_ms']:>6.0f} ms   item-level (inside generate)")
    share = np.mean([(r["generate_ms"] + r["features_build_ms"]) / r["total_ms"] for r in rows])
    print(f"\ngenerate + features.build = {share:.0%} of the request on average.")
    print(f"Item-level work (item_features + covisitation_matrix): {item_level + rows[0]['covisit_matrix_ms']:.0f} ms;")
    print(f"user-side aggregations computed over every user, though one user asked: {user_level:.0f} ms.")
    print("Caching item-level frames saves the first number; filtering events to the requesting")
    print("user before the user-side aggregations is needed to remove the second.")
    common.save_results("ch13_profile", {"repeats": REPEATS, "rows": rows, "generate_plus_features_share": share,
                                         "features_build_parts_ms": parts, "item_level_ms": item_level + rows[0]["covisit_matrix_ms"],
                                         "user_side_all_users_ms": user_level})  # fmt: skip


if __name__ == "__main__":
    main()
