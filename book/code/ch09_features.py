"""Chapter 9 - one feature row in full, positives vs negatives, and the leakage demo."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402
import ltr_data  # noqa: E402
from recsys import pipeline, ranker, schema  # noqa: E402

DIST_FEATURES = [
    "cand_recent_score", "item_ctr_30d", "uc_click_share_30d",
    "item_clicks_7d", "item_rating_count", "price_ratio_to_user",
]  # fmt: skip


def label_window_clicks(frame: pd.DataFrame, start: pd.Timestamp) -> pd.Series:
    """The LEAK: how often the user clicked this item inside [start, start+7d) - the future."""
    ev = common.load_tables()["events"]
    win = ev[(ev["ts"] >= start) & (ev["ts"] < start + pd.Timedelta(days=7)) & (ev["event_type"] == "click")]
    n = win.groupby(["user_id", "item_id"]).size().rename("n").reset_index()
    m = frame[["user_id", "item_id"]].merge(n, on=["user_id", "item_id"], how="left")
    return pd.Series(m["n"].fillna(0).to_numpy(), index=frame.index)


def main() -> None:
    train, ev = ltr_data.frames()

    common.section("Frames")
    print(f"train frame: {len(train):>7} rows, {train['user_id'].nunique()} users, "
          f"{int((train['label'] > 0).sum())} positive rows  (cutoff {common.CUTOFF - pd.Timedelta(days=7)})")
    print(f"eval  frame: {len(ev):>7} rows, {ev['user_id'].nunique()} users, "
          f"{int((ev['label'] > 0).sum())} positive rows  (cutoff {common.CUTOFF})")
    print(f"number of features: {len(schema.FEATURES)} in {len({f.group for f in schema.FEATURES})} groups")

    common.section("One candidate row, every feature (user 42, best heuristic score)")
    u = ev[ev["user_id"] == 42].copy()
    u["h"] = pipeline.heuristic_score(u)
    row = u.sort_values("h", ascending=False).iloc[0]
    print(f"user_id={int(row['user_id'])} item_id={int(row['item_id'])} category_id={int(row['category_id'])} "
          f"shop_id={int(row['shop_id'])} label={int(row['label'])}")
    width = max(len(f.name) for f in schema.FEATURES)
    last_group = None
    one_row = []
    for f in schema.FEATURES:
        if f.group != last_group:
            print(f"  [{f.group}]")
            last_group = f.group
        v = row[f.name]
        print(f"    {f.name:<{width}} = {v:>10.3f}   {f.description}")
        one_row.append({"feature": f.name, "group": f.group, "value": float(v), "description": f.description})

    common.section("Positives vs negatives (eval frame): mean of selected features")
    pos = ev["label"] > 0
    stats = []
    for name in DIST_FEATURES:
        stats.append({
            "feature": name,
            "mean_positive": float(ev.loc[pos, name].mean()),
            "mean_negative": float(ev.loc[~pos, name].mean()),
            "median_positive": float(ev.loc[pos, name].median()),
            "median_negative": float(ev.loc[~pos, name].median()),
        })
    print(pd.DataFrame(stats).set_index("feature").round(3).to_string())
    print(f"base rate of positives among candidates: {pos.mean():.4f}")

    common.section("Leakage demo: add the user's clicks on the item *inside the label window*")
    t_cut = common.CUTOFF - pd.Timedelta(days=7)
    train_l = train.assign(leaky_future_clicks=label_window_clicks(train, t_cut))
    ev_l = ev.assign(leaky_future_clicks=label_window_clicks(ev, common.CUTOFF))
    honest_names = schema.FEATURE_NAMES
    leaky_names = honest_names + ["leaky_future_clicks"]

    honest = ranker.train(train, honest_names)
    leaky = ranker.train(train_l, leaky_names)
    res_h = common.evaluate(ltr_data.top_by(ev, ranker.score(honest, ev)), "honest ranker")
    res_l = common.evaluate(ltr_data.top_by(ev_l, ranker.score(leaky, ev_l)), "leaky ranker")
    common.print_table([res_h, res_l])
    imp = ranker.importance(leaky)
    print(f"\nleaky_future_clicks share of total gain: {imp['leaky_future_clicks']:.1%} "
          f"(rank {list(imp.index).index('leaky_future_clicks') + 1} of {len(imp)})")
    print(f"ndcg@20 inflation: {res_l['ndcg@20'] / res_h['ndcg@20']:.2f}x")
    print("The leaky model looks brilliant offline and would be useless live: in production the future is not available.")

    common.save_results("ch09_features", {
        "n_features": len(schema.FEATURES),
        "train_rows": len(train), "eval_rows": len(ev),
        "positive_rate": float(pos.mean()),
        "example_row": one_row,
        "dist_stats": stats,
        "dist_features": DIST_FEATURES,
        "results": [res_h, res_l],
        "leaky_gain_share": float(imp["leaky_future_clicks"]),
    })


if __name__ == "__main__":
    main()
