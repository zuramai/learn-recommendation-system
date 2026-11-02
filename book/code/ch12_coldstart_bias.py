"""Chapter 12 - position bias, user-activity buckets (cold start) and new-item exposure."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402
import ltr_data  # noqa: E402
from recsys import candidates, evaluate as rs_eval, ranker  # noqa: E402

BUCKETS = [("0-2 clicks", 0, 2), ("3-10 clicks", 3, 10), (">10 clicks", 11, 10**9)]


def main() -> None:
    tables = common.load_tables()
    hist = common.history()
    labels = common.labels()
    users = common.eval_users()

    # ---- position bias -------------------------------------------------------------
    common.section("Click-through rate by display position (all impressions before the cutoff)")
    ev = hist.events
    imp = ev[ev["event_type"] == "impression"].groupby("position").size()
    clk = ev[ev["event_type"] == "click"].groupby("position").size()
    ctr = (clk / imp).reindex(range(12))
    pos_df = pd.DataFrame({"impressions": imp, "clicks": clk, "ctr": ctr}).reindex(range(12))
    pos_df.index.name = "position"
    print(pos_df.round(4).to_string())
    print(f"\nposition 0 vs position 11: {ctr[0] / ctr[11]:.1f}x higher CTR")
    print("Part of this is *position bias* (people look at the top) and part is that better items are placed higher.")

    # ---- models --------------------------------------------------------------------
    train, evf = ltr_data.frames()
    model = ranker.train(train)
    lm = ltr_data.top_by(evf, ranker.score(model, evf))

    avail = common.available_items()
    pop7 = candidates.item_popularity(hist, 7)
    ranked = pop7[pop7.index.isin(avail["item_id"])].sort_values(ascending=False).head(common.TOP_N).index
    pop_recs = pd.DataFrame({"user_id": np.repeat(users, len(ranked)), "item_id": np.tile(ranked, len(users))})

    # ---- activity buckets ----------------------------------------------------------
    common.section("Metrics by user activity (clicks before the cutoff)")
    clicks = ev[ev["event_type"] == "click"].groupby("user_id").size()
    prior = pd.Series(clicks.reindex(users).fillna(0).astype(int).to_numpy(), index=users)
    rows = []
    for name, lo, hi in BUCKETS:
        bu = prior.index[(prior >= lo) & (prior <= hi)]
        lab = labels[labels["user_id"].isin(bu)]
        for method, recs in [("pop_7d", pop_recs), ("LambdaMART", lm)]:
            m = rs_eval.metrics(recs[recs["user_id"].isin(bu)], lab, common.K)
            rows.append({"bucket": name, "method": method, "users": len(set(bu) & set(lab["user_id"])), **m})
    bdf = pd.DataFrame(rows).set_index(["bucket", "method"])
    print(bdf.round(4).to_string())
    for name, *_ in BUCKETS:
        a, b = bdf.loc[(name, "LambdaMART"), "ndcg@20"], bdf.loc[(name, "pop_7d"), "ndcg@20"]
        print(f"  {name:<12} LambdaMART / pop_7d ndcg@20 = {a / b:.2f}x")

    # ---- new item exposure ---------------------------------------------------------
    common.section("Share of NEW items (listed < 7 days before the cutoff) in the top 20")
    items = hist.items
    is_new = items.set_index("item_id")["listed_at"] > common.CUTOFF - pd.Timedelta(days=7)
    cat_new = float(is_new[avail["item_id"]].mean())
    exposure = {"catalogue (available items)": cat_new}
    for method, recs in [("pop_7d", pop_recs), ("LambdaMART", lm)]:
        top = recs.groupby("user_id").head(common.K)
        exposure[method] = float(top["item_id"].map(is_new).fillna(False).mean())
    pos_new = labels["item_id"].map(is_new).fillna(False).mean()
    exposure["future positives"] = float(pos_new)
    for k, v in exposure.items():
        print(f"  {k:<28} {v:6.2%}")

    common.save_results("ch12_coldstart_bias", {
        "position": pos_df.reset_index().to_dict(orient="records"),
        "buckets": bdf.reset_index().to_dict(orient="records"),
        "new_item_share": exposure,
    })


if __name__ == "__main__":
    main()
