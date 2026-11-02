"""Command line: `uv run recsys {schema,simulate,train,recommend}`."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from . import data, pipeline, ranker, schema, simulate

DATA_DIR = Path("data")
MODEL_PATH = Path("artifacts/ranker.txt")
SOURCE_LABEL = {
    "cand_recent_score": "you viewed it",
    "cand_covisit_score": "viewed together with your items",
    "cand_catpop_score": "popular in a category you like",
}


def cmd_schema(_: argparse.Namespace) -> None:
    print(schema.describe())


def cmd_simulate(args: argparse.Namespace) -> None:
    cfg = simulate.SimConfig(n_users=args.users, n_items=args.items, seed=args.seed)
    t = time.time()
    tables = simulate.simulate(cfg)
    data.save(tables, cfg, DATA_DIR)
    print(f"Simulated {cfg.n_days} days in {time.time() - t:.1f}s -> {DATA_DIR}/")
    for name, df in tables.items():
        print(f"  {name:<11} {len(df):>8,} rows")
    print("\nEvent funnel:")
    print(tables["events"]["event_type"].value_counts().to_string())


def cmd_train(args: argparse.Namespace) -> None:
    tables, meta = data.load(DATA_DIR)
    t = time.time()
    model, rep = pipeline.train_and_evaluate(tables, meta, ablation=args.ablation)
    ranker.save(model, MODEL_PATH)
    pd.set_option("display.width", 120)
    print(f"Trained in {time.time() - t:.1f}s; model -> {MODEL_PATH}")
    print(f"train rows {rep['train_rows']:,} | eval rows {rep['eval_rows']:,} | eval users {rep['eval_users']:,}")
    print(f"\nCandidate recall (ceiling for any ranker): {rep['candidate_recall']:.3f}\n")
    print(rep["results"].round(4).to_string())
    print("\nTop features by gain:")
    print(rep["importance"].head(15).round(3).to_string())
    if "ablation" in rep:
        print("\nAblation (metric change when a feature group is removed; more negative = more important):")
        print(rep["ablation"].round(4).to_string())


def cmd_recommend(args: argparse.Namespace) -> None:
    tables, meta = data.load(DATA_DIR)
    model = ranker.load(MODEL_PATH)
    items = tables["items"].merge(tables["categories"][["category_id", "name"]], on="category_id")
    items = items.set_index("item_id")

    ev = tables["events"]
    mine = ev[(ev["user_id"] == args.user) & ev["event_type"].isin(["click", "add_to_cart", "order"])]
    print(f"User {args.user} — last actions:")
    for r in mine.tail(8).itertuples():
        it = items.loc[r.item_id]
        print(f"  {r.ts:%m-%d %H:%M}  {r.event_type:<12} {it['title']:<36} [{it['name']}]")
    se = tables["searches"]
    q = se.loc[se["user_id"] == args.user, "query"].tail(5).tolist()
    print(f"Recent searches: {q}\n")

    recs = pipeline.recommend(tables, meta, model, args.user, args.k)
    print(f"Top {args.k} recommendations:")
    for i, r in enumerate(recs.itertuples(), 1):
        it = items.loc[r.item_id]
        why = max(SOURCE_LABEL, key=lambda c: getattr(r, c))
        why = SOURCE_LABEL[why] if getattr(r, why) > 0 else "trending now"
        print(f"  {i:>2}. {it['title']:<36} {it['name']:<18} Rp{it['price']:>10,.0f}  score {r.score:6.2f}  ({why})")


def main() -> None:
    p = argparse.ArgumentParser(prog="recsys", description=__doc__)
    sub = p.add_subparsers(required=True)
    sub.add_parser("schema", help="print raw tables and features").set_defaults(fn=cmd_schema)
    s = sub.add_parser("simulate", help="generate a synthetic marketplace into data/")
    s.add_argument("--users", type=int, default=1500)
    s.add_argument("--items", type=int, default=2500)
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(fn=cmd_simulate)
    t = sub.add_parser("train", help="train ranker, evaluate vs baselines, save model")
    t.add_argument("--ablation", action="store_true", help="also retrain without each feature group")
    t.set_defaults(fn=cmd_train)
    r = sub.add_parser("recommend", help="show recommendations for one user")
    r.add_argument("user", type=int)
    r.add_argument("-k", type=int, default=10)
    r.set_defaults(fn=cmd_recommend)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
