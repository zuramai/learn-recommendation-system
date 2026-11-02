"""Command line: `uv run recsys <command>`.

Offline CSV lab: schema, simulate, train, recommend.
Live marketplace: seed-db, serve, retrain, gen-proto.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from . import data, pipeline, ranker, schema, simulate

DATA_DIR = Path("data")
MODEL_PATH = Path("artifacts/ranker.txt")


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
    model, rep = pipeline.train_and_evaluate(tables, meta.end, ablation=args.ablation)
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

    recs, _ = pipeline.recommend(tables, meta.end, model, args.user, args.k)
    recs = recs.assign(source=pipeline.source_of(recs))
    print(f"Top {args.k} recommendations:")
    for i, r in enumerate(recs.itertuples(), 1):
        it = items.loc[r.item_id]
        why = pipeline.SOURCES[r.source]
        print(f"  {i:>2}. {it['title']:<36} {it['name']:<18} Rp{it['price']:>10,.0f}  score {r.score:6.2f}  ({why})")


def cmd_seed_db(args: argparse.Namespace) -> None:
    from . import db, seed

    if not args.yes:
        raise SystemExit(f"This deletes ALL marketplace data in {db.DATABASE_URL}. Re-run with --yes.")
    t = time.time()
    seed.seed_db(simulate.SimConfig(n_users=args.users, n_items=args.items, seed=args.seed))
    print(f"\nDone in {time.time() - t:.1f}s")


def cmd_serve(_: argparse.Namespace) -> None:
    from . import service

    service.serve()


def cmd_retrain(_: argparse.Namespace) -> None:
    from . import db, service

    store = db.Store()
    store.refresh()
    t = time.time()
    _, meta, rep = service.retrain(store, service.MODEL_DIR)
    pd.set_option("display.width", 120)
    print(f"Trained model {meta['version']} on {store.events_loaded:,} events in {time.time() - t:.1f}s")
    print(f"train rows {rep['train_rows']:,} | eval rows {rep['eval_rows']:,} | eval users {rep['eval_users']:,}")
    print(f"\nCandidate recall (ceiling for any ranker): {rep['candidate_recall']:.3f}\n")
    print(rep["results"].round(4).to_string())
    print(f"\nSaved {service.MODEL_DIR}/ranker.txt and model_meta.json (a running `recsys serve` loads it on restart).")


PROTO_ARGS = ["-I", "recsys/pb=../proto", "--python_out=.", "--grpc_python_out=.", "../proto/recommender.proto"]


def cmd_gen_proto(_: argparse.Namespace) -> None:
    """Regenerate recsys/pb/*_pb2*.py from ../proto/recommender.proto (run from recommender/)."""
    from grpc_tools import protoc

    code = protoc.main(["grpc_tools.protoc", *PROTO_ARGS])
    if code:
        raise SystemExit(code)
    print("generated recsys/pb/recommender_pb2.py, recommender_pb2_grpc.py")


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
    sd = sub.add_parser("seed-db", help="replace the marketplace DB contents with a simulated history")
    sd.add_argument("--users", type=int, default=1500)
    sd.add_argument("--items", type=int, default=2500)
    sd.add_argument("--seed", type=int, default=7)
    sd.add_argument("--yes", action="store_true", help="confirm deleting all marketplace data")
    sd.set_defaults(fn=cmd_seed_db)
    sub.add_parser("serve", help="run the gRPC recommender service").set_defaults(fn=cmd_serve)
    sub.add_parser("retrain", help="train the ranker on the marketplace DB and save it").set_defaults(fn=cmd_retrain)
    sub.add_parser("gen-proto", help="regenerate gRPC stubs from ../proto").set_defaults(fn=cmd_gen_proto)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
