#!/usr/bin/env python3
"""Generate every chart: imports tools/figures/ch*.py and calls make(book/assets/charts).

    uv run python tools/make_figures.py [ch07 ch10 ...]

Modules that fail are reported and skipped; exit status is 1 if any failed or if a
chart promised by plan 9.5 is missing afterwards.
"""
from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
BOOK = TOOLS.parent
OUT = BOOK / "assets" / "charts"

EXPECTED = """ch01-funnel ch01-events-per-day ch01-long-tail ch02-dcg-discount ch03-decay-curve ch03-results
ch04-similarity-heatmap ch05-results ch06-als-loss ch06-bpr-loss ch07-embeddings-pca ch07-ann-tradeoff
ch08-source-recall ch08-recall-vs-k ch09-feature-distributions ch09-leakage ch10-importance ch10-shap-example
ch10-ablation ch11-mmr-tradeoff ch12-position-ctr ch12-activity-buckets ch13-latency-hist
ch14-leaderboard""".split()


def main() -> int:
    only = set(sys.argv[1:])
    sys.path.insert(0, str(BOOK / "code"))
    sys.path.insert(0, str(TOOLS))
    import matplotlib
    matplotlib.use("Agg")
    OUT.mkdir(parents=True, exist_ok=True)
    failed = 0
    for path in sorted((TOOLS / "figures").glob("ch*.py")):
        if only and path.stem not in only:
            continue
        try:
            mod = importlib.import_module(f"figures.{path.stem}")
            made = mod.make(OUT)
        except Exception:
            failed += 1
            print(f"!! {path.stem} FAILED", file=sys.stderr)
            traceback.print_exc()
            continue
        for p in made:
            print(f"{path.stem}: {Path(p).relative_to(BOOK)}")
    if not only:
        missing = [n for n in EXPECTED if not (OUT / f"{n}.png").is_file()]
        if missing:
            print("missing charts: " + ", ".join(missing), file=sys.stderr)
            return 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
