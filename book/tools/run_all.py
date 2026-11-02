#!/usr/bin/env python3
"""Run every companion script in order; stdout -> results/<stem>.txt.

    uv run python tools/run_all.py [--skip-live] [--only ch05_cf ch06_mf ...]

Each script writes its own results/<stem>.json. Stops at the first non-zero exit.
`--skip-live` skips ch13_latency.py (needs the running gRPC recommender).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

BOOK = Path(__file__).resolve().parent.parent
LIVE = {"ch13_latency"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-live", action="store_true", help="skip scripts that need live services (ch13_latency)")
    ap.add_argument("--only", nargs="*", help="run only these script stems")
    args = ap.parse_args()

    scripts = sorted((BOOK / "code").glob("ch*.py"))
    if args.only:
        scripts = [s for s in scripts if s.stem in args.only]
    (BOOK / "results").mkdir(exist_ok=True)
    t_all = time.time()
    for script in scripts:
        if args.skip_live and script.stem in LIVE:
            print(f"-- skip {script.stem} (--skip-live)")
            continue
        out = BOOK / "results" / f"{script.stem}.txt"
        print(f"== {script.stem} ...", end=" ", flush=True)
        t0 = time.time()
        with out.open("w") as fh:
            proc = subprocess.run([sys.executable, str(script.relative_to(BOOK))], cwd=BOOK, stdout=fh)
        dt = time.time() - t0
        if proc.returncode != 0:
            print(f"FAILED (exit {proc.returncode}) after {dt:.0f}s; partial output in {out.relative_to(BOOK)}")
            return proc.returncode
        print(f"ok {dt:.0f}s -> {out.relative_to(BOOK)}")
    print(f"done in {time.time() - t_all:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
