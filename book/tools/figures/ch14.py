"""Chapter 14 figure: ndcg@20 of every method from every chapter's results JSON."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ._util import BOOK


def _walk(obj):
    if isinstance(obj, dict):
        if "method" in obj and "ndcg@20" in obj and "bucket" not in obj:
            yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def make(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: dict[str, tuple[float, int]] = {}
    for f in sorted((BOOK / "results").glob("ch*.json")):
        m = re.match(r"ch(\d+)", f.name)
        chapter = int(m.group(1))
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        for r in _walk(data):
            name = str(r["method"])
            if "leaky" in name.lower() or name.lower().startswith("mmr lambda=1.0"):
                continue  # leaked metrics are not a real result; MMR lambda=1 duplicates the plain ranker
            key = f"{name} (ch{chapter})"
            if key not in rows or r["ndcg@20"] > rows[key][0]:
                rows[key] = (float(r["ndcg@20"]), chapter)
    items = sorted(rows.items(), key=lambda kv: kv[1][0])
    fig, ax = plt.subplots(figsize=(9, max(5, 0.28 * len(items))))
    cmap = plt.get_cmap("viridis")
    colors = [cmap(c / 14) for _, (_, c) in items]
    bars = ax.barh([k for k, _ in items], [v for _, (v, _) in items], color=colors)
    ax.bar_label(bars, fmt="%.3f", fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlabel("ndcg@20 (same users, same labels, same candidates budget of 50)")
    ax.set_title("Chapter 14: leaderboard - every method in the book")
    ax.set_xlim(0, max(v for v, _ in rows.values()) * 1.12)
    fig.tight_layout()
    p = out_dir / "ch14-leaderboard.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return [p]
