"""Chapter 5 figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parents[2] / "results"


def make(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    res = json.loads((RESULTS / "ch05_cf.json").read_text())["results"]
    names = [r["method"] for r in res]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ("ndcg@20", "recall@20")):
        vals = [r[metric] for r in res]
        bars = ax.bar(names, vals, color=["#9aa5b1", "#3b82f6", "#10b981", "#f59e0b"][: len(names)])
        ax.bar_label(bars, fmt="%.3f")
        ax.set_title(f"{metric} by collaborative-filtering method")
        ax.set_ylabel(metric)
        ax.set_ylim(0, max(vals) * 1.18)
    fig.suptitle("Chapter 5: collaborative filtering vs. the 7-day popularity baseline")
    fig.tight_layout()
    p = out_dir / "ch05-results.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return [p]
