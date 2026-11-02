"""Chapter 9 figures: positives vs negatives for six features, and the leakage demo."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ._util import BOOK, results


def make(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(BOOK / "code"))
    import ltr_data

    r = results("ch09_features")
    _, ev = ltr_data.frames()
    pos = ev["label"] > 0
    paths = []

    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, name in zip(axes.ravel(), r["dist_features"]):
        v = ev[name].to_numpy(dtype=float)
        lo, hi = np.nanpercentile(v, [1, 99])
        if hi <= lo:
            hi = lo + 1
        bins = np.linspace(lo, hi, 25)
        ax.hist(np.clip(v[~pos], lo, hi), bins=bins, density=True, alpha=0.55, label="negatives", color="#9aa5b1")
        ax.hist(np.clip(v[pos], lo, hi), bins=bins, density=True, alpha=0.6, label="positives", color="#ef4444")
        if name in ("cand_recent_score", "item_clicks_7d"):
            ax.set_yscale("log")
        ax.set_title(name, fontsize=10)
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.suptitle("Chapter 9: six features, candidates that were later clicked/carted/ordered vs. not")
    fig.tight_layout()
    p = out_dir / "ch09-feature-distributions.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths.append(p)

    res = r["results"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, metric in zip(axes, ("ndcg@20", "recall@20")):
        vals = [x[metric] for x in res]
        bars = ax.bar([x["method"] for x in res], vals, color=["#3b82f6", "#ef4444"])
        ax.bar_label(bars, fmt="%.3f")
        ax.set_title(metric)
        ax.set_ylim(0, max(vals) * 1.18)
    fig.suptitle("Chapter 9: one leaked feature (clicks in the label window) inflates offline metrics")
    fig.tight_layout()
    p = out_dir / "ch09-leakage.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths.append(p)
    return paths
