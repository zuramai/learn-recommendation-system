"""Chapter 6 figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parents[2] / "results"


def make(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    r = json.loads((RESULTS / "ch06_mf.json").read_text())
    paths = []
    for key, name, xlabel, ylabel, title in (
        ("als_loss", "ch06-als-loss.png", "ALS iteration", "Weighted squared loss + regularisation",
         "Implicit ALS: loss per iteration"),
        ("bpr_loss", "ch06-bpr-loss.png", "BPR epoch", "Mean -log sigmoid(score gap)",
         "BPR: pairwise loss per epoch"),
    ):
        y = r[key]
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(range(1, len(y) + 1), y, marker="o", color="#3b82f6")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = out_dir / name
        fig.savefig(p, dpi=200)
        plt.close(fig)
        paths.append(p)
    return paths
