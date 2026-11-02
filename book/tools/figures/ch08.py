"""Chapter 8 figures: per-source recall and recall vs K."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ._util import results


def make(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    r = results("ch08_candidates")
    paths = []

    src = [s for s in r["sources"] if not s["source"].startswith("UNION")]
    union = next(s for s in r["sources"] if s["source"].startswith("UNION"))
    names = [s["source"] for s in src]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(names))
    b1 = ax.bar([i - 0.2 for i in x], [s["recall"] for s in src], 0.4, label="recall of the source alone", color="#3b82f6")
    b2 = ax.bar([i + 0.2 for i in x], [s["unique_pair_recall"] for s in src], 0.4, label="found by this source only (pooled pairs)", color="#f59e0b")
    ax.bar_label(b1, fmt="%.3f")
    ax.bar_label(b2, fmt="%.3f")
    ax.axhline(union["recall"], color="#10b981", ls="--", label=f"union of all four = {union['recall']:.3f}")
    ax.set_xticks(list(x), names)
    ax.set_ylabel("share of future positives")
    ax.set_ylim(0, union["recall"] * 1.3)
    ax.set_title("Chapter 8: what each retriever finds, and what only it finds")
    ax.legend(loc="upper left")
    fig.tight_layout()
    p = out_dir / "ch08-source-recall.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ks = r["ks"]
    for name, vals in r["recall_vs_k"].items():
        style = {"ls": "--", "color": "#10b981", "lw": 2} if "ceiling" in name else {"marker": "o", "lw": 2 if "union" in name else 1.2}
        ax.plot(ks, vals, label=name, **style)
    ax.set_xlabel("K (list length)")
    ax.set_ylabel("recall@K")
    ax.set_title("Chapter 8: recall vs. list length, per retriever and for the union")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "ch08-recall-vs-k.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths.append(p)
    return paths
