"""Chapter 11 figure: the accuracy / diversity trade-off of MMR."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ._util import results


def make(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = results("ch11_rerank")["results"]
    mmr = [r for r in rows if r["method"].startswith("MMR")]
    other = [r for r in rows if not r["method"].startswith("MMR")]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([r["distinct_categories@20"] for r in mmr], [r["ndcg@20"] for r in mmr], "o-", color="#3b82f6", label="MMR")
    for r in mmr:
        ax.annotate(r["method"].replace("MMR ", ""), (r["distinct_categories@20"], r["ndcg@20"]), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.scatter([r["distinct_categories@20"] for r in other], [r["ndcg@20"] for r in other], color="#ef4444", marker="s", zorder=3, label="repo rules")
    for r in other:
        ax.annotate(r["method"], (r["distinct_categories@20"], r["ndcg@20"]), textcoords="offset points", xytext=(6, -12), fontsize=8)
    ax.set_xlabel("mean distinct categories in the top 20")
    ax.set_ylabel("ndcg@20")
    ax.set_title("Chapter 11: more diverse lists cost offline accuracy")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = out_dir / "ch11-mmr-tradeoff.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return [p]
