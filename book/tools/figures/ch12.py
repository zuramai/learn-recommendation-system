"""Chapter 12 figures: position-dependent CTR and metrics by user activity."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ._util import results


def make(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    r = results("ch12_coldstart_bias")
    paths = []

    pos = pd.DataFrame(r["position"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(pos["position"], pos["ctr"], color="#3b82f6")
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in pos["ctr"]], fontsize=8)
    ax.set_xticks(pos["position"])
    ax.set_xlabel("display position (0 = top)")
    ax.set_ylabel("click-through rate")
    ax.set_title("Chapter 12: the same page, very different CTR by position")
    fig.tight_layout()
    p = out_dir / "ch12-position-ctr.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths.append(p)

    b = pd.DataFrame(r["buckets"])
    order = list(dict.fromkeys(b["bucket"]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (method, color) in enumerate([("pop_7d", "#9aa5b1"), ("LambdaMART", "#3b82f6")]):
        s = b[b["method"] == method].set_index("bucket").loc[order]
        bars = ax.bar([x + (i - 0.5) * 0.4 for x in range(len(order))], s["ndcg@20"], 0.4, label=method, color=color)
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
    users = b[b["method"] == "pop_7d"].set_index("bucket").loc[order]["users"]
    ax.set_xticks(range(len(order)), [f"{o}\n({u} users)" for o, u in zip(order, users)])
    ax.set_ylabel("ndcg@20")
    ax.set_title("Chapter 12: personalisation pays off as history grows")
    ax.legend()
    fig.tight_layout()
    p = out_dir / "ch12-activity-buckets.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths.append(p)
    return paths
