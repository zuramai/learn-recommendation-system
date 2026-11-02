"""Chapter 2 figure: the DCG position discount."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recsys.evaluate import GAIN


def make(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pos = np.arange(1, 21)
    disc = 1 / np.log2(pos + 1)  # position 1 -> 1/log2(2), matching log2(pos_0based + 2)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(pos, disc, "o-", color="#3182bd", label=r"$1/\log_2(\mathrm{pos}+1)$ (NDCG)")
    axes[0].plot(pos, 1 / pos, "s--", color="grey", ms=4, label="1/pos (harsher)")
    axes[0].set_xlabel("Position in the list (1 = top)")
    axes[0].set_ylabel("Discount applied to the gain")
    axes[0].set_title("Position discount")
    axes[0].legend()
    labels = ["click\n(1)", "cart\n(2)", "order\n(3)"]
    g = [GAIN[1], GAIN[2], GAIN[3]]
    axes[1].bar(labels, g, color=["#9ecae1", "#6baed6", "#08519c"])
    for i, v in enumerate(g):
        axes[1].text(i, v + 0.1, f"{v:.0f}", ha="center")
    axes[1].set_ylabel("Gain")
    axes[1].set_title("Gain per label (same as the ranker)")
    fig.tight_layout()
    p = out_dir / "ch02-dcg-discount.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return [p]
