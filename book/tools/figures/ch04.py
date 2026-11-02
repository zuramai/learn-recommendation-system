"""Chapter 4 figure: TF-IDF cosine similarity heat map of 15 items in 3 categories."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ._util import results


def make(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    h = results("ch04_content")["heatmap"]
    S = np.array(h["similarity"])
    labels = h["labels"]
    cats = h["categories"]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(S, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), [l.split(" (")[0] for l in labels], rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), [l.split(" (")[0] for l in labels], fontsize=7)
    for i in range(1, len(cats)):
        if cats[i] != cats[i - 1]:
            ax.axhline(i - 0.5, color="k", lw=1)
            ax.axvline(i - 0.5, color="k", lw=1)
    ax.set_title("TF-IDF cosine similarity: 15 items, 3 categories\n(black lines separate Coffee & Tea, Snacks, Sneakers)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="cosine similarity")
    fig.tight_layout()
    p = out_dir / "ch04-similarity-heatmap.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return [p]
