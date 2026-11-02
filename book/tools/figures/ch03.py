"""Chapter 3 figures: decay curve and baseline results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ._util import results


def make(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    r = results("ch03_popularity")
    paths = []

    fig, ax = plt.subplots(figsize=(7, 3.8))
    c = r["decay_curve"]
    ax.plot(c["age_days"], c["weight"], color="#3182bd")
    hl = r["half_life_days"]
    for k in (1, 2, 3):
        ax.plot([hl * k], [0.5**k], "o", color="#d62728")
        ax.annotate(f"{0.5**k:.3g}", (hl * k, 0.5**k), xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel("Age of the event (days)")
    ax.set_ylabel("Weight")
    ax.set_title(f"Exponential decay, half-life {hl:.0f} days")
    fig.tight_layout()
    p = out_dir / "ch03-decay-curve.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths.append(p)

    rows = sorted(r["results"], key=lambda x: x["ndcg@20"])
    names = [x["method"] for x in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(y - 0.2, [x["recall@20"] for x in rows], height=0.4, label="recall@20", color="#9ecae1")
    ax.barh(y + 0.2, [x["ndcg@20"] for x in rows], height=0.4, label="ndcg@20", color="#08519c")
    for yi, x in zip(y, rows):
        ax.text(x["ndcg@20"], yi + 0.2, f" {x['ndcg@20']:.3f}", va="center", fontsize=8)
    ax.set_yticks(y, names)
    ax.set_xlabel("Score (higher is better)")
    ax.set_title("Popularity baselines on the held-out week")
    ax.legend(loc="lower right")
    fig.tight_layout()
    p = out_dir / "ch03-results.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths.append(p)
    return paths
