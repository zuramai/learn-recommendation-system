"""Chapter 7 figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parents[2] / "results"


def make(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    r = json.loads((RESULTS / "ch07_two_tower.json").read_text())
    paths = []

    pca = r["pca"]
    x, y = np.array(pca["x"]), np.array(pca["y"])
    parents = np.array(pca["parent"])
    fig, ax = plt.subplots(figsize=(8, 6))
    for k, par in enumerate(sorted(set(parents))):
        m = parents == par
        ax.scatter(x[m], y[m], s=10, alpha=0.7, label=par, color=plt.cm.tab10(k))
    ax.set_xlabel(f"PC1 ({pca['explained_variance'][0]:.0%} of variance)")
    ax.set_ylabel(f"PC2 ({pca['explained_variance'][1]:.0%} of variance)")
    ax.set_title("Two-tower item embeddings projected to 2-D, coloured by parent category")
    ax.legend(fontsize=8, markerscale=2)
    fig.tight_layout()
    p = out_dir / "ch07-embeddings-pca.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths.append(p)

    ann = r["ann"]
    probes = [a["n_probe"] for a in ann]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([a["share_scanned"] * 100 for a in ann], [a["recall_vs_brute_force@20"] for a in ann],
            marker="o", color="#3b82f6")
    for a in ann:
        ax.annotate(f"probe {a['n_probe']}\n{a['ms_per_query']:.3f} ms", (a["share_scanned"] * 100, a["recall_vs_brute_force@20"]),
                    textcoords="offset points", xytext=(6, -22), fontsize=8)
    ax.set_xlabel("Share of items scanned per query (%)")
    ax.set_ylabel("Recall@20 vs. brute force")
    ax.set_ylim(0.6, 1.03)
    ax.set_title("IVF approximate search: recall vs. work (32 clusters)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "ch07-ann-tradeoff.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths.append(p)
    return paths
