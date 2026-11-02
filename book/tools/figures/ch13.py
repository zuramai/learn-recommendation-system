"""Chapter 13 figure: latency histogram of the live Recommend RPC."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ._util import results


def make(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    r = results("ch13_latency")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(r["server_ms"], bins=30, color="#3b82f6", alpha=0.85)
    for q, color in (("p50", "#10b981"), ("p90", "#f59e0b"), ("p99", "#ef4444")):
        ax.axvline(r["server"][q], color=color, ls="--", label=f"{q} = {r['server'][q]:.0f} ms")
    ax.set_xlabel("server-side latency_ms per Recommend call")
    ax.set_ylabel("requests")
    ax.set_title(f"Chapter 13: Recommend latency over {r['n_users']} random users (k=20)")
    ax.legend()
    fig.tight_layout()
    p = out_dir / "ch13-latency-hist.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return [p]
