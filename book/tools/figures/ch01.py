"""Chapter 1 figures: funnel, events per day, long tail."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._util import results


def make(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    r = results("ch01_explore")
    paths = []

    # Funnel
    steps = ["impression", "click", "add_to_cart", "order"]
    vals = [r["event_counts"][s] for s in steps]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ys = np.arange(len(steps))[::-1]
    ax.barh(ys, vals, color=["#9ecae1", "#6baed6", "#3182bd", "#08519c"])
    for y, v in zip(ys, vals):
        ax.text(v, y, f"  {v:,} ({100 * v / vals[0]:.1f}%)", va="center")
    ax.set_yticks(ys, steps)
    ax.set_xlim(0, vals[0] * 1.35)
    ax.set_xlabel("Number of events (60 simulated days)")
    ax.set_title("Event funnel: impressions to orders")
    fig.tight_layout()
    p = out_dir / "ch01-funnel.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths.append(p)

    # Events per day
    d = r["per_day"]
    dates = pd.to_datetime(d["date"])
    clicks = np.array(d["clicks"])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(dates, clicks, color="#6baed6", width=0.85, label="clicks")
    ax.plot(dates, d["orders"], color="#d62728", marker="o", ms=3, lw=1, label="orders")
    for dt, pd_ in zip(dates, d["payday"]):
        if pd_:
            ax.axvspan(dt - pd.Timedelta(hours=12), dt + pd.Timedelta(hours=12), color="#fdd0a2", alpha=0.6, lw=0)
    for dt, s, c in zip(dates, d["sale"], clicks):
        if s:
            ax.annotate(f"{dt.month}.{dt.day} sale", (dt, c), xytext=(0, 14), textcoords="offset points",
                        ha="center", arrowprops=dict(arrowstyle="->"), fontsize=9)
    ax.axvspan(dates[0], dates[0], color="#fdd0a2", alpha=0.6, label="payday window (25th to 1st)")
    ax.set_ylim(0, clicks.max() * 1.2)
    ax.set_ylabel("Events per day")
    ax.set_title("Traffic over time: weekends, paydays and mega-sales")
    ax.legend(loc="upper center", ncol=3)
    fig.autofmt_xdate()
    fig.tight_layout()
    p = out_dir / "ch01-events-per-day.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths.append(p)

    # Long tail
    counts = np.array(r["long_tail_counts"])
    n = len(counts)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(np.arange(1, n + 1), counts, color="#3182bd")
    axes[0].fill_between(np.arange(1, n + 1), counts, color="#3182bd", alpha=0.2)
    axes[0].axvline(n * 0.1, color="grey", ls="--")
    axes[0].text(n * 0.11, counts.max() * 0.8, "top 10% of items", color="grey")
    axes[0].set_xlabel("Item rank by interactions")
    axes[0].set_ylabel("Interactions (click+cart+order)")
    axes[0].set_title("Long tail of item popularity")
    cum = np.cumsum(counts) / counts.sum()
    axes[1].plot(np.arange(1, n + 1) / n * 100, cum * 100, color="#d62728")
    axes[1].plot([0, 100], [0, 100], color="grey", ls=":")
    axes[1].set_xlabel("% of items (most popular first)")
    axes[1].set_ylabel("% of all interactions")
    axes[1].set_title(f"Top 10% of items get {100 * r['top10pct_share']:.0f}% of interactions")
    fig.tight_layout()
    p = out_dir / "ch01-long-tail.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths.append(p)
    return paths
