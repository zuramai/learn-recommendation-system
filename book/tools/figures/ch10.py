"""Chapter 10 figures: importance, SHAP example, group ablation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ._util import results

GROUP_COLORS = {
    "candidate": "#8b5cf6", "user": "#3b82f6", "item": "#10b981", "shop": "#f59e0b",
    "user_item": "#ef4444", "user_category": "#ec4899", "user_shop": "#14b8a6", "context": "#9aa5b1",
}  # fmt: skip


def make(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    r = results("ch10_ltr")
    paths = []

    top = r["importance"][:15][::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh([t["feature"] for t in top], [t["gain_share"] for t in top], color=[GROUP_COLORS[t["group"]] for t in top])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for g, c in GROUP_COLORS.items()]
    ax.legend(handles, GROUP_COLORS, fontsize=8, title="feature group", loc="lower right")
    ax.set_xlabel("share of total gain")
    ax.set_title("Chapter 10: top 15 features of the LambdaMART ranker")
    fig.tight_layout()
    p = out_dir / "ch10-importance.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths.append(p)

    sh = r["shap_user42"]
    rows = sh["contributions"][::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    vals = [c["contribution"] for c in rows]
    ax.barh([f"{c['feature']} = {c['value']:.3g}" for c in rows], vals, color=["#10b981" if v > 0 else "#ef4444" for v in vals])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("contribution to the raw score (SHAP)")
    ax.set_title(f"Chapter 10: why item {sh['item_id']} is #1 for user 42 (score {sh['score']:.2f}, base {sh['bias']:.2f})")
    fig.tight_layout()
    p = out_dir / "ch10-shap-example.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths.append(p)

    abl = r["ablation"]
    names = [k for k in abl if not k.startswith("seed noise")]
    noise = abl["seed noise (±1 std)"]["ndcg@20"]
    d = [abl[k]["ndcg@20"] for k in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(names[::-1], d[::-1], color=["#ef4444" if v < -noise else "#10b981" if v > noise else "#9aa5b1" for v in d[::-1]])
    ax.axvspan(-noise, noise, color="#3b82f6", alpha=0.15, label=f"seed noise (±1 std = {noise:.4f})")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("change in ndcg@20 vs. the full model")
    ax.set_title("Chapter 10: ablation - retrain without one feature group")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    p = out_dir / "ch10-ablation.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths.append(p)
    return paths
