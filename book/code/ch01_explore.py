"""Chapter 1: what the simulated marketplace looks like (tables, funnel, sparsity, long tail, time)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

import common
from common import section

tables = common.load_tables()
res: dict = {}

section("Table sizes")
sizes = {name: len(df) for name, df in tables.items()}
for name, n in sizes.items():
    print(f"{name:<12}{n:>10,}")
res["table_sizes"] = sizes

ev = tables["events"]
section("Event funnel (all 60 days)")
order = ["impression", "click", "add_to_cart", "order", "hide", "return"]
counts = ev["event_type"].value_counts().reindex(order).fillna(0).astype(int)
res["event_counts"] = counts.to_dict()
imp = counts["impression"]
print(f"{'event_type':<14}{'count':>10}{'% of impressions':>20}{'% of previous step':>22}")
prev = None
funnel = ["impression", "click", "add_to_cart", "order"]
for e in order:
    c = counts[e]
    step = f"{100 * c / prev:>21.2f}%" if e in funnel[1:] else f"{'':>22}"
    print(f"{e:<14}{c:>10,}{100 * c / imp:>19.2f}%{step}")
    if e in funnel:
        prev = c
res["funnel_rates"] = {
    "ctr": counts["click"] / imp,
    "click_to_cart": counts["add_to_cart"] / counts["click"],
    "cart_to_order": counts["order"] / counts["add_to_cart"],
}

section("Events per day")
day = ev["ts"].dt.floor("D")
per_day = ev.groupby([day, "event_type"]).size().unstack(fill_value=0)
pos_day = per_day[["click", "add_to_cart", "order"]].sum(axis=1)
orders_day = per_day["order"]
cfg_start = pd.Timestamp(common.END) - pd.Timedelta(days=60)
dates = pd.date_range(cfg_start, periods=60, freq="D")
pos_day = pos_day.reindex(dates, fill_value=0)
orders_day = orders_day.reindex(dates, fill_value=0)
clicks_day = per_day["click"].reindex(dates, fill_value=0)
payday = [bool(d.day >= 25 or d.day == 1) for d in dates]
sale = [bool(d.month == d.day) for d in dates]
base = np.median(clicks_day[[not (p or s) for p, s in zip(payday, sale)]])
print(f"median clicks/day on ordinary days : {base:,.0f}")
print(f"mean clicks/day on payday days     : {clicks_day[payday].mean():,.0f}")
for d, s in zip(dates, sale):
    if s:
        print(f"sale day {d.date()} (month == day): {clicks_day[d]:,} clicks, {orders_day[d]:,} orders")
wk = np.array([d.day_of_week >= 5 for d in dates])
print(f"weekend vs weekday mean clicks/day  : {clicks_day[wk].mean():,.0f} vs {clicks_day[~wk].mean():,.0f}")
res["per_day"] = {
    "date": [str(d.date()) for d in dates],
    "clicks": clicks_day.tolist(),
    "orders": orders_day.tolist(),
    "payday": payday,
    "sale": sale,
}

section("Popularity concentration (clicks + carts + orders per item)")
pos = ev[ev["event_type"].isin(["click", "add_to_cart", "order"])]
n_items = len(tables["items"])
per_item = pos.groupby("item_id").size().reindex(tables["items"]["item_id"], fill_value=0).sort_values(ascending=False)
cum = per_item.cumsum() / per_item.sum()
for pct in (1, 10, 20):
    n = max(1, int(round(n_items * pct / 100)))
    share = per_item.iloc[:n].sum() / per_item.sum()
    print(f"top {pct:>2}% of items ({n:>4}) get {100 * share:5.1f}% of interactions")
    res[f"top{pct}pct_share"] = share
print(f"items with zero interactions: {(per_item == 0).sum()} of {n_items}")
res["items_zero_interactions"] = int((per_item == 0).sum())
res["long_tail_counts"] = per_item.to_numpy()

section("Sparsity of the user x item matrix (before cutoff)")
hist = common.history()
m, ui, ii = common.interaction_matrix(hist)
dens = m.nnz / (m.shape[0] * m.shape[1])
print(f"users with interactions : {m.shape[0]:,}")
print(f"items with interactions : {m.shape[1]:,}")
print(f"non-zero cells          : {m.nnz:,}")
print(f"density                 : {100 * dens:.3f}%  (sparsity {100 * (1 - dens):.3f}%)")
res["matrix"] = {"users": m.shape[0], "items": m.shape[1], "nnz": m.nnz, "density": dens}

section("Clicks per user (before cutoff)")
cpu = hist.events[hist.events["event_type"] == "click"].groupby("user_id").size()
cpu = cpu.reindex(tables["users"]["user_id"], fill_value=0)
q = cpu.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
for k, v in q.items():
    print(f"p{int(k * 100):<3}: {v:6.0f}")
print(f"mean: {cpu.mean():.1f}   max: {cpu.max()}   users with 0 clicks: {(cpu == 0).sum()}")
res["clicks_per_user"] = {"mean": cpu.mean(), "max": int(cpu.max()), "zero": int((cpu == 0).sum()),
                          **{f"p{int(k * 100)}": v for k, v in q.items()}}

common.save_results("ch01_explore", res)
