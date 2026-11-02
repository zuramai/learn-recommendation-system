"""Stage 2 input — feature engineering for (user, item) candidates.

Every feature is computed from `History` (events strictly before the cutoff),
so the same code builds training rows (cutoff in the past, labels after it)
and serving rows (cutoff = now). Output columns == schema.FEATURE_NAMES.

Patterns worth noticing:
* Multiple time windows (3d/7d/30d) let the model separate "trending now" from
  "steadily good".
* Rates are Bayesian-smoothed: an item with 1 click / 1 impression is not a
  100% CTR item; we shrink it toward the global average until data accumulates.
* Cross features (user x item, user x category, user x shop) carry most of the
  personalisation; pure item features carry most of the quality signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema
from .data import History

NEVER_HOURS = 9999.0
NEVER_DAYS = 999.0


def _count(ev: pd.DataFrame, keys: list[str], event_type: str, max_age: float | None = None, name: str = "n") -> pd.DataFrame:
    sel = ev[ev["event_type"] == event_type]
    if max_age is not None:
        sel = sel[sel["age_days"] < max_age]
    return sel.groupby(keys).size().rename(name).reset_index()


def _smoothed(num: pd.Series, den: pd.Series, prior: float, strength: float) -> pd.Series:
    return (num + prior * strength) / (den + strength)


def user_features(hist: History) -> pd.DataFrame:
    ev = hist.events
    u = hist.users[["user_id", "signup_at", "device_tier", "age_bucket"]].copy()
    for name, et, age in [
        ("user_clicks_7d", "click", 7),
        ("user_clicks_30d", "click", 30),
        ("user_orders_30d", "order", 30),
        ("_impr_30d", "impression", 30),
    ]:
        u = u.merge(_count(ev, ["user_id"], et, age, name), on="user_id", how="left")
    u = u.fillna({c: 0 for c in ["user_clicks_7d", "user_clicks_30d", "user_orders_30d", "_impr_30d"]})
    global_ctr = u["user_clicks_30d"].sum() / max(u["_impr_30d"].sum(), 1)
    u["user_ctr_30d"] = _smoothed(u["user_clicks_30d"], u["_impr_30d"], global_ctr, 20)
    last = ev.groupby("user_id")["age_days"].min().rename("user_days_since_active")
    u = u.join(last, on="user_id").fillna({"user_days_since_active": NEVER_DAYS})
    u["user_account_age_days"] = (hist.cutoff - u["signup_at"]) / pd.Timedelta(days=1)
    return u.rename(columns={"device_tier": "user_device_tier", "age_bucket": "user_age_bucket"}).drop(
        columns=["signup_at", "_impr_30d"]
    )


def item_features(hist: History) -> pd.DataFrame:
    ev = hist.events
    it = hist.items.copy()
    it["item_price"] = it["price"]
    it["item_discount_pct"] = it["discount_pct"]
    it["item_price_vs_category"] = it["price"] / it["median_price"]
    it["item_free_shipping"] = it["free_shipping"].astype(int)
    it["item_rating_avg"] = it["rating_avg"]
    it["item_rating_count"] = np.log1p(it["rating_count"])
    it["item_age_days"] = (hist.cutoff - it["listed_at"]) / pd.Timedelta(days=1)
    it["item_is_consumable"] = it["is_consumable"].astype(int)

    counts = [
        ("item_impressions_7d", "impression", 7),
        ("item_clicks_7d", "click", 7),
        ("item_orders_7d", "order", 7),
        ("_impr_30d", "impression", 30),
        ("_clicks_30d", "click", 30),
        ("_orders_30d", "order", 30),
        ("_clicks_3d", "click", 3),
        ("_clicks_14d", "click", 14),
        ("_orders_all", "order", None),
        ("_returns_all", "return", None),
    ]
    for name, et, age in counts:
        it = it.merge(_count(ev, ["item_id"], et, age, name), on="item_id", how="left")
    it = it.fillna({name: 0 for name, _, _ in counts})

    ctr = it["_clicks_30d"].sum() / max(it["_impr_30d"].sum(), 1)
    cvr = it["_orders_30d"].sum() / max(it["_clicks_30d"].sum(), 1)
    ret = it["_returns_all"].sum() / max(it["_orders_all"].sum(), 1)
    it["item_ctr_30d"] = _smoothed(it["_clicks_30d"], it["_impr_30d"], ctr, 50)
    it["item_cvr_30d"] = _smoothed(it["_orders_30d"], it["_clicks_30d"], cvr, 20)
    it["item_return_rate"] = _smoothed(it["_returns_all"], it["_orders_all"], ret, 10)
    it["item_trend"] = (it["_clicks_3d"] / 3 + 0.1) / (it["_clicks_14d"] / 14 + 0.1)

    sh = hist.shops.rename(columns={"is_mall": "shop_is_mall", "cancel_rate": "shop_cancel_rate", "response_rate": "shop_response_rate"})
    sh["shop_is_mall"] = sh["shop_is_mall"].astype(int)
    it = it.merge(sh[["shop_id", "city", "shop_rating", "shop_is_mall", "shop_cancel_rate", "shop_response_rate"]], on="shop_id")
    return it.rename(columns={"city": "shop_city"})


def user_item_features(hist: History) -> pd.DataFrame:
    ev = hist.events
    keys = ["user_id", "item_id"]
    ui = ev[keys].drop_duplicates()
    for name, et in [("ui_clicks", "click"), ("ui_carts", "add_to_cart"), ("ui_orders", "order"), ("_impr", "impression")]:
        ui = ui.merge(_count(ev, keys, et, None, name), on=keys, how="left")
    ui = ui.fillna(0)
    pos = ev[ev["event_type"].isin(schema.EVENT_WEIGHT)]
    last = (pos.groupby(keys)["age_days"].min() * 24).rename("ui_hours_since_last")
    dwell = ev.groupby(keys)["dwell_sec"].max().rename("ui_max_dwell_sec")
    ui = ui.join(last, on=keys).join(dwell, on=keys)
    ui["ui_ignored_impressions"] = np.where(ui["ui_clicks"] == 0, ui["_impr"], 0)
    return ui.drop(columns="_impr")


def user_category_features(hist: History) -> pd.DataFrame:
    ev = hist.events.merge(hist.items[["item_id", "category_id"]], on="item_id")
    keys = ["user_id", "category_id"]
    clicks30 = _count(ev, keys, "click", 30, "_c")
    clicks30["uc_click_share_30d"] = clicks30["_c"] / clicks30.groupby("user_id")["_c"].transform("sum")
    se = hist.searched_categories()
    searches = se[se["age_days"] < 30].groupby(keys).size().rename("uc_searches_30d").reset_index()
    orders = ev[ev["event_type"] == "order"].groupby(keys)["age_days"].agg(uc_orders="size", uc_days_since_order="min").reset_index()
    uc = clicks30[keys + ["uc_click_share_30d"]].merge(searches, on=keys, how="outer").merge(orders, on=keys, how="outer")
    return uc


def user_shop_features(hist: History) -> pd.DataFrame:
    ev = hist.events.merge(hist.items[["item_id", "shop_id"]], on="item_id")
    keys = ["user_id", "shop_id"]
    us = _count(ev, keys, "click", None, "us_clicks").merge(_count(ev, keys, "order", None, "us_orders"), on=keys, how="outer")
    return us


def user_price_level(hist: History) -> pd.Series:
    """User's typical price level: median of (price / category median) over clicked items. 1.0 = average."""
    clicks = hist.events[hist.events["event_type"] == "click"]
    lvl = clicks.merge(hist.items[["item_id", "price", "median_price"]], on="item_id")
    return (lvl["price"] / lvl["median_price"]).groupby(lvl["user_id"]).median().rename("_user_price_level")


def build(cands: pd.DataFrame, hist: History) -> pd.DataFrame:
    """Attach every feature in schema.FEATURES to the candidate rows."""
    df = cands.merge(item_features(hist), on="item_id", how="left")
    df = df.merge(user_features(hist), on="user_id", how="left")
    df = df.merge(user_item_features(hist), on=["user_id", "item_id"], how="left")
    df = df.merge(user_category_features(hist), on=["user_id", "category_id"], how="left")
    df = df.merge(user_shop_features(hist), on=["user_id", "shop_id"], how="left")
    df = df.join(user_price_level(hist), on="user_id")
    df = df.merge(hist.users[["user_id", "city"]].rename(columns={"city": "user_city"}), on="user_id", how="left")

    df["price_ratio_to_user"] = df["item_price_vs_category"] / df["_user_price_level"].fillna(1.0)
    df["same_city"] = (df["user_city"] == df["shop_city"]).astype(int)
    fill = {
        "ui_hours_since_last": NEVER_HOURS,
        "uc_days_since_order": NEVER_DAYS,
        "user_days_since_active": NEVER_DAYS,
    }
    df = df.fillna(fill)
    zero = ["ui_clicks", "ui_carts", "ui_orders", "ui_ignored_impressions", "ui_max_dwell_sec", "uc_click_share_30d",
            "uc_searches_30d", "uc_orders", "us_clicks", "us_orders", "user_clicks_7d", "user_clicks_30d",
            "user_orders_30d"]  # fmt: skip
    df[zero] = df[zero].fillna(0)

    # Keep ids for grouping/re-ranking, then exactly the registered features, in order.
    out = df[["user_id", "item_id", "category_id", "shop_id"] + schema.FEATURE_NAMES]
    assert list(out.columns[4:]) == schema.FEATURE_NAMES
    return out.reset_index(drop=True)
