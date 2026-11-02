"""Synthetic marketplace: generates the RAW tables in `schema.py`.

How behaviour is generated (the "hidden truth" the model must rediscover):

* Each user has a sparse category preference, a latent taste vector, a price
  sensitivity, and a preferred hour of day.
* Each item has a latent quality and a taste vector; shops have a city.
* A session picks an *intent* category (or a re-buy of a consumable), shows
  12 items (search = mostly intent category; home feed = half popular stuff),
  and the user clicks / carts / orders each one with probabilities driven by:
      taste match + category preference + quality + price fit + discount
      + free shipping + same-city - position bias
* Time matters: more sessions on weekends, paydays (25th-1st) and the 9.9
  mega-sale, which also boosts discounts and conversion.
* Negatives: users hide irrelevant feed items; low-quality items get returned;
  a durable item the user already bought loses almost all appeal.

None of the latent variables are written to disk — only what an app would log.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import schema

# (top-level, leaf, is_consumable, median price IDR)
CATEGORY_TREE: list[tuple[str, str, bool, float]] = [
    ("Electronics", "Phone Cases", False, 45_000),
    ("Electronics", "Earphones", False, 180_000),
    ("Electronics", "Chargers & Cables", False, 70_000),
    ("Electronics", "Smartwatches", False, 450_000),
    ("Fashion", "Men's T-Shirts", False, 80_000),
    ("Fashion", "Women's Dresses", False, 150_000),
    ("Fashion", "Sneakers", False, 350_000),
    ("Fashion", "Bags", False, 200_000),
    ("Beauty", "Skincare", True, 120_000),
    ("Beauty", "Makeup", True, 90_000),
    ("Beauty", "Shampoo", True, 45_000),
    ("Home & Living", "Kitchenware", False, 110_000),
    ("Home & Living", "Bedding", False, 250_000),
    ("Home & Living", "Storage", False, 75_000),
    ("Mother & Baby", "Diapers", True, 130_000),
    ("Mother & Baby", "Baby Formula", True, 220_000),
    ("Mother & Baby", "Baby Clothes", False, 70_000),
    ("Groceries", "Snacks", True, 30_000),
    ("Groceries", "Coffee & Tea", True, 60_000),
    ("Groceries", "Instant Noodles", True, 25_000),
    ("Sports", "Yoga Mats", False, 150_000),
    ("Sports", "Dumbbells", False, 200_000),
    ("Sports", "Cycling Gear", False, 300_000),
    ("Pets", "Cat Food", True, 90_000),
    ("Pets", "Dog Toys", False, 40_000),
]
CITIES = ["Jakarta", "Surabaya", "Bandung", "Medan", "Makassar"]
CITY_P = [0.40, 0.20, 0.18, 0.12, 0.10]
BRANDS = ["Aurora", "Nusa", "Kirana", "Volt", "Bamboo", "Lumi", "Sakura", "Titan", "Garuda", "Mega"]
ADJECTIVES = ["Premium", "Original", "Murah", "Best Seller", "Import", "Lokal", "Pro", "Lite"]
DEVICE_OS = ["android", "ios", "web"]
DAY = 86_400  # seconds
SLOTS_PER_SESSION = 12
TASTE_DIM = 8


@dataclass(frozen=True)
class SimConfig:
    n_users: int = 1500
    n_items: int = 2500
    n_shops: int = 150
    n_days: int = 60
    start_date: str = "2026-08-01"  # 60 days -> includes 9.9 sale and two paydays
    seed: int = 7


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _day_weights(start: pd.Timestamp, n_days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Traffic multiplier per day + flags for payday and mega-sale."""
    days = pd.date_range(start, periods=n_days, freq="D")
    payday = np.array([d.day >= 25 or d.day == 1 for d in days])
    sale = np.array([d.month == d.day for d in days])  # 9.9, 10.10, ... style sales
    w = np.ones(n_days)
    w[days.day_of_week >= 5] *= 1.2
    w[payday] *= 1.4
    w[sale] *= 3.0
    return w / w.sum(), payday, sale


def simulate(cfg: SimConfig = SimConfig()) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    start = pd.Timestamp(cfg.start_date)

    # ---------------- catalog ---------------- #
    categories = pd.DataFrame(CATEGORY_TREE, columns=["parent_name", "name", "is_consumable", "median_price"])
    categories.insert(0, "category_id", np.arange(len(categories)))
    categories = categories[schema.CATEGORIES.column_names]
    n_cat = len(categories)
    parents = categories["parent_name"].unique()

    shop_parent = rng.choice(parents, cfg.n_shops)
    shop_mall = rng.random(cfg.n_shops) < 0.15
    shops = pd.DataFrame(
        {
            "shop_id": np.arange(cfg.n_shops),
            "city": rng.choice(CITIES, cfg.n_shops, p=CITY_P),
            "is_mall": shop_mall,
            "shop_rating": np.clip(rng.normal(4.6, 0.25, cfg.n_shops) + 0.15 * shop_mall, 3.0, 5.0).round(2),
            "response_rate": np.clip(rng.beta(8, 2, cfg.n_shops), 0, 1).round(3),
            "cancel_rate": np.clip(rng.beta(1, 40, cfg.n_shops), 0, 1).round(4),
        }
    )

    item_shop = rng.integers(0, cfg.n_shops, cfg.n_items)
    item_cat = np.empty(cfg.n_items, dtype=int)
    for i, s in enumerate(item_shop):  # a shop sells inside its own top-level category
        leaves = categories.index[categories["parent_name"] == shop_parent[s]]
        item_cat[i] = rng.choice(leaves)
    is_mall_item = shop_mall[item_shop]
    quality = rng.normal(0, 1, cfg.n_items) + 0.4 * is_mall_item  # hidden
    item_vec = rng.normal(0, 1, (cfg.n_items, TASTE_DIM))  # hidden
    base_pop = rng.lognormal(0, 1, cfg.n_items)  # hidden: platform's exposure prior
    median_price = categories["median_price"].to_numpy()[item_cat]
    price = (median_price * rng.lognormal(0, 0.45, cfg.n_items) * np.where(is_mall_item, 1.3, 1.0)).round(-2)
    discount = np.where(rng.random(cfg.n_items) < 0.4, rng.choice([0.05, 0.1, 0.2, 0.3, 0.5], cfg.n_items), 0.0)
    listed_day = np.where(
        rng.random(cfg.n_items) < 0.85,
        rng.integers(-365, 0, cfg.n_items),
        rng.integers(0, cfg.n_days - 5, cfg.n_items),  # 15% new listings during the window
    )
    oos_day = np.where(rng.random(cfg.n_items) < 0.08, rng.integers(5, cfg.n_days, cfg.n_items), 10**6)
    oos_day = np.maximum(oos_day, listed_day + 3)
    brand = rng.choice(BRANDS, cfg.n_items)
    cat_names = categories["name"].to_numpy()
    items = pd.DataFrame(
        {
            "item_id": np.arange(cfg.n_items),
            "shop_id": item_shop,
            "category_id": item_cat,
            "brand": brand,
            "title": [f"{b} {cat_names[c]} {rng.choice(ADJECTIVES)}" for b, c in zip(brand, item_cat)],
            "price": price,
            "discount_pct": discount,
            "free_shipping": rng.random(cfg.n_items) < 0.5,
            "rating_avg": np.clip(4.3 + 0.35 * quality + rng.normal(0, 0.15, cfg.n_items), 1, 5).round(1),
            "rating_count": rng.poisson(base_pop * 40 * np.clip(-listed_day, 1, None) / 100),
            "listed_at": start + pd.to_timedelta(listed_day, unit="D"),
            "out_of_stock_at": pd.Series(
                np.where(oos_day < 10**6, start + pd.to_timedelta(np.minimum(oos_day, 10**4), unit="D"), pd.NaT),
                dtype="datetime64[ns]",
            ),
        }
    )

    # ---------------- users ---------------- #
    cat_pref = rng.dirichlet(np.full(n_cat, 0.15), cfg.n_users)  # hidden, sparse
    taste = rng.normal(0, 1, (cfg.n_users, TASTE_DIM))  # hidden
    price_sens = rng.uniform(0.2, 1.5, cfg.n_users)  # hidden
    peak_hour = rng.choice([8, 12, 20, 22], cfg.n_users, p=[0.15, 0.25, 0.35, 0.25])  # hidden
    activity = rng.gamma(2.0, 7.0, cfg.n_users)  # hidden: expected sessions in window
    device_tier = rng.choice([1, 2, 3], cfg.n_users, p=[0.45, 0.4, 0.15])
    signup_day = np.where(rng.random(cfg.n_users) < 0.85, rng.integers(-700, 0, cfg.n_users), rng.integers(0, cfg.n_days - 10, cfg.n_users))
    user_city = rng.choice(CITIES, cfg.n_users, p=CITY_P)
    users = pd.DataFrame(
        {
            "user_id": np.arange(cfg.n_users),
            "signup_at": start + pd.to_timedelta(signup_day, unit="D"),
            "city": user_city,
            "age_bucket": rng.choice([0, 1, 2, 3], cfg.n_users, p=[0.15, 0.45, 0.28, 0.12]),
            "device_tier": device_tier,
        }
    )
    # Richer phones -> less price sensitive (gives device_tier real signal).
    price_sens = price_sens * np.array([0, 1.2, 1.0, 0.6])[device_tier]

    # ---------------- behaviour ---------------- #
    day_p, payday, sale = _day_weights(start, cfg.n_days)
    shop_city = shops["city"].to_numpy()
    consumable = categories["is_consumable"].to_numpy()
    items_by_cat = [np.flatnonzero(item_cat == c) for c in range(n_cat)]
    log_pop = np.log(base_pop)
    log_price_ratio = np.log(price / median_price)
    free_ship = items["free_shipping"].to_numpy()

    sessions: list[tuple] = []
    searches: list[tuple] = []
    events: list[tuple] = []

    avail_by_day = [np.flatnonzero((listed_day <= d) & (oos_day > d)) for d in range(cfg.n_days)]

    def available(idx: np.ndarray, day: int) -> np.ndarray:
        return idx[(listed_day[idx] <= day) & (oos_day[idx] > day)]

    def sample_top(idx: np.ndarray, k: int, boost=None, exclude: np.ndarray | None = None) -> np.ndarray:
        """Platform's existing (non-personal) ranking: popularity + noise (Gumbel top-k)."""
        if exclude is not None and len(exclude):
            idx = idx[~np.isin(idx, exclude)]
        k = min(k, len(idx))
        if k <= 0:
            return idx[:0]
        score = log_pop[idx] + 0.3 * quality[idx] + rng.gumbel(size=len(idx))
        if boost is not None:
            score = score + boost
        top = np.argpartition(-score, k - 1)[:k]
        return idx[top[np.argsort(-score[top])]]

    for u in range(cfg.n_users):
        first_day = max(signup_day[u], 0)
        p = day_p[first_day:] / day_p[first_day:].sum()
        n_sess = rng.poisson(activity[u] * p.size / cfg.n_days)
        days = np.sort(rng.choice(np.arange(first_day, cfg.n_days), n_sess, p=p)) if n_sess else []
        consumable_orders: dict[int, int] = {}  # item -> day ordered
        owned_durables: list[int] = []  # bought once, rarely wanted again

        for day in days:
            # Timestamps are kept as seconds since `start` and converted once at the end (fast).
            hour = int(rng.normal(peak_hour[u], 2.5)) % 24
            t0 = int(day) * DAY + hour * 3600 + int(rng.integers(0, 3600))
            sid = len(sessions)
            surface = "search" if rng.random() < 0.45 else "home_feed"
            sessions.append((sid, u, t0, surface, DEVICE_OS[rng.choice(3, p=[0.7, 0.2, 0.1])]))

            # Intent: re-buy a consumable that is "due", else a preferred category.
            due = [i for i, d in consumable_orders.items() if day - d >= 10]
            rebuy_item = -1
            if due and rng.random() < 0.35:
                rebuy_item = int(rng.choice(due))
                intent = item_cat[rebuy_item]
            elif rng.random() < 0.15:
                intent = int(rng.integers(n_cat))  # exploration
            else:
                intent = int(rng.choice(n_cat, p=cat_pref[u]))

            pool = available(items_by_cat[intent], day)
            if surface == "search":
                query_brand = rng.choice(BRANDS) if rng.random() < 0.3 else None
                query = f"{query_brand + ' ' if query_brand else ''}{cat_names[intent].lower()}"
                searches.append((len(searches), sid, u, t0, query))
                boost = 1.5 * (brand[pool] == query_brand) if query_brand else None
                shown = sample_top(pool, SLOTS_PER_SESSION - 1, boost)
            else:
                shown = sample_top(pool, SLOTS_PER_SESSION // 2)
            # Fill remaining slots with globally popular items (search ads / feed filler).
            shown = np.concatenate([shown, sample_top(avail_by_day[day], SLOTS_PER_SESSION - len(shown), exclude=shown)])
            if rebuy_item >= 0 and oos_day[rebuy_item] > day:
                shown = np.concatenate([[rebuy_item], shown[shown != rebuy_item]])[:SLOTS_PER_SESSION]

            # ---- user response, vectorised over the shown slate ---- #
            c = item_cat[shown]
            eff_disc = np.minimum(discount[shown] + (0.2 if sale[day] else 0.0), 0.7)
            utility = (
                1.0 * (item_vec[shown] @ taste[u]) / np.sqrt(TASTE_DIM)
                + 0.6 * np.log((cat_pref[u, c] + 0.01) * n_cat)
                + 1.0 * (c == intent)
                + 0.5 * quality[shown]
                - price_sens[u] * log_price_ratio[shown]
                + 1.5 * eff_disc
                + 0.3 * free_ship[shown]
                + 0.3 * (shop_city[item_shop[shown]] == user_city[u])
                + 2.5 * (shown == rebuy_item)
                - 3.0 * np.isin(shown, owned_durables)
            )
            pos = np.arange(len(shown))
            p_click = _sigmoid(utility - 4.2 - 0.12 * pos)
            clicked = rng.random(len(shown)) < p_click

            for j, it in enumerate(shown):
                ts = t0 + 5 * j + int(rng.integers(0, 5))
                events.append((ts, u, sid, it, "impression", j, np.nan, np.nan))
                if not clicked[j]:
                    if surface == "home_feed" and utility[j] < 0 and rng.random() < 0.05:
                        events.append((ts + 2, u, sid, it, "hide", j, np.nan, np.nan))
                    continue
                dwell = float(np.clip(rng.exponential(15 * np.exp(0.3 * utility[j])), 1, 600))
                tc = ts + 10
                events.append((tc, u, sid, it, "click", j, round(dwell, 1), np.nan))
                if rng.random() >= _sigmoid(utility[j] - 4.0):
                    continue
                tcart = tc + int(dwell)
                events.append((tcart, u, sid, it, "add_to_cart", j, np.nan, np.nan))
                p_order = _sigmoid(-0.4 + 0.6 * payday[day] + 0.8 * sale[day] + 0.4 * quality[it])
                if rng.random() >= p_order:
                    continue
                to = tcart + 60 * int(rng.integers(1, 30))
                paid = round(price[it] * (1 - eff_disc[j]), -2)
                events.append((to, u, sid, it, "order", j, np.nan, paid))
                if consumable[item_cat[it]]:
                    consumable_orders[int(it)] = int(day)
                else:
                    owned_durables.append(int(it))
                if rng.random() < _sigmoid(-3.0 - 1.2 * quality[it]):
                    events.append((to + DAY * int(rng.integers(3, 8)), u, sid, it, "return", j, np.nan, np.nan))

    def to_ts(sec) -> pd.Series:
        return start + pd.to_timedelta(sec, unit="s")

    sessions_df = pd.DataFrame(sessions, columns=schema.SESSIONS.column_names)
    sessions_df["started_at"] = to_ts(sessions_df["started_at"])
    searches_df = pd.DataFrame(searches, columns=schema.SEARCHES.column_names)
    searches_df["ts"] = to_ts(searches_df["ts"])
    ev = pd.DataFrame(events, columns=[c for c in schema.EVENTS.column_names if c != "event_id"])
    ev = ev[ev["ts"] < cfg.n_days * DAY]  # drop returns that land past the horizon
    ev = ev.sort_values("ts", kind="stable").reset_index(drop=True)
    ev["ts"] = to_ts(ev["ts"])
    ev.insert(0, "event_id", np.arange(len(ev)))

    tables = {
        "categories": categories,
        "shops": shops,
        "items": items,
        "users": users,
        "sessions": sessions_df,
        "searches": searches_df,
        "events": ev,
    }
    for t in schema.RAW_TABLES:  # enforce the contract
        assert list(tables[t.name].columns) == t.column_names, t.name
    return tables
