"""Fill the marketplace database with a simulated history, so recommendations work immediately.

The simulated 60 days are shifted so they end "now"; every simulator id becomes
id + 1 (database ids start at 1; user_id 0 means "anonymous" in the gRPC API).
This is the only code in the recommender that writes to the app database.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from argon2 import PasswordHasher

from . import db
from .simulate import SimConfig, simulate

TABLES = ["users", "categories", "shops", "items", "sessions", "searches", "events"]
SERIAL_TABLES = ["users", "shops", "items", "sessions", "searches", "events"]
DEMO_PASSWORD = "password"


def _none(v):
    """NaN/NaT -> NULL for COPY."""
    return None if v is None or (not isinstance(v, str) and pd.isna(v)) else v


def _rows(df: pd.DataFrame):
    for row in df.itertuples(index=False, name=None):
        yield tuple(_none(v.item() if isinstance(v, np.generic) else v) for v in row)


def build_rows(cfg: SimConfig) -> dict[str, pd.DataFrame]:
    """Simulated tables -> DB-shaped frames (column order = COPY column order)."""
    t = simulate(cfg)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("min")
    shift = now - (pd.Timestamp(cfg.start_date) + pd.Timedelta(days=cfg.n_days))
    for name, cols in {"items": ["listed_at", "out_of_stock_at"], "users": ["signup_at"],
                       "sessions": ["started_at"], "searches": ["ts"], "events": ["ts"]}.items():  # fmt: skip
        for c in cols:
            t[name][c] = t[name][c] + shift
    sim_start = pd.Timestamp(cfg.start_date) + shift

    pw = PasswordHasher().hash(DEMO_PASSWORD)
    u = t["users"]
    buyers = pd.DataFrame({"id": u["user_id"] + 1})
    buyers["email"] = "user" + buyers["id"].astype(str) + "@demo.test"
    buyers["password_hash"] = pw
    buyers["name"] = "Demo User " + buyers["id"].astype(str)
    buyers["city"] = u["city"]
    buyers["age_bucket"] = u["age_bucket"]
    buyers["device_tier"] = u["device_tier"]
    buyers["is_admin"] = False
    buyers["created_at"] = u["signup_at"]

    s = t["shops"]
    shop_id = s["shop_id"] + 1
    sellers = pd.DataFrame({"id": cfg.n_users + shop_id})
    sellers["email"] = "seller" + shop_id.astype(str) + "@demo.test"
    sellers["password_hash"] = pw
    sellers["name"] = "Seller " + shop_id.astype(str)
    sellers["city"] = s["city"]
    sellers["age_bucket"] = 1
    sellers["device_tier"] = 2
    sellers["is_admin"] = False
    sellers["created_at"] = sim_start
    admin = pd.DataFrame([{"id": cfg.n_users + cfg.n_shops + 1, "email": "admin@demo.test", "password_hash": pw,
                           "name": "Admin", "city": "Jakarta", "age_bucket": 1, "device_tier": 2, "is_admin": True,
                           "created_at": sim_start}])  # fmt: skip
    users = pd.concat([buyers, sellers, admin], ignore_index=True)

    shops = pd.DataFrame({"id": shop_id, "owner_user_id": sellers["id"]})
    shops["name"] = s["city"] + " Store " + shop_id.astype(str)
    for c in ["city", "is_mall", "shop_rating", "response_rate", "cancel_rate"]:
        shops[c] = s[c]
    shops["created_at"] = sim_start

    c = t["categories"]
    categories = pd.DataFrame({"id": c["category_id"] + 1, "name": c["name"], "parent_name": c["parent_name"],
                               "is_consumable": c["is_consumable"], "median_price": c["median_price"].astype(int)})  # fmt: skip

    it = t["items"]
    city_of_shop = dict(zip(s["shop_id"], s["city"]))
    oos = it["out_of_stock_at"].where(it["out_of_stock_at"] <= now)
    items = pd.DataFrame({"id": it["item_id"] + 1, "shop_id": it["shop_id"] + 1, "category_id": it["category_id"] + 1})
    for col in ["brand", "title"]:
        items[col] = it[col]
    items["description"] = it["title"] + ". Ships from " + it["shop_id"].map(city_of_shop) + "."
    items["price"] = it["price"].astype(int)
    for col in ["discount_pct", "free_shipping", "rating_avg", "rating_count"]:
        items[col] = it[col]
    items["stock"] = np.where(oos.isna(), 100, 0)
    items["is_active"] = True
    items["listed_at"] = it["listed_at"]
    items["out_of_stock_at"] = oos

    se = t["sessions"]
    sessions = pd.DataFrame({"id": se["session_id"] + 1, "user_id": se["user_id"] + 1, "started_at": se["started_at"],
                             "last_seen_at": se["started_at"], "surface": se["surface"], "device_os": se["device_os"]})  # fmt: skip
    sq = t["searches"]
    searches = pd.DataFrame({"id": sq["search_id"] + 1, "session_id": sq["session_id"] + 1,
                             "user_id": sq["user_id"] + 1, "ts": sq["ts"], "query": sq["query"]})  # fmt: skip
    ev = t["events"]
    events = pd.DataFrame({"id": ev["event_id"] + 1, "ts": ev["ts"], "user_id": ev["user_id"] + 1,
                           "session_id": ev["session_id"] + 1, "item_id": ev["item_id"] + 1,
                           "event_type": ev["event_type"],
                           "position": ev["position"].astype("Int64"), "dwell_sec": ev["dwell_sec"],
                           "price_paid": ev["price_paid"], "surface": None})  # fmt: skip
    return {"users": users, "categories": categories, "shops": shops, "items": items,
            "sessions": sessions, "searches": searches, "events": events}  # fmt: skip


def seed_db(cfg: SimConfig) -> None:
    rows = build_rows(cfg)
    with db.connect(read_only=False) as conn, conn.transaction(), conn.cursor() as cur:
        cur.execute(
            "TRUNCATE order_items, orders, cart_items, events, searches, sessions, items, shops, categories, users "
            "RESTART IDENTITY CASCADE"
        )
        for name in TABLES:
            df = rows[name]
            # Timestamps are naive UTC; tell Postgres so regardless of the session time zone.
            df = df.copy()
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].dt.tz_localize("UTC")
            with cur.copy(f"COPY {name} ({', '.join(df.columns)}) FROM STDIN") as copy:
                for r in _rows(df):
                    copy.write_row(r)
        for name in SERIAL_TABLES:
            cur.execute(f"SELECT setval(pg_get_serial_sequence('{name}', 'id'), (SELECT MAX(id) FROM {name}))")

    print("Seeded marketplace database:")
    for name in TABLES:
        print(f"  {name:<11} {len(rows[name]):>8,} rows")
    print("\nDemo logins (password: password):")
    print("  buyer   user43@demo.test   (any user1..user%d)" % cfg.n_users)
    print("  seller  seller1@demo.test  (any seller1..seller%d)" % cfg.n_shops)
    print("  admin   admin@demo.test")
