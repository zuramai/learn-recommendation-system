"""Read the marketplace Postgres database into the same tables the simulator produces.

The recommender never writes to the app database: `connect(read_only=True)` sets
`default_transaction_read_only`, so an accidental INSERT fails loudly.

`Store` keeps an in-memory copy and refreshes it incrementally (events and
searches by id), so a click logged a moment ago is visible to the next request.
"""

from __future__ import annotations

import os
import threading
import time

import pandas as pd
import psycopg

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/marketplace")

CATEGORIES_SQL = (
    "SELECT id AS category_id, name, parent_name, is_consumable, median_price::float8 AS median_price "
    "FROM categories ORDER BY id"
)
SHOPS_SQL = "SELECT id AS shop_id, city, is_mall, shop_rating, response_rate, cancel_rate FROM shops"
ITEMS_SQL = (
    "SELECT id AS item_id, shop_id, category_id, brand, title, price::float8 AS price, discount_pct, free_shipping, "
    "rating_avg, rating_count, listed_at, out_of_stock_at FROM items WHERE is_active"
)
USERS_SQL = "SELECT id AS user_id, created_at AS signup_at, city, age_bucket, device_tier FROM users"
SEARCHES_SQL = "SELECT id AS search_id, session_id, user_id, ts, query FROM searches WHERE id > %s ORDER BY id"
EVENTS_SQL = (
    "SELECT id AS event_id, ts, user_id, session_id, item_id, event_type, position, dwell_sec, price_paid "
    "FROM events WHERE id > %s ORDER BY id"
)

DATETIME_COLS = ("listed_at", "out_of_stock_at", "signup_at", "ts")
FLOAT_COLS = ("position", "dwell_sec", "price_paid")
OVERLAP_IDS = 1000  # re-read this many ids back: catches rows committed out of id order
MIN_REFRESH_SEC = 1.0


def connect(read_only: bool) -> psycopg.Connection:
    options = "-c default_transaction_read_only=on" if read_only else ""
    return psycopg.connect(DATABASE_URL, autocommit=True, options=options)


def _frame(cur: psycopg.Cursor, sql: str, params=None) -> pd.DataFrame:
    cur.execute(sql, params)
    df = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
    for c in df.columns:
        if c in DATETIME_COLS:
            df[c] = pd.to_datetime(df[c], utc=True).dt.tz_convert(None)
        elif c in FLOAT_COLS:
            df[c] = df[c].astype(float)
    return df


class Store:
    """In-memory snapshot of the DB, shaped like `simulate()` output (minus sessions)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conn: psycopg.Connection | None = None
        self._tables: dict[str, pd.DataFrame] = {}
        self._last_event_id = 0
        self._last_search_id = 0
        self._last_refresh = 0.0

    def _cursor(self) -> psycopg.Cursor:
        if self._conn is None or self._conn.closed:
            self._conn = connect(read_only=True)
        return self._conn.cursor()

    def refresh(self) -> None:
        with self._lock:
            if self._tables and time.monotonic() - self._last_refresh < MIN_REFRESH_SEC:
                return
            try:
                self._refresh()
            except psycopg.OperationalError:
                self._conn = None  # reconnect once (e.g. Postgres restarted)
                self._refresh()
            self._last_refresh = time.monotonic()

    def _refresh(self) -> None:
        with self._cursor() as cur:
            t = {
                "categories": _frame(cur, CATEGORIES_SQL),
                "shops": _frame(cur, SHOPS_SQL),
                "items": _frame(cur, ITEMS_SQL),
                "users": _frame(cur, USERS_SQL),
            }
            new_searches = _frame(cur, SEARCHES_SQL, (self._last_search_id - OVERLAP_IDS,))
            new_events = _frame(cur, EVENTS_SQL, (self._last_event_id - OVERLAP_IDS,))
        t["searches"] = self._merge("searches", new_searches, "search_id")
        t["events"] = self._merge("events", new_events, "event_id")
        if len(t["searches"]):
            self._last_search_id = int(t["searches"]["search_id"].max())
        if len(t["events"]):
            self._last_event_id = int(t["events"]["event_id"].max())
        self._tables = t

    def _merge(self, name: str, new: pd.DataFrame, key: str) -> pd.DataFrame:
        old = self._tables.get(name)
        if old is None or old.empty:
            df = new
        elif new.empty:
            return old
        else:
            df = pd.concat([old, new], ignore_index=True)
        df = df.drop_duplicates(subset=key, keep="last")
        return df.sort_values("ts", kind="stable").reset_index(drop=True)

    def snapshot(self) -> dict[str, pd.DataFrame]:
        with self._lock:
            return dict(self._tables)

    @property
    def events_loaded(self) -> int:
        return len(self._tables.get("events", ()))
