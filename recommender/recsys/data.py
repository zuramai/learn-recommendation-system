"""Load/save raw tables and slice them at a cutoff time.

The single most important rule in recsys feature engineering: when you build
features "as of" time T, you may only look at events with ts < T. Breaking it
("leakage") makes offline metrics look great and production results bad.
`History` is the only way the rest of the code sees events.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import schema
from .simulate import SimConfig

DATETIME_COLS = {
    "items": ["listed_at", "out_of_stock_at"],
    "users": ["signup_at"],
    "sessions": ["started_at"],
    "searches": ["ts"],
    "events": ["ts"],
}


@dataclass(frozen=True)
class Meta:
    start: pd.Timestamp
    n_days: int

    def day(self, d: int) -> pd.Timestamp:
        return self.start + pd.Timedelta(days=d)

    @property
    def end(self) -> pd.Timestamp:
        return self.start + pd.Timedelta(days=self.n_days)


def save(tables: dict[str, pd.DataFrame], cfg: SimConfig, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
    (out / "meta.json").write_text(json.dumps({"start_date": cfg.start_date, "n_days": cfg.n_days}))


def load(src: Path) -> tuple[dict[str, pd.DataFrame], Meta]:
    if not (src / "meta.json").exists():
        raise SystemExit(f"No dataset in {src}/ — run `uv run recsys simulate` first.")
    tables = {}
    for t in schema.RAW_TABLES:
        tables[t.name] = pd.read_csv(src / f"{t.name}.csv", parse_dates=DATETIME_COLS.get(t.name, []))
    m = json.loads((src / "meta.json").read_text())
    return tables, Meta(pd.Timestamp(m["start_date"]), m["n_days"])


@dataclass(frozen=True)
class History:
    """Everything knowable at `cutoff`: catalog snapshot + events/searches strictly before it."""

    cutoff: pd.Timestamp
    events: pd.DataFrame  # with extra column age_days = (cutoff - ts) in days
    searches: pd.DataFrame  # with extra column age_days
    items: pd.DataFrame  # only items already listed at cutoff, joined with category info
    users: pd.DataFrame
    shops: pd.DataFrame
    categories: pd.DataFrame

    @classmethod
    def at(cls, tables: dict[str, pd.DataFrame], cutoff: pd.Timestamp) -> History:
        ev = tables["events"]
        ev = ev[ev["ts"] < cutoff].copy()
        ev["age_days"] = (cutoff - ev["ts"]) / pd.Timedelta(days=1)
        se = tables["searches"]
        se = se[se["ts"] < cutoff].copy()
        se["age_days"] = (cutoff - se["ts"]) / pd.Timedelta(days=1)
        cats = tables["categories"]
        items = tables["items"]
        items = items[items["listed_at"] <= cutoff].merge(
            cats[["category_id", "is_consumable", "median_price"]], on="category_id"
        )
        # Stock status is known at cutoff; the *future* stock-out time is not.
        items["in_stock"] = items["out_of_stock_at"].isna() | (items["out_of_stock_at"] > cutoff)
        items = items.drop(columns="out_of_stock_at")
        return cls(cutoff, ev, se, items, tables["users"], tables["shops"], cats)

    def interactions(self) -> pd.DataFrame:
        """Positive actions (click/cart/order) with their strength weight."""
        pos = self.events[self.events["event_type"].isin(schema.EVENT_WEIGHT)].copy()
        pos["w"] = pos["event_type"].map(schema.EVENT_WEIGHT)
        return pos

    def searched_categories(self) -> pd.DataFrame:
        """Map raw queries to categories (here: naive substring match on the category name).

        Production systems use a query classifier; the output shape is the same:
        (user_id, category_id, age_days) per search.
        """
        q = self.searches["query"].str.lower()
        parts = []
        for cid, name in zip(self.categories["category_id"], self.categories["name"].str.lower()):
            hit = self.searches[q.str.contains(name, regex=False)]
            parts.append(hit[["user_id", "age_days"]].assign(category_id=cid))
        return pd.concat(parts, ignore_index=True)


def labels(tables: dict[str, pd.DataFrame], cutoff: pd.Timestamp, days: int) -> pd.DataFrame:
    """What each user did in [cutoff, cutoff + days): graded label per (user, item)."""
    ev = tables["events"]
    win = ev[(ev["ts"] >= cutoff) & (ev["ts"] < cutoff + pd.Timedelta(days=days))]
    win = win[win["event_type"].isin(schema.LABEL_GAIN)]
    out = win.assign(label=win["event_type"].map(schema.LABEL_GAIN))
    return out.groupby(["user_id", "item_id"], as_index=False)["label"].max()
