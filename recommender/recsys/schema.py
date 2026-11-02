"""Data contracts for the prototype — the single source of truth.

Two layers:

1. RAW TABLES: what the app logs (events, searches, sessions) and what the
   catalog DB holds (users, items, shops, categories). Raw = facts, never
   aggregates. Hidden "truth" used by the simulator (a user's real taste, an
   item's real quality) is deliberately NOT here: a real platform never sees it.

2. FEATURES: numbers computed from raw tables *as of a cutoff time* for one
   (user, item) candidate pair. The ranker sees only these. `features.py`
   asserts its output columns equal `FEATURE_NAMES`, so this file can't drift.

Run `uv run recsys schema` to print everything below.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str
    description: str


@dataclass(frozen=True)
class Table:
    name: str
    grain: str  # what one row represents
    columns: tuple[Column, ...]

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


# --------------------------------------------------------------------------- #
# Raw tables
# --------------------------------------------------------------------------- #

CATEGORIES = Table(
    "categories",
    "one leaf category",
    (
        Column("category_id", "int", "Leaf category key"),
        Column("name", "str", "Leaf name, e.g. 'Phone Cases'"),
        Column("parent_name", "str", "Top-level category, e.g. 'Electronics'"),
        Column("is_consumable", "bool", "Bought repeatedly (diapers, coffee) -> re-purchase is expected"),
        Column("median_price", "float", "Typical price in the category (IDR)"),
    ),
)

SHOPS = Table(
    "shops",
    "one seller",
    (
        Column("shop_id", "int", "Seller key"),
        Column("city", "str", "Ship-from city"),
        Column("is_mall", "bool", "Official/Mall store badge"),
        Column("shop_rating", "float", "Average seller rating 1-5"),
        Column("response_rate", "float", "Share of chats answered"),
        Column("cancel_rate", "float", "Share of orders cancelled by seller"),
    ),
)

ITEMS = Table(
    "items",
    "one product listing (catalog snapshot)",
    (
        Column("item_id", "int", "Listing key"),
        Column("shop_id", "int", "Seller"),
        Column("category_id", "int", "Leaf category"),
        Column("brand", "str", "Brand name"),
        Column("title", "str", "Listing title"),
        Column("price", "float", "List price (IDR)"),
        Column("discount_pct", "float", "Standing discount 0-0.5"),
        Column("free_shipping", "bool", "Eligible for free-shipping voucher"),
        Column("rating_avg", "float", "Average product rating 1-5"),
        Column("rating_count", "int", "Number of ratings"),
        Column("listed_at", "datetime", "When the listing went live"),
        Column("out_of_stock_at", "datetime|null", "When stock ran out (null = still in stock)"),
    ),
)

USERS = Table(
    "users",
    "one buyer account",
    (
        Column("user_id", "int", "Buyer key"),
        Column("signup_at", "datetime", "Account creation time"),
        Column("city", "str", "Delivery city"),
        Column("age_bucket", "int", "0:<20 1:20-29 2:30-39 3:40+"),
        Column("device_tier", "int", "Phone price tier 1(low)-3(high); proxy for spending power"),
    ),
)

SESSIONS = Table(
    "sessions",
    "one app visit (request-time context lives here)",
    (
        Column("session_id", "int", "Visit key"),
        Column("user_id", "int", "Buyer"),
        Column("started_at", "datetime", "Visit start"),
        Column("surface", "str", "'home_feed' or 'search' — where the items were shown"),
        Column("device_os", "str", "'android' | 'ios' | 'web'"),
    ),
)

SEARCHES = Table(
    "searches",
    "one submitted search query",
    (
        Column("search_id", "int", "Query key"),
        Column("session_id", "int", "Visit"),
        Column("user_id", "int", "Buyer"),
        Column("ts", "datetime", "When submitted"),
        Column("query", "str", "Raw text the user typed"),
    ),
)

EVENTS = Table(
    "events",
    "one user action on one item (the main interaction log)",
    (
        Column("event_id", "int", "Event key"),
        Column("ts", "datetime", "When it happened"),
        Column("user_id", "int", "Buyer"),
        Column("session_id", "int", "Visit"),
        Column("item_id", "int", "Item acted on"),
        Column(
            "event_type",
            "str",
            "impression | click | add_to_cart | order | hide | return  (see EVENT_TYPES)",
        ),
        Column("position", "int", "Slot the item was shown in (0 = top); position bias matters"),
        Column("dwell_sec", "float|null", "Seconds on product page (clicks only)"),
        Column("price_paid", "float|null", "Effective unit price (orders only, after discounts)"),
    ),
)

RAW_TABLES: tuple[Table, ...] = (CATEGORIES, SHOPS, ITEMS, USERS, SESSIONS, SEARCHES, EVENTS)

# Implicit-feedback strength. Used for candidate scores and for labels.
EVENT_TYPES = ("impression", "click", "add_to_cart", "order", "hide", "return")
EVENT_WEIGHT = {"click": 1.0, "add_to_cart": 3.0, "order": 5.0}
# Graded relevance for the ranker label (what happened in the label window).
LABEL_GAIN = {"click": 1, "add_to_cart": 2, "order": 3}


# --------------------------------------------------------------------------- #
# Features (one row = one (user, item) candidate at a cutoff time)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Feature:
    name: str
    group: str
    description: str


FEATURES: tuple[Feature, ...] = (
    # -- candidate generation: which retriever proposed it and how strongly --
    Feature("cand_recent_score", "candidate", "Time-decayed weight of the user's own past actions on this item"),
    Feature("cand_covisit_score", "candidate", "Co-visitation: how often this item is clicked in the same session as the user's recent items"),
    Feature("cand_catpop_score", "candidate", "Popularity of the item inside the user's favourite categories"),
    Feature("cand_global_pop_rank", "candidate", "Rank in global trending list (999 = not in list)"),
    Feature("cand_n_sources", "candidate", "How many retrievers proposed this item"),
    # -- user: who is asking --
    Feature("user_clicks_7d", "user", "User's clicks in the last 7 days (activity level)"),
    Feature("user_clicks_30d", "user", "User's clicks in the last 30 days"),
    Feature("user_orders_30d", "user", "User's orders in the last 30 days"),
    Feature("user_ctr_30d", "user", "User's click-through rate (smoothed)"),
    Feature("user_days_since_active", "user", "Days since the user's last event"),
    Feature("user_account_age_days", "user", "Days since signup"),
    Feature("user_device_tier", "user", "Phone tier 1-3"),
    Feature("user_age_bucket", "user", "Age bucket 0-3"),
    # -- item: what is being recommended --
    Feature("item_price", "item", "List price"),
    Feature("item_discount_pct", "item", "Standing discount"),
    Feature("item_price_vs_category", "item", "Price / category median (cheap vs premium)"),
    Feature("item_free_shipping", "item", "Free-shipping eligible"),
    Feature("item_rating_avg", "item", "Average rating"),
    Feature("item_rating_count", "item", "Number of ratings (log1p)"),
    Feature("item_age_days", "item", "Days since listed (freshness / cold start)"),
    Feature("item_impressions_7d", "item", "Times shown in last 7 days"),
    Feature("item_clicks_7d", "item", "Clicks in last 7 days"),
    Feature("item_orders_7d", "item", "Orders in last 7 days"),
    Feature("item_ctr_30d", "item", "Clicks / impressions, Bayesian-smoothed"),
    Feature("item_cvr_30d", "item", "Orders / clicks, Bayesian-smoothed"),
    Feature("item_trend", "item", "Clicks last 3d vs last 14d rate (>1 = heating up)"),
    Feature("item_return_rate", "item", "Returns / orders, smoothed (quality signal)"),
    Feature("item_is_consumable", "item", "Category is repeat-purchase"),
    # -- shop / seller --
    Feature("shop_rating", "shop", "Seller rating"),
    Feature("shop_is_mall", "shop", "Mall / official store"),
    Feature("shop_cancel_rate", "shop", "Seller cancellation rate"),
    Feature("shop_response_rate", "shop", "Seller chat response rate"),
    # -- user x item: personal history with this exact item --
    Feature("ui_clicks", "user_item", "Times this user clicked this item"),
    Feature("ui_carts", "user_item", "Times this user carted this item"),
    Feature("ui_orders", "user_item", "Times this user ordered this item"),
    Feature("ui_hours_since_last", "user_item", "Hours since user last touched the item (9999 = never)"),
    Feature("ui_ignored_impressions", "user_item", "Times shown but never clicked (fatigue)"),
    Feature("ui_max_dwell_sec", "user_item", "Longest product-page dwell on this item"),
    # -- user x category --
    Feature("uc_click_share_30d", "user_category", "Share of user's clicks in this item's category"),
    Feature("uc_searches_30d", "user_category", "User's searches that map to this category"),
    Feature("uc_orders", "user_category", "User's orders in this category"),
    Feature("uc_days_since_order", "user_category", "Days since last order in the category (re-purchase cycle; 999 = never)"),
    # -- user x shop --
    Feature("us_clicks", "user_shop", "User's clicks on this shop's items"),
    Feature("us_orders", "user_shop", "User's orders from this shop"),
    # -- price fit & context --
    Feature("price_ratio_to_user", "user_item", "Item price / user's median clicked price in the category"),
    Feature("same_city", "context", "Ship-from city == user's city (faster, cheaper delivery)"),
)

FEATURE_NAMES: list[str] = [f.name for f in FEATURES]


def describe() -> str:
    """Human-readable dump of the schema (used by `recsys schema`)."""
    out: list[str] = ["RAW TABLES", "=========="]
    for t in RAW_TABLES:
        out.append(f"\n{t.name}  (one row = {t.grain})")
        for c in t.columns:
            out.append(f"  {c.name:<18} {c.dtype:<14} {c.description}")
    out += ["", "FEATURES (one row = one (user, item) candidate at cutoff time)", "=" * 62]
    group = None
    for f in FEATURES:
        if f.group != group:
            group = f.group
            out.append(f"\n[{group}]")
        out.append(f"  {f.name:<26} {f.description}")
    return "\n".join(out)
