"""Stage 3 — re-ranking: business rules on top of model scores.

The model maximises predicted engagement per item; it doesn't know about the
page as a whole. Re-ranking fixes that:

* Don't recommend a durable item the user already bought (the fridge problem).
  Consumables (coffee, diapers) are exempt: re-buying them is the point.
* Diversity: cap items per category and per shop so one hot category or one
  big seller doesn't fill the whole page.

These rules usually cost a little offline NDCG and win online (users get bored
of 20 phone cases). `evaluate` reports both so you can see the trade-off.
"""

from __future__ import annotations

import pandas as pd

MAX_PER_CATEGORY = 8
MAX_PER_SHOP = 4


def rerank(scored: pd.DataFrame, k: int) -> pd.DataFrame:
    """`scored` needs user_id, item_id, category_id, shop_id, ui_orders, item_is_consumable, score."""
    bought_durable = (scored["ui_orders"] > 0) & (scored["item_is_consumable"] == 0)
    scored = scored[~bought_durable].sort_values(["user_id", "score"], ascending=[True, False])
    # Keep an item only if it is within the user's top-N for its category AND top-M for its shop.
    cat_rank = scored.groupby(["user_id", "category_id"]).cumcount()
    shop_rank = scored.groupby(["user_id", "shop_id"]).cumcount()
    kept = scored[(cat_rank < MAX_PER_CATEGORY) & (shop_rank < MAX_PER_SHOP)]
    return kept.groupby("user_id").head(k)
