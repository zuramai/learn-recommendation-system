"""Leakage guard: features/candidates built at a cutoff must not depend on anything after it.

If someone adds a feature that accidentally reads future events, offline metrics
silently inflate. This test rewrites the entire future and checks nothing changes.
"""

import pandas as pd
import pytest

from recsys import candidates, data, features
from recsys.simulate import SimConfig, simulate

CUTOFF_DAY = 30


@pytest.fixture(scope="module")
def tables():
    return simulate(SimConfig(n_users=200, n_items=400, n_shops=30, seed=1))


def _build(tables, cutoff):
    hist = data.History.at(tables, cutoff)
    users = tables["users"]["user_id"].to_numpy()
    feats = features.build(candidates.generate(hist, users), hist)
    return feats.sort_values(["user_id", "item_id"]).reset_index(drop=True)


def test_features_ignore_events_after_cutoff(tables):
    cutoff = pd.Timestamp(SimConfig().start_date) + pd.Timedelta(days=CUTOFF_DAY)
    before = _build(tables, cutoff)

    # Rewrite the future: every post-cutoff event becomes an order of item 0, and
    # every post-cutoff search becomes a query for a real category.
    future = tables["events"]["ts"] >= cutoff
    assert future.any()
    tampered = dict(tables)
    ev = tables["events"].copy()
    ev.loc[future, ["event_type", "item_id"]] = ["order", 0]
    tampered["events"] = ev
    se = tables["searches"].copy()
    se.loc[se["ts"] >= cutoff, "query"] = "diapers"
    tampered["searches"] = se

    pd.testing.assert_frame_equal(before, _build(tampered, cutoff))


def test_items_listed_after_cutoff_are_never_candidates(tables):
    cutoff = pd.Timestamp(SimConfig().start_date) + pd.Timedelta(days=CUTOFF_DAY)
    feats = _build(tables, cutoff)
    listed = tables["items"].set_index("item_id")["listed_at"]
    assert (listed.loc[feats["item_id"]] <= cutoff).all()
