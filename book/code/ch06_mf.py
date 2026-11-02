"""Chapter 6 -- matrix factorisation: implicit ALS (Hu-Koren-Volinsky) and BPR, in plain numpy."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common
import dense_recs

FACTORS = 32


def als_implicit(R: np.ndarray, factors=FACTORS, reg=0.1, alpha=20.0, iters=15, seed=0):
    """Minimise  sum_ui c_ui (p_ui - x_u . y_i)^2 + reg (|X|^2 + |Y|^2)
    with p = 1[r > 0] and c = 1 + alpha * r, by alternating exact ridge solves per row."""
    rng = np.random.default_rng(seed)
    n_u, n_i = R.shape
    P = (R > 0).astype(np.float64)
    C = 1.0 + alpha * R
    X = rng.normal(0, 0.1, (n_u, factors))
    Y = rng.normal(0, 0.1, (n_i, factors))
    eye = reg * np.eye(factors)

    def solve(fixed, conf, pref):
        gram = fixed.T @ fixed  # computed once per sweep: the HKV trick
        out = np.empty((conf.shape[0], factors))
        for r in range(conf.shape[0]):
            nz = np.flatnonzero(conf[r] > 1.0)
            Fn = fixed[nz]
            cu = conf[r, nz]
            A = gram + Fn.T @ ((cu - 1.0)[:, None] * Fn) + eye
            b = Fn.T @ (cu * pref[r, nz])
            out[r] = np.linalg.solve(A, b)
        return out

    losses = []
    for _ in range(iters):
        X = solve(Y, C, P)
        Y = solve(X, C.T, P.T)
        err = P - X @ Y.T
        losses.append(float((C * err**2).sum() + reg * ((X**2).sum() + (Y**2).sum())))
    return X, Y, losses


def bpr(R: np.ndarray, factors=FACTORS, lr=0.05, reg=0.01, epochs=30, batch=256, seed=0):
    """Bayesian Personalised Ranking: for each positive (u, i) draw a negative j and push
    x_u.y_i above x_u.y_j (maximise log sigmoid of the gap). Mini-batch SGD in numpy."""
    rng = np.random.default_rng(seed)
    n_u, n_i = R.shape
    pos_mask = R > 0
    users, items = np.nonzero(pos_mask)
    X = rng.normal(0, 0.1, (n_u, factors))
    Y = rng.normal(0, 0.1, (n_i, factors))
    losses = []
    for _ in range(epochs):
        order = rng.permutation(len(users))
        total = 0.0
        for s in range(0, len(order), batch):
            b = order[s : s + batch]
            u, i = users[b], items[b]
            j = rng.integers(0, n_i, len(b))
            for _retry in range(5):  # resample negatives that are actually positives
                bad = pos_mask[u, j]
                if not bad.any():
                    break
                j[bad] = rng.integers(0, n_i, bad.sum())
            xu, yi, yj = X[u], Y[i], Y[j]
            gap = (xu * (yi - yj)).sum(1)
            sig = 1.0 / (1.0 + np.exp(gap))  # = 1 - sigmoid(gap) = -d loss / d gap
            total += float(np.log1p(np.exp(-gap)).sum())
            gu = sig[:, None] * (yi - yj) - reg * xu
            gi = sig[:, None] * xu - reg * yi
            gj = -sig[:, None] * xu - reg * yj
            np.add.at(X, u, lr * gu)
            np.add.at(Y, i, lr * gi)
            np.add.at(Y, j, lr * gj)
        losses.append(total / len(users))
    return X, Y, losses


def main() -> None:
    hist = common.history()
    users = common.eval_users()
    avail = common.available_items()["item_id"].to_numpy()
    m_sp, user_index, item_index = common.interaction_matrix(hist)
    R = m_sp.toarray().astype(np.float64)

    common.section("Setup")
    print(f"matrix {R.shape[0]} users x {R.shape[1]} items, {m_sp.nnz} observed cells")
    print(f"factors={FACTORS}")

    user_rows = user_index.get_indexer(users)
    item_cols = item_index.get_indexer(avail)
    ok_i = item_cols >= 0

    def scores_for(X, Y):
        s = np.zeros((len(users), len(avail)))
        ur = user_rows >= 0
        full = X[user_rows[ur]] @ Y[item_cols[ok_i]].T
        tmp = np.zeros((ur.sum(), len(avail)))
        tmp[:, ok_i] = full
        s[ur] = tmp
        return s

    results, padding, out = [], {}, {}

    common.section("ALS (implicit): factors 32, lambda 0.1, alpha 20, 15 iterations")
    t = time.perf_counter()
    X, Y, als_loss = als_implicit(R)
    print(f"trained in {time.perf_counter() - t:.1f}s")
    for it, l in enumerate(als_loss, 1):
        print(f"iteration {it:>2d}  loss {l:>14.1f}")
    recs, st = dense_recs.scores_to_recs(scores_for(X, Y), users, avail)
    results.append(common.evaluate(recs, "als_implicit"))
    padding["als_implicit"] = st

    common.section("BPR: factors 32, lr 0.05, reg 0.01, 30 epochs, 1 negative per positive")
    t = time.perf_counter()
    Xb, Yb, bpr_loss = bpr(R)
    print(f"trained in {time.perf_counter() - t:.1f}s")
    for ep, l in enumerate(bpr_loss, 1):
        print(f"epoch {ep:>2d}  mean -log sigmoid(gap) {l:.4f}")
    recs, st = dense_recs.scores_to_recs(scores_for(Xb, Yb), users, avail)
    results.append(common.evaluate(recs, "bpr"))
    padding["bpr"] = st

    common.section("Padding with 7-day popularity")
    for name, s in padding.items():
        print(f"{name:<13s} users without scores {s['users_without_model_scores']:>4d}  "
              f"slots padded {s['share_of_slots_from_popularity_padding']:.1%}")

    common.section("Results (top-50 lists, evaluated at K=20)")
    common.print_table(results)

    common.save_results("ch06_mf", {
        "results": results,
        "padding": padding,
        "als_loss": als_loss,
        "bpr_loss": bpr_loss,
        "settings": {"factors": FACTORS, "als": {"reg": 0.1, "alpha": 20, "iters": 15},
                     "bpr": {"lr": 0.05, "reg": 0.01, "epochs": 30}},
    })


if __name__ == "__main__":
    main()
