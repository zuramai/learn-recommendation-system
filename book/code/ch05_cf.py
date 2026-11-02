"""Chapter 5 -- collaborative filtering: user-kNN, item-kNN and session co-visitation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common
import dense_recs
from recsys import candidates


def row_normalise(m: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(norm, 1e-12)


def main() -> None:
    hist = common.history()
    users = common.eval_users()
    avail = common.available_items()["item_id"].to_numpy()

    common.section("Setup")
    m_sp, user_index, item_index = common.interaction_matrix(hist)
    R = m_sp.toarray().astype(np.float64)  # users x items, summed event weights
    print(f"interaction matrix: {R.shape[0]} users x {R.shape[1]} items, "
          f"{m_sp.nnz} non-zero cells, density {m_sp.nnz / (R.shape[0] * R.shape[1]):.3%}")
    print(f"eval users: {len(users)}   recommendable (in-stock) items: {len(avail)}")
    has_hist = np.isin(users, user_index.to_numpy())
    print(f"eval users with history before the cutoff: {has_hist.sum()}  (the rest get 7-day popularity)")

    # Dense item columns restricted to what we may recommend.
    item_cols = item_index.get_indexer(avail)  # -1 when an available item has no interactions
    in_matrix = item_cols >= 0
    user_rows = user_index.get_indexer(users)  # -1 for cold users
    R_eval = np.zeros((len(users), R.shape[1]))
    R_eval[user_rows >= 0] = R[user_rows[user_rows >= 0]]

    def to_avail(score_items: np.ndarray) -> np.ndarray:
        out = np.zeros((score_items.shape[0], len(avail)))
        out[:, in_matrix] = score_items[:, item_cols[in_matrix]]
        return out

    results, padding = [], {}

    def run(name: str, scores: np.ndarray) -> None:
        recs, stats = dense_recs.scores_to_recs(scores, users, avail)
        res = common.evaluate(recs, name)
        results.append(res)
        padding[name] = stats

    # --- baseline for reference
    pop = dense_recs.pop7_vector(avail)
    run("pop_7d", np.tile(pop + 1e-9, (len(users), 1)))

    # --- user-kNN
    common.section("User-kNN (cosine, neighbours k)")
    Rn = row_normalise(R)
    sim_users = row_normalise(R_eval) @ Rn.T  # eval users x all users
    for j, uid in enumerate(users):
        if user_rows[j] >= 0:
            sim_users[j, user_rows[j]] = 0.0  # a user is not their own neighbour
    sweep = {}
    for k in (10, 25, 50, 100):
        idx = np.argpartition(-sim_users, k, axis=1)[:, :k]
        w = np.zeros_like(sim_users)
        np.put_along_axis(w, idx, np.take_along_axis(sim_users, idx, axis=1), axis=1)
        scores = to_avail(w @ R)
        recs, _ = dense_recs.scores_to_recs(scores, users, avail)
        sweep[k] = common.evaluate(recs, f"user_knn_k{k}")
        print(f"k={k:<4d} ndcg@20 {sweep[k]['ndcg@20']:.4f}  recall@20 {sweep[k]['recall@20']:.4f}")
        if k == 50:
            run("user_knn", scores)
    print("example: strongest neighbours of the first user with history")
    j0 = int(np.flatnonzero(user_rows >= 0)[0])
    top_nb = np.argsort(-sim_users[j0])[:3]
    print(f"  user {users[j0]}: " + ", ".join(f"user {user_index[t]} (cos {sim_users[j0, t]:.3f})" for t in top_nb))

    # --- item-kNN
    common.section("Item-kNN (cosine between item columns)")
    Cn = row_normalise(R.T)  # items x users, unit rows
    S = Cn @ Cn.T
    np.fill_diagonal(S, 0.0)
    print(f"item-item similarity: {S.shape[0]} x {S.shape[1]}, "
          f"mean non-zero similarity {S[S > 0].mean():.4f}, max {S.max():.3f}")
    scores = to_avail(R_eval @ S)  # score(u, i) = sum_j sim(i, j) * weight(u, j)
    run("item_knn", scores)
    j0 = int(np.flatnonzero(user_rows >= 0)[0])
    items = hist.items.set_index("item_id")
    seed_item = item_index[int(np.argmax(R[user_rows[j0]]))]
    nb = np.argsort(-S[item_index.get_loc(seed_item)])[:3]
    print(f"example: most similar items to item {seed_item} ('{items.loc[seed_item, 'title']}'):")
    for t in nb:
        iid = item_index[t]
        print(f"  item {iid:<5d} cos {S[item_index.get_loc(seed_item), t]:.3f}  '{items.loc[iid, 'title']}'")

    # --- co-visitation
    common.section("Session co-visitation (seeded with the user's decayed recent items)")
    seeds = candidates._top_per_user(candidates.recent(hist), "cand_recent_score", candidates.N_SEEDS)
    cov = candidates.covisitation_matrix(hist)
    print(f"co-visitation pairs: {len(cov)}   seeds per user <= {candidates.N_SEEDS}")
    j = seeds.merge(cov, left_on="item_id", right_on="item_id_a")
    j["s"] = j["cand_recent_score"] * j["w"]
    sc = j.groupby(["user_id", "item_id_b"], as_index=False)["s"].sum()
    sc = sc[sc["user_id"].isin(users) & sc["item_id_b"].isin(avail)]
    cov_scores = np.zeros((len(users), len(avail)))
    cov_scores[pd.Index(users).get_indexer(sc["user_id"]), pd.Index(avail).get_indexer(sc["item_id_b"])] = sc["s"].to_numpy()
    run("covisit", cov_scores)

    common.section("Padding with 7-day popularity (users/slots the method could not fill)")
    for name, st in padding.items():
        print(f"{name:<10s} no-score users {st['users_without_model_scores']:>4d}  "
              f"partially padded {st['users_padded_partially']:>4d}  "
              f"slots padded {st['share_of_slots_from_popularity_padding']:.1%}")

    common.section("Results (top-50 lists, evaluated at K=20)")
    common.print_table(results)

    common.save_results("ch05_cf", {
        "results": results,
        "padding": padding,
        "user_knn_sweep": {str(k): v for k, v in sweep.items()},
        "matrix": {"users": R.shape[0], "items": R.shape[1], "nnz": int(m_sp.nnz)},
        "eval_users": int(len(users)),
        "eval_users_with_history": int(has_hist.sum()),
    })


if __name__ == "__main__":
    main()
