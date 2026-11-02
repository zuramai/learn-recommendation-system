"""Chapter 4: content-based filtering with TF-IDF item vectors and a decay-weighted user profile."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import common
from common import TOP_N, section
from recsys import candidates

HALF_LIFE = 7.0
# book:start data_prep
hist = common.history()
avail = common.available_items()
users = common.eval_users()
cats = hist.categories.set_index("category_id")

items = hist.items.copy()
items["category_name"] = items["category_id"].map(cats["name"])
items["parent_name"] = items["category_id"].map(cats["parent_name"])
# book:end data_prep
# book:start item_text
items["text"] = items["title"] + " " + items["category_name"] + " " + items["brand"] + " " + items["parent_name"]
# book:end item_text

section("Item text and TF-IDF")
print("Example documents:")
for t in items["text"].head(3):
    print(f"  {t!r}")
# book:start tfidf
vec = TfidfVectorizer(lowercase=True)
X = vec.fit_transform(items["text"])  # rows are L2-normalised already
print(f"\nitems: {X.shape[0]}, vocabulary: {X.shape[1]} terms, non-zeros per item: {X.nnz / X.shape[0]:.1f}")
idf = pd.Series(vec.idf_, index=vec.get_feature_names_out()).sort_values()
print("\nLowest IDF (common, uninformative):")
print(idf.head(5).round(3).to_string())
print("Highest IDF (rare, discriminative):")
print(idf.tail(5).round(3).to_string())
item_row = {it: i for i, it in enumerate(items["item_id"])}
# book:end tfidf

section("User profiles (decay-weighted mean of interacted item vectors)")
# book:start profile_build
pos = hist.interactions()
pos = pos[pos["item_id"].isin(item_row)]
pos = pos.assign(s=pos["w"] * np.power(0.5, pos["age_days"] / HALF_LIFE))
uidx = {u: i for i, u in enumerate(users)}
p = pos[pos["user_id"].isin(uidx)]
p = p.groupby(["user_id", "item_id"], as_index=False)["s"].sum()
W = sp.csr_matrix((p["s"], (p["user_id"].map(uidx), p["item_id"].map(item_row))), shape=(len(users), X.shape[0]))
wsum = np.asarray(W.sum(axis=1)).ravel()
# book:end profile_build
# book:start profile_norm
profile = normalize(sp.diags(1 / np.where(wsum > 0, wsum, 1)) @ W @ X)
# book:end profile_norm
n_cold = int((wsum == 0).sum())
print(f"profiles built for {len(users) - n_cold} of {len(users)} eval users; {n_cold} fall back to pop_7d")
u0 = users[np.argmax(wsum)]
row = profile[uidx[u0]].toarray().ravel()
top_terms = pd.Series(row, index=vec.get_feature_names_out()).nlargest(6)
print(f"\nStrongest profile terms for user {u0}:")
print(top_terms.round(3).to_string())

section("Results (cosine(profile, item), top 50 per user)")
# book:start scoring
item_ids = avail["item_id"].to_numpy()
Xa = X[[item_row[i] for i in item_ids]]
S = np.asarray((profile @ Xa.T).todense())
pop7 = candidates.item_popularity(hist, 7).reindex(item_ids).fillna(0).to_numpy()
S = S + 1e-6 * pop7 / max(pop7.max(), 1e-9)  # cold users: tie-break by popularity
top = np.argsort(-S, axis=1, kind="stable")[:, :TOP_N]
# book:end scoring
recs = pd.DataFrame({"user_id": np.repeat(users, TOP_N), "item_id": item_ids[top].ravel()})
row_res = common.evaluate(recs, "content_tfidf")
common.print_table([row_res])

section("Five nearest items for three example items")
sim_all = X @ X.T
nearest = {}
for cname in ("Coffee & Tea", "Sneakers", "Cat Food"):
    cid = cats.index[cats["name"] == cname][0]
    ex = items[(items["category_id"] == cid)].iloc[0]
    i = item_row[ex["item_id"]]
    s = np.asarray(sim_all[i].todense()).ravel()
    s[i] = -1
    nn = np.argsort(-s, kind="stable")[:5]
    print(f"\nQuery: item {ex['item_id']} '{ex['title']}' ({ex['category_name']})")
    lst = []
    for j in nn:
        r = items.iloc[j]
        print(f"   {s[j]:.3f}  item {r['item_id']:<5} {r['title']}  [{r['category_name']}]")
        lst.append({"item_id": int(r["item_id"]), "title": r["title"], "category": r["category_name"], "cosine": float(s[j])})
    nearest[cname] = {"query": {"item_id": int(ex["item_id"]), "title": ex["title"]}, "neighbours": lst}

# 15 items (5 in each of 3 categories) for the similarity heatmap chart.
heat_items = []
for cname in ("Coffee & Tea", "Snacks", "Sneakers"):
    cid = cats.index[cats["name"] == cname][0]
    heat_items += items[items["category_id"] == cid].head(5).index.tolist()
sub = items.loc[heat_items]
rows_ = [item_row[i] for i in sub["item_id"]]
H = np.asarray(X[rows_] @ X[rows_].T.todense())
section("Mean cosine similarity within / across the three heat-map categories")
names = sub["category_name"].tolist()
for a in ("Coffee & Tea", "Snacks", "Sneakers"):
    line = []
    for b in ("Coffee & Tea", "Snacks", "Sneakers"):
        ia = [k for k, n in enumerate(names) if n == a]
        ib = [k for k, n in enumerate(names) if n == b]
        blk = H[np.ix_(ia, ib)]
        mean = blk[~np.eye(len(ia), dtype=bool)].mean() if a == b else blk.mean()
        line.append(f"{mean:.3f}")
    print(f"{a:<14}" + "  ".join(f"{x:>7}" for x in line))

common.save_results("ch04_content", {
    "results": row_res,
    "vocabulary": int(X.shape[1]),
    "n_profiles": len(users) - n_cold,
    "n_cold_users": n_cold,
    "nearest": nearest,
    "heatmap": {"labels": [f"{t} ({c})" for t, c in zip(sub["title"], sub["category_name"])],
                "categories": names, "similarity": H.tolist()},
})
