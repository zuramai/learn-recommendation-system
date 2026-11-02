# Content-based filtering

It is the week before the 9.9 sale and Sari, a product manager on the marketplace's discovery team, is staring at a dashboard with one number she does not like. Every day a few hundred new products are listed by sellers: a ceramic coffee dripper from a shop in Bandung, a pair of sneakers from a brand nobody has searched for yet, a cat-food bundle that went live ten minutes ago. None of them has a single click. The popularity baselines from the previous chapter cannot see them, because popularity is *history*, and these products have none. Yet a buyer who spent last week browsing pour-over coffee gear would very probably love that dripper.

What Sari needs is a recommender that can look at what an item *is*, not only at who touched it. If the dripper's title says "coffee", its category says "Coffee & Tea", and the buyer has been clicking on coffee things, then the two can be matched without a single interaction on the new product. That idea is **content-based filtering**, and it is the subject of this chapter.

**What you'll learn**

- How to turn an item's text and metadata into a numeric vector, and why we weight words with TF-IDF instead of just counting them.
- What cosine similarity measures, and why it is the default distance for sparse text vectors.
- How to build a user profile as a decay-weighted average of the item vectors the user touched, and how to score every item against it.
- How the companion script `ch04_content.py` performs on the simulated marketplace, and why it beats global popularity but not category-aware popularity.
- Where content-based methods shine (cold start) and where they fail (over-specialisation, shallow metadata).

## Intuition: the librarian who never watches the readers

Imagine a librarian who is not allowed to look at the lending records. She cannot say "people who borrowed this also borrowed that". All she may do is read the covers: the title, the subject shelf, the author. When you hand her three novels you enjoyed, she walks to the shelves, finds books whose covers *sound like* the three you gave her, and hands them over.

That librarian is a content-based recommender. She has two skills:

1. She can describe every book as a short list of descriptive words (an **item representation**).
2. She can summarise your taste as a list of the same kind of words (a **user profile**), and compare the two lists.

Notice what she does not need: other readers. This is the superpower and the weakness at once. A book that arrived this morning is recommendable immediately, as soon as someone writes its cover. But she will also never surprise you. If you read three crime novels she will keep walking you to the crime shelf, forever, because nothing on a cover says "readers who like crime also secretly like gardening books".

In our marketplace the "covers" are the columns of the `items` table: `title`, `brand`, and the category of the item (plus its parent category, like "Electronics" or "Pets"). In a richer system they would also include descriptions, images, seller text, and attributes such as colour or size. The principle is the same.

The chapter's pipeline looks like this:

```mermaid
flowchart TB
    A[Item table: title, brand, category] --> B[Build one text string per item]
    B --> C[TF-IDF vectors, L2-normalised]
    D[User history: clicks, carts, orders] --> E[Weight each event: strength x time decay]
    C --> F[User profile = weighted mean of item vectors]
    E --> F
    F --> G[Cosine of profile vs every available item]
    C --> G
    G --> H[Top 50 items per user]
```

*Figure: The content-based pipeline. Item vectors are computed once; user profiles are recomputed from the latest history.*

Everything in the diagram is cheap, deterministic and explainable, which is why content-based scoring remains a workhorse for "similar items" widgets and cold-start fallbacks in production systems.

## The algorithm

### Step 1: item representation as text

The simplest, and surprisingly effective, way to represent an item is to glue its descriptive fields into one string and treat that string as a short document. In the companion script this is one line:

```python
items["text"] = items["title"] + " " + items["category_name"] + " " + items["brand"] + " " + items["parent_name"]
```

*Excerpt from `book/code/ch04_content.py`.*

An item titled "Volt Skincare Lokal" in category "Skincare" with brand "Volt" and parent category "Beauty" becomes the document `Volt Skincare Lokal Skincare Volt Beauty`. Note that category and brand words are repeated on purpose: titles in this simulator are short, and repeating the structured fields lets them count more than once.

Once every item is a document, we need to turn the document into a vector. The most naive way is a **bag of words**: pick a vocabulary of $V$ words and for every item write down how many times each word appears. Two items are similar if they use similar words.

### Step 2: TF-IDF weighting

Bag-of-words has an obvious flaw. The word "home" appears in every item of the category "Home & Living" (and in the parent-category text of each one), so it tells us almost nothing about *which* home item a buyer wants, while the word "sneakers" is specific and carries much more information. We want rare words to count more than common words.

**TF-IDF** (term frequency–inverse document frequency) does exactly that. For a term $t$ in document $d$ out of a collection of $N$ documents:

$$
\text{tfidf}(t, d) = \text{tf}(t, d) \times \text{idf}(t)
$$

where $\text{tf}(t,d)$ is the number of times $t$ occurs in $d$, and the inverse document frequency downweights terms that occur in many documents. The version used by scikit-learn's `TfidfVectorizer`, which our script uses, is the *smoothed* one:

$$
\text{idf}(t) = \ln\frac{1 + N}{1 + \text{df}(t)} + 1
$$

with $\text{df}(t)$ the number of documents that contain $t$. The "+1" terms keep the logarithm finite and guarantee that even a term present in every document keeps a small positive weight.

Finally each document vector $\mathbf{x}_d$ is **L2-normalised** to unit length:

$$
\hat{\mathbf{x}}_d = \frac{\mathbf{x}_d}{\lVert \mathbf{x}_d \rVert_2}
$$

so that a long title does not automatically look more important than a short one.

### Step 3: cosine similarity

To compare two vectors we measure the angle between them rather than the distance between their tips:

$$
\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\lVert \mathbf{a} \rVert \, \lVert \mathbf{b} \rVert}
$$

Because TF-IDF vectors have no negative entries, cosine lies between 0 (no shared terms at all) and 1 (identical direction). When both vectors are already unit length, as ours are after step 2, the denominator is 1 and **cosine similarity is simply the dot product**. That is why the script can compute the similarity of every pair of items with one sparse matrix multiplication, `X @ X.T`.

Why angle and not Euclidean distance? Imagine one item whose text is repeated twice, producing a vector twice as long but pointing in the same direction. Euclidean distance says it is far from the original; cosine correctly says the two are identical in meaning.

### Step 4: the user profile

Items now live in a vector space. A user is described by what they did to items, so the natural profile is a **weighted average of the vectors of the items they interacted with**. Two kinds of weight apply to each event:

- **Strength.** In this repo a click counts 1, an add-to-cart 3 and an order 5 (`EVENT_WEIGHT` in `recommender/recsys/schema.py`). Buying something is stronger evidence of taste than glancing at it.
- **Recency.** Taste drifts; a click from last week matters more than a click from last month. We apply exponential decay with a half-life $h$ of 7 days: an event of age $a$ days is multiplied by $2^{-a/h}$.

Writing $w_e$ for the strength of event $e$, $a_e$ for its age and $\hat{\mathbf{x}}_{i(e)}$ for the vector of the item it touched, the profile is

$$
\mathbf{p}_u = \frac{\sum_{e \in E_u} w_e \, 2^{-a_e/h} \, \hat{\mathbf{x}}_{i(e)}}{\sum_{e \in E_u} w_e \, 2^{-a_e/h}},
\qquad
\hat{\mathbf{p}}_u = \frac{\mathbf{p}_u}{\lVert \mathbf{p}_u \rVert_2}.
$$

The denominator makes it a proper average, so a user with 100 events does not get a bigger vector than a user with 5. The decay function is the same one the repo already uses for its "recently viewed" retriever:

```include
path: recommender/recsys/candidates.py
symbol: _decay
```

### Step 5: scoring and ranking

For every available item $j$, the score for user $u$ is

$$
\text{score}(u, j) = \hat{\mathbf{p}}_u \cdot \hat{\mathbf{x}}_j .
$$

Sort descending, keep the top 50 per user (the common evaluation contract defined by `book/code/common.py`), done. Users with no history at all have an all-zero profile; the script breaks their ties with a tiny popularity term so that they get the `pop_7d` list instead of an arbitrary one.

### A worked example by hand

Let us run the whole thing on a toy catalogue of four items, small enough to check with a pocket calculator. The vocabulary has four words and each item has just a few of them:

| Item | Text | aurora | coffee | sneakers | volt |
|---|---|---|---|---|---|
| A | `aurora coffee` | 1 | 1 | 0 | 0 |
| B | `volt coffee` | 0 | 1 | 0 | 1 |
| C | `aurora sneakers` | 1 | 0 | 1 | 0 |
| D | `volt sneakers sneakers` | 0 | 0 | 2 | 1 |

Every word occurs in exactly two of the four items, so $\text{idf} = \ln(5/3) + 1 \approx 1.511$ for all of them. (With uniform IDF, IDF scaling cancels out after normalisation; in the real catalogue IDF differs between words, as we will see below.) After L2 normalisation:

- A $= (0.707,\ 0.707,\ 0,\ 0)$ and B $= (0,\ 0.707,\ 0,\ 0.707)$ and C $= (0.707,\ 0,\ 0.707,\ 0)$ (coordinates in the order aurora, coffee, sneakers, volt).
- D has raw TF-IDF $(0,\ 0,\ 2\cdot1.511,\ 1.511)$, length $1.511\sqrt{5}$, so $\hat{\mathbf{x}}_D = (0,\ 0,\ 0.894,\ 0.447)$. The word repeated twice dominates the vector.

Cosine similarities follow directly. A and B share only "coffee": $0.707 \times 0.707 = 0.5$. A and C share only "aurora": also 0.5. A and D share nothing: 0. C and D share "sneakers": $0.707 \times 0.894 = 0.632$.

Now a user clicked item A today (strength 1, age 0) and ordered item C seven days ago (strength 5, decay $2^{-7/7} = 0.5$). The effective weights are $1 \times 1 = 1$ and $5 \times 0.5 = 2.5$, sum 3.5. The profile is

$$
\mathbf{p} = \frac{1 \cdot \mathbf{x}_A + 2.5 \cdot \mathbf{x}_C}{3.5} = (0.707,\ 0.202,\ 0.505,\ 0),
$$

and after normalisation $\hat{\mathbf{p}} = (0.793,\ 0.226,\ 0.566,\ 0)$. Scores:

| Item | Score |
|---|---|
| A | 0.721 |
| B | 0.160 |
| C | 0.961 |
| D | 0.506 |

C scores highest because it carries both of the user's strongest words, "aurora" and "sneakers", with the heavier (order) weight on sneakers. D outranks B even though the user never saw an item named "volt": the strong sneakers signal pulls D up, while B only shares the weak "coffee" signal. (A and C are already seen; a production system would filter them out. The simple baseline in this chapter does not, which slightly depresses its measured quality.)

::: tip
Hand-calculate a tiny example like this one before trusting a library call. When your toy numbers match what `TfidfVectorizer` prints, you know you understand the normalisation and the smoothing; when they do not, the discrepancy is almost always the "+1" in IDF or the L2 step.
:::

## Code walkthrough

The full script is `book/code/ch04_content.py` (118 lines). We walk through it in four pieces. The excerpts below are copied verbatim from the file; they run top to bottom as one script, not as functions.

### Preparing the item text

The script starts by loading the shared simulated data through `common.py`, so every algorithm in the book sees exactly the same history, cutoff and eval users. It then joins each item to its category and parent category names and builds the text:

```python
hist = common.history()
avail = common.available_items()
users = common.eval_users()
cats = hist.categories.set_index("category_id")

items = hist.items.copy()
items["category_name"] = items["category_id"].map(cats["name"])
items["parent_name"] = items["category_id"].map(cats["parent_name"])
items["text"] = items["title"] + " " + items["category_name"] + " " + items["brand"] + " " + items["parent_name"]
```

Note that `hist` is `History.at(tables, CUTOFF)`: only the data visible at the cutoff. Items listed after the cutoff do not exist for this model, which keeps the evaluation honest.

### Fitting TF-IDF

```python
vec = TfidfVectorizer(lowercase=True)
X = vec.fit_transform(items["text"])  # rows are L2-normalised already
idf = pd.Series(vec.idf_, index=vec.get_feature_names_out()).sort_values()
item_row = {it: i for i, it in enumerate(items["item_id"])}
```

`fit_transform` does three jobs at once: it learns the vocabulary and the document frequencies, computes the IDF for each word, and returns a sparse matrix `X` with one L2-normalised row per item. `item_row` maps an item id to its row number so we can look items up later. Keeping the matrix sparse matters: with a real vocabulary of tens of thousands of words, a dense matrix would be unmanageable, but each item only touches a handful of columns.

### Building profiles with sparse algebra

This is the heart of the script, and it avoids any Python loop over users:

```python
pos = hist.interactions()
pos = pos[pos["item_id"].isin(item_row)]
pos = pos.assign(s=pos["w"] * np.power(0.5, pos["age_days"] / HALF_LIFE))
uidx = {u: i for i, u in enumerate(users)}
p = pos[pos["user_id"].isin(uidx)]
p = p.groupby(["user_id", "item_id"], as_index=False)["s"].sum()
W = sp.csr_matrix((p["s"], (p["user_id"].map(uidx), p["item_id"].map(item_row))), shape=(len(users), X.shape[0]))
wsum = np.asarray(W.sum(axis=1)).ravel()
profile = normalize(sp.diags(1 / np.where(wsum > 0, wsum, 1)) @ W @ X)
```

`hist.interactions()` returns the click, add-to-cart and order events with a column `w` holding the strength weight:

```include
path: recommender/recsys/data.py
symbol: History.interactions
```

The script multiplies `w` by the decay factor to get the effective weight `s`, then sums the effective weight per (user, item) pair. These sums fill a sparse **user × item weight matrix** `W`. Then the formula for the profile becomes pure matrix algebra:

- `W @ X` computes, for every user, $\sum_e s_e\,\hat{\mathbf{x}}_{i(e)}$, a weighted sum of item vectors.
- `sp.diags(1 / wsum)` divides each user's row by their total weight, giving the weighted average.
- `normalize(...)` rescales each profile to unit length, so the final dot product is a true cosine.
- `np.where(wsum > 0, wsum, 1)` is a guard: a user with no positive events has `wsum == 0`, and dividing by zero would produce `inf`. Replacing zero by one leaves their all-zero row at zero.

### Scoring, tie-breaking and evaluation

```python
Xa = X[[item_row[i] for i in item_ids]]
S = np.asarray((profile @ Xa.T).todense())
pop7 = candidates.item_popularity(hist, 7).reindex(item_ids).fillna(0).to_numpy()
S = S + 1e-6 * pop7 / max(pop7.max(), 1e-9)  # cold users: tie-break by popularity
top = np.argsort(-S, axis=1, kind="stable")[:, :TOP_N]
```

`profile @ Xa.T` produces the full user × available-item score matrix in one product. Only items that are in stock at the cutoff (`avail`) are scored. The popularity tie-break is scaled to $10^{-6}$ so it can only separate scores that are otherwise equal: it decides among cold users (all zeros) and among near-duplicate items, but never overrides a real content difference. The final `argsort` picks the top 50 per user, and `common.evaluate` computes exactly the same metrics used for all other chapters.

## Run it

From the `book/` directory:

```bash
uv run python code/ch04_content.py
```

The first thing the script prints is a look at the text documents and the learned vocabulary:

```output
file: results/ch04_content.txt
lines: 2-21
```

Three things stand out. First, the documents are short: only 4.8 non-zero terms per item, which is a tiny amount of information. Second, the entire vocabulary is just 64 terms, because the simulator generates titles from a small template (brand + category + a suffix such as "Lokal" or "Pro"). A real marketplace has millions of distinct words, model numbers and typos. Third, the IDF ranking behaves exactly as the theory promises: "living" and "home" (shared by every item of a whole parent category) have the lowest weights, near 2.86, while "sneakers", "shampoo" and "skincare" (specific, narrow categories) have the highest, around 4.5 to 4.7.

## Results

### What the profiles look like

```output
file: results/ch04_content.txt
lines: 23-32
```

Of the 947 eval users, 919 have a usable profile, and 28 have no positive history before the cutoff, so they fall back to the popularity tie-break. For user 894, the strongest profile term is "diapers" (0.809), followed by "baby" and "mother" (both 0.301): a parent buying baby supplies. The profile is directly readable by a human, one of the great selling points of content-based methods.

### How well does it rank?

```output
file: results/ch04_content.txt
lines: 34-37
```

To judge this number, put it next to the baselines from the previous chapter:

```output
file: results/ch03_popularity.txt
lines: 15-22
```

Three conclusions:

1. **Content-based beats global popularity.** The NDCG@20 of `content_tfidf` is 0.0909 against 0.0475 for `pop_7d`, nearly double. Personal taste, even from category-and-brand words alone, carries real information.
2. **It loses to category-aware popularity.** `pop_in_my_categories` scores 0.1302, clearly above our 0.0909. Why? Both methods exploit the same fact: users shop within categories. But `pop_in_my_categories` also ranks *inside* the category by popularity, which separates good items from poor ones. Our TF-IDF vector contains only category and brand words, so every item of a category looks almost the same, and the choice among them is a coin flip (decided only by the $10^{-6}$ popularity tie-break).
3. **Coverage is excellent.** `coverage@20` of 0.9892 means that nearly the whole in-stock catalogue shows up in somebody's top 20, against 0.0087 for popularity, which recommends the same few dozen items to everyone. Content-based methods are a natural antidote to the "rich get richer" effect we met in chapter 3. The coverage figure is slightly inflated by the random-ish choice among near-ties, but the direction is real.

### Nearest neighbours and the "cosine = 1.0" lesson

```output
file: results/ch04_content.txt
lines: 39-60
```

Look at the first query. The item "Aurora Coffee & Tea Import" has a nearest neighbour with cosine **1.000**: a different item (id 2399) with exactly the same title and category. The next ones, 0.953, differ only in the suffix "Lokal", "Lite" or "Premium". The suffix word contributes so little that these items are, as far as the vector space is concerned, interchangeable.

This is not a bug, but it is a teaching point. When two items have identical text, their TF-IDF vectors are identical and no content-based method can tell them apart. The marketplace has plenty of such near-duplicates: sellers copy listings, or list the same product several times. A content-only recommender will happily fill the top 5 slots with five copies of the same product from five sellers. Distinguishing between them needs *behavioural* signals (price, rating, popularity) which arrive in the learning-to-rank stage of later chapters.

The similarity structure is easiest to see as a picture. The heat map shows the cosine between 15 items, five from each of three categories:

![Cosine similarity between 15 items from three categories (Coffee & Tea, Snacks, Sneakers). Black lines separate the categories. Blocks on the diagonal are bright; everything off-diagonal is near zero.](../assets/charts/ch04-similarity-heatmap.png)

The picture has three dark blocks on the diagonal: items from the same category are similar, items from different categories are not. Inside a block the shade varies a little (the "Aurora Snacks" pair is darker than the others, because they share the brand word too), but the variation is small. Numbers confirm it:

```output
file: results/ch04_content.txt
lines: 62-65
```

The rows and columns are Coffee & Tea, Snacks and Sneakers. On the diagonal the mean similarity between two *different* items in the same category is 0.752, 0.633 and 0.681; off-diagonal it is at most 0.084. (The faint light-blue squares in the heat map off the diagonal, such as between a coffee item and a snack item, come from a shared brand word like "Aurora" or "Nusa".)

::: fact
**TF-IDF is over fifty years old.** The idea of weighting a term by the inverse of how many documents contain it was published by Karen Spärck Jones in 1972, as "a statistical interpretation of term specificity". It has been used in search engines and recommender systems ever since, and it is still a strong baseline today.

Source: Spärck Jones, Journal of Documentation, 1972
:::

::: fact
**Pandora's "Music Genome" is content-based filtering done by hand.** Pandora's founders assembled a team of musicologists who analysed and classified songs on roughly 400 attributes, such as vocal style or rhythm, and used those attribute vectors to find songs similar to ones a listener liked, instead of relying on other listeners' histories.

Source: Westergren, NPR interview, 2011
:::

::: fact
**Profiles as centroids come from the 1970s.** In the SMART retrieval system, Rocchio's relevance-feedback algorithm moved a query vector toward the centroid of the documents a user marked relevant and away from the non-relevant ones. Averaging item vectors into a user profile, as we do in this chapter, is the same geometric idea applied to recommendations.

Source: Rocchio, The SMART Retrieval System, 1971
:::

### The same idea inside the app

![The marketplace's search page for the query "coffee". Results are matched on title and category text and then sorted by popularity.](../assets/screenshots/search.png)

The marketplace app already contains a tiny content-based component, and you can see it in the screenshot. When a buyer types "coffee" in the search box, the backend matches the query against product titles and category names (an `ILIKE` filter in `backend/src/catalog.rs`). That is item representation by text, with the simplest possible similarity function: "does the string occur?". The ordering of the results, "Popular", is not content-based at all. A TF-IDF or BM25 ranking would be the natural next step for search; for recommendations, the profile approach of this chapter replaces the typed query with a *query built from the user's own behaviour*.

## Pitfalls and how it is done in production

### Over-specialisation

The librarian who only reads covers keeps recommending more of the same. Our profile for user 894 is dominated by "diapers", so all 50 recommendations will be diapers or diaper-adjacent items. If that user needs a stroller or a baby bottle next, no word overlap with the history exists to suggest it. In recommender terminology the system has **no serendipity**: it cannot recommend what the user did not already show an interest in.

Mitigations are worth knowing, because every production system uses some of them:

- **Blend with collaborative signals.** Chapter 5 introduces co-visitation, which finds that people who buy diapers also buy strollers, without any shared words.
- **Re-rank for diversity.** Chapter 11 introduces maximal marginal relevance and category caps that limit how many items from one category may appear in the top 20.
- **Explore deliberately.** Reserve a few slots for items outside the profile, and learn from the response.

### Shallow metadata, noisy text

Content-based methods are only as good as the content. In our simulator, titles are generated from a template, so the vocabulary is 64 words and the near-duplicates abound. Real catalogues have the opposite problem: seller-written titles are stuffed with keywords ("MURAH!!! Best Seller Sneakers Pria Wanita Original 100%"), misspell brand names, mix languages, and use abbreviations. Production systems clean and normalise titles, extract structured attributes (brand, colour, size) with models, and increasingly replace TF-IDF with learned embeddings from text and image encoders. Chapter 7 shows what a learned item representation looks like.

### The cold-start strength

Here the picture turns positive. A brand-new item has no clicks, so popularity, co-visitation and matrix factorisation (chapters 3, 5, 6) cannot recommend it. But its title, brand and category exist from the first second, so a content-based scorer can place it in front of the right buyers *immediately*. This is why content similarity is a standard part of the cold-start toolkit, which we revisit in chapter 12. A new *user* with a single click is similarly well served: one click already produces a non-zero profile pointing at a category, whereas collaborative methods need many interactions before neighbours or factors become reliable.

::: warning
Do not confuse "works for new items" with "works well for new items". A new item with a generic title looks like thousands of existing items and will get only a generic share of the traffic. Content features tell the system *where to show* the item; only real clicks tell it *whether the item is good*.
:::

### Scalability and freshness

The item matrix $X$ changes slowly (only when sellers edit listings) and can be computed offline. Profiles depend on the latest behaviour. A production implementation rarely recomputes a dense similarity matrix; it stores item vectors in an approximate-nearest-neighbour index (chapter 7) and looks up the profile vector at request time. For a fresh signal, many systems maintain the profile incrementally: on each new click, `p_new = decay * p_old + w * x_item`, which is the streaming form of the weighted mean.

### Tuning the profile

Three knobs matter. The *half-life* decides how quickly interests fade: a short half-life reacts quickly to a new interest but forgets stable long-term taste; a long one does the reverse. The *event weights* (click 1, cart 3, order 5 here) encode how much stronger a purchase is than a glance. And the *text recipe* (which fields, how often repeated) decides what "similar" even means. A good practice is to treat these as hyperparameters and tune them against the same time-split evaluation you use for everything else, as you will do in the hands-on exercise.

### How this fits the repo's ranking system

The content-based score is not a retriever in the final repo pipeline; the four retrievers are `recent`, `covisit`, `catpop` and `global_trending`. Content knowledge enters the ranker differently: as *features* such as `uc_click_share_30d` (the share of the user's clicks that fall in the item's category) and `item_price_vs_category` (price divided by the category median), defined in `schema.py` and computed in `features.py`. That is a common production pattern. A cheap content score becomes one input among many into a learned ranker, rather than the whole recommender. The ranker then learns, from data, how much to trust content relative to popularity or recency.

## Summary

- Content-based filtering represents every item by its own attributes (text, category, brand) and recommends items that resemble what the user already liked.
- TF-IDF weights words by term frequency times inverse document frequency, so specific words ("sneakers") matter more than generic ones ("home"); L2 normalisation makes cosine similarity equal to a dot product.
- A user profile is the weighted average of the vectors of items the user touched, with weights equal to event strength times exponential time decay (half-life 7 days here).
- On the simulated marketplace the method reaches an NDCG@20 of 0.0909, almost twice the 0.0475 of `pop_7d`, but below the 0.1302 of `pop_in_my_categories`, because category and brand words cannot separate items inside a category.
- Coverage is very high (0.9892 versus 0.0087 for popularity): content-based scoring spreads exposure over the catalogue instead of concentrating it on a few hits.
- Near-duplicate titles produce a cosine similarity of exactly 1.0, so content alone cannot tell copies apart; behavioural signals must do that.
- The main strength is cold start (new items and new users can be scored immediately); the main weakness is over-specialisation, as the method never suggests anything unlike the history.
- In the repo's final system, content signals are used as ranker features rather than as a standalone retriever.

## Quiz

**Q4.1** Why does TF-IDF multiply the term frequency by an *inverse* document frequency?

A. To make long documents score higher than short ones.
B. To give more weight to terms that appear in few documents and less to terms that appear in many.
C. To make all vectors orthogonal to each other.
D. To remove stop-words from the vocabulary automatically.

**Q4.2** After TF-IDF vectors are L2-normalised, which statement about cosine similarity is correct?

A. It equals the Euclidean distance between the two vectors.
B. It equals the dot product of the two vectors.
C. It can be negative for any two items.
D. It depends on how many words each item's text contains.

**Q4.3** A seller lists a new item at 09:00 and nobody has clicked it yet. Which method can still recommend it at 09:01?

A. Global popularity over the last 7 days.
B. Item-to-item co-visitation counted from sessions.
C. A content-based scorer that uses the item's title, brand and category.
D. A recommender using only past orders.

**Q4.4** In the results, two different items have a cosine similarity of exactly 1.000. What does this tell you about their text, and why is it a problem for a content-only recommender?

**Q4.5** Explain "over-specialisation" in content-based filtering and name two ways to reduce it.

**Q4.6** Read this line from the script:

```python
profile = normalize(sp.diags(1 / np.where(wsum > 0, wsum, 1)) @ W @ X)
```

What would go wrong without `np.where(wsum > 0, wsum, 1)`, and what would the profile row of a user with no positive events be after the line runs?

**Q4.7** A user clicked item A today (strength 1, age 0 days) and added item B to the cart 14 days ago (strength 3). With a half-life of 7 days, what fraction of the final profile's total weight comes from item A? (Give the answer to three decimals.)

**Q4.8** Two items have unit-length vectors $\mathbf{a} = (0.6,\ 0.8,\ 0)$ and $\mathbf{b} = (0.8,\ 0,\ 0.6)$. Compute their cosine similarity.

**Q4.9** *Hands-on.* In `book/code/ch04_content.py`, change `HALF_LIFE = 7.0` to `1.0`, run the script, and then repeat with `30.0`. Compare the `ndcg@20` and `recall@20` in the results table with the baseline of 0.0909 and 0.1378. What do you expect to observe about how quickly the profile should forget, and does the outcome match? (Restore the value afterwards.)

