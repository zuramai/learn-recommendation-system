# Content-based filtering

It is the week before the 9.9 sale and Sari, a product manager on the marketplace's discovery team, is staring at a dashboard with one number she does not like. Every day a few hundred new products are listed by sellers: a ceramic coffee dripper from a shop in Bandung, a pair of sneakers from a brand nobody has searched for yet, a cat-food bundle that went live ten minutes ago. None of them has a single click. The popularity baselines from the previous chapter cannot see them, because popularity is *history*, and these products have none. Yet a buyer who spent last week browsing pour-over coffee gear would very probably love that dripper.

What Sari needs is a recommender that looks at what an item *is*, not only at who touched it. If the dripper's title says "coffee" and the buyer has been clicking on coffee things, the two can be matched without a single interaction on the new product. That idea is **content-based filtering**.

Sari's problem has a name: **cold start**. A *cold* item or user is one with no history yet, so every method that learns from past clicks has nothing to learn from. Content-based filtering is the standard answer to cold start, because it only needs facts that exist from the first second: the title, the brand and the category.

**What you'll learn**

- How to turn an item's text and metadata into a numeric vector, and why we weight words with TF-IDF instead of just counting them.
- What cosine similarity measures, and why it is the default distance for sparse text vectors.
- How to build a user profile as a decay-weighted average of the item vectors the user touched, and how to score every item against it.
- How the companion script `ch04_content.py` performs on the simulated marketplace, and why it beats global popularity but not category-aware popularity.
- Where content-based methods shine (cold start) and where they fail (over-specialisation, shallow metadata).

Here is a quick map of the words this chapter uses. Each one is explained again, slowly, where it first appears.

| Word | Plain meaning |
|---|---|
| **Vector** | A list of numbers in a fixed order. We turn each product into one, so a computer can compare products. |
| **TF-IDF** | A way to fill that list so that rare, specific words count more than common, generic ones. |
| **Cosine similarity** | A score from 0 to 1 for how alike two vectors are. 1 means the same direction, 0 means nothing in common. |
| **User profile** | A taste summary of one shopper, built by blending the vectors of the products they touched. |
| **Cold start** | Recommending for a new item or new user who has no history yet. |
| **Filter bubble** | When a recommender keeps showing more of the same, so the shopper never sees anything different. |

## Intuition: the librarian who never watches the readers

Imagine a librarian who is not allowed to look at the lending records. She cannot say "people who borrowed this also borrowed that". All she may do is read the covers: the title, the subject shelf, the author. When you hand her three novels you enjoyed, she walks to the shelves, finds books whose covers *sound like* the three you gave her, and hands them over.

That librarian is a content-based recommender. She has two skills:

1. She can describe every book as a short list of descriptive words (an **item representation**).
2. She can summarise your taste as a list of the same kind of words (a **user profile**), and compare the two lists.

Notice what she does not need: other readers. That is both the superpower and the weakness: a book that arrived this morning is recommendable at once, but she will keep walking you to the crime shelf forever.

In our marketplace the "covers" are the `title`, `brand` and category (plus parent category) columns of the `items` table.

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

Everything here is cheap, deterministic and explainable, which is why content-based scoring remains a workhorse for "similar items" widgets and cold-start fallbacks.

## The algorithm

### Step 1: item representation as text

The simplest, and surprisingly effective, way to represent an item is to glue its descriptive fields into one string and treat that string as a short document. In the companion script this is one line:

```include
path: book/code/ch04_content.py
region: item_text
```

An item titled "Volt Skincare Lokal" in category "Skincare" with brand "Volt" and parent category "Beauty" becomes the document `Volt Skincare Lokal Skincare Volt Beauty`. Note that category and brand words are repeated on purpose: titles in this simulator are short, and repeating the structured fields lets them count more than once.

Next we turn each document into a **vector**. A vector is just a **list of numbers in a fixed order**. To build one, we first collect the **vocabulary**: the list of every distinct word that appears in any item. Then the vector holds one number per vocabulary word. The simplest recipe is a **bag of words**: shake the words out of the title as if from a bag, and write down how many times each vocabulary word appears.

For example, take a vocabulary of four words in the order (`aurora`, `coffee`, `sneakers`, `volt`). The title text `volt sneakers sneakers` becomes the vector $(0, 0, 2, 1)$: zero `aurora`, zero `coffee`, two `sneakers`, one `volt`. You will meet this exact vector again as item D in the hand-worked example below.

Why bother? A computer cannot tell that two titles "feel alike", but it can easily compare two lists of numbers. Real vocabularies have thousands of words, and any one title uses only a handful of them, so nearly every entry is zero. Such a mostly-zero vector is called **sparse**, and we store only its non-zero entries to save memory.

### Step 2: TF-IDF weighting

Bag-of-words has an obvious flaw. The word "home" appears in every item of the category "Home & Living" (and in the parent-category text of each one), so it tells us almost nothing about *which* home item a buyer wants, while the word "sneakers" is specific and carries much more information. We want **rare words to count more than common words**.

Think of a detective game where you must guess a suspect from clues. "Wears shoes" is a useless clue: nearly everyone does. "Walks with a limp" is a great clue: very few people do. A good clue is one that most suspects do *not* have. TF-IDF scores words the same way, using two parts:

- **TF, term frequency**: how many times the word appears in *this* item's text. More repeats, more important to this item.
- **IDF, inverse document frequency**: how *unusual* the word is across the whole catalogue. Here a "document" is one item's text. A word found in almost every item gets a low IDF; a word found in only a few items gets a high IDF.

Multiply the two and you get the **TF-IDF** (term frequency–inverse document frequency) weight. Without the IDF part, common words such as "home" would drown out the specific ones. In symbols, for a term $t$ in document $d$ out of a collection of $N$ documents:

$$
\text{tfidf}(t, d) = \text{tf}(t, d) \times \text{idf}(t)
$$

where $\text{tf}(t,d)$ is the number of times $t$ occurs in $d$, and the inverse document frequency downweights terms that occur in many documents. The version used by scikit-learn's `TfidfVectorizer` (a ready-made TF-IDF tool in the scikit-learn Python library), which our script uses, is the *smoothed* one:

$$
\text{idf}(t) = \ln\frac{1 + N}{1 + \text{df}(t)} + 1
$$

with $\text{df}(t)$ the **document frequency**: the number of documents that contain $t$. $\ln$ is the natural logarithm, a function that grows slowly: a bigger ratio inside gives a bigger result, but never explosively. The "+1" terms keep the logarithm finite and guarantee that even a term present in every document keeps a small positive weight.

Let us plug numbers into a catalogue of $N = 4$ items:

| The word appears in... | df | idf = ln((1+4)/(1+df)) + 1 |
|---|---|---|
| all 4 items (generic) | 4 | $\ln(5/5) + 1 = 1.000$ |
| 2 items | 2 | $\ln(5/3) + 1 \approx 1.511$ |
| 1 item (rare) | 1 | $\ln(5/2) + 1 \approx 1.916$ |

The rarer the word, the bigger the multiplier. If "sneakers" appears twice in an item (tf = 2) and is rare (idf 1.916), its weight is $2 \times 1.916 = 3.832$.

Finally we **L2-normalise** each document vector: we rescale it so that its length becomes exactly 1. The L2 length is the ordinary straight-line length you know from Pythagoras: square every entry, add them up, take the square root. For the vector $(3, 4)$ the length is $\sqrt{9 + 16} = 5$, so dividing every entry by 5 gives $(0.6, 0.8)$, which has length 1 and points the same way. Formally, each document vector $\mathbf{x}_d$ becomes

$$
\hat{\mathbf{x}}_d = \frac{\mathbf{x}_d}{\lVert \mathbf{x}_d \rVert_2}
$$

where $\lVert \mathbf{x}_d \rVert_2$ is that length (the little hat on $\hat{\mathbf{x}}$ means "length 1"). This makes sure a long title does not automatically look more important than a short one: only the *proportions* between its words remain.

### Step 3: cosine similarity

Cosine similarity answers one question: **do these two vectors point the same way?** Picture each vector as an arrow from the same starting point. Two arrows pointing in the same direction are "the same taste" (score 1); two arrows at right angles share nothing (score 0). We measure the angle between the arrows, rather than the distance between their tips:

$$
\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\lVert \mathbf{a} \rVert \, \lVert \mathbf{b} \rVert}
$$

Two pieces of notation. The **dot product** $\mathbf{a} \cdot \mathbf{b}$ means: multiply the numbers in matching positions, then add them up. And $\lVert \mathbf{a} \rVert$ is the length of $\mathbf{a}$ from the previous step. By hand with two small vectors, $\mathbf{a} = (1, 2, 0)$ and $\mathbf{b} = (2, 1, 3)$:

- dot product: $1\times2 + 2\times1 + 0\times3 = 4$.
- lengths: $\lVert \mathbf{a} \rVert = \sqrt{5} \approx 2.236$ and $\lVert \mathbf{b} \rVert = \sqrt{14} \approx 3.742$.
- cosine: $4 / (2.236 \times 3.742) \approx 0.478$.

Two sanity checks. Vectors $(1, 0)$ and $(0, 1)$ share no word at all, so their dot product is 0 and the cosine is 0. Vectors $(1, 2, 0)$ and $(2, 4, 0)$ are the same direction, one just twice as long: dot product 10, lengths product $2.236 \times 4.472 = 10$, cosine exactly 1.

Because TF-IDF vectors have no negative entries, cosine lies between 0 (no shared terms at all, the arrows are at right angles) and 1 (identical direction). When both vectors are already unit length, as ours are after step 2, the denominator is 1 and **cosine similarity is simply the dot product**. That is why the script can compute the similarity of every pair of items with one sparse matrix multiplication, `X @ X.T`: `X` has one row per item, and the product is a big table with the similarity of every item to every other item.

Why angle and not Euclidean distance (the straight-line distance between the two arrow tips)? A text repeated twice gives a vector twice as long but pointing the same way, as in our $(1, 2, 0)$ and $(2, 4, 0)$ example. The tips are far apart, yet the content is the same, and cosine correctly calls it identical.

### Step 4: the user profile

The profile is **a taste summary of one shopper, written as a vector in the same word-space as the items**. Items now live in a **vector space** (just "the place where all these vectors sit"). A user is described by what they did to items, so the natural profile is a **weighted average of the vectors of the items they interacted with**.

A weighted average is like a course mark where the exam counts more than a quiz. If a quiz scores 80 with weight 1 and an exam scores 90 with weight 5, the final mark is $(1\times80 + 5\times90)/(1+5) = 88.3$: pulled towards the exam because it weighs more. In the same way, the items that matter most pull the profile towards themselves. Two kinds of weight apply to each event (a click, an add-to-cart or an order):

- **Strength.** In this repo a click counts 1, an add-to-cart 3 and an order 5 (`EVENT_WEIGHT` in `recommender/recsys/schema.py`). Buying something is stronger evidence of taste than glancing at it.
- **Recency.** Taste drifts; a click from last week matters more than a click from last month. We apply exponential decay with a half-life $h$ of 7 days: an event of age $a$ days is multiplied by $2^{-a/h}$.

What is **exponential decay with a half-life**? It means an event loses *half* of its weight every $h$ days. With $h = 7$:

| Age of the event | Multiplier $2^{-a/7}$ |
|---|---|
| today (0 days) | 1 |
| 7 days | 0.5 |
| 14 days | 0.25 |
| 21 days | 0.125 |

So the weight fades smoothly: an order placed a week ago (strength 5) counts as 5 × 0.5 = 2.5, instead of vanishing abruptly or counting at full strength forever.

Now the formula, symbol by symbol. $E_u$ is the set of user $u$'s events. For one event $e$: $w_e$ is its strength, $a_e$ is its age in days, and $\hat{\mathbf{x}}_{i(e)}$ is the unit-length vector of the item $i(e)$ it touched. The top of the fraction adds up all the item vectors, each scaled by its strength and freshness; the bottom adds up those same scales. The profile is

$$
\mathbf{p}_u = \frac{\sum_{e \in E_u} w_e \, 2^{-a_e/h} \, \hat{\mathbf{x}}_{i(e)}}{\sum_{e \in E_u} w_e \, 2^{-a_e/h}},
\qquad
\hat{\mathbf{p}}_u = \frac{\mathbf{p}_u}{\lVert \mathbf{p}_u \rVert_2}.
$$

In words: multiply each touched item's vector by "strength × freshness", add the results, divide by the total of "strength × freshness", then scale the result to length 1 (the second equation).

The denominator makes it a proper average, so a user with 100 events does not get a bigger vector than a user with 5. The decay function is the same one the repo already uses for its "recently viewed" retriever (a retriever is the quick first step of a recommender that picks a shortlist of candidate items):

```include
path: recommender/recsys/candidates.py
symbol: _decay
```

### Step 5: scoring and ranking

Now we **score** items: we give every item a number saying how well it fits this user's taste. We use cosine similarity between the profile and the item. For every available item $j$ (one that is in stock at the cutoff), the score for user $u$ is

$$
\text{score}(u, j) = \hat{\mathbf{p}}_u \cdot \hat{\mathbf{x}}_j .
$$

Sort descending and keep the top 50 per user. Users with no history have an all-zero profile, so every item scores exactly 0, a complete tie. The script **breaks the tie** with a tiny popularity term (a very small extra number added to the score, which only matters when everything else is equal), so these users get the `pop_7d` list: the globally most popular items of the last 7 days, from chapter 3.

### A worked example by hand

Let us run the whole thing on a toy catalogue of four items, small enough to check with a pocket calculator. The vocabulary has four words and each item has just a few of them:

| Item | Text | aurora | coffee | sneakers | volt |
|---|---|---|---|---|---|
| A | `aurora coffee` | 1 | 1 | 0 | 0 |
| B | `volt coffee` | 0 | 1 | 0 | 1 |
| C | `aurora sneakers` | 1 | 0 | 1 | 0 |
| D | `volt sneakers sneakers` | 0 | 0 | 2 | 1 |

Every word occurs in exactly two of the four items, so $\text{idf} = \ln(5/3) + 1 \approx 1.511$ for all of them and cancels out after normalisation. After L2 normalisation:

- A $= (0.707,\ 0.707,\ 0,\ 0)$ and B $= (0,\ 0.707,\ 0,\ 0.707)$ and C $= (0.707,\ 0,\ 0.707,\ 0)$ (coordinates in the order aurora, coffee, sneakers, volt).
- D has raw TF-IDF $(0,\ 0,\ 2\cdot1.511,\ 1.511)$, length $1.511\sqrt{5}$, so $\hat{\mathbf{x}}_D = (0,\ 0,\ 0.894,\ 0.447)$. The word repeated twice dominates the vector.

Cosine similarities follow directly. A and B share only "coffee": $0.707 \times 0.707 = 0.5$. A and C share only "aurora": also 0.5. A and D share nothing: 0. C and D share "sneakers": $0.707 \times 0.894 = 0.632$.

Now a user clicked item A today (strength 1, age 0) and ordered item C seven days ago (strength 5, decay $2^{-7/7} = 0.5$). The effective weights are $1 \times 1 = 1$ and $5 \times 0.5 = 2.5$, sum 3.5. The profile is

$$
\mathbf{p} = \frac{1 \cdot \mathbf{x}_A + 2.5 \cdot \mathbf{x}_C}{3.5} = (0.707,\ 0.202,\ 0.505,\ 0),
$$

and after normalisation $\hat{\mathbf{p}} = (0.793,\ 0.226,\ 0.566,\ 0)$. The scores are A 0.721, B 0.160, C 0.961 and D 0.506. C wins because it carries both of the user's strongest words; D outranks B because the sneakers signal is strong while B shares only the weak "coffee" signal. (A and C are already seen; a production system filters them out, the simple baseline here does not.)

How to read this result: the profile $(0.793,\ 0.226,\ 0.566,\ 0)$ says "mostly `aurora`, then `sneakers`, a little `coffee`, nothing about `volt`". It is a readable taste summary, and each item's score is just how closely its own word-list lines up with it.


## Code walkthrough

The full script is `book/code/ch04_content.py`; the snippets below are included straight from it and run top to bottom.

### Preparing the item text

The script starts by loading the shared simulated data through `common.py`, so every algorithm in the book sees exactly the same history, **cutoff** (the moment in time we freeze; the model may only see what happened before it) and eval users (the shoppers on whom we measure the results). It then joins each item to its category and parent category names and builds the text:

```include
path: book/code/ch04_content.py
region: data_prep
```

Then the text column is built:

```include
path: book/code/ch04_content.py
region: item_text
```

Note that `hist` is `History.at(tables, CUTOFF)`: only the data visible at the cutoff. Items listed after the cutoff do not exist for this model, which keeps the evaluation honest.

### Fitting TF-IDF

```include
path: book/code/ch04_content.py
region: tfidf
```

`fit_transform` does three jobs at once: it learns the vocabulary and the document frequencies, computes the IDF for each word, and returns a sparse matrix `X` with one L2-normalised row per item. `item_row` maps an item id to its row number so we can look items up later. A **sparse matrix** is a table that stores only its non-zero cells. Keeping it sparse matters: with a real vocabulary of tens of thousands of words, a table with a cell for every item-word pair would be unmanageable, but each item only touches a handful of columns.

### Building profiles with sparse algebra

This is the heart of the script, and it avoids any Python loop over users:

```include
path: book/code/ch04_content.py
region: profile_build
```

The last line of the profile construction is the normalisation step:

```include
path: book/code/ch04_content.py
region: profile_norm
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

```include
path: book/code/ch04_content.py
region: scoring
```

`profile @ Xa.T` produces the full user × available-item score matrix in one product. Only items that are in stock at the cutoff (`avail`) are scored. The popularity tie-break is scaled to $10^{-6}$ (one millionth) so it can only separate scores that are otherwise equal: it decides among cold users (all zeros) and among near-duplicate items, but never overrides a real content difference. The final `argsort` returns item positions sorted by score, from which we pick the top 50 per user, and `common.evaluate` computes exactly the same metrics used for all other chapters.

## Run it

From the `book/` directory:

```bash
uv run python code/ch04_content.py
```

The script first prints the vocabulary statistics and IDF extremes:

```output
file: results/ch04_content.txt
lines: 8-21
```

The documents are short (4.8 non-zero terms per item) and the vocabulary is just 64 terms, because the simulator builds titles from a template; a real marketplace has millions of words, model numbers and typos. The IDF ranking behaves as theory promises: "living" and "home" have the lowest weights, near 2.86, while "sneakers", "shampoo" and "skincare" have the highest, around 4.5 to 4.7.

## Results

### What the profiles look like

```output
file: results/ch04_content.txt
lines: 26-32
```

Of the 947 eval users, 919 have a usable profile; 28 have no positive history and fall back to the popularity tie-break. For user 894 the strongest term is "diapers" (0.809), followed by "baby" and "mother" (both 0.301): a parent buying baby supplies, and the profile is directly readable by a human.

### How well does it rank?

```output
file: results/ch04_content.txt
lines: 34-37
```

Here is a reminder of the metrics in that table. **NDCG@20** (normalised discounted cumulative gain) is a score from 0 to 1 that rewards placing the item the shopper really bought near the *top* of their 20 recommendations. **recall@20** is the share of the items a shopper really bought that appear anywhere in the 20. **coverage@20** is the share of the whole catalogue that appears in *anyone's* top 20. For every metric, higher is better.

Chapter 3 reported `pop_7d` (most popular items of the last 7 days) at 0.0475 and `pop_in_my_categories` (most popular items inside the categories the user already shops in) at 0.1302 on the same labels.

Three conclusions:

1. **Content-based beats global popularity.** The NDCG@20 of `content_tfidf` is 0.0909 against 0.0475 for `pop_7d`, nearly double. Personal taste, even from category-and-brand words alone, carries real information.
2. **It loses to category-aware popularity.** `pop_in_my_categories` scores 0.1302, clearly above our 0.0909. Why? Both methods exploit the same fact: users shop within categories. But `pop_in_my_categories` also ranks *inside* the category by popularity, which separates good items from poor ones. Our TF-IDF vector contains only category and brand words, so every item of a category looks almost the same, and the choice among them is a coin flip (decided only by the $10^{-6}$ popularity tie-break).
3. **Coverage is excellent.** `coverage@20` of 0.9892 means that nearly the whole in-stock catalogue shows up in somebody's top 20, against 0.0087 for popularity, which recommends the same few dozen items to everyone. Content-based methods are a natural antidote to the "rich get richer" effect we met in chapter 3. The coverage figure is slightly inflated by the random-ish choice among near-ties, but the direction is real.

### Nearest neighbours and the "cosine = 1.0" lesson

```output
file: results/ch04_content.txt
lines: 39-47
```

Look at the first query. "Aurora Coffee & Tea Import" has a nearest neighbour with cosine **1.000**: a different item (id 2399) with exactly the same title and category. The next ones, 0.953, differ only in the suffix "Lokal", "Lite" or "Premium", which contributes almost nothing.

This is a teaching point, not a bug. Identical text gives identical vectors, so no content-based method can tell the items apart; a content-only recommender will happily fill the top 5 slots with copies of the same product. Telling them apart needs *behavioural* signals such as price, rating and popularity.

A **heat map** is a table where each cell is coloured by its number: bright means high, dark means low. This one shows the cosine between 15 items, five from each of three categories:

![Cosine similarity between 15 items from three categories (Coffee & Tea, Snacks, Sneakers). Black lines separate the categories. Blocks on the diagonal are bright; everything off-diagonal is near zero.](../assets/charts/ch04-similarity-heatmap.png)

Items of the same category are similar; items of different categories are not. Numbers confirm it:

```output
file: results/ch04_content.txt
lines: 62-65
```

The rows and columns are Coffee & Tea, Snacks and Sneakers. The mean similarity between two different items of the same category is 0.752, 0.633 and 0.681 on the diagonal; off-diagonal it is at most 0.084. The faint squares off the diagonal come from shared brand words such as "Aurora".

::: fact
**TF-IDF is over fifty years old.** The idea of weighting a term by the inverse of how many documents contain it was published by Karen Spärck Jones in 1972, as "a statistical interpretation of term specificity". It has been used in search engines and recommender systems ever since, and it is still a strong baseline today.

Source: Spärck Jones, Journal of Documentation, 1972
:::

::: fact
**Pandora's "Music Genome" is content-based filtering done by hand.** Pandora's founders assembled a team of musicologists who analysed and classified songs on roughly 400 attributes, such as vocal style or rhythm, and used those attribute vectors to find songs similar to ones a listener liked, instead of relying on other listeners' histories.

Source: Westergren, NPR interview, 2011
:::

### The same idea inside the app

![The marketplace's search page for the query "coffee". Results are matched on title and category text and then sorted by popularity.](../assets/screenshots/search.png)

The app already contains a tiny content-based component: the backend matches the typed query against product titles and category names (an `ILIKE` filter, meaning a case-insensitive "contains this text" match, in `backend/src/catalog.rs`), and the "Popular" ordering is not content-based at all. The profile approach of this chapter replaces the typed query with a *query built from the user's own behaviour*.

## Pitfalls and how it is done in production

### Over-specialisation

The librarian who only reads covers keeps recommending more of the same. Our profile for user 894 is dominated by "diapers", so the recommendations will be diaper-adjacent; if that user needs a stroller next, no word overlap suggests it. This is **over-specialisation**, and when it happens to a whole feed it is often called a **filter bubble**: the shopper sees only what resembles their past, and the system never learns whether they would have liked something different. The system has no **serendipity** (the pleasant surprise of finding something you did not know you wanted).

Mitigations: blend with collaborative signals such as co-visitation (items viewed together in the same session; chapter 5), re-rank for diversity with MMR (maximal marginal relevance: a re-ranking rule that penalises an item that is too similar to ones already chosen) and category caps (chapter 11), and reserve a few slots for exploration.

### Shallow metadata, noisy text

Content-based methods are only as good as the content. Here titles come from a template; real seller titles are stuffed with keywords ("MURAH!!! Best Seller Sneakers Pria Wanita Original 100%"), misspell brands and mix languages. Production systems clean titles, extract structured attributes, and increasingly replace TF-IDF with learned text and image **embeddings** (vectors a model has learned so that similar things end up close together; chapter 7).

### The cold-start strength

A brand-new item has no clicks, so popularity, co-visitation and matrix factorisation (chapters 3, 5, 6) cannot recommend it. But its title, brand and category exist from the first second, so a content-based scorer can place it in front of the right buyers *immediately*; we revisit this in chapter 12. A new *user* with a single click is similarly served: one click already yields a non-zero profile. Remember that content tells the system *where to show* an item; only real clicks tell it *whether the item is good*.

### Scalability and tuning

Item vectors are computed offline and stored in an approximate-nearest-neighbour index (a lookup structure that finds items with similar vectors quickly, without comparing against every single item, at the cost of occasionally missing the very best match; chapter 7); the profile is looked up at request time. Tune the half-life, the event weights and the text recipe on the same time-split evaluation as everything else; you will vary the half-life in the hands-on exercise.

### How this fits the repo's ranking system

The content-based score is not a retriever in the final repo pipeline; the four retrievers are `recent`, `covisit`, `catpop` and `global_trending`. Content knowledge enters the ranker (the slower, smarter model that sorts the shortlist) differently: as *features* (input numbers the model learns from) such as `uc_click_share_30d` (the share of the user's clicks that fall in the item's category) and `item_price_vs_category` (price divided by the category median), defined in `schema.py` and computed in `features.py`. That is a common production pattern. A cheap content score becomes one input among many into a learned ranker, rather than the whole recommender. The ranker then learns, from data, how much to trust content relative to popularity or recency.

## Summary

- Content-based filtering represents every item by its own attributes (text, category, brand) and recommends items that resemble what the user already liked.
- TF-IDF weights words by term frequency times inverse document frequency, so specific words ("sneakers") matter more than generic ones ("home"); L2 normalisation makes cosine similarity equal to a dot product.
- A user profile is the weighted average of the vectors of items the user touched, with weights equal to event strength times exponential time decay (half-life 7 days here).
- On the simulated marketplace the method reaches an NDCG@20 of 0.0909, almost twice the 0.0475 of `pop_7d`, but below the 0.1302 of `pop_in_my_categories`, because category and brand words cannot separate items inside a category.
- Coverage is very high (0.9892 versus 0.0087 for popularity), but near-duplicate titles have cosine similarity exactly 1.0, so content alone cannot tell copies apart; behavioural signals must.
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

```include
path: book/code/ch04_content.py
region: profile_norm
```

What would go wrong without `np.where(wsum > 0, wsum, 1)`, and what would the profile row of a user with no positive events be after the line runs?

**Q4.7** A user clicked item A today (strength 1, age 0 days) and added item B to the cart 14 days ago (strength 3). With a half-life of 7 days, what fraction of the final profile's total weight comes from item A? (Give the answer to three decimals.)

**Q4.8** Two items have unit-length vectors $\mathbf{a} = (0.6,\ 0.8,\ 0)$ and $\mathbf{b} = (0.8,\ 0,\ 0.6)$. Compute their cosine similarity.

**Q4.9** *Hands-on.* In `book/code/ch04_content.py`, change `HALF_LIFE = 7.0` to `1.0`, run the script, and then repeat with `30.0`. Compare the `ndcg@20` and `recall@20` in the results table with the baseline of 0.0909 and 0.1378. What do you expect to observe about how quickly the profile should forget, and does the outcome match? (Restore the value afterwards.)

