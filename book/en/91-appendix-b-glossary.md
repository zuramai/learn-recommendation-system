# Appendix B — Glossary

Terms are listed alphabetically. Each entry gives a plain-language meaning, sometimes with a tiny example. The chapter where a term is introduced and explained in full is in brackets.

| Term | Meaning |
|---|---|
| **A/B test** | An experiment where a random half of shoppers see the old system (control) and the other half see the new one (treatment), and a statistical test tells us whether the difference in a metric is real or just luck. [14] |
| **Ablation** | Retraining a model with one group of features removed to see how much worse it gets; the drop shows how much that group was worth. [10] |
| **Add-to-cart** | A logged event for a shopper putting an item in the cart; it shows stronger interest than a click. [1] |
| **ALS (Alternating Least Squares)** | A way to learn taste cards (see *Matrix factorisation*): fix the item cards and solve for the best user cards, then fix the user cards and solve for the items, and repeat. [6] |
| **ANN (Approximate Nearest Neighbour)** | A search that finds items that are *almost* the closest to a query, much faster than comparing against every single item. [7] |
| **AP (Average Precision)** | A score for one list: at every rank where a relevant item appears, note the share of relevant items so far, then average those shares. [2] |
| **Bayesian average** | A rate pulled toward the overall average until an item has enough data, so 1 click out of 1 view is not treated as a perfect 100%. Also called *smoothing*. [3] |
| **Beacon** | A browser feature that hands a small message to the browser to deliver after the page has closed, like dropping a letter in a postbox on your way out. [13] |
| **Beta distribution** | A curve that shows how plausible each click rate is for an item, given its clicks and non-clicks so far; it is narrow when we have lots of data and wide when we have little. [11] |
| **BPR (Bayesian Personalised Ranking)** | A training goal that says: for each shopper, score an item they interacted with higher than one they did not. [6] |
| **Candidate** | An item that a first-stage retriever proposes for a shopper; only candidates are scored by the ranker. [8] |
| **Candidate recall** | The share of the items a shopper will actually engage with that made it into the candidate list. It is the recall ceiling: the ranker cannot recommend what was never proposed. [8] |
| **Co-visitation** | Counting how often two items are used in the same session; it powers "people who viewed this also viewed that". [5] |
| **Cold start** | A new shopper or item with little or no history, so there is nothing yet for a personal method to learn from. [12] |
| **Collaborative filtering** | Recommending from the behaviour of many shoppers (similar shoppers, or items used together), without reading what the items are. [5] |
| **Content-based filtering** | Recommending items whose descriptions (text, category, brand) look like items the shopper already liked. [4] |
| **Cosine similarity** | A score from 0 to 1 for how alike two vectors are, based on the angle between them: 1 means the same direction, 0 means nothing in common. [4] |
| **Coverage** | The share of the catalogue that appears in anyone's top K; low coverage means many items never get shown. [2] |
| **CSR (Compressed Sparse Row)** | A way to store a mostly empty table by keeping only the non-empty cells and where they sit, row by row. [5] |
| **CTR (Click-through rate)** | Clicks divided by impressions: out of 100 views, 5 clicks is a CTR of 5%. [1] |
| **Cross feature** | A feature about a pair, such as *this shopper × this item*, rather than about the shopper or the item alone. [9] |
| **Cutoff** | The moment we freeze time at: features may use only events before it, and what happens after it is the answer key. [2] |
| **DCG / NDCG** | DCG gives credit for each relevant item, with less credit the further down it sits ($1/\log_2(\text{rank}+1)$); NDCG divides by the best score possible, so it runs from 0 to 1. [2] |
| **Decay (half-life)** | Counting older events for less. With a 7-day half-life, an action a week old counts half as much as one today. [3] |
| **Dense** | Stored in full, every cell written down even when it is zero; the opposite of *sparse*. [6] |
| **Dot product** | Multiply two lists of numbers position by position, then add the results: $(1,2)\cdot(3,4) = 1\times3 + 2\times4 = 11$. A bigger result means the two lists point the same way. [4] |
| **Dwell time** | How many seconds a shopper stays on a product page; longer usually means more interest. [1] |
| **Embedding** | A short list of numbers learned for a shopper or an item, arranged so that similar things get similar lists. [7] |
| **Epoch** | One full pass through all the training data. [6] |
| **Event** | One logged action by a shopper on an item: impression, click, add-to-cart, order, hide or return. [1] |
| **Explicit feedback** | Ratings or reviews that shoppers give on purpose. [1] |
| **Exploration** | Deliberately showing items we are unsure about so we can learn about them, at a small short-term cost (ε-greedy and Thompson sampling are two ways). [11] |
| **Exposure** | Whether, and how often, an item was actually shown to a shopper; an item never shown cannot get clicks. [1] |
| **Fallback** | What the system shows when the recommender is unavailable or the shopper is unknown; here, recent best sellers. [12] |
| **Feature** | A number that describes a shopper–item pair at one moment, such as `user_clicks_30d`, and that the model reads as input. [9] |
| **Feature store** | A shared place that defines each feature once and serves the same numbers to training and to live requests. [14] |
| **Feedback loop** | The recommender decides what is shown, what is shown decides the clicks, and the clicks become its next training data, so it can end up reinforcing its own habits. [12] |
| **Gradient** | For each number the model can change, a signal that says which direction (up or down) makes the loss smaller, and by how much. [6] |
| **Gradient boosting** | Building a model from many small decision trees, each one trained to fix the mistakes left by the trees before it. [10] |
| **Guardrail metric** | A number that must not get worse (such as coverage or page speed) even while we try to improve the main metric. [2] |
| **Hard filter** | A rule that removes candidates outright, such as out of stock or hidden by the shopper. [8] |
| **Hit rate@K** | The share of shoppers for whom at least one of the top K items was a hit; if 4 of 10 shoppers found something, it is 0.4. [2] |
| **HNSW (Hierarchical Navigable Small World)** | An ANN method that links each vector to its neighbours in a layered graph and finds matches by walking along the links. [7] |
| **Hyperparameter** | A setting you choose before training (such as the embedding size or the window length in days), as opposed to a parameter that training learns. [3] |
| **Impression** | A moment when an item was shown on screen. [1] |
| **Implicit feedback** | Behaviour that hints at interest without being a rating: views, clicks, purchases. [1] |
| **In-batch negatives** | When training on a batch of examples, treating the other shoppers' positive items in that batch as the "wrong answers" for each shopper. [7] |
| **IPW (Inverse propensity weighting)** | Giving each logged click more weight when it was unlikely to be seen in the first place (weight = 1 ÷ chance of being shown), to correct for position and exposure bias. [12] |
| **Item-to-item** | Recommending items that are similar to the ones a shopper used, using a table of how alike item pairs are. [5] |
| **IVF (Inverted file index)** | An ANN method that sorts vectors into groups and searches only the few groups nearest to the query. [7] |
| **k-means** | A method that sorts points into $k$ groups: pick $k$ centres, assign each point to its nearest centre, move each centre to the average of its points, and repeat. [7] |
| **Label** | What actually happened after the cutoff (0 nothing, 1 click, 2 cart, 3 order); the answer the model learns to predict. [9] |
| **Label window** | The stretch of time right after the cutoff (here, one week) whose events become the labels; nothing from it may be used for training or features. [2] |
| **LambdaMART** | A gradient-boosted ranker whose training signal is larger for pairs of items where swapping them would change NDCG a lot. [10] |
| **Latency, p50, p99** | Latency is how long one request takes. p50 is the typical (median) time; p99 is the time that 99 of every 100 requests beat, so it shows the slow ones. [13] |
| **Latency budget** | The time each stage of a request is allowed to take so that the whole response arrives before its deadline. [14] |
| **Latent factor** | A hidden taste dimension (such as "strong and black" versus "sweet and milky") that nobody labelled but the model discovers from the data. [6] |
| **Leakage** | Using information from after the cutoff when building features; it is cheating, and it inflates offline scores. [9] |
| **Learning rate** | How big a step the model takes each time it adjusts its numbers; too big overshoots, too small is slow. [6] |
| **Learning to rank** | Training a model to put items in the right order for a shopper, rather than to predict each item's value on its own. [10] |
| **Logit** | A raw score before it is turned into a probability. [7] |
| **logQ correction** | Subtracting the log of how often an item is picked as a negative from its score, so popular items are not unfairly punished when training with in-batch negatives. [7] |
| **Long tail** | The many items that each get little attention but together make up a large share of the catalogue. [1] |
| **Loss** | A single number for "how wrong is the model right now"; training tries to make it small. If it predicts 3.5 and the truth is 5, the error behind the loss is 1.5. [6] |
| **Matrix factorisation** | Summarising the big shopper × item table as two thin tables of short "taste cards", one per shopper and one per item, whose dot product guesses the missing cells. [6] |
| **MMR (Maximal Marginal Relevance)** | A re-ranking rule that picks the next item by balancing how relevant it is against how similar it is to items already chosen. [11] |
| **MRR (Mean Reciprocal Rank)** | For each shopper take 1 ÷ the rank of their first hit (rank 1 gives 1, rank 2 gives 0.5, rank 4 gives 0.25), then average over shoppers. [2] |
| **Multi-task learning** | One model that predicts several outcomes (click, cart, order) at once, sharing its inner layers. [14] |
| **Overfitting** | A model memorising its training data, like a student who memorises last year's exam answers and then fails this year's exam. [6] |
| **PCA (Principal Component Analysis)** | A way to squash many-number vectors down to two numbers by keeping the directions along which the points spread out most, so they can be drawn on a page. [7] |
| **Point-in-time correctness** | Computing every feature using only the data that existed at its cutoff, as if we were standing in that moment. [9] |
| **Popularity bias** | Recommenders favouring items that are already popular, which then get more exposure and become even more popular. [3] |
| **Position bias** | Items shown higher on the page get more clicks even if they are no better. [12] |
| **PQ (Product quantisation)** | Compressing a vector by cutting it into short pieces and replacing each piece with the number of its closest entry in a small palette, like painting by numbers. [14] |
| **Precision@K / Recall@K** | Precision@K is the share of the top K that is relevant; recall@K is the share of all the relevant items that landed in the top K. [2] |
| **Profiling** | Measuring where a program spends its time, stage by stage, so that you speed up the part that is actually slow. [13] |
| **Propensity** | The chance that an item was shown to a shopper in the first place. [12] |
| **Query group** | All the rows that belong to one shopper's list; ranking is compared only inside a group. [10] |
| **Recall ceiling** | See *Candidate recall*. [8] |
| **Recency** | How recently something happened; newer events usually matter more. [4] |
| **Regularisation** | Adding a penalty for large numbers to the loss so the model stays simple and does not overfit. [6] |
| **Re-ranking** | A final step that applies page-level rules (limits per category, removing already-bought items) to the model's scores. [11] |
| **Retrieval** | The first stage: cheaply picking a few hundred candidates out of the whole catalogue. [8] |
| **Sentinel value** | A deliberately extreme number (such as 9999) that means "this never happened". [9] |
| **Session** | One burst of shopper activity; here a gap of 30 minutes starts a new session. [13] |
| **Shard** | One slice of the catalogue kept on its own machine, like one branch of a library; a query asks every shard and merges the answers. [14] |
| **SHAP** | A method that splits one prediction into a contribution from each feature, and those contributions add up to the model's output. [10] |
| **Smoothing** | See *Bayesian average*. [3] |
| **Softmax** | A recipe that turns any list of scores into probabilities that add up to 1, giving bigger scores bigger shares. [7] |
| **Sparse matrix** | A big table in which almost every cell is empty, so we store only the non-empty cells. [4] |
| **Sparsity** | The share of the shopper × item table that is empty; usually above 99%. [1] |
| **SRM (Sample ratio mismatch)** | The two groups of an A/B test came out a different size than planned (for example 50.6/49.4 instead of 50/50 on millions of users), which signals a bug, so the results cannot be trusted. [14] |
| **Temperature** | A dial in softmax: a low temperature makes the model sharper and more confident, a high one makes it flatter and more cautious. [7] |
| **TF-IDF** | A word weight: high for a word that is common in one description but rare across all descriptions, so it describes that item. [4] |
| **Thompson sampling** | An exploration method: draw one random plausible click rate for each item from its Beta distribution and show the items with the highest draws. [11] |
| **Training/serving skew** | A mismatch between how features are computed for training and how they are computed live, which quietly hurts the model. [14] |
| **Two-tower model** | A retrieval model with one small neural network for shoppers and one for items, whose output vectors are compared with a dot product. [7] |
| **TTL (Time to live)** | How long a saved (cached) answer may be reused before it must be recomputed. [14] |
| **Vector** | A list of numbers in a fixed order; we turn each product or shopper into one so a computer can compare them. [4] |
| **Weak learner** | A very simple model, such as a tree with one question and two answers, that is only slightly better than guessing and is useful only in large numbers. [10] |
