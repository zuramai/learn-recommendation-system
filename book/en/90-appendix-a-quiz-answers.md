# Appendix A — Quiz answers

Answers to every end-of-chapter quiz, with a short explanation. Hands-on exercises describe what you should observe; your exact numbers depend on the change you make.

## Chapter 1 — What a Recommender Does

**Q1.1** — B. Retrieval cheaply narrows millions of items to a few hundred candidates and is judged on recall. Option A describes ranking, C describes re-ranking, and D is unrelated to serving. Whatever retrieval misses can never be recovered by the stages after it.

**Q1.2** — C. Dwell time and add-to-cart are by-products of normal behaviour, not deliberate statements of opinion. Star ratings, written reviews and thumbs-up taps are all explicit feedback because the user chose to express a preference.

**Q1.3** — B. A real platform only sees logs, so the simulator writes only what an app would log and keeps the latent variables private. Methods must therefore recover the signal from the same kind of data they would see in production. Because the truth exists, we also know the data is learnable, which helps when diagnosing a failing method.

**Q1.4** — CTR is clicks divided by impressions, and the impression count is the denominator that says how often an item was actually shown. If only clicks are logged, a popular-looking item might simply have been shown very often, and you cannot tell "shown and ignored" from "never shown". You would also have no negatives for training and no way to study position bias.

**Q1.5** — Sparsity is the fraction of user-item cells with no recorded interaction (here 99.388%). Each user has touched very few items, so an algorithm must infer taste from a handful of filled cells. Consequently methods share statistical strength across users and items: neighbourhood methods (chapter 5), low-rank factorisation (chapter 6), and content features. Any answer naming one such consequence is correct.

**Q1.6** — A Saturday sale day has raw weight 1.2 × 3.0 = 3.6. A Sunday payday that is not a sale day has 1.2 × 1.4 = 1.68. The flags are returned because the simulation loop reuses them: on sale days the effective discount is raised by 0.2, and the order probability has payday and sale bonuses, so time changes behaviour as well as traffic volume.

**Q1.7** — 251 orders. Multiply the three rates: 0.1059 × 0.4295 × 0.5517 ≈ 0.02509 (the 2.51% of impressions that end in an order), and 10,000 × 0.02509 ≈ 250.9, which rounds to 251.

**Q1.8** — Density = 50,000 / (1,000 × 10,000) = 50,000 / 10,000,000 = 0.5%, so sparsity = 99.5%. Even this "dense-looking" example leaves 199 of every 200 cells empty.

**Q1.9** — After the change, the script prints the top 50% (1,250 items) receiving 92.8% of interactions, so the bottom half of the catalog shares only about 7% of them. The top 5% (125 items) gets 35.1%, which lies between the 12.5% of the top 1% and the 50.3% of the top 10%. This shows how steeply interactions concentrate in the head and how thin the tail is, which is why popularity baselines are strong.

**Q1.10** — Two good reasons: (1) the user may never have noticed the item, because it sat low in the list or off-screen (exposure and position bias), so no click does not mean no interest; (2) the user may have been interested but chose a different item first or intended to come back later. Treating these as hard negatives teaches the model to repeat what was shown before and penalises items for being badly placed.

## Chapter 2 — Measuring success

**Q2.1** — B. With a random shuffle, some test events are older than training events, so the model learns from the future (popularity spikes, a user's later taste, other events from the same session). A is not the problem, C is nonsense, and D is unrelated to how data is split. A time-based split removes the leak.

**Q2.2** — C. Both lists contain the same set of items, so recall@10, precision@10 and coverage are identical. NDCG discounts by position, so the list that puts the ordered item (gain 7) first scores higher than the one that puts it last.

**Q2.3** — B. Candidate recall is the share of future positives that appear in the candidate set, so no ranker that only reorders those candidates can exceed it. It is not "always lower" than final recall (A is wrong: it is an upper bound), it is not a breadth measure (C is coverage), and improving it needs better retrievers, not a better ranker (D).

**Q2.4** — Because AP should also punish positives that never appear. Missing positives add nothing to the numerator but still count in the denominator $R$. Dividing by the number of hits would reward a list that finds one item at rank 1 and ignores four others as much as one that finds all five.

**Q2.5** — With an inner join (or a merge that starts from `recs`), future positives that were never recommended would disappear from the table. The number of rows per user would then be the number of hits instead of $R$, so recall would be hits/hits = 1 for everyone who has any hit, and users with no hits would vanish rather than score 0. The left merge from the labels keeps every positive in the denominator.

**Q2.6** — `dropna(subset=["pos"])` removes the future positives that were not recommended in the top $k$ (their `pos` is NaN after the left merge), so only hits contribute to DCG. The constant is `+ 2` because `pos` is zero-based while the formula uses the one-based rank $i$: $\log_2(i+1) = \log_2(\text{pos}+2)$. Using `+ 1` would divide by $\log_2 1 = 0$ at the first slot.

**Q2.7** — DCG@3 = 4.131, IDCG@3 = 7.631, NDCG@3 = 0.541. DCG = $0/\log_2 2 + 1/\log_2 3 + 7/\log_2 4 = 0.631 + 3.5 = 4.131$. The ideal order puts the order first: $7/\log_2 2 + 1/\log_2 3 = 7 + 0.631 = 7.631$. The ratio is $4.131/7.631 \approx 0.541$.

**Q2.8** — precision@5 = 0.4 and recall@5 = 0.5. Two hits in five slots gives $2/5 = 0.4$; two of the four future positives were found, giving $2/4 = 0.5$.

**Q2.9** — NDCG is an average over users of ranking quality and does not see breadth or the pipeline ceiling. Coverage reveals a model that shows the same few bestsellers to everyone (good NDCG on a popularity-dominated week, tiny share of the catalogue ever shown). Candidate recall reveals that the first stage never retrieves many of the items users later buy, so the ranker has nothing to choose from and NDCG improvements from re-tuning it will be capped; NDCG alone does not tell you whether the bottleneck is retrieval or ranking.

**Q2.10** — Coverage@20 should drop to about $20/2311 \approx 0.0087$, because only 20 distinct items are ever shown across all users (the number of available items is 2311 in the printed output). Accuracy numbers will differ from the random run, but they will not be personalised. The exercise shows that coverage and accuracy are separate axes: random had coverage 1.0 with near-zero accuracy, and a constant list has near-zero coverage.

## Chapter 3 — Popularity baselines

**Q3.1** — B. In the simulator each item has a fixed hidden quality, so popularity does not drift. When the underlying popularity is stable, the 53-day count is a less noisy estimate of it than the 7-day count. On real marketplaces with flash sales and seasonality the window often wins, so tune it on held-out data.

**Q3.2** — C. With $v=0$ the weight on the item's own rating, $v/(v+m)$, is zero, so the score equals the global mean $C$ (4.324 in the printout). The stated 4.4 or 4.5 is just a placeholder written to new items. Smoothing treats it as "no information".

**Q3.3** — B. Coverage@20 is the number of distinct recommended items divided by the available catalogue. A global ranking gives the same 20 items to everybody, so only 20 items are ever shown. This is the root of popularity bias and the feedback loop.

**Q3.4** — Pass `num` = total stars (R · v), `den` = number of ratings v, `prior` = the global mean C and `strength` = m. The function then returns (R·v + C·m)/(v + m). Splitting the fraction gives v/(v+m)·R + m/(v+m)·C, which is the Bayesian average. The same call works for CTR (clicks, impressions, global CTR, pseudo-impressions).

**Q3.5** — Items shown at the top get more impressions and clicks, which raises their popularity score, which keeps them at the top, while items never shown cannot collect evidence. Any of these weakens it: personalisation, exploration (ε-greedy, Thompson sampling), exposure-normalised rates such as smoothed CTR, position-bias correction or diversity re-ranking.

**Q3.6** — No. The filter is `age_days < window_days`, strict, so an event with age exactly 7.0 is excluded. Every surviving interaction keeps its full weight `w` (1, 3 or 5) with no decay, because this is a hard window. Changing `<` to `<=` would include an event aged exactly 7 days; to include the whole 8th day you would call it with `window_days=8` (or use `<= 8`).

**Q3.7** — About 4.98. Order: 5 · 2^(−7/7) = 5 · 0.5 = 2.5. Click: 1 · 2^(−14/7) = 0.25. Add-to-cart: 3 · 2^(−3/7) ≈ 3 · 0.743 = 2.229. Sum = 2.5 + 0.25 + 2.229 ≈ 4.98.

**Q3.8** — With v = 20: 20/120·4.8 + 100/120·4.0 = 0.8 + 3.333 = 4.133. With v = 400: 400/500·4.8 + 100/500·4.0 = 3.84 + 0.80 = 4.64. More ratings move the estimate from the prior (4.0) towards the item's own mean (4.8).

**Q3.9** — Half-life = 10 days / 2 = 5 days, because 25% = 2^(−2), i.e. two half-lives have elapsed at age 10. A smaller half-life concentrates the weight on the most recent events, so the score reacts quickly to new behaviour but rests on fewer effective events, which makes it noisier.

**Q3.10** — With a very short half-life (2 days) pop_decay behaves like a short window: its behaviour should move towards the short-memory pop_7d row, since it rests on few effective events. With a very long half-life (30 days) the weights are close to 1 for the whole 53-day history, so pop_decay approaches pop_all_time. This matches the explanation that, with stable simulated popularity, more effective history means less noise. Exact values depend on your run; the ordering is what to check.

## Chapter 4 — Content-based filtering

**Q4.1** — B. IDF is large for rare terms and small for terms present in many documents, so a word like "sneakers" counts for more than "home". Length preference (A) is not a goal; normalisation actually removes it, and TF-IDF does not make vectors orthogonal or remove stop-words by itself.

**Q4.2** — B. For unit vectors the denominator of cosine is 1, so cosine equals the dot product, which is why one sparse product `X @ X.T` yields all pairwise similarities. It is not Euclidean distance (although the two are monotonically related for unit vectors), it is non-negative for TF-IDF vectors, and normalisation removes the dependence on text length.

**Q4.3** — C. The content-based scorer needs only the item's own attributes, which exist from the moment it is listed. Popularity, co-visitation and order-based methods all need interactions that a brand-new item does not have.

**Q4.4** — The two items have identical text after tokenisation (same title, category and brand words), so their TF-IDF vectors are identical and the angle between them is zero. A content-only recommender cannot tell them apart and may fill the top slots with copies of the same product; breaking the tie needs behavioural signals such as price, rating, popularity or seller quality, or a diversity step.

**Q4.5** — Over-specialisation means the system only recommends items that resemble the user's history, so it never surprises them or covers a new need. Remedies: blend with collaborative signals such as co-visitation, re-rank for diversity (MMR or category caps), or reserve slots for exploration. Any two of these are acceptable.

**Q4.6** — Without the guard, a user with no positive events has `wsum == 0` and `1 / wsum` becomes infinity, so multiplying the all-zero row by it yields NaN. With the guard the divisor is 1, the row of `W @ X` stays all zeros, and `normalize` leaves a zero vector. The profile is therefore all zeros and every item scores 0, which is why the script adds a tiny popularity tie-break for these users.

**Q4.7** — 0.571. Item A has weight 1 × 2^0 = 1. Item B has weight 3 × 2^(−14/7) = 3 × 0.25 = 0.75. The share of A is 1 / (1 + 0.75) = 0.5714, i.e. 0.571. The older cart event, despite its higher strength, contributes less than the fresh click.

**Q4.8** — 0.48. The vectors are unit length, so cosine is the dot product: 0.6 × 0.8 + 0.8 × 0 + 0 × 0.6 = 0.48. They share only the first dimension, which is why the similarity is moderate rather than high.

**Q4.9** — Expected observation: very short half-lives such as 1 day throw away most of the history, so profiles rest on a few events and quality typically drops; very long ones such as 30 days approach an unweighted average and may react slowly, so a moderate half-life near the baseline of 7 days is typically best. Your actual numbers may differ, because the simulator's interests are fairly stable. Whatever you observe, compare ndcg@20 and recall@20 to 0.0909 and 0.1378 and treat differences within a few thousandths with caution.

## Chapter 5 — Collaborative filtering

**Q5.1** — B. User-based CF assumes that people who agreed in the past will probably agree again, so the items liked by users with overlapping histories are good recommendations. It does not look at titles (A), it is not about popularity alone (C), and it does not assume static taste (D) — which is why time decay is used elsewhere in the book.

**Q5.2** — C. Item-to-item CF computes the expensive item–item similar-items table offline. At request time it only looks up neighbours of the items in the user's history, so cost is independent of catalogue size and customer count. User-kNN must compare the target to all users per request.

**Q5.3** — B. Jaccard only asks "present or not", which fits binary signals such as same-session membership and stops heavy users from dominating through counts. When weights carry information (order = 5 clicks), cosine on the weighted matrix keeps that information. Both measures are symmetric.

**Q5.4** — Each row is normalised so that the outgoing weights of every source item sum to 1. Without it, pairs involving very popular items would have huge raw counts and would dominate every candidate list regardless of how strongly the items are actually related. Normalising makes weights comparable across source items and ranks neighbours of each item by relative strength; it also makes the matrix asymmetric.

**Q5.5** — Both items have few interactions and a couple of shared users, so cosine is inflated by tiny support (the table's maximum similarity is 1.000, typically a single common user). Fixes: require a minimum number of common users (support threshold), shrink similarity by n/(n+λ), or cap/sample the data. Any one of these is acceptable.

**Q5.6** — It collapses all events of the same item inside one session into the single strongest weight, so refreshing a page ten times counts once. With `.sum()` an item clicked, carted and ordered in one session would count 1+3+5 = 9 instead of 5, and repeat-clicking loops or bots would inflate pair weights, distorting the intended strength scale.

**Q5.7** — Cosine 0.632, Jaccard 0.333. The dot product is 2·1 + 0·1 + 1·0 = 2; the norms are √5 = 2.236 and √2 = 1.414, so cosine = 2/3.162 = 0.632. The user sets are X = {u1, u3} and Y = {u1, u2}: intersection 1, union 3, so Jaccard = 1/3 = 0.333.

**Q5.8** — 0.55. Score(C) = Σ recency(seed) · w(seed→C) = 0.5·0.3 + 1.0·0.4 = 0.15 + 0.40 = 0.55. This is the sum in `covisit` (`cand_recent_score * w`, grouped and summed per user and candidate). Two seeds pointing to the same candidate reinforce each other.

**Q5.9** — B. item_knn has coverage@20 0.8399 (highest), and ndcg@20 0.0925, below user_knn (0.1662) and covisit (0.1197). user_knn does not have the highest coverage (A false); covisit's recall@5 (0.0795) is below user_knn's (0.1230) (C false); pop_7d's ndcg (0.0475) is below item_knn's (D false).

**Q5.10** — In a scratch run, ndcg@20 went 0.1662 (k=50), 0.1702 (k=100), 0.1706 (k=200), 0.1700 (k=400) and recall@20 0.2273, 0.2419, 0.2409, 0.2405: gains stop after about k=100 and the curve is flat or slightly down. Beyond a point extra neighbours are only weakly similar users whose votes add noise rather than signal, and the similarity weights make them contribute little anyway.

## Chapter 6 — Matrix factorisation

**Q6.1** — B. A latent factor is one learned dimension of the user and item vectors. Nobody labels it in advance; the optimiser discovers it because giving every user and item a value on that dimension helps reconstruct the interactions. It is not a raw matrix column, a hyperparameter or a seller-entered attribute.

**Q6.2** — B. $p_{ui}$ is the binary target "did this user show interest" ($1[R_{ui}>0]$), and $c_{ui}=1+\alpha R_{ui}$ weights how much the squared error on that cell matters. Unobserved cells keep $c=1$, so they are weak zeros, while orders and carts are strong positives.

**Q6.3** — C. $Y^\top Y$ does not depend on the user, so it is computed once per sweep. The correction $Y^\top(C^u-I)Y$ has non-zero rows only for the $n_u$ items the user touched, so the per-user cost scales with their history, not the catalogue. The loss is still not jointly convex, and the trick says nothing about regularisation or explicit ratings.

**Q6.4** — Empty cells mostly mean "not seen", not "disliked". Fitting them as true zeros would claim that almost all of the catalogue is hated by every user and would push the model to predict near zero everywhere. Implicit ALS keeps them as zeros but gives them only the floor confidence $c=1$, while observed cells get $c=1+\alpha R$, so the loss mainly cares about explaining the interactions we are sure about.

**Q6.5** — With near-zero vectors every gap $x_u\cdot(y_i-y_j)$ is about 0. Then $\sigma(0)=0.5$ and the loss $-\ln 0.5=\ln 2\approx 0.693$ for every triple. Seeing this value at epoch 1 is a sanity check that the model starts as an uninformed coin flip and the loss is wired correctly.

**Q6.6** — $x_u = 23/34 \approx 0.676$. Confidences are $c=1+4R=(5,13)$. Numerator: $\sum c\,p\,y = 5\cdot1\cdot2+13\cdot1\cdot1=23$. Denominator: $\sum c\,y^2+\lambda = 5\cdot4+13\cdot1+1=34$. Dividing gives 0.676.

**Q6.7** — $\sigma(-\ln 3)=1/(1+3)=0.25$; loss $=-\ln0.25=\ln4\approx1.386$; $s=\sigma(\ln 3)=0.75$. The gap is negative, so the pair is ordered wrongly (the negative outscores the positive). The multiplier $s=0.75$ is large, so the update is large, which is exactly the behaviour BPR wants for badly ordered pairs.

**Q6.8** — `sig` equals $\sigma(-\text{gap}) = 1-\sigma(\text{gap})$, which is $-\partial\,\text{loss}/\partial\,\text{gap}$. By the chain rule every gradient is `sig` times the derivative of the gap with respect to that parameter: $y_i-y_j$ for the user, $x_u$ for the positive item, and $-x_u$ for the negative item. `gj` is negative because raising the gap requires lowering the negative item's score, so $y_j$ moves away from the user vector. Pairs already ordered correctly have `sig` near 0 and barely change anything.

**Q6.9** — Recall@20 only asks whether the user's future items appear anywhere in the top 20; NDCG also rewards placing them near the top. ALS finds more of the right items, but orders them less sharply than user-kNN. That profile suits a candidate generator (judged by recall) feeding a later ranking stage (judged by NDCG), such as the LightGBM ranker in this repo.

**Q6.10** — Exact numbers depend on your run (the script is seeded and deterministic), so compare against the 0.1581 ndcg@20 for `als_implicit` in `results/ch06_mf.txt`. Expect `alpha=1.0` to change ndcg@20 noticeably, because with a small alpha observed and unobserved cells carry almost the same confidence and the model behaves differently; the direction is not guaranteed, so check it. Expect `factors=8` to drop well below the 32-factor run: 8 dimensions cannot separate the tastes spread over many categories and thousands of items. The final ALS loss is not comparable across `alpha` values because the loss itself changes with the weights. The lesson is that sweeping hyperparameters against the evaluation labels tunes on the test set; use an earlier validation window and report the test week once.

## Chapter 7 — Embeddings and two-tower retrieval

**Q7.1** — B. Dividing cosines by a small $\tau$ stretches the gaps between scores, so the softmax becomes sharper and the loss concentrates on the top of the ranking, punishing hard negatives. A flatter softmax (A) is what a high temperature gives. The temperature does not change the embedding dimension or the batch contents.

**Q7.2** — C. A batch is a random sample of positive pairs, so each item appears in proportion to its interaction count. Popular items therefore show up as negatives in many rows and receive more downward push, although serving scores all items uniformly. L2 normalisation fixes norms to 1, so it is not the cause.

**Q7.3** — A. Item vectors depend only on item attributes, so they can be computed once and stored in an index. The score is an inner product, which is what ANN structures are built for. A tower that took the user's history as an item input (D) would break this.

**Q7.4** — It subtracts $\log q_j$, the log of the probability that item $j$ is drawn into the batch, from the logit of column $j$. Popular items (large $q_j$) get a small boost and rare ones a large boost, which cancels the extra push-down popular items get from being over-sampled as negatives. The correction is only used in training; at serving the plain dot product is used.

**Q7.5** — It measures how many of the exact brute-force top-20 items the approximate search still returns. An ANN index only needs to approximate exact search, so fidelity to brute force isolates the index's error from the model's quality. NDCG would mix both and could not tell you which one lost accuracy.

**Q7.6** — It replaces occurrences of the target item in the user's own history with the padding index, so the user tower cannot see the answer. If removed, the model could copy the target from the history, so training loss would fall faster and look better, but evaluation on the held-out week, where no such shortcut exists, would be worse. This is a small instance of target leakage.

**Q7.7** — About 43,159 dot products, or about 0.43% of brute force. Centroid scoring costs 4,096; each cluster has $10{,}000{,}000 / 4{,}096 \approx 2{,}441$ items, so 16 probes scan $16 \times 2{,}441 = 39{,}062.5$. The total is $4{,}096 + 39{,}062.5 = 43{,}158.5$, and $43{,}158.5 / 10{,}000{,}000 \approx 0.43\%$, roughly 230 times less work.

**Q7.8** — By $\ln(0.04 / 0.0004) = \ln 100 \approx 4.605$ logit units. The correction adds $-\ln q$: $-\ln 0.04 = 3.219$ for the popular item and $-\ln 0.0004 = 7.824$ for the rare one, and the difference is 4.605. In probability terms, the rarer item's exponentiated score is multiplied by 100 relative to the popular one.

**Q7.9** — Expected observations (verify on your run): the spec `two_tower` stays well below `two_tower_logq` in ndcg@20 at every temperature, because the popularity bias comes from the sampling scheme, not the temperature. Temperature changes how sharp the loss is, so ndcg and especially coverage@20 will shift between runs; a very low value such as 0.02 tends to be harder to optimise. The point of the exercise is that the logQ gap persists and temperature is a tuning knob, not the cure. Restore results with `git checkout book/results`.

## Chapter 8 — Multi-source candidate generation

**Q8.1** — B. An item missing from the candidate set can never reach the ranker, so the first stage is judged on how many future positives it contains. False positives are cheap because the second stage, with richer features, removes them. The ranker does not need a clean input (D is false), and precision can be computed on implicit feedback (A is false).

**Q8.2** — C, `catpop`. Its unique pair recall is 0.0968, against 0.0524 for `covisit`, 0.0298 for `recent` and 0.0253 for `global_trending`. It also has the highest standalone recall (0.3274), because it covers users through both clicks and searches.

**Q8.3** — B. The positive rate rises from 0.0090 to 0.1817 as more sources agree, so the number of sources and each source's score are informative features. Dropping single-source candidates (A, C) would throw away most of the recall, since 76,612 candidates have a single source. Counting sources cannot replace a ranker (D), which also uses user, item and cross features.

**Q8.4** — C. Only `global_trending`. `recent` is empty without history, `covisit` is seeded from `recent`, so it is empty too, and `catpop` needs category affinity from clicks or searches. The outer join only fills missing score columns; it does not invent candidates (A, D).

**Q8.5** — The union contains 103.94 candidates per user on average, so cutting the ranked list at K=100 drops the candidates at the end of the longer lists, and some of those are positives. The ceiling of 0.4645 is the recall of the whole candidate set with nothing cut. Recall@K can only reach it when K is at least the size of the pool for every user.

**Q8.6** — Hiding is explicit negative feedback, and a learned feature only changes probabilities: the model could still show the hidden item occasionally. A hard filter on the pool guarantees it never reappears, is cheaper (fewer rows to score) and is easy to audit. Users tolerate an irrelevant recommendation but not one they have explicitly rejected.

**Q8.7** — It would be 3 instead of 1. After the swap the three score columns are already filled with 0.0, so `notna().sum(axis=1)` counts all three of them for every row, and the trending term adds nothing because the item is not in that list. The feature would no longer separate single-source from multi-source candidates and would lose the signal that raised the positive rate from 0.0090 to 0.1817.

**Q8.8** — The out-of-stock items in the top 30 would be removed after `nlargest`, leaving fewer than 30 trending items, and the list would shrink whenever a popular item sells out. Filtering first and then taking the top 30 always returns 30 in-stock items (if enough exist).

**Q8.9** — 12.1880 + 35.8321 + 54.9102 + 30.0000 = 132.9303, minus the union size of 103.9366 gives 28.9937, i.e. about 29.0 duplicate (user, item) proposals per user, which is 28.9937 / 132.9303 = 21.8% of the sum. That is the overlap between sources, which is what `cand_n_sources` counts.

**Q8.10** — Expect `catpop` `cands_per_user` to rise (more categories joined, up to 5 × 20 items per user instead of 3 × 20) and its recall to rise slightly, since a user's future positives sometimes lie in the 4th and 5th categories. The union's candidates per user and recall rise a little, with diminishing returns because the extra categories have a low affinity share. The ranker then has to score more rows per request, which increases latency, and most of the added candidates are negatives, so the ranker's job of separating positives from negatives gets harder. The results are overwritten by the run, hence the `git checkout` to restore them.

## Chapter 9 — Feature engineering for ranking

**Q9.1** — B. Point-in-time correctness means every feature for a training example is computed only from information that existed before that example's cutoff time. Normalisation, equal event counts and refresh frequency are unrelated engineering choices. Violating the rule is called leakage and inflates offline metrics.

**Q9.2** — B. Events stamped exactly at the cutoff belong to the label window `[cutoff, cutoff + days)`, which `data.labels` selects with `ts >= cutoff`. The strict `<` in `History.at` therefore puts each event on exactly one side of the boundary, so no event is both a feature input and a label.

**Q9.3** — B. With smoothing, the single impression and click are blended with `s` imaginary impressions at the global rate, so the estimate lands close to `p0`. It moves away only as real impressions accumulate and outweigh the prior. The rate still depends on the item's own clicks; it just trusts them in proportion to the evidence.

**Q9.4** — Zero would mean "touched just now", which is the strongest possible recency signal, so it would confuse "never" with "this very minute". A large sentinel such as 9999 lies far outside the real range, so a tree can separate "never touched" from every real value with a single threshold. (Linear and neural models need an explicit missing flag instead.)

**Q9.5** — An item feature describes the item for every user (e.g. `item_ctr_30d`). A user feature describes the user regardless of the item (e.g. `user_clicks_30d`). A cross feature is an aggregate over a pair of keys (e.g. `ui_clicks`, `uc_click_share_30d`, `us_orders`). Cross features carry most of the personalisation, because only they differ between two users for the same item. Item features mostly carry quality signal.

**Q9.6** — Yes, the test would fail. The test rewrites every post-cutoff event into an order of item 0 in the tampered tables. A feature reading the full unfiltered table would therefore change its value (item 0 would gain many orders, and other items would lose theirs), so the rebuilt frame would differ from the original. `pd.testing.assert_frame_equal(before, _build(tampered, cutoff))` compares the two frames cell by cell and raises as soon as one differs.

**Q9.7** — Smoothed CTR = (3 + 20·0.10) / (10 + 20) = 5/30 ≈ 0.167. The raw CTR is 3/10 = 0.30. Smoothing pulls the estimate more than half-way back to the global 0.10 because the item has only 10 impressions against a prior strength of 20.

**Q9.8** — item_trend = (6/3 + 0.1) / (14/14 + 0.1) = 2.1 / 1.1 ≈ 1.91. The value is above 1, so the item is heating up: the last three days average 2 clicks a day against 1 click a day over the last two weeks.

**Q9.9** — Expected: with a 1-day leak window the leaked feature covers only a seventh of the label window, so most positives get a zero value and the leak weakens; NDCG@20 of the leaky ranker should fall below the 0.5035 of the 7-day leak, and the gain share of `leaky_future_clicks` should fall too. It will most likely still stay clearly above the honest 0.2116, and the feature will still rank near the top. Even a small slice of the future is enough to make the offline metric untrustworthy, which is why the only safe rule is a hard cutoff with no exceptions.

## Chapter 10 — Learning to rank

**Q10.1** — B. Pointwise losses score each row independently, pairwise losses compare two items from the same user, and listwise losses look at the whole ranked list. Model family (trees, networks) is orthogonal to this distinction, and all three can use graded or binary labels.

**Q10.2** — C. Each new tree is fit to the residuals, which for squared error equal the negative gradient of the loss at the current predictions. The prediction is then updated by $\eta$ times the tree's leaf values.

**Q10.3** — D. The NDCG change from swapping a pair is large near the top and for big gain gaps, so those pairs get large lambdas. The multiplication does not make NDCG differentiable; it replaces the gradient by a hand-defined one.

**Q10.4** — LightGBM's `group=` argument is a list of consecutive group sizes, not user ids. If a user's rows are not adjacent, the boundaries cut across users, so "queries" mix different users' rows and the pairs and NDCG are computed over meaningless lists. There is usually no error, only a worse model, which makes the bug hard to notice.

**Q10.5** — Few positives per list (about a hundred candidates, few actions) give list-aware losses little extra signal over a simple binary label; graded labels (orders) are rare and noisy; and the candidate recall ceiling squeezes differences between good rankers. The gap is about one seed standard deviation (0.2147 vs 0.2125 mean ndcg@20, std 0.0014 and 0.0021). I would switch only if a larger-data offline test with many seeds, or an online A/B test, showed a consistent advantage.

**Q10.6** — `pred_contrib=True` returns one column per feature plus a final bias column (the baseline score). Slicing `[:, :-1]` removes the bias so only features remain. Without it the bias column, a large constant like −2.9, would often be the largest absolute value and be reported as a "feature" at an index beyond the feature names, causing an index error or a bogus top explanation.

**Q10.7** — 0.393. Gains are 1, 0, 7; IDCG = 7·1 + 1·0.6309 + 0 = 7.631. Swapping positions 1 and 3: gain gap 7 − 1 = 6, discount gap 1 − 0.5 = 0.5, so the DCG change is 3.0 and 3.0/7.631 = 0.393.

**Q10.8** — About +1.47. Contributions plus bias equal the score, so the total contribution is 1.277 − (−2.917) = 4.194. Subtracting the eight shown (2.726) leaves 1.468, which rounds to +1.47 (±0.003 from rounding).

**Q10.9** — C. The drop is about 1.6 noise standard deviations, which is suggestive, not definitive. A fixed "exact effect" (A) ignores seed noise, the 0.01 threshold (B) is arbitrary, and the negative sign (D) is exactly what a useful group would produce when removed.

**Q10.10** — Expected (hypothesis, to be checked by running): making orders worth the same as clicks removes the extra reward for the strongest actions, so the model may rank carted and ordered items less sharply and `order_recall@20` could drop. Whether ndcg@20 changes by more than the seed standard deviation of 0.0021 is what you must judge from your run; a smaller change is noise. The exercise teaches that `label_gain` encodes business value and that a change in it must be judged against seed noise.

## Chapter 11 — Re-ranking, diversity and business rules

**Q11.1** — B. With $\lambda = 1$ the term $(1-\lambda)\max\text{sim}$ is zero, so every step picks the highest remaining relevance. The result is the original ordering; this is why the `MMR lambda=1.0` row in the results equals the unmodified LambdaMART baseline.

**Q11.2** — C. Re-buying coffee or diapers is the normal behaviour that the recommender should encourage, while a user who already owns a refrigerator does not need that exact one again. The rule therefore drops an item only when `ui_orders > 0` and `item_is_consumable == 0`.

**Q11.3** — C. Thompson sampling draws from each item's posterior, so items with little data have wide distributions and sometimes win a slot, whereas well-measured poor items almost never do. Exploration therefore shrinks automatically as evidence accumulates, with no fixed $\varepsilon$.

**Q11.4** — Rules should be explicit, time-boxed and auditable, which you cannot guarantee inside a trained model. If boosted or sponsored impressions enter the training data as ordinary evidence, the next retraining bakes the campaign into the model permanently and distorts relevance estimates. In the re-ranker a boost is a visible term with a start and end date that can be switched off and tested without retraining.

**Q11.5** — The caps remove items and do not backfill, so for users whose candidate pool is concentrated in a few categories or shops the surviving list can be shorter than 50, and deep recall falls. `recall@5` is nearly unchanged because the top few items rarely hit a cap. (This is the most likely explanation; the script does not log list lengths.)

**Q11.6** — 8 items, and the per-category cap is responsible. All 30 candidates share one category, so only those whose position within the category is below `MAX_PER_CATEGORY = 8` survive; the shop cap (at most 3 items per shop here, under 4) removes nothing, and `head(20)` has fewer than 20 rows to keep.

**Q11.7** — Order: item 1, item 3, item 2. Step 1 scores are $0.5 \times$ rel $= 0.45, 0.40, 0.25$, so item 1 (0.9) is chosen. Step 2: item 2 scores $0.5 \cdot 0.8 - 0.5 \cdot 1 = -0.10$ (same category as item 1), item 3 scores $0.5 \cdot 0.5 - 0 = 0.25$, so item 3 is chosen and item 2 comes last.

**Q11.8** — Beta(6, 16), mean $6/22 \approx 0.273$, standard deviation $\approx 0.093$. The variance is $ab/((a+b)^2(a+b+1)) = 96/(484 \cdot 23) \approx 0.00862$. A new item has Beta(1, 1) with standard deviation $\approx 0.289$, about three times larger, so its samples vary far more and it will often draw high enough to win a slot until data narrows it.

**Q11.9** — A. The results show $-0.0035$ ndcg and $+1.45$ categories at $\lambda = 0.9$, and $-0.0715$ ndcg at $\lambda = 0.3$. B is false (coverage@20 fell from 0.4245 to 0.3129), C is false (the filter alone cost 0.0007), and D is false (the caps raised categories from 5.33 to 6.58).

**Q11.10** — Expected (not run here): with sibling categories no longer penalised, items from the same parent group stop being pushed down, so ndcg@20 at $\lambda = 0.5$ should be higher than the 0.1603 in the table, while the mean distinct categories can stay high, since distinct categories are still counted per category. The page may however concentrate in fewer parent groups (for example only Pets). Compare your measured numbers with this prediction and with the baseline row.

## Chapter 12 — Cold start, context and bias

**Q12.1** — B. `global_trending` takes no user argument and `generate` cross-joins its result onto every user id, so even a user with no history, or the anonymous `user_id = 0`, receives the globally trending items. It is not session-based, it runs on every request rather than only on failure, and it ranks by weighted recent interactions, not discounts.

**Q12.2** — C. With $x = 0$ and $n = 0$ the formula reduces to $p_0 s / s = p_0$, the catalogue-average CTR. This treats a new item as average until evidence says otherwise, instead of as zero or undefined. Smoothing therefore removes the penalty for having no data.

**Q12.3** — C. The ranker predicts what a user will do in the next 7 days, with labels aggregated over a week. The request hour is identical for every candidate in a user's group and has no per-request label to learn from. Context belongs in a second, real-time model trained on the impression log.

**Q12.4** — The ranker is trained on clicks, so an item with a few more clicks gets a higher score and is shown more. More exposure produces more clicks, which widens the gap, while items never shown never get a chance to prove themselves. Techniques that reduce the loop include exploration (epsilon-greedy or Thompson sampling, reserving slots for unproven items), diversity re-ranking, and logging impressions with positions so propensity weighting can correct for exposure. Any two of these are acceptable.

**Q12.5** — A slow recommender harms users just as a dead one does, so a timeout bounds the worst-case wait to five seconds and then serves something useful. The JSON carries `"fallback": true` and `"model_version": "fallback"`. The frontend reads the flag and shows the "Recommender offline" banner.

**Q12.6** — Q ranks higher. The query orders first by the count of `order` events in the last seven days descending (Q has 1, P has 0) and only uses `rating_count` to break ties. The final `i.id` makes the order deterministic when both earlier keys tie, so repeated requests return the same list.

**Q12.7** — Raw CTR: A is $40/200 = 0.20$ and B is $20/200 = 0.10$. Debiased relevance: A is $0.20/0.8 = 0.25$ and B is $0.10/0.2 = 0.50$ (equivalently, each click on A is weighted 1.25 and each click on B is weighted 5). B is twice as relevant as A even though its raw CTR is half. The raw numbers were misleading because B was shown in a position that few users examine.

**Q12.8** — Hands-on. Expect the new "0 clicks" bucket to contain far fewer users than the other buckets, so its metrics are noisy and the `LambdaMART / pop_7d` ratio there should be trusted less than the ratio for the large bucket. With no history, the ranker can only use trending candidates and item-level features, so look for whether its advantage over `pop_7d` shrinks relative to the 5.22x of the heavy-user bucket. If the bucket is empty, the script's evaluation users all have at least one prior click and the row will not appear. The point is that cold cohorts are small and noisy.

**Q12.9** — B. A click at a position with a tiny propensity gets weight $1/\theta_k$, which can be dozens or hundreds, so one lucky click can dominate training. Clipping bounds the weight, trading a small bias for a large variance reduction. Propensities are probabilities and cannot be negative, weights need not sum to one, and LightGBM accepts large weights.

## Chapter 13 — Building the system

**Q13.1** — B. A read-only default transaction means any `INSERT`, `UPDATE` or `DELETE` from the recommender fails with `ReadOnlySqlTransaction`. This protects the system of record from bugs in feature or model code and keeps the Rust API as the single writer. It does not speed up reads and has nothing to do with gRPC.

**Q13.2** — A. The browser may report only `impression`, `click` and `hide`, which are observations of the shopper's own screen. `order` (and `add_to_cart`) are consequences of authenticated server-side actions, so the API writes them itself; accepting them from the client would let anyone forge strong positive signals for any item.

**Q13.3** — B. The tracker's `pagehide` listener was registered first, so it emptied the queue and sent the beacon before the product page's own listener queued the click. The click was added after the last flush opportunity and was lost. The fix is the `onPageHide` hook registry: hooks run first, then the queue is drained and beaconed.

**Q13.4** — Auto-increment ids are assigned at insert time but rows become visible at commit time. If a transaction holding id 500 commits after one holding id 501, a refresh that already saw 501 would never fetch 500 with `id > last`. Re-reading the last 1000 ids and de-duplicating by `event_id` catches such late commits at the cost of a few redundant rows.

**Q13.5** — Each `Recommend` call copies the references to the current booster and version into local variables at the start, so it keeps a consistent model/version pair for its whole duration. `retrain` builds the new booster separately and then swaps both attributes with plain reference assignments. In-flight requests finish with the old model, new requests use the new one, and no lock is needed on the read path.

**Q13.6** — Two sessions. The 10:00 event creates session A; at 10:20 the latest session was seen 20 minutes ago (inside the 30-minute window), so A is reused and its `last_seen_at` becomes 10:20. At 11:05 the gap since 10:20 is 45 minutes, which exceeds 30, so a new session B is inserted. The first session ends the day with `last_seen_at` = 10:20.

**Q13.7** — About 20.2% of the budget and 3.985 s. The fraction is 1010.6 / 5000 = 0.2021. The worst-case staleness is 2 + 1 + 0.985 = 3.985 seconds (flush delay + refresh throttle + request latency).

**Q13.8** — It returns active, in-stock items ordered by the number of `order` events in the last 7 days, then by rating count. Items are labelled `source: "trending"` with score 0 and no contributions because they come from a non-personalised popularity query and no model scored them; this lets the frontend reuse the same card rendering. The response also carries `fallback: true` and `model_version: "fallback"` so the UI can show the offline banner.

**Q13.9** — B. Protobuf identifies fields on the wire by number, not by name, so two sides with different numbering for the same field disagree about which bytes mean what and data is dropped or misread without a compile error. That is why a field number must never change or be reused once the message is in use; delete fields by reserving their numbers instead.

**Q13.10** — Expect `p50` latency to change very little (all runs stay in roughly the same band as the k=20 run), and the candidates-per-request line to be identical in all three runs, because `candidates.generate` and `features.build` do not depend on `k`; only the final re-rank cut and the number of SHAP rows change. This indicates that the time is dominated by fixed per-request work (history and feature building), not by scoring or by response size.

## Chapter 14 — From prototype to Shopee scale

**Q14.1** — B. Skew means the feature is legitimate in both places but computed differently (different code, different filters such as bot traffic), so the model learns on one distribution and is queried with another. Dataset size, retrain frequency and ANN approximation are separate issues. A feature store with one definition and logging the served features are the standard defences.

**Q14.2** — B. DIN's local activation unit weights each past behaviour by its relevance to the candidate being scored, so the user's interest vector changes with the candidate. Option A describes the Embedding-and-MLP recipe that DIN was designed to improve on. The other options describe unrelated components.

**Q14.3** — A. Each task has its own softmax gate over a shared pool of experts, so related tasks can pick similar experts and conflicting tasks can specialise. Gates do not change the datasets, equalise the losses or cache anything.

**Q14.4** — Users experience the slowest part of a request, and a page that fans out to many services is as slow as the slowest one, so tail latency (p99) decides the experience; the median hides the slow requests (here 985.0 ms at p99 versus 839.9 ms at p50). The Rust API gives the gRPC call a 5 second timeout; on error or timeout it returns a popularity-based fallback list with `fallback: true` instead of failing the page.

**Q14.5** — Offline numbers come from logs of a previous system, so they can be inflated by leakage (the leaky ranker scored 0.5035 versus 0.2116 for the honest one, and would be useless live). They also inherit position bias and ignore revenue, diversity and seller effects. Another valid reason: selection bias and feedback loops (the model is only scored on items the old system showed). Only a randomised A/B test measures the real effect.

**Q14.6** — Pooled proportion $\hat p = (5000+5150)/100000 = 0.1015$. SE $=\sqrt{0.1015 \times 0.8985 \times 2/50000} \approx 0.001910$. The difference is $0.1030 - 0.1000 = 0.0030$, so $z = 0.0030/0.001910 \approx 1.57$. The two-sided p-value is about 0.116, which is above 0.05, so the lift is not statistically significant: the data cannot rule out zero effect, and the test was probably underpowered.

**Q14.7** — Raw: $2\times10^9 \times 128 \times 2 = 5.12\times10^{11}$ bytes $= 512$ GB. Compressed: $2\times10^9 \times 32 = 6.4\times10^{10}$ bytes $= 64$ GB. Product quantisation therefore gives an 8x reduction here, at the price of some recall, which is tuned with the number of probes and re-scoring.

**Q14.8** — Base $= 1.0\times0.2 + 2.0\times0.05 + 3.0\times0.01 = 0.2 + 0.10 + 0.03 = 0.33$. With the boost, $0.33 \times (1 + 0.1) = 0.363$ (no ads term). With boost 0 the score is 0.33. The boost multiplies the whole blend, so a campaign lifts items proportionally to how good the model already thinks they are.

**Q14.9** — B. With millions of users, a 0.6-point deviation from 50/50 is far outside chance and indicates a sample ratio mismatch: a bug in randomisation, logging, bot filtering or a redirect that loses one arm. Results from such a test are not trustworthy until the cause is found.

**Q14.10** — Expect the $\lambda=0.8$ row to lie between the 0.9 and 0.7 rows on both measures: ndcg@20 between about 0.2081 and 0.1884, and more distinct categories than at 0.9 but fewer than at 0.7 (between about 6.78 and 11.03). The monotone trade-off supports the claim: every step towards more diversity costs relevance, so the diversity weight is a product decision to be tested online.
