# Appendix B — Glossary

Terms are listed alphabetically. The chapter where a term is introduced is in brackets.

| Term | Meaning |
|---|---|
| **A/B test** | An online experiment that randomly splits users between a control and a treatment and compares a metric, with a significance test deciding whether the difference is real. [14] |
| **Ablation** | Retraining a model with one part removed (here: a feature group) to measure how much that part contributes. [10] |
| **Add-to-cart** | An event logged when a user puts an item in the cart; a stronger interest signal than a click. [1] |
| **ALS (Alternating Least Squares)** | A matrix-factorisation algorithm that fixes item vectors and solves for user vectors exactly, then the reverse, repeatedly. [6] |
| **ANN (Approximate Nearest Neighbour)** | Search that finds *almost* the closest vectors far faster than checking every vector. [7] |
| **AP (Average Precision)** | The mean of precision@k taken at every rank k where a relevant item appears. [2] |
| **Bayesian average** | A rating or rate shrunk towards a global prior until enough data accumulates, so 1 click in 1 view is not treated as a 100% rate. [3] |
| **BPR (Bayesian Personalised Ranking)** | A pairwise objective: for each user, score an item they interacted with above one they did not. [6] |
| **Candidate** | An item proposed by a retriever for a user; only candidates reach the ranker. [8] |
| **Candidate recall** | The share of a user's future positives present in the candidate set; the ceiling for any ranker. [8] |
| **Co-visitation** | Counting how often two items are interacted with in the same session; the basis of "viewed together". [5] |
| **Cold start** | A new user or item with little or no history, for which personalised methods have nothing to learn from. [12] |
| **Collaborative filtering** | Recommending from the behaviour of similar users or the co-consumption of items, without looking at item content. [5] |
| **Content-based filtering** | Recommending items whose attributes (text, category, brand) resemble those a user liked. [4] |
| **Cosine similarity** | The cosine of the angle between two vectors: 1 for the same direction, 0 for unrelated. [4] |
| **Coverage** | The share of the catalogue that appears in anyone's top-K; a simple diversity measure. [2] |
| **CTR (Click-through rate)** | Clicks divided by impressions. [1] |
| **Cutoff** | The moment as of which features are computed; only events before it may be used. [2] |
| **DCG / NDCG** | Discounted cumulative gain: relevance summed with a $1/\log_2(\text{rank}+1)$ discount; NDCG divides by the best possible DCG, giving 0–1. [2] |
| **Decay (half-life)** | Down-weighting old events exponentially; with a 7-day half-life an action a week old counts half. [3] |
| **Dwell time** | Seconds a user spends on a product page; longer usually means more interest. [1] |
| **Embedding** | A learned vector representing a user or item so that similar things are close together. [7] |
| **Event** | One logged action of a user on an item: impression, click, add-to-cart, order, hide or return. [1] |
| **Explicit feedback** | Ratings or reviews users give on purpose. [1] |
| **Exploration** | Deliberately showing uncertain items to learn about them, at a small short-term cost (ε-greedy, Thompson sampling). [11] |
| **Fallback** | What the system serves when the recommender is unavailable or the user is unknown; here, recent best sellers. [12] |
| **Feature** | A number describing a (user, item) pair at a moment in time, used as model input. [9] |
| **Feature store** | A system that computes, stores and serves features consistently for training and serving. [14] |
| **Feedback loop** | The model's recommendations shape the data it is later trained on, which can amplify its own biases. [12] |
| **Gradient boosting** | Building a model as a sum of small decision trees, each fitted to the errors of the previous ones. [10] |
| **Hard filter** | A rule that removes candidates outright (out of stock, hidden by the user). [8] |
| **Impression** | An item was shown on screen. [1] |
| **Implicit feedback** | Behaviour that signals interest without being a rating: views, clicks, purchases. [1] |
| **In-batch negatives** | Using the other positives in a training batch as negatives for each example. [7] |
| **IPW (Inverse propensity weighting)** | Re-weighting logged feedback by the inverse probability that an item was seen, to correct exposure and position bias. [12] |
| **Item-to-item** | Collaborative filtering that recommends items similar to those the user interacted with, using item–item similarity. [5] |
| **IVF (Inverted file index)** | An ANN method that clusters vectors and searches only the clusters nearest to the query. [7] |
| **Label** | What actually happened after the cutoff (0 nothing, 1 click, 2 cart, 3 order); the model's training target. [9] |
| **LambdaMART** | A gradient-boosted tree ranker whose gradients are weighted by how much swapping two items would change NDCG. [10] |
| **Latency budget** | The time each stage of a request is allowed to take so the whole response meets its deadline. [14] |
| **Leakage** | Using information from after the cutoff when building features; inflates offline metrics. [9] |
| **Learning to rank** | Training a model to order items for a query or user rather than to predict each item's value in isolation. [10] |
| **logQ correction** | Subtracting the log of an item's sampling probability from its logit when training with in-batch negatives, so popular items are not over-penalised. [7] |
| **Long tail** | The many items that each receive little attention but together make up a large share of the catalogue. [1] |
| **Matrix factorisation** | Approximating the user–item matrix as a product of low-dimensional user and item factor matrices. [6] |
| **MMR (Maximal Marginal Relevance)** | A re-ranking rule that trades relevance against similarity to items already chosen. [11] |
| **Multi-task learning** | One model predicting several outcomes (click, cart, order) with shared layers. [14] |
| **Point-in-time correctness** | Computing every feature only from data available at its cutoff. [9] |
| **Popularity bias** | Recommenders favouring already-popular items, which then get more exposure and become more popular. [3] |
| **Position bias** | Items shown higher get more clicks regardless of relevance. [12] |
| **Precision@K / Recall@K** | Share of the top K that is relevant / share of the relevant items that are in the top K. [2] |
| **Re-ranking** | A final stage applying page-level rules (diversity caps, already-bought filter) to model scores. [11] |
| **Retrieval** | The first stage: cheaply selecting a few hundred candidates out of the whole catalogue. [8] |
| **Session** | A burst of user activity; here a gap of 30 minutes starts a new one. [13] |
| **SHAP** | A method that splits one prediction into per-feature contributions that sum to the model output. [10] |
| **Smoothing** | See *Bayesian average*. [3] |
| **Sparsity** | The share of the user–item matrix that is empty; usually above 99%. [1] |
| **TF-IDF** | Term frequency × inverse document frequency: a word weight that is high for words frequent in one document but rare overall. [4] |
| **Training/serving skew** | Differences between how features are computed for training and at serving time, which silently degrade a model. [14] |
| **Two-tower model** | A retrieval model with separate neural networks for users and items whose output vectors are compared by dot product. [7] |
