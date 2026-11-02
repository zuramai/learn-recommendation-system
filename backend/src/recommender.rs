use std::{collections::HashMap, time::Duration};

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sqlx::{PgPool, QueryBuilder};

use crate::{
    AppState,
    auth::{AdminUser, MaybeUser},
    catalog::{CARD_COLS, CARD_FROM, ProductCard, fetch_cards},
    error::{AppError, AppResult},
};

pub mod pb {
    tonic::include_proto!("recommender.v1");
}

pub type RecClient = pb::recommender_client::RecommenderClient<tonic::transport::Channel>;

#[derive(Serialize)]
struct ContributionJson {
    feature: String,
    value: f64,
    contribution: f64,
}

#[derive(Serialize)]
struct RecCard {
    #[serde(flatten)]
    card: ProductCard,
    score: f64,
    source: String,
    rank: i32,
    contributions: Vec<ContributionJson>,
}

// book:start hydrate
/// Turn gRPC items (ids only) into full cards, in the recommender's order.
/// Ids that no longer exist or are inactive are skipped.
async fn hydrate(pool: &PgPool, items: Vec<pb::RecommendedItem>) -> Result<Vec<RecCard>, sqlx::Error> {
    let ids: Vec<i64> = items.iter().map(|i| i.item_id).collect();
    let mut cards: HashMap<i64, ProductCard> = fetch_cards(pool, &ids).await?.into_iter().map(|c| (c.id, c)).collect();
    Ok(items
        .into_iter()
        .filter_map(|it| {
            cards.remove(&it.item_id).map(|card| RecCard {
                card,
                score: it.score,
                source: it.source,
                rank: it.rank,
                contributions: it
                    .contributions
                    .into_iter()
                    .map(|c| ContributionJson { feature: c.feature, value: c.value, contribution: c.contribution })
                    .collect(),
            })
        })
        .collect())
}
// book:end hydrate

// book:start fallback
/// Used when the recommender is down: recent best-sellers, padded by rating count.
async fn fallback(
    pool: &PgPool,
    k: i64,
    category_id: Option<i64>,
    exclude_id: Option<i64>,
) -> Result<Vec<RecCard>, sqlx::Error> {
    let mut qb = QueryBuilder::new(format!("SELECT {CARD_COLS} {CARD_FROM} WHERE i.is_active AND i.stock > 0"));
    if let Some(c) = category_id {
        qb.push(" AND i.category_id = ").push_bind(c);
    }
    if let Some(x) = exclude_id {
        qb.push(" AND i.id <> ").push_bind(x);
    }
    qb.push(
        " ORDER BY (SELECT count(*) FROM events e WHERE e.item_id = i.id AND e.event_type = 'order'
                     AND e.ts > now() - interval '7 days') DESC, i.rating_count DESC, i.id LIMIT ",
    )
    .push_bind(k);
    let cards = qb.build_query_as::<ProductCard>().fetch_all(pool).await?;
    Ok(cards
        .into_iter()
        .enumerate()
        .map(|(i, card)| RecCard {
            card,
            score: 0.0,
            source: "trending".into(),
            rank: i as i32 + 1,
            contributions: vec![],
        })
        .collect())
}
// book:end fallback

#[derive(Deserialize)]
struct RecsQuery {
    k: Option<i64>,
    surface: Option<String>,
}

// book:start recommend_route
async fn recommendations(
    State(st): State<AppState>,
    MaybeUser(user): MaybeUser,
    Query(q): Query<RecsQuery>,
) -> AppResult<Json<Value>> {
    let k = q.k.unwrap_or(20).clamp(1, 50);
    let req = pb::RecommendRequest {
        user_id: user.unwrap_or(0),
        k: k as i32,
        surface: q.surface.unwrap_or_else(|| "home_feed".into()),
    };
    let mut rec = st.rec.clone();
    let res = tokio::time::timeout(Duration::from_secs(5), rec.recommend(req)).await;
    match res {
        Ok(Ok(resp)) => {
            let r = resp.into_inner();
            let items = hydrate(&st.pool, r.items).await?;
            Ok(Json(json!({
                "items": items,
                "model_version": r.model_version,
                "latency_ms": r.latency_ms,
                "candidates_considered": r.candidates_considered,
                "fallback": false,
            })))
        }
        other => {
            match other {
                Ok(Err(e)) => tracing::warn!("recommender error, using fallback: {e}"),
                _ => tracing::warn!("recommender timed out, using fallback"),
            }
            let items = fallback(&st.pool, k, None, None).await?;
            Ok(Json(json!({
                "items": items,
                "model_version": "fallback",
                "latency_ms": 0,
                "candidates_considered": 0,
                "fallback": true,
            })))
        }
    }
}
// book:end recommend_route

#[derive(Deserialize)]
struct SimilarQuery {
    k: Option<i64>,
}

async fn similar(
    State(st): State<AppState>,
    MaybeUser(user): MaybeUser,
    Path(id): Path<i64>,
    Query(q): Query<SimilarQuery>,
) -> AppResult<Json<Value>> {
    let (category_id,): (i64,) = sqlx::query_as("SELECT category_id FROM items WHERE id = $1 AND is_active")
        .bind(id)
        .fetch_one(&st.pool)
        .await?;
    let k = q.k.unwrap_or(12).clamp(1, 30);
    let req = pb::SimilarItemsRequest { item_id: id, k: k as i32, user_id: user.unwrap_or(0) };
    let mut rec = st.rec.clone();
    match tokio::time::timeout(Duration::from_secs(5), rec.similar_items(req)).await {
        Ok(Ok(resp)) => {
            let items = hydrate(&st.pool, resp.into_inner().items).await?;
            Ok(Json(json!({ "items": items, "fallback": false })))
        }
        other => {
            if let Ok(Err(e)) = other {
                tracing::warn!("similar_items error, using fallback: {e}");
            } else {
                tracing::warn!("similar_items timed out, using fallback");
            }
            let items = fallback(&st.pool, k, Some(category_id), Some(id)).await?;
            Ok(Json(json!({ "items": items, "fallback": true })))
        }
    }
}

async fn info(State(st): State<AppState>) -> AppResult<Json<Value>> {
    let mut rec = st.rec.clone();
    let r = tokio::time::timeout(Duration::from_secs(10), rec.model_info(pb::ModelInfoRequest {}))
        .await
        .ok()
        .and_then(|r| r.ok())
        .ok_or_else(|| AppError::Upstream("recommender unavailable".into()))?
        .into_inner();
    let features: Vec<Value> = r
        .features
        .iter()
        .map(|f| json!({ "name": f.name, "group": f.group, "description": f.description, "gain_share": f.gain_share }))
        .collect();
    Ok(Json(json!({
        "model_version": r.model_version,
        "trained_at": r.trained_at,
        "metrics": r.metrics,
        "features": features,
        "events_loaded": r.events_loaded,
        "train_rows": r.train_rows,
    })))
}

async fn retrain(State(st): State<AppState>, _admin: AdminUser) -> AppResult<Json<Value>> {
    let mut rec = st.rec.clone();
    let res = tokio::time::timeout(Duration::from_secs(180), rec.retrain(pb::RetrainRequest {}))
        .await
        .map_err(|_| AppError::Upstream("retrain timed out".into()))?;
    match res {
        Ok(r) => {
            let r = r.into_inner();
            Ok(Json(json!({ "model_version": r.model_version, "metrics": r.metrics, "train_rows": r.train_rows })))
        }
        Err(s) if s.code() == tonic::Code::Aborted => Err(AppError::Conflict("retrain already running".into())),
        Err(s) => Err(AppError::Upstream(format!("recommender error: {}", s.message()))),
    }
}

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/recommendations", get(recommendations))
        .route("/products/{id}/similar", get(similar))
        .route("/recsys/info", get(info))
        .route("/admin/retrain", post(retrain))
}
