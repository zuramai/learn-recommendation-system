use axum::{
    Json, Router,
    extract::{Query, State},
    http::StatusCode,
    routing::{get, post},
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::{
    AppState,
    auth::{AuthUser, MaybeUser},
    error::{AppError, AppResult},
    sessions::{insert_event, touch_session},
};

#[derive(Deserialize)]
struct ClientEvent {
    item_id: i64,
    event_type: String,
    position: Option<i32>,
    surface: Option<String>,
    dwell_sec: Option<f64>,
}

#[derive(Deserialize)]
struct EventBatch {
    events: Vec<ClientEvent>,
}

// book:start post_events
async fn post_events(
    State(st): State<AppState>,
    MaybeUser(user): MaybeUser,
    Json(b): Json<EventBatch>,
) -> AppResult<StatusCode> {
    let Some(uid) = user else { return Ok(StatusCode::NO_CONTENT) };
    if b.events.len() > 100 {
        return Err(AppError::BadRequest("too many events (max 100)".into()));
    }
    if b.events.iter().any(|e| !matches!(e.event_type.as_str(), "impression" | "click" | "hide")) {
        return Err(AppError::BadRequest("event_type not allowed from client".into()));
    }
    let Some(first) = b.events.first() else { return Ok(StatusCode::NO_CONTENT) };
    let mut tx = st.pool.begin().await?;
    let sid = touch_session(&mut tx, uid, first.surface.as_deref().unwrap_or("direct")).await?;
    for e in &b.events {
        let r = insert_event(
            &mut tx,
            uid,
            sid,
            e.item_id,
            &e.event_type,
            e.position,
            e.dwell_sec,
            None,
            e.surface.as_deref(),
        )
        .await;
        match r {
            Ok(()) => {}
            Err(sqlx::Error::Database(d)) if d.is_foreign_key_violation() => {
                return Err(AppError::BadRequest("unknown item".into()));
            }
            Err(e) => return Err(e.into()),
        }
    }
    tx.commit().await?;
    Ok(StatusCode::NO_CONTENT)
}
// book:end post_events

#[derive(Deserialize)]
struct MeQuery {
    limit: Option<i64>,
}

#[derive(Serialize, sqlx::FromRow)]
struct EventRow {
    id: i64,
    ts: DateTime<Utc>,
    event_type: String,
    item_id: i64,
    item_title: String,
    surface: Option<String>,
    position: Option<i32>,
    dwell_sec: Option<f64>,
    price_paid: Option<f64>,
}

#[derive(Serialize, sqlx::FromRow)]
struct SearchRow {
    ts: DateTime<Utc>,
    query: String,
}

async fn my_events(
    State(st): State<AppState>,
    u: AuthUser,
    Query(q): Query<MeQuery>,
) -> AppResult<Json<serde_json::Value>> {
    let limit = q.limit.unwrap_or(30).clamp(1, 100);
    let events = sqlx::query_as::<_, EventRow>(
        "SELECT e.id, e.ts, e.event_type, e.item_id, i.title AS item_title, e.surface, e.position,
                e.dwell_sec, e.price_paid
         FROM events e JOIN items i ON i.id = e.item_id
         WHERE e.user_id = $1 ORDER BY e.ts DESC, e.id DESC LIMIT $2",
    )
    .bind(u.id)
    .bind(limit)
    .fetch_all(&st.pool)
    .await?;
    let searches = sqlx::query_as::<_, SearchRow>(
        "SELECT ts, query FROM searches WHERE user_id = $1 ORDER BY ts DESC, id DESC LIMIT 10",
    )
    .bind(u.id)
    .fetch_all(&st.pool)
    .await?;
    Ok(Json(serde_json::json!({ "events": events, "searches": searches })))
}

pub fn router() -> Router<AppState> {
    Router::new().route("/events", post(post_events)).route("/me/events", get(my_events))
}
