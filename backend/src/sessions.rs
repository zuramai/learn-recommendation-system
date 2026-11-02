use sqlx::PgConnection;

// book:start touch_session
/// Returns the id of the user's current session, creating one if the latest
/// session has been idle for more than 30 minutes.
pub async fn touch_session(conn: &mut PgConnection, user_id: i64, surface: &str) -> Result<i64, sqlx::Error> {
    let existing: Option<(i64,)> = sqlx::query_as(
        "UPDATE sessions SET last_seen_at = now() WHERE id = (
             SELECT id FROM sessions
             WHERE user_id = $1 AND last_seen_at > now() - interval '30 minutes'
             ORDER BY last_seen_at DESC LIMIT 1)
         RETURNING id",
    )
    .bind(user_id)
    .fetch_optional(&mut *conn)
    .await?;
    if let Some((id,)) = existing {
        return Ok(id);
    }
    let (id,): (i64,) = sqlx::query_as(
        "INSERT INTO sessions (user_id, surface, device_os) VALUES ($1, $2, 'web') RETURNING id",
    )
    .bind(user_id)
    .bind(surface)
    .fetch_one(&mut *conn)
    .await?;
    Ok(id)
}
// book:end touch_session

#[allow(clippy::too_many_arguments)]
pub async fn insert_event(
    conn: &mut PgConnection,
    user_id: i64,
    session_id: i64,
    item_id: i64,
    event_type: &str,
    position: Option<i32>,
    dwell_sec: Option<f64>,
    price_paid: Option<f64>,
    surface: Option<&str>,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        "INSERT INTO events (user_id, session_id, item_id, event_type, position, dwell_sec, price_paid, surface)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
    )
    .bind(user_id)
    .bind(session_id)
    .bind(item_id)
    .bind(event_type)
    .bind(position)
    .bind(dwell_sec)
    .bind(price_paid)
    .bind(surface)
    .execute(conn)
    .await?;
    Ok(())
}
