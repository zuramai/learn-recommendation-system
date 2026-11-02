use std::collections::{BTreeMap, HashMap};

use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    routing::{get, patch, post},
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;

use crate::{
    AppState,
    auth::AuthUser,
    catalog::{CARD_COLS, CARD_FROM, ProductCard},
    error::{AppError, AppResult},
    sessions::{insert_event, touch_session},
};

const SHIPPING_FEE: i64 = 10_000;

#[derive(sqlx::FromRow)]
struct CartRow {
    #[sqlx(flatten)]
    card: ProductCard,
    quantity: i32,
}

#[derive(Serialize)]
pub struct CartLine {
    item: ProductCard,
    quantity: i32,
    line_total: i64,
}

#[derive(Serialize)]
pub struct Cart {
    lines: Vec<CartLine>,
    subtotal: i64,
    shipping_fee: i64,
    total: i64,
}

/// 0 if every line from the shop ships free, else the flat fee; summed over shops.
fn shipping_total<'a>(lines: impl Iterator<Item = (i64, bool)> + 'a) -> i64 {
    let mut per_shop: BTreeMap<i64, bool> = BTreeMap::new();
    for (shop, free) in lines {
        let e = per_shop.entry(shop).or_insert(true);
        *e = *e && free;
    }
    per_shop.values().filter(|all_free| !**all_free).count() as i64 * SHIPPING_FEE
}

async fn load_cart(pool: &PgPool, user_id: i64) -> AppResult<Cart> {
    let rows = sqlx::query_as::<_, CartRow>(&format!(
        "SELECT {CARD_COLS}, ci.quantity {CARD_FROM} JOIN cart_items ci ON ci.item_id = i.id
         WHERE ci.user_id = $1 AND i.is_active ORDER BY ci.added_at, i.id"
    ))
    .bind(user_id)
    .fetch_all(pool)
    .await?;
    let subtotal = rows.iter().map(|r| r.card.final_price * r.quantity as i64).sum();
    let shipping_fee = shipping_total(rows.iter().map(|r| (r.card.shop_id, r.card.free_shipping)));
    let lines = rows
        .into_iter()
        .map(|r| CartLine { line_total: r.card.final_price * r.quantity as i64, quantity: r.quantity, item: r.card })
        .collect();
    Ok(Cart { lines, subtotal, shipping_fee, total: subtotal + shipping_fee })
}

async fn get_cart(State(st): State<AppState>, u: AuthUser) -> AppResult<Json<Cart>> {
    Ok(Json(load_cart(&st.pool, u.id).await?))
}

#[derive(Deserialize)]
struct AddBody {
    item_id: i64,
    quantity: i32,
}

async fn add(State(st): State<AppState>, u: AuthUser, Json(b): Json<AddBody>) -> AppResult<Json<Cart>> {
    if b.quantity < 1 {
        return Err(AppError::BadRequest("quantity must be at least 1".into()));
    }
    let stock: Option<(i32,)> = sqlx::query_as("SELECT stock FROM items WHERE id = $1 AND is_active")
        .bind(b.item_id)
        .fetch_optional(&st.pool)
        .await?;
    let (stock,) = stock.ok_or(AppError::NotFound)?;
    let existing: Option<(i32,)> =
        sqlx::query_as("SELECT quantity FROM cart_items WHERE user_id = $1 AND item_id = $2")
            .bind(u.id)
            .bind(b.item_id)
            .fetch_optional(&st.pool)
            .await?;
    let have = existing.map(|(q,)| q).unwrap_or(0);
    if have as i64 + b.quantity as i64 > stock as i64 {
        return Err(AppError::Conflict(format!("only {stock} in stock")));
    }
    let mut conn = st.pool.acquire().await?;
    sqlx::query(
        "INSERT INTO cart_items (user_id, item_id, quantity) VALUES ($1, $2, $3)
         ON CONFLICT (user_id, item_id) DO UPDATE SET quantity = cart_items.quantity + EXCLUDED.quantity",
    )
    .bind(u.id)
    .bind(b.item_id)
    .bind(b.quantity)
    .execute(&mut *conn)
    .await?;
    let sid = touch_session(&mut conn, u.id, "product").await?;
    insert_event(&mut conn, u.id, sid, b.item_id, "add_to_cart", None, None, None, Some("product")).await?;
    drop(conn);
    Ok(Json(load_cart(&st.pool, u.id).await?))
}

#[derive(Deserialize)]
struct PatchBody {
    quantity: i32,
}

async fn update(
    State(st): State<AppState>,
    u: AuthUser,
    Path(item_id): Path<i64>,
    Json(b): Json<PatchBody>,
) -> AppResult<Json<Cart>> {
    if b.quantity < 0 {
        return Err(AppError::BadRequest("quantity must not be negative".into()));
    }
    if b.quantity == 0 {
        sqlx::query("DELETE FROM cart_items WHERE user_id = $1 AND item_id = $2")
            .bind(u.id)
            .bind(item_id)
            .execute(&st.pool)
            .await?;
    } else {
        let (stock,): (i32,) = sqlx::query_as("SELECT stock FROM items WHERE id = $1 AND is_active")
            .bind(item_id)
            .fetch_one(&st.pool)
            .await?;
        if b.quantity > stock {
            return Err(AppError::Conflict(format!("only {stock} in stock")));
        }
        let n = sqlx::query("UPDATE cart_items SET quantity = $3 WHERE user_id = $1 AND item_id = $2")
            .bind(u.id)
            .bind(item_id)
            .bind(b.quantity)
            .execute(&st.pool)
            .await?
            .rows_affected();
        if n == 0 {
            return Err(AppError::NotFound);
        }
    }
    Ok(Json(load_cart(&st.pool, u.id).await?))
}

async fn remove(State(st): State<AppState>, u: AuthUser, Path(item_id): Path<i64>) -> AppResult<Json<Cart>> {
    sqlx::query("DELETE FROM cart_items WHERE user_id = $1 AND item_id = $2")
        .bind(u.id)
        .bind(item_id)
        .execute(&st.pool)
        .await?;
    Ok(Json(load_cart(&st.pool, u.id).await?))
}

#[derive(Deserialize)]
struct CheckoutBody {
    shipping_address: String,
}

#[derive(sqlx::FromRow)]
struct LockedItem {
    id: i64,
    shop_id: i64,
    title: String,
    final_price: i64,
    free_shipping: bool,
    stock: i32,
    is_active: bool,
}

#[derive(Serialize)]
struct PlacedOrder {
    id: i64,
    shop_id: i64,
    total: i64,
}

// book:start checkout
async fn checkout(
    State(st): State<AppState>,
    u: AuthUser,
    Json(b): Json<CheckoutBody>,
) -> AppResult<(StatusCode, Json<serde_json::Value>)> {
    let address = b.shipping_address.trim();
    if address.is_empty() {
        return Err(AppError::BadRequest("shipping address is required".into()));
    }
    let mut tx = st.pool.begin().await?;
    let cart: Vec<(i64, i32)> = sqlx::query_as("SELECT item_id, quantity FROM cart_items WHERE user_id = $1")
        .bind(u.id)
        .fetch_all(&mut *tx)
        .await?;
    if cart.is_empty() {
        return Err(AppError::BadRequest("cart is empty".into()));
    }
    let ids: Vec<i64> = cart.iter().map(|(i, _)| *i).collect();
    // Row locks serialise concurrent checkouts of the same items.
    let locked: HashMap<i64, LockedItem> = sqlx::query_as::<_, LockedItem>(
        "SELECT id, shop_id, title, ROUND(price * (1 - discount_pct))::bigint AS final_price,
                free_shipping, stock, is_active
         FROM items WHERE id = ANY($1) ORDER BY id FOR UPDATE",
    )
    .bind(&ids)
    .fetch_all(&mut *tx)
    .await?
    .into_iter()
    .map(|i| (i.id, i))
    .collect();

    let mut by_shop: BTreeMap<i64, Vec<(&LockedItem, i32)>> = BTreeMap::new();
    for (item_id, qty) in &cart {
        let it = locked.get(item_id).ok_or(AppError::NotFound)?;
        if !it.is_active || *qty > it.stock {
            return Err(AppError::Conflict(format!("insufficient stock for {}", it.title)));
        }
        by_shop.entry(it.shop_id).or_default().push((it, *qty));
    }

    let sid = touch_session(&mut tx, u.id, "checkout").await?;
    let mut placed = Vec::new();
    for (shop_id, lines) in &by_shop {
        let subtotal: i64 = lines.iter().map(|(it, q)| it.final_price * *q as i64).sum();
        let shipping_fee = shipping_total(lines.iter().map(|(it, _)| (it.shop_id, it.free_shipping)));
        let total = subtotal + shipping_fee;
        let (order_id,): (i64,) = sqlx::query_as(
            "INSERT INTO orders (buyer_id, shop_id, status, shipping_address, subtotal, shipping_fee, total)
             VALUES ($1, $2, 'paid', $3, $4, $5, $6) RETURNING id",
        )
        .bind(u.id)
        .bind(shop_id)
        .bind(address)
        .bind(subtotal)
        .bind(shipping_fee)
        .bind(total)
        .fetch_one(&mut *tx)
        .await?;
        for (it, qty) in lines {
            sqlx::query("INSERT INTO order_items (order_id, item_id, quantity, unit_price) VALUES ($1, $2, $3, $4)")
                .bind(order_id)
                .bind(it.id)
                .bind(qty)
                .bind(it.final_price)
                .execute(&mut *tx)
                .await?;
            sqlx::query(
                "UPDATE items SET stock = stock - $2,
                        out_of_stock_at = CASE WHEN stock - $2 = 0 THEN now() ELSE out_of_stock_at END
                 WHERE id = $1",
            )
            .bind(it.id)
            .bind(qty)
            .execute(&mut *tx)
            .await?;
            insert_event(
                &mut tx,
                u.id,
                sid,
                it.id,
                "order",
                None,
                None,
                Some((it.final_price * *qty as i64) as f64),
                Some("checkout"),
            )
            .await?;
        }
        placed.push(PlacedOrder { id: order_id, shop_id: *shop_id, total });
    }
    sqlx::query("DELETE FROM cart_items WHERE user_id = $1").bind(u.id).execute(&mut *tx).await?;
    tx.commit().await?;
    Ok((StatusCode::CREATED, Json(serde_json::json!({ "orders": placed }))))
}
// book:end checkout

#[derive(Serialize, sqlx::FromRow)]
pub struct OrderRow {
    pub id: i64,
    pub shop_id: i64,
    pub shop_name: String,
    pub status: String,
    pub subtotal: i64,
    pub shipping_fee: i64,
    pub total: i64,
    pub created_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub buyer_name: Option<String>,
}

#[derive(Serialize, sqlx::FromRow)]
struct OrderLine {
    #[serde(skip)]
    order_id: i64,
    item_id: i64,
    title: String,
    quantity: i32,
    unit_price: i64,
}

#[derive(Serialize)]
pub struct OrderJson {
    #[serde(flatten)]
    row: OrderRow,
    items: Vec<OrderLine>,
}

pub async fn attach_lines(pool: &PgPool, rows: Vec<OrderRow>) -> AppResult<Vec<OrderJson>> {
    let ids: Vec<i64> = rows.iter().map(|r| r.id).collect();
    let lines = sqlx::query_as::<_, OrderLine>(
        "SELECT oi.order_id, oi.item_id, i.title, oi.quantity, oi.unit_price
         FROM order_items oi JOIN items i ON i.id = oi.item_id
         WHERE oi.order_id = ANY($1) ORDER BY oi.order_id, oi.item_id",
    )
    .bind(&ids)
    .fetch_all(pool)
    .await?;
    let mut map: HashMap<i64, Vec<OrderLine>> = HashMap::new();
    for l in lines {
        map.entry(l.order_id).or_default().push(l);
    }
    Ok(rows
        .into_iter()
        .map(|r| OrderJson { items: map.remove(&r.id).unwrap_or_default(), row: r })
        .collect())
}

async fn orders(State(st): State<AppState>, u: AuthUser) -> AppResult<Json<Vec<OrderJson>>> {
    let rows = sqlx::query_as::<_, OrderRow>(
        "SELECT o.id, o.shop_id, s.name AS shop_name, o.status, o.subtotal, o.shipping_fee, o.total,
                o.created_at, NULL::text AS buyer_name
         FROM orders o JOIN shops s ON s.id = o.shop_id
         WHERE o.buyer_id = $1 ORDER BY o.created_at DESC, o.id DESC",
    )
    .bind(u.id)
    .fetch_all(&st.pool)
    .await?;
    Ok(Json(attach_lines(&st.pool, rows).await?))
}

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/cart", get(get_cart).post(add))
        .route("/cart/{item_id}", patch(update).delete(remove))
        .route("/checkout", post(checkout))
        .route("/orders", get(orders))
}
