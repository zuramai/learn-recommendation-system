use axum::{
    Json, Router,
    extract::{Path, Query, State},
    routing::get,
};
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, QueryBuilder};

use crate::{
    AppState,
    auth::MaybeUser,
    error::{AppError, AppResult},
    sessions::touch_session,
};

/// Columns of a ProductCard; pair with `CARD_FROM`.
pub const CARD_COLS: &str = "i.id, i.title, i.brand, i.price, i.discount_pct,
    ROUND(i.price * (1 - i.discount_pct))::bigint AS final_price, i.free_shipping,
    i.rating_avg, i.rating_count, i.category_id, c.name AS category_name,
    i.shop_id, s.name AS shop_name, s.city AS shop_city, (i.stock > 0) AS in_stock";
pub const CARD_FROM: &str =
    "FROM items i JOIN categories c ON c.id = i.category_id JOIN shops s ON s.id = i.shop_id";

#[derive(Serialize, sqlx::FromRow, Clone)]
pub struct ProductCard {
    pub id: i64,
    pub title: String,
    pub brand: String,
    pub price: i64,
    pub discount_pct: f64,
    pub final_price: i64,
    pub free_shipping: bool,
    pub rating_avg: f64,
    pub rating_count: i32,
    pub category_id: i64,
    pub category_name: String,
    pub shop_id: i64,
    pub shop_name: String,
    pub shop_city: String,
    pub in_stock: bool,
}

#[derive(Serialize, sqlx::FromRow)]
struct Category {
    id: i64,
    name: String,
    parent_name: String,
}

async fn categories(State(st): State<AppState>) -> AppResult<Json<Vec<Category>>> {
    Ok(Json(
        sqlx::query_as::<_, Category>("SELECT id, name, parent_name FROM categories ORDER BY id")
            .fetch_all(&st.pool)
            .await?,
    ))
}

#[derive(Deserialize)]
struct ProductsQuery {
    q: Option<String>,
    category_id: Option<i64>,
    shop_id: Option<i64>,
    sort: Option<String>,
    page: Option<i64>,
    page_size: Option<i64>,
}

fn push_filters(qb: &mut QueryBuilder<'_, sqlx::Postgres>, q: &ProductsQuery, term: &Option<String>) {
    qb.push(" WHERE i.is_active");
    if let Some(t) = term {
        let pat = format!("%{}%", t.replace('\\', "\\\\").replace('%', "\\%").replace('_', "\\_"));
        qb.push(" AND (i.title ILIKE ").push_bind(pat.clone());
        qb.push(" OR c.name ILIKE ").push_bind(pat).push(")");
    }
    if let Some(c) = q.category_id {
        qb.push(" AND i.category_id = ").push_bind(c);
    }
    if let Some(s) = q.shop_id {
        qb.push(" AND i.shop_id = ").push_bind(s);
    }
}

#[derive(Serialize)]
struct ProductsResponse {
    items: Vec<ProductCard>,
    total: i64,
    page: i64,
    page_size: i64,
}

async fn products(
    State(st): State<AppState>,
    MaybeUser(user): MaybeUser,
    Query(q): Query<ProductsQuery>,
) -> AppResult<Json<ProductsResponse>> {
    let term = q.q.as_deref().map(str::trim).filter(|s| !s.is_empty()).map(str::to_owned);
    let page = q.page.unwrap_or(1).max(1);
    let page_size = q.page_size.unwrap_or(24).clamp(1, 60);
    let order = match q.sort.as_deref().unwrap_or("popular") {
        "popular" | "relevance" => "i.rating_count DESC, i.id",
        "price_asc" => "final_price ASC, i.id",
        "price_desc" => "final_price DESC, i.id",
        "newest" => "i.listed_at DESC, i.id",
        other => return Err(AppError::BadRequest(format!("unknown sort {other}"))),
    };

    let mut qb = QueryBuilder::new(format!("SELECT {CARD_COLS} {CARD_FROM}"));
    push_filters(&mut qb, &q, &term);
    qb.push(format!(" ORDER BY {order} LIMIT ")).push_bind(page_size);
    qb.push(" OFFSET ").push_bind((page - 1) * page_size);
    let items = qb.build_query_as::<ProductCard>().fetch_all(&st.pool).await?;

    let mut cb = QueryBuilder::new(format!("SELECT count(*) {CARD_FROM}"));
    push_filters(&mut cb, &q, &term);
    let (total,): (i64,) = cb.build_query_as().fetch_one(&st.pool).await?;

    if let (Some(uid), Some(t)) = (user, &term) {
        let mut conn = st.pool.acquire().await?;
        let sid = touch_session(&mut conn, uid, "search").await?;
        sqlx::query("INSERT INTO searches (session_id, user_id, query) VALUES ($1, $2, $3)")
            .bind(sid)
            .bind(uid)
            .bind(t)
            .execute(&mut *conn)
            .await?;
    }
    Ok(Json(ProductsResponse { items, total, page, page_size }))
}

#[derive(Serialize, sqlx::FromRow)]
struct ShopBrief {
    id: i64,
    name: String,
    city: String,
    is_mall: bool,
    shop_rating: f64,
}

#[derive(sqlx::FromRow)]
struct DetailRow {
    #[sqlx(flatten)]
    card: ProductCard,
    description: String,
    stock: i32,
}

#[derive(Serialize)]
struct ProductDetail {
    #[serde(flatten)]
    card: ProductCard,
    description: String,
    stock: i32,
    shop: ShopBrief,
}

async fn product(State(st): State<AppState>, Path(id): Path<i64>) -> AppResult<Json<ProductDetail>> {
    let row = sqlx::query_as::<_, DetailRow>(&format!(
        "SELECT {CARD_COLS}, i.description, i.stock {CARD_FROM} WHERE i.id = $1 AND i.is_active"
    ))
    .bind(id)
    .fetch_one(&st.pool)
    .await?;
    let shop = sqlx::query_as::<_, ShopBrief>("SELECT id, name, city, is_mall, shop_rating FROM shops WHERE id = $1")
        .bind(row.card.shop_id)
        .fetch_one(&st.pool)
        .await?;
    Ok(Json(ProductDetail { card: row.card, description: row.description, stock: row.stock, shop }))
}

#[derive(Serialize)]
struct ShopJson {
    id: i64,
    name: String,
    city: String,
    is_mall: bool,
    shop_rating: f64,
    product_count: i64,
}

async fn shop(State(st): State<AppState>, Path(id): Path<i64>) -> AppResult<Json<ShopJson>> {
    let s = sqlx::query_as::<_, ShopBrief>("SELECT id, name, city, is_mall, shop_rating FROM shops WHERE id = $1")
        .bind(id)
        .fetch_one(&st.pool)
        .await?;
    let (n,): (i64,) = sqlx::query_as("SELECT count(*) FROM items WHERE shop_id = $1 AND is_active")
        .bind(id)
        .fetch_one(&st.pool)
        .await?;
    Ok(Json(ShopJson {
        id: s.id,
        name: s.name,
        city: s.city,
        is_mall: s.is_mall,
        shop_rating: s.shop_rating,
        product_count: n,
    }))
}

pub async fn fetch_cards(pool: &PgPool, ids: &[i64]) -> Result<Vec<ProductCard>, sqlx::Error> {
    sqlx::query_as::<_, ProductCard>(&format!("SELECT {CARD_COLS} {CARD_FROM} WHERE i.is_active AND i.id = ANY($1)"))
        .bind(ids)
        .fetch_all(pool)
        .await
}

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/categories", get(categories))
        .route("/products", get(products))
        .route("/products/{id}", get(product))
        .route("/shops/{id}", get(shop))
}
