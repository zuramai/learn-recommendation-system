use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    routing::{get, patch},
};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;

use crate::{
    AppState,
    auth::{AuthUser, CITIES},
    cart::{OrderJson, OrderRow, attach_lines},
    catalog::{CARD_COLS, CARD_FROM, ProductCard},
    error::{AppError, AppResult},
};

#[derive(Serialize, sqlx::FromRow)]
struct ShopFull {
    id: i64,
    name: String,
    city: String,
    is_mall: bool,
    shop_rating: f64,
    response_rate: f64,
    cancel_rate: f64,
}

const SHOP_COLS: &str = "id, name, city, is_mall, shop_rating, response_rate, cancel_rate";

#[derive(Deserialize)]
struct NewShop {
    name: String,
    city: String,
}

async fn create_shop(
    State(st): State<AppState>,
    u: AuthUser,
    Json(b): Json<NewShop>,
) -> AppResult<(StatusCode, Json<ShopFull>)> {
    if b.name.trim().is_empty() {
        return Err(AppError::BadRequest("name is required".into()));
    }
    if !CITIES.contains(&b.city.as_str()) {
        return Err(AppError::BadRequest(format!("city must be one of {}", CITIES.join(", "))));
    }
    let res = sqlx::query_as::<_, ShopFull>(&format!(
        "INSERT INTO shops (owner_user_id, name, city) VALUES ($1, $2, $3) RETURNING {SHOP_COLS}"
    ))
    .bind(u.id)
    .bind(b.name.trim())
    .bind(&b.city)
    .fetch_one(&st.pool)
    .await;
    match res {
        Ok(s) => Ok((StatusCode::CREATED, Json(s))),
        Err(sqlx::Error::Database(e)) if e.is_unique_violation() => {
            Err(AppError::Conflict("you already have a shop".into()))
        }
        Err(e) => Err(e.into()),
    }
}

async fn get_shop(State(st): State<AppState>, u: AuthUser) -> AppResult<Json<ShopFull>> {
    Ok(Json(
        sqlx::query_as::<_, ShopFull>(&format!("SELECT {SHOP_COLS} FROM shops WHERE owner_user_id = $1"))
            .bind(u.id)
            .fetch_one(&st.pool)
            .await?,
    ))
}

async fn own_shop(pool: &PgPool, user_id: i64) -> AppResult<i64> {
    let r: Option<(i64,)> = sqlx::query_as("SELECT id FROM shops WHERE owner_user_id = $1")
        .bind(user_id)
        .fetch_optional(pool)
        .await?;
    r.map(|(i,)| i).ok_or_else(|| AppError::NotFoundMsg("create a shop first".into()))
}

#[derive(sqlx::FromRow)]
struct SellerProductRow {
    #[sqlx(flatten)]
    card: ProductCard,
    stock: i32,
    is_active: bool,
    description: String,
}

#[derive(Serialize)]
struct SellerProduct {
    #[serde(flatten)]
    card: ProductCard,
    stock: i32,
    is_active: bool,
    description: String,
}

const SELLER_EXTRA: &str = "i.stock, i.is_active, i.description";

async fn seller_product(pool: &PgPool, id: i64) -> AppResult<SellerProduct> {
    let r = sqlx::query_as::<_, SellerProductRow>(&format!(
        "SELECT {CARD_COLS}, {SELLER_EXTRA} {CARD_FROM} WHERE i.id = $1"
    ))
    .bind(id)
    .fetch_one(pool)
    .await?;
    Ok(SellerProduct { card: r.card, stock: r.stock, is_active: r.is_active, description: r.description })
}

async fn list_products(State(st): State<AppState>, u: AuthUser) -> AppResult<Json<Vec<SellerProduct>>> {
    let shop = own_shop(&st.pool, u.id).await?;
    let rows = sqlx::query_as::<_, SellerProductRow>(&format!(
        "SELECT {CARD_COLS}, {SELLER_EXTRA} {CARD_FROM} WHERE i.shop_id = $1 ORDER BY i.id DESC"
    ))
    .bind(shop)
    .fetch_all(&st.pool)
    .await?;
    Ok(Json(
        rows.into_iter()
            .map(|r| SellerProduct { card: r.card, stock: r.stock, is_active: r.is_active, description: r.description })
            .collect(),
    ))
}

#[derive(Deserialize)]
struct NewProduct {
    title: String,
    #[serde(default)]
    description: String,
    brand: String,
    category_id: i64,
    price: i64,
    #[serde(default)]
    discount_pct: f64,
    #[serde(default)]
    free_shipping: bool,
    #[serde(default)]
    stock: i32,
}

#[derive(sqlx::FromRow)]
struct Fields {
    title: String,
    description: String,
    brand: String,
    category_id: i64,
    price: i64,
    discount_pct: f64,
    free_shipping: bool,
    stock: i32,
    is_active: bool,
    shop_id: i64,
}

async fn validate(pool: &PgPool, f: &Fields) -> AppResult<()> {
    if f.title.trim().is_empty() {
        return Err(AppError::BadRequest("title is required".into()));
    }
    if f.brand.trim().is_empty() {
        return Err(AppError::BadRequest("brand is required".into()));
    }
    if f.price <= 0 {
        return Err(AppError::BadRequest("price must be positive".into()));
    }
    if !(0.0..=0.5).contains(&f.discount_pct) {
        return Err(AppError::BadRequest("discount_pct must be between 0 and 0.5".into()));
    }
    if f.stock < 0 {
        return Err(AppError::BadRequest("stock must not be negative".into()));
    }
    let ok: Option<(i64,)> = sqlx::query_as("SELECT id FROM categories WHERE id = $1")
        .bind(f.category_id)
        .fetch_optional(pool)
        .await?;
    if ok.is_none() {
        return Err(AppError::BadRequest("unknown category".into()));
    }
    Ok(())
}

async fn create_product(
    State(st): State<AppState>,
    u: AuthUser,
    Json(b): Json<NewProduct>,
) -> AppResult<(StatusCode, Json<SellerProduct>)> {
    let shop = own_shop(&st.pool, u.id).await?;
    let f = Fields {
        title: b.title.trim().into(),
        description: b.description,
        brand: b.brand.trim().into(),
        category_id: b.category_id,
        price: b.price,
        discount_pct: b.discount_pct,
        free_shipping: b.free_shipping,
        stock: b.stock,
        is_active: true,
        shop_id: shop,
    };
    validate(&st.pool, &f).await?;
    let (id,): (i64,) = sqlx::query_as(
        "INSERT INTO items (shop_id, category_id, brand, title, description, price, discount_pct,
                            free_shipping, stock, out_of_stock_at)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, CASE WHEN $9 = 0 THEN now() END) RETURNING id",
    )
    .bind(shop)
    .bind(f.category_id)
    .bind(&f.brand)
    .bind(&f.title)
    .bind(&f.description)
    .bind(f.price)
    .bind(f.discount_pct)
    .bind(f.free_shipping)
    .bind(f.stock)
    .fetch_one(&st.pool)
    .await?;
    Ok((StatusCode::CREATED, Json(seller_product(&st.pool, id).await?)))
}

#[derive(Deserialize)]
struct PatchProduct {
    title: Option<String>,
    description: Option<String>,
    brand: Option<String>,
    category_id: Option<i64>,
    price: Option<i64>,
    discount_pct: Option<f64>,
    free_shipping: Option<bool>,
    stock: Option<i32>,
    is_active: Option<bool>,
}

async fn load_owned(pool: &PgPool, user_id: i64, item_id: i64) -> AppResult<Fields> {
    let f = sqlx::query_as::<_, Fields>(
        "SELECT title, description, brand, category_id, price, discount_pct, free_shipping, stock, is_active, shop_id
         FROM items WHERE id = $1",
    )
    .bind(item_id)
    .fetch_one(pool)
    .await?;
    let owner: Option<(i64,)> = sqlx::query_as("SELECT id FROM shops WHERE id = $1 AND owner_user_id = $2")
        .bind(f.shop_id)
        .bind(user_id)
        .fetch_optional(pool)
        .await?;
    if owner.is_none() {
        return Err(AppError::Forbidden);
    }
    Ok(f)
}

async fn patch_product(
    State(st): State<AppState>,
    u: AuthUser,
    Path(id): Path<i64>,
    Json(b): Json<PatchProduct>,
) -> AppResult<Json<SellerProduct>> {
    let old = load_owned(&st.pool, u.id, id).await?;
    let old_stock = old.stock;
    let f = Fields {
        title: b.title.map(|s| s.trim().to_owned()).unwrap_or(old.title),
        description: b.description.unwrap_or(old.description),
        brand: b.brand.map(|s| s.trim().to_owned()).unwrap_or(old.brand),
        category_id: b.category_id.unwrap_or(old.category_id),
        price: b.price.unwrap_or(old.price),
        discount_pct: b.discount_pct.unwrap_or(old.discount_pct),
        free_shipping: b.free_shipping.unwrap_or(old.free_shipping),
        stock: b.stock.unwrap_or(old.stock),
        is_active: b.is_active.unwrap_or(old.is_active),
        shop_id: old.shop_id,
    };
    validate(&st.pool, &f).await?;
    // 0 -> >0 clears out_of_stock_at; >0 -> 0 sets it; otherwise unchanged.
    sqlx::query(
        "UPDATE items SET title=$2, description=$3, brand=$4, category_id=$5, price=$6, discount_pct=$7,
                free_shipping=$8, stock=$9, is_active=$10,
                out_of_stock_at = CASE WHEN $11 = 0 AND $9 > 0 THEN NULL
                                       WHEN $11 > 0 AND $9 = 0 THEN now()
                                       ELSE out_of_stock_at END
         WHERE id=$1",
    )
    .bind(id)
    .bind(&f.title)
    .bind(&f.description)
    .bind(&f.brand)
    .bind(f.category_id)
    .bind(f.price)
    .bind(f.discount_pct)
    .bind(f.free_shipping)
    .bind(f.stock)
    .bind(f.is_active)
    .bind(old_stock)
    .execute(&st.pool)
    .await?;
    Ok(Json(seller_product(&st.pool, id).await?))
}

async fn delete_product(State(st): State<AppState>, u: AuthUser, Path(id): Path<i64>) -> AppResult<StatusCode> {
    load_owned(&st.pool, u.id, id).await?;
    sqlx::query("UPDATE items SET is_active = false WHERE id = $1").bind(id).execute(&st.pool).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn seller_orders(State(st): State<AppState>, u: AuthUser) -> AppResult<Json<Vec<OrderJson>>> {
    let shop = own_shop(&st.pool, u.id).await?;
    let rows = sqlx::query_as::<_, OrderRow>(
        "SELECT o.id, o.shop_id, s.name AS shop_name, o.status, o.subtotal, o.shipping_fee, o.total,
                o.created_at, b.name AS buyer_name
         FROM orders o JOIN shops s ON s.id = o.shop_id JOIN users b ON b.id = o.buyer_id
         WHERE o.shop_id = $1 ORDER BY o.created_at DESC, o.id DESC",
    )
    .bind(shop)
    .fetch_all(&st.pool)
    .await?;
    Ok(Json(attach_lines(&st.pool, rows).await?))
}

#[derive(Deserialize)]
struct StatusBody {
    status: String,
}

async fn patch_order(
    State(st): State<AppState>,
    u: AuthUser,
    Path(id): Path<i64>,
    Json(b): Json<StatusBody>,
) -> AppResult<Json<serde_json::Value>> {
    let shop = own_shop(&st.pool, u.id).await?;
    let (order_shop, current): (i64, String) = sqlx::query_as("SELECT shop_id, status FROM orders WHERE id = $1")
        .bind(id)
        .fetch_one(&st.pool)
        .await?;
    if order_shop != shop {
        return Err(AppError::Forbidden);
    }
    match (current.as_str(), b.status.as_str()) {
        ("paid", "shipped") | ("shipped", "completed") => {}
        _ => return Err(AppError::BadRequest("invalid status transition".into())),
    }
    sqlx::query("UPDATE orders SET status = $2 WHERE id = $1")
        .bind(id)
        .bind(&b.status)
        .execute(&st.pool)
        .await?;
    Ok(Json(serde_json::json!({ "id": id, "status": b.status })))
}

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/seller/shop", get(get_shop).post(create_shop))
        .route("/seller/products", get(list_products).post(create_product))
        .route("/seller/products/{id}", patch(patch_product).delete(delete_product))
        .route("/seller/orders", get(seller_orders))
        .route("/seller/orders/{id}", patch(patch_order))
}
