use argon2::{
    Argon2, PasswordHash, PasswordHasher, PasswordVerifier,
    password_hash::{SaltString, rand_core::OsRng},
};
use axum::{
    Json, Router,
    extract::{FromRequestParts, State},
    http::{HeaderMap, StatusCode, header::SET_COOKIE, request::Parts},
    response::IntoResponse,
    routing::{get, post},
};
use jsonwebtoken::{DecodingKey, EncodingKey, Header, Validation, decode, encode};
use serde::{Deserialize, Serialize};

use crate::{
    AppState,
    error::{AppError, AppResult},
};

pub const CITIES: [&str; 5] = ["Jakarta", "Surabaya", "Bandung", "Medan", "Makassar"];
const WEEK_SECS: usize = 7 * 24 * 3600;

#[derive(Serialize, Deserialize)]
struct Claims {
    sub: i64,
    exp: usize,
}

#[derive(Serialize, sqlx::FromRow)]
pub struct UserJson {
    pub id: i64,
    pub email: String,
    pub name: String,
    pub city: String,
    pub is_admin: bool,
    pub shop_id: Option<i64>,
}

pub fn hash_password(password: &str) -> anyhow::Result<String> {
    let salt = SaltString::generate(&mut OsRng);
    Argon2::default()
        .hash_password(password.as_bytes(), &salt)
        .map(|h| h.to_string())
        .map_err(|e| anyhow::anyhow!("hash failed: {e}"))
}

fn verify_password(password: &str, stored: &str) -> bool {
    PasswordHash::new(stored)
        .map(|h| Argon2::default().verify_password(password.as_bytes(), &h).is_ok())
        .unwrap_or(false)
}

fn make_token(user_id: i64, secret: &str) -> anyhow::Result<String> {
    let exp = chrono::Utc::now().timestamp() as usize + WEEK_SECS;
    Ok(encode(
        &Header::default(),
        &Claims { sub: user_id, exp },
        &EncodingKey::from_secret(secret.as_bytes()),
    )?)
}

fn cookie_header(token: &str, max_age: usize) -> [(axum::http::HeaderName, String); 1] {
    [(SET_COOKIE, format!("auth={token}; HttpOnly; SameSite=Lax; Path=/; Max-Age={max_age}"))]
}

fn token_from_headers(headers: &HeaderMap) -> Option<&str> {
    let raw = headers.get(axum::http::header::COOKIE)?.to_str().ok()?;
    raw.split(';').map(str::trim).find_map(|p| p.strip_prefix("auth="))
}

/// Valid JWT whose user still exists -> (id, is_admin).
async fn identify(parts: &Parts, state: &AppState) -> AppResult<Option<(i64, bool)>> {
    let Some(token) = token_from_headers(&parts.headers) else { return Ok(None) };
    let Ok(data) = decode::<Claims>(
        token,
        &DecodingKey::from_secret(state.jwt_secret.as_bytes()),
        &Validation::default(),
    ) else {
        return Ok(None);
    };
    let row: Option<(bool,)> = sqlx::query_as("SELECT is_admin FROM users WHERE id = $1")
        .bind(data.claims.sub)
        .fetch_optional(&state.pool)
        .await?;
    Ok(row.map(|(a,)| (data.claims.sub, a)))
}

pub struct AuthUser {
    pub id: i64,
}
pub struct MaybeUser(pub Option<i64>);
pub struct AdminUser {
    #[allow(dead_code)]
    pub id: i64,
}

impl FromRequestParts<AppState> for AuthUser {
    type Rejection = AppError;
    async fn from_request_parts(parts: &mut Parts, state: &AppState) -> Result<Self, AppError> {
        match identify(parts, state).await? {
            Some((id, _)) => Ok(AuthUser { id }),
            None => Err(AppError::Unauthorized),
        }
    }
}

impl FromRequestParts<AppState> for MaybeUser {
    type Rejection = AppError;
    async fn from_request_parts(parts: &mut Parts, state: &AppState) -> Result<Self, AppError> {
        Ok(MaybeUser(identify(parts, state).await.ok().flatten().map(|(id, _)| id)))
    }
}

impl FromRequestParts<AppState> for AdminUser {
    type Rejection = AppError;
    async fn from_request_parts(parts: &mut Parts, state: &AppState) -> Result<Self, AppError> {
        match identify(parts, state).await? {
            Some((id, true)) => Ok(AdminUser { id }),
            Some(_) => Err(AppError::Forbidden),
            None => Err(AppError::Unauthorized),
        }
    }
}

pub async fn load_user(pool: &sqlx::PgPool, id: i64) -> AppResult<UserJson> {
    Ok(sqlx::query_as::<_, UserJson>(
        "SELECT u.id, u.email, u.name, u.city, u.is_admin, s.id AS shop_id
         FROM users u LEFT JOIN shops s ON s.owner_user_id = u.id WHERE u.id = $1",
    )
    .bind(id)
    .fetch_one(pool)
    .await?)
}

#[derive(Deserialize)]
struct RegisterBody {
    email: String,
    password: String,
    name: String,
    city: String,
}

async fn register(State(st): State<AppState>, Json(b): Json<RegisterBody>) -> AppResult<impl IntoResponse> {
    let email = b.email.trim().to_lowercase();
    if !email.contains('@') {
        return Err(AppError::BadRequest("invalid email".into()));
    }
    if b.password.chars().count() < 8 {
        return Err(AppError::BadRequest("password must be at least 8 characters".into()));
    }
    if b.name.trim().is_empty() {
        return Err(AppError::BadRequest("name is required".into()));
    }
    if !CITIES.contains(&b.city.as_str()) {
        return Err(AppError::BadRequest(format!("city must be one of {}", CITIES.join(", "))));
    }
    let hash = hash_password(&b.password)?;
    let res: Result<(i64,), sqlx::Error> = sqlx::query_as(
        "INSERT INTO users (email, password_hash, name, city) VALUES ($1, $2, $3, $4) RETURNING id",
    )
    .bind(&email)
    .bind(&hash)
    .bind(b.name.trim())
    .bind(&b.city)
    .fetch_one(&st.pool)
    .await;
    let id = match res {
        Ok((id,)) => id,
        Err(sqlx::Error::Database(e)) if e.is_unique_violation() => {
            return Err(AppError::Conflict("email already registered".into()));
        }
        Err(e) => return Err(e.into()),
    };
    let token = make_token(id, &st.jwt_secret)?;
    let user = load_user(&st.pool, id).await?;
    Ok((StatusCode::CREATED, cookie_header(&token, WEEK_SECS), Json(user)))
}

#[derive(Deserialize)]
struct LoginBody {
    email: String,
    password: String,
}

async fn login(State(st): State<AppState>, Json(b): Json<LoginBody>) -> AppResult<impl IntoResponse> {
    let row: Option<(i64, String)> = sqlx::query_as("SELECT id, password_hash FROM users WHERE email = $1")
        .bind(b.email.trim().to_lowercase())
        .fetch_optional(&st.pool)
        .await?;
    let bad = || AppError::UnauthorizedMsg("invalid email or password".into());
    let (id, hash) = row.ok_or_else(bad)?;
    if !verify_password(&b.password, &hash) {
        return Err(bad());
    }
    let token = make_token(id, &st.jwt_secret)?;
    let user = load_user(&st.pool, id).await?;
    Ok((StatusCode::OK, cookie_header(&token, WEEK_SECS), Json(user)))
}

async fn logout() -> impl IntoResponse {
    (StatusCode::NO_CONTENT, cookie_header("", 0))
}

async fn me(State(st): State<AppState>, user: AuthUser) -> AppResult<Json<UserJson>> {
    Ok(Json(load_user(&st.pool, user.id).await?))
}

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/auth/register", post(register))
        .route("/auth/login", post(login))
        .route("/auth/logout", post(logout))
        .route("/auth/me", get(me))
}
