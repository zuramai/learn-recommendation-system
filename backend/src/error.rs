use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::json;

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("{0}")]
    BadRequest(String),
    #[error("not logged in")]
    Unauthorized,
    /// 401 with a specific message (e.g. bad login).
    #[error("{0}")]
    UnauthorizedMsg(String),
    #[error("forbidden")]
    Forbidden,
    #[error("not found")]
    NotFound,
    /// 404 with a specific message (e.g. "create a shop first").
    #[error("{0}")]
    NotFoundMsg(String),
    #[error("{0}")]
    Conflict(String),
    #[error("{0}")]
    Upstream(String),
    #[error("internal error")]
    Internal(#[from] anyhow::Error),
}

pub type AppResult<T> = Result<T, AppError>;

impl From<sqlx::Error> for AppError {
    fn from(e: sqlx::Error) -> Self {
        match e {
            sqlx::Error::RowNotFound => AppError::NotFound,
            other => AppError::Internal(other.into()),
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = match &self {
            AppError::BadRequest(_) => StatusCode::BAD_REQUEST,
            AppError::Unauthorized | AppError::UnauthorizedMsg(_) => StatusCode::UNAUTHORIZED,
            AppError::Forbidden => StatusCode::FORBIDDEN,
            AppError::NotFound | AppError::NotFoundMsg(_) => StatusCode::NOT_FOUND,
            AppError::Conflict(_) => StatusCode::CONFLICT,
            AppError::Upstream(_) => StatusCode::BAD_GATEWAY,
            AppError::Internal(e) => {
                tracing::error!("internal error: {e:#}");
                StatusCode::INTERNAL_SERVER_ERROR
            }
        };
        (status, Json(json!({ "error": self.to_string() }))).into_response()
    }
}
