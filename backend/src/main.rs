mod auth;
mod cart;
mod catalog;
mod error;
mod events;
mod recommender;
mod sessions;
mod seller;

use axum::Router;
use sqlx::{PgPool, postgres::PgPoolOptions};
use tower_http::trace::TraceLayer;

#[derive(Clone)]
pub struct AppState {
    pub pool: PgPool,
    pub rec: recommender::RecClient,
    pub jwt_secret: String,
}

fn env_var(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| {
        eprintln!("missing env var {name} (see backend/.env.example)");
        std::process::exit(1);
    })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    let database_url = env_var("DATABASE_URL");
    let jwt_secret = env_var("JWT_SECRET");
    let recommender_url = env_var("RECOMMENDER_URL");
    let bind_addr = env_var("BIND_ADDR");

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let pool = PgPoolOptions::new().max_connections(10).connect(&database_url).await?;
    sqlx::migrate!("./migrations").run(&pool).await?;
    if std::env::args().nth(1).as_deref() == Some("migrate") {
        tracing::info!("migrations applied");
        return Ok(());
    }

    let channel = tonic::transport::Endpoint::from_shared(recommender_url)?.connect_lazy();
    let state = AppState { pool, rec: recommender::RecClient::new(channel), jwt_secret };

    let api = Router::new()
        .merge(auth::router())
        .merge(catalog::router())
        .merge(cart::router())
        .merge(seller::router())
        .merge(events::router())
        .merge(recommender::router());
    let app = Router::new().nest("/api", api).with_state(state).layer(TraceLayer::new_for_http());

    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    tracing::info!("listening on {bind_addr}");
    axum::serve(listener, app).await?;
    Ok(())
}
