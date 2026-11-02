mod recommender;

use sqlx::postgres::PgPoolOptions;

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
    let _jwt_secret = env_var("JWT_SECRET");
    let _recommender_url = env_var("RECOMMENDER_URL");
    let _bind_addr = env_var("BIND_ADDR");

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let pool = PgPoolOptions::new().max_connections(10).connect(&database_url).await?;
    sqlx::migrate!("./migrations").run(&pool).await?;
    if std::env::args().nth(1).as_deref() == Some("migrate") {
        tracing::info!("migrations applied");
        return Ok(());
    }
    Ok(())
}
