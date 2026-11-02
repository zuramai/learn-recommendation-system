# Marketplace with a visible recommender

A small Shopee-style marketplace — browse, search, cart, checkout, seller center —
whose home feed is driven by a multi-stage recommender you can inspect from the UI.

- `frontend/` — Vue 3 + Vite + TypeScript + Pinia + Tailwind web app
- `backend/` — Rust (Axum) API; the only writer to PostgreSQL
- `recommender/` — Python recommender (candidates → features → LightGBM → re-rank), served over gRPC, read-only on the DB
- `proto/recommender.proto` — the gRPC contract shared by Rust and Python
- `book/` — "Recommender Systems: From Scratch to Shopee Scale" (EN + ID PDFs)

**Start here: [docs/GUIDE.md](docs/GUIDE.md)** — how to run it (section "Run it") and how the recommendations work.
