# Appendix C — Reproducing every result

Every number, chart and screenshot in this book is produced by code in this
repository. This appendix lists the exact commands, in order. All paths are
relative to the repository root.

## 1. Tools

| Tool | Used for |
|---|---|
| PostgreSQL at `localhost:5432` (user `postgres`, password `postgres`) | the marketplace database (chapter 13, screenshots, latency) |
| Rust toolchain (`cargo`) and `protoc` | the API server in `backend/` |
| Node.js and `npm` | the web app in `frontend/`, and mermaid/KaTeX for the PDF build |
| `uv` | every Python environment: `recommender/` and `book/` |

## 2. The offline companion scripts (chapters 1–12)

These need nothing but Python. They all use the same simulated marketplace
(seed 7), cached in `book/build/sim.pkl` on first use.

```sh
cd book
uv sync                                  # installs recsys from ../recommender in editable mode
uv run python tools/run_all.py --skip-live
```

`run_all.py` runs `code/ch01_explore.py` … `code/ch12_coldstart_bias.py` in
order. Each script prints the text quoted in its chapter, which is saved to
`results/<script>.txt`, and writes its numbers to `results/<script>.json`. The
run stops at the first script that fails. To re-run a single chapter's script:

```sh
uv run python tools/run_all.py --only ch06_mf
```

The scripts are deterministic: the same machine gives identical files. Across
machines, timings (milliseconds per query in chapter 7) differ, and LightGBM
and PyTorch may differ in the last decimal places.

## 3. The live system (chapter 13)

One-time setup:

```sh
PGPASSWORD=postgres createdb -h localhost -U postgres marketplace
cp backend/.env.example backend/.env
(cd backend && cargo run -- migrate)
(cd recommender && uv sync && uv run recsys seed-db --yes)
(cd frontend && npm i)
```

Then start the three services, each in its own terminal:

```sh
cd recommender && uv run recsys serve
cd backend && cargo run
cd frontend && npm run dev
```

With the recommender running, the latency script calls it directly over gRPC:

```sh
cd book && uv run python tools/run_all.py --only ch13_latency
```

Because `seed-db` shifts the simulated history so that it ends at the moment
you run it, the live model's metrics and latencies will differ slightly from
the ones printed in this book.

## 4. Charts

```sh
cd book && uv run python tools/make_figures.py
```

Reads `results/*.json` and writes every PNG in `assets/charts/`. Each chapter's
charts are drawn by `tools/figures/chNN.py`.

## 5. Screenshots

With all three services running and the app at `http://localhost:5173`:

```sh
cd book && uv run python tools/screenshots.py
```

This logs in through the real UI as `user43@demo.test`, places an order, then
logs in as that shop's seller and as `admin@demo.test`, and writes the PNGs in
`assets/screenshots/`. The fallback banner needs the recommender to be
**stopped**:

```sh
# stop `recsys serve`, then:
cd book && uv run python tools/screenshots.py --only fallback
```

## 6. The PDFs

```sh
cd book
uv run playwright install chromium
npm i                                    # mermaid + KaTeX, rendered inside the page
uv run python tools/check.py --lang en
uv run python tools/check.py --lang id
uv run python tools/build.py --lang all
```

`check.py` verifies every include, output file and image exists, and that each
chapter has its fact boxes, diagram, figures, quiz and minimum length.
`build.py` renders Markdown to HTML, opens it in headless Chromium, waits for
the diagrams and formulas to render, and prints `dist/recsys-book-en.pdf` and
`dist/recsys-book-id.pdf` with their page counts.
