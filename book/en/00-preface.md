# Preface

Open the Shopee app on a Tuesday evening. Before you type a letter, the home screen already shows forty products: a phone case for the phone you bought last month, cat food because you browsed pet products last week, a flash-sale kettle. Nobody wrote those slots by hand. A pipeline looked at millions of products, discarded almost all of them in milliseconds, scored the survivors, and shuffled the winners so the page did not look like a wall of identical phone cases.

That pipeline is a **recommender system**. This book teaches how one works by building one: first a list of the most popular items, then, chapter by chapter, every idea that turns that list into a modern multi-stage system.

## The big picture in plain words

A recommender answers one question: **"Out of everything we sell, which few items should this person see right now?"** It does this using clues the shop already has: what the person clicked and bought before, and what similar people did.

The difficulty is size. A marketplace like ours has thousands of items (millions in a real app), and the screen has room for about forty. We cannot carefully judge every item for every visitor within the fraction of a second before the page loads. So real systems work like a shop assistant fetching things from a huge warehouse for a customer:

1. **Retrieval (candidate generation): grab a trolley-load from the warehouse.** The assistant does not inspect every box. They walk to the pet aisle and the phone-accessory aisle, because Sari's history says she likes those, and pull a few hundred boxes quickly. This step must be *fast*, and it may be a little rough. The few hundred items it returns are the **candidates**.
2. **Ranking: look closely and put the trolley in order.** Now the assistant has only a few hundred boxes, so they can afford to examine each one carefully: how likely is Sari to click this, given her price range, her past orders and the time of day? Each candidate gets a score and the list is sorted from best to worst. This step is *precise* but slow per item, which is why it only sees candidates.
3. **Re-ranking: arrange the shop window.** Sorted by score alone, the top ten might be ten nearly identical phone cases. The assistant reshuffles for variety, pushes out-of-stock products down, and follows shop rules, such as not showing the same seller five times in a row. This final polish is **re-ranking**.

```mermaid
flowchart LR
    W["All items<br/>(the warehouse)"] -->|"1. Retrieval<br/>fast and rough"| C["A few hundred<br/>candidates"]
    C -->|"2. Ranking<br/>slow and precise"| R["Scored and sorted list"]
    R -->|"3. Re-ranking<br/>variety and rules"| P["The ~40 items<br/>Sari sees"]
```

*Figure: The three-stage funnel. Each stage looks at fewer items than the one before, so it can afford to do more work per item.*

The funnel exists because of a trade-off: careful judgement costs time, and time is limited, so we spend cheap effort on many items and expensive effort on few. Without it, you would either show slow pages (judging everything carefully) or poor recommendations (judging everything roughly).

We build exactly this shape step by step. First, in Chapters 3 to 7, one method tries to do the whole job. Then Chapters 8 to 12 split the work into the three stages above.

## Who this book is for

The book is written for **mid-level software engineers**: you read Python comfortably, know REST and SQL, and have shipped production code. No machine-learning background is needed; each piece of mathematics is explained the first time it appears and used on a tiny worked example. Everything runs on a laptop in minutes, with no GPU.

Two promises shape everything that follows:

1. **Every algorithm is explained step by step.** You will see the intuition, the formula, a worked example, and then the real code that implements it.
2. **Every number in the text comes from a program you can run.** No result in this book was typed by hand. Each chapter has a companion script; its output is pasted into the book directly from a file captured when the script ran. If you re-run the script you will get the same numbers, because all random seeds are fixed. A **random seed** is like rolling dice in a game where everyone rolls the same numbers in the same order: the "random" parts of the simulation and training come out identical on every run, so your numbers match the book's.

## The shape of the book

The book follows the historical and practical path of the field. Each chapter adds machinery only after the previous approach has shown its limits on the same data.

- **Foundations (Chapters 1–2).** The problem, the data, and the ruler: precision, recall, NDCG and time-based evaluation. In short, these are three scores for "how good is this list?" and the habit of testing a recommender on days it has not seen.
- **Single-model recommenders (Chapters 3–7).** One method does the whole job. Popularity (show what everyone likes), content-based filtering (suggest items whose description resembles what you liked), collaborative filtering (suggest what people similar to you liked), matrix factorisation (compress everyone's tastes into a few hidden numbers), and embeddings with two-tower retrieval (turn users and items into lists of numbers so that "close" means "a good match"). Each term is explained from scratch in its own chapter.
- **The multi-stage system (Chapters 8–12).** Real marketplaces run a funnel, described in the next section: a cheap stage fetches a few hundred **candidates**, a precise stage **ranks** them, and a final stage **re-ranks** for diversity and business rules. These chapters cover candidate generation, feature engineering (turning raw logs into useful numbers for the model), learning to rank, re-ranking, and cold start, context and bias.
- **Production (Chapters 13–14).** A working Vue (web page), Rust (server), Postgres (database) and Python gRPC application in which you follow one click end to end, then what changes at tens of millions of users.

The appendices hold quiz answers, a glossary, reproduction commands and the bibliography.

## What is in the repository

The book lives inside the repository of a complete working project. The repository contains the book, the code the book teaches, and the application that uses it.

```mermaid
flowchart LR
    subgraph repo["recommendation-system/"]
        proto["proto/<br/>recommender.proto<br/>(gRPC contract)"]
        backend["backend/<br/>Rust + Axum API<br/>(writes events)"]
        recsys["recommender/<br/>Python 'recsys' package<br/>(gRPC service, reads DB)"]
        frontend["frontend/<br/>Vue 3 storefront"]
        book["book/<br/>this book:<br/>chapters, code, results"]
        docs["docs/<br/>GUIDE.md"]
    end
    frontend -->|"HTTP /api"| backend
    backend -->|"gRPC"| recsys
    proto -.->|"generates client"| backend
    proto -.->|"generates server"| recsys
    backend --> pg[("PostgreSQL")]
    recsys -->|"read-only"| pg
    book -.->|"imports and runs"| recsys
    book -.->|"includes code from"| backend
    docs -.->|"explains"| repo
```

*Figure: The repository at a glance. Solid arrows are runtime calls; dotted arrows are build-time or documentation relationships.*

Each directory has one job:

| Directory | What it holds | Where it appears in the book |
|---|---|---|
| `recommender/` | The Python package `recsys`: the marketplace simulator, candidate retrievers, feature builders, the LightGBM ranker, the re-ranker and the gRPC service | Chapters 1–3 and 8–13 |
| `book/code/` | One companion script per algorithm chapter, plus `common.py` shared by all of them | Every chapter |
| `book/results/` | The captured output (`.txt`) and metrics (`.json`) of those scripts | Every `output` block |
| `backend/` | The Rust and Axum API: authentication, catalogue, cart, checkout, event logging, the gRPC client | Chapter 13 |
| `frontend/` | The Vue 3 storefront with the "Why this?" explanations and the live event log | Chapter 13 |
| `proto/` | `recommender.proto`, the single file that defines the contract between Rust and Python | Chapter 13 |
| `docs/` | `GUIDE.md`, a plain-language walkthrough of the running application | Chapter 13 |
| `book/` | The chapters, charts, screenshots and build tooling for the PDF you are reading | – |

Two design decisions recur throughout. **Rust writes all user behaviour to Postgres; Python only reads it**, through a read-only connection (one that can look at the data but cannot change it, so the recommender can never corrupt the shop's records). And **recommendations are computed on request**, meaning the list is built at the moment you open the page, so a click made five seconds ago can already change the home page, while the model (the learned part) is retrained periodically in the background.

A few words in the table above, in plain terms. **gRPC** is a way for two programs, here the Rust one and the Python one, to call each other's functions over the network. A **`.proto` file** is the written agreement describing exactly which calls exist and what they carry. **LightGBM** is a library that learns to score items; Chapter 10 explains it.

## Setting up

You can read the book without running anything, but you will learn more if you change a number and watch what happens.

### What you need

| Tool | Used for |
|---|---|
| [`uv`](https://docs.astral.sh/uv/) | Python environments for `recommender/` and `book/` |
| Node.js and `npm` | The Vue frontend, and the `mermaid` and `katex` libraries used to render diagrams and math in the PDF |
| PostgreSQL | Only for the live application of Chapter 13. Listening on `localhost:5432`, user `postgres`, password `postgres` |
| Rust toolchain (`cargo`) | Only for the backend |
| `protoc` | Only to compile the gRPC contract for Rust |

The algorithm chapters (1–12) need **only `uv`**. Postgres, Rust and Node are needed only when you want to run the full web application.

### Companion scripts and the book tooling

Clone the repository, then from the repository root run:

```bash
cd book && uv sync && uv run playwright install chromium && npm i
```

`uv sync` installs Python dependencies, including the `recsys` package from `../recommender` in editable mode, so the scripts import the same code the chapters show. The `playwright install` step downloads a headless Chromium, which is used both to render the PDF and to take the screenshots. `npm i` fetches `mermaid` and `katex`.

Every companion script runs from the `book/` directory:

```bash
uv run python code/ch01_explore.py
```

To regenerate every result file in order, use `uv run python tools/run_all.py`. Appendix C lists all commands, including those that rebuild the figures and the PDF.

### The web application

For Chapter 13 you need the three services running. The authoritative, step-by-step instructions are in `docs/GUIDE.md`, section "Run it". In outline:

1. Create the database and apply the schema: `createdb marketplace`, copy `backend/.env.example` to `backend/.env`, then `cargo run -- migrate` inside `backend/`.
2. Fill the database with the simulated marketplace: `uv run recsys seed-db --yes` inside `recommender/`.
3. Start three terminals: `uv run recsys serve` (Python, port 50051), `cargo run` (Rust, port 8080) and `npm run dev` (Vue, port 5173).

Then open `http://localhost:5173`. The seed data includes demo accounts such as `user43@demo.test` with the password `password`; the login page lists the others.

## The running example: a simulated marketplace

Real marketplace logs are private and public datasets lack positions (where on the page an item was shown), dwell time (how long the user stayed on it) and stock, so the book uses a **simulated marketplace** shipped with the repository: a small pretend store, run by a program, with pretend shoppers who browse and buy. Because we wrote the simulator we also know the hidden truth behind the behaviour, which Chapter 1 uses to show how logged data distorts what users want.

The default simulation, configured in `recommender/recsys/simulate.py`, has 1,500 users, 2,500 items, 150 shops and 60 days of activity, with fixed seeds. The Chapter 1 script confirms the table sizes:

```output
file: results/ch01_explore.txt
lines: 2-9
```

Beyond sheer size, the simulator reproduces the patterns that make recommendation interesting. Some terms below are new, so each comes with a plain meaning:

- **Popularity is lopsided.** A small share of items attracts a large share of interactions, exactly as in a real store. Plot items from most to least popular and you get a few tall bars followed by a very long, low tail; hence the name **long tail**. The danger is that a recommender which only repeats the tall bars never shows shoppers the rest of the shelf.
- **Tastes are personal.** Each user has hidden preferences for categories, price range and brands. Sari might love cat food and cheap phone cases; another user might prefer premium electronics.
- **Time matters.** Traffic is higher on weekends and around paydays (the 25th through the 1st of each month), and the simulated window contains the **9.9 mega-sale**, a day on which traffic and orders spike. Chapter 1 quantifies these effects and charts them.
- **Behaviour is a funnel.** Every action is logged as an *event*. A user sees an item on screen (an **impression**), some impressions become **clicks**, fewer become **add-to-carts** (the item goes into the shopping cart), and fewer still become **orders** (the purchase). Each step has fewer people than the one before, like a funnel narrowing. Some items get *hidden* (the user taps "not interested") and some purchases are *returned*.
- **Stock runs out and new items appear.** Sellers list products during the window and some sell out. A brand-new item that nobody has clicked yet is hard to recommend, because we have no behaviour to learn from; this is the **cold-start problem**, together with the availability problem, in Chapter 12.

### One time split for the whole book

The most important convention in the book is a fixed point in time called the **cutoff**. The simulated marketplace covers 60 days. We pick **day 53 as the cutoff** and apply one rule everywhere:

> Every algorithm may learn from, and compute features from, events that happened **before** the cutoff. It is then judged on what users actually did in the **7 days after** it.

Think of a weather forecaster. You give them records up to 1 March and ask, "What will 2 to 8 March look like?" You already know the answer because it is in your records, but you hide it, then compare the forecast with what really happened. That comparison is the score. *Cheating* would be letting the forecaster read the 5 March newspaper first. In our marketplace, the **history** is the records, the cutoff is "1 March", and the **label window** (the days after the cutoff) is the hidden answer key. The same rule applies to our running shopper Sari: the recommender sees only what Sari did before day 53 and is then checked against what she did in days 53 to 60. If it were allowed to see her day-55 purchase while learning, its score would look wonderful and mean nothing.

```mermaid
flowchart LR
    A["Day 0<br/>simulation starts"] --> B["Days 0 to 53<br/>HISTORY<br/>algorithms learn here"]
    B --> C["Day 53<br/>CUTOFF"]
    C --> D["Days 53 to 60<br/>LABEL WINDOW<br/>what users really did"]
    D --> E["Day 60<br/>simulation ends"]
```

*Figure: The time split used by every companion script. Nothing from the label window is ever visible during training or feature building.*

The shared file `book/code/common.py` enforces the split and provides one `evaluate` function, so numbers are comparable across chapters.

Think of `evaluate` as the single answer key that every algorithm is marked against. If each chapter used its own marking scheme, you could not tell whether a new idea was better or just graded more kindly.

## How each chapter is organised

Chapters 1 to 14 follow the same template, so you always know where you are:

1. **An opening story** set in a Shopee-like scenario, and a short list of what you will learn.
2. **Intuition**, using an everyday analogy before any formula.
3. **The algorithm**: the formulas, numbered steps, and a worked example small enough to do with a pencil.
4. **Code walkthrough**: real code from the repository, explained piece by piece.
5. **Run it**: the command to execute and its captured output.
6. **Results**: charts and an interpretation tied to the actual numbers.
7. **Pitfalls and production practice**: what goes wrong, and how large systems handle it.
8. **Summary** and **Quiz**.

### Visual and typographic conventions

Code is pulled directly from the repository files, and output blocks are copied from captured results files. Any number in a paragraph also appears in an output block or a JSON file under `book/results/`. Inline math looks like $a \cdot b$; Chapter 2 builds NDCG from nothing.

Three quality scores appear from Chapter 2 onward. You do not need them yet, but here is the one-line meaning of each so they feel familiar. **Precision** asks: of the items we showed, what share did the user really want? **Recall** asks: of all the items the user really wanted, what share did we manage to show? **NDCG** (Normalised Discounted Cumulative Gain) is one score that rewards putting the right item near the *top* of the list rather than at the bottom. Chapter 2 builds each of them with a tiny example.

Four kinds of callout boxes break up the text. They look like this in the PDF:

**"Did you know?" boxes** (blue) carry a verifiable fact from the research literature or the history of the field, with its source named at the bottom of the box and listed in the bibliography. Every chapter has at least two of them.

::: tip
**A tip** is a practical shortcut or a habit that will save you time.
:::

::: warning
**A warning** marks a mistake that is easy to make and expensive to discover, such as evaluating on data the model has already seen.
:::

::: try
**A "Try it yourself" box** suggests a small experiment: change one parameter in a script, re-run it and compare. These are the best way to turn reading into understanding.
:::

Finally, every chapter ends with a **quiz** of eight to ten questions: multiple choice, short answer, a code-reading question, a calculation and one hands-on exercise that modifies a companion script. Answers with short explanations are in Appendix A. Try to answer before you look; the effort of retrieving an answer is what makes it stick.

## A note on scale

Where we describe how large platforms work we rely on published papers and invent no internal numbers. When a simple method beats a fancier one on this dataset, the book says so and explains why.

With the map in hand, turn the page. Chapter 1 begins where every recommender project begins: with the data, and with an honest look at what it can and cannot tell us.
