-- Marketplace schema. Rust (marketplace-api) is the only writer; the Python
-- recommender connects read-only. Prices are integer IDR.

CREATE TABLE users (
    id            BIGSERIAL PRIMARY KEY,
    email         TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    name          TEXT NOT NULL,
    city          TEXT NOT NULL,
    age_bucket    INT NOT NULL DEFAULT 1,
    device_tier   INT NOT NULL DEFAULT 2,
    is_admin      BOOLEAN NOT NULL DEFAULT false,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE categories (
    id            BIGINT PRIMARY KEY,
    name          TEXT NOT NULL,
    parent_name   TEXT NOT NULL,
    is_consumable BOOLEAN NOT NULL,
    median_price  BIGINT NOT NULL
);

CREATE TABLE shops (
    id            BIGSERIAL PRIMARY KEY,
    owner_user_id BIGINT UNIQUE NOT NULL REFERENCES users,
    name          TEXT NOT NULL,
    city          TEXT NOT NULL,
    is_mall       BOOLEAN NOT NULL DEFAULT false,
    shop_rating   DOUBLE PRECISION NOT NULL DEFAULT 4.5,
    response_rate DOUBLE PRECISION NOT NULL DEFAULT 0.9,
    cancel_rate   DOUBLE PRECISION NOT NULL DEFAULT 0.02,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE items (
    id              BIGSERIAL PRIMARY KEY,
    shop_id         BIGINT NOT NULL REFERENCES shops,
    category_id     BIGINT NOT NULL REFERENCES categories,
    brand           TEXT NOT NULL,
    title           TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    price           BIGINT NOT NULL CHECK (price > 0),
    discount_pct    DOUBLE PRECISION NOT NULL DEFAULT 0 CHECK (discount_pct BETWEEN 0 AND 0.5),
    free_shipping   BOOLEAN NOT NULL DEFAULT false,
    rating_avg      DOUBLE PRECISION NOT NULL DEFAULT 0,
    rating_count    INT NOT NULL DEFAULT 0,
    stock           INT NOT NULL DEFAULT 0 CHECK (stock >= 0),
    is_active       BOOLEAN NOT NULL DEFAULT true,
    listed_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    out_of_stock_at TIMESTAMPTZ NULL
);
CREATE INDEX items_category_id_idx ON items (category_id);
CREATE INDEX items_shop_id_idx ON items (shop_id);

CREATE TABLE sessions (
    id           BIGSERIAL PRIMARY KEY,
    user_id      BIGINT NOT NULL REFERENCES users,
    started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    surface      TEXT NOT NULL,
    device_os    TEXT NOT NULL
);
CREATE INDEX sessions_user_last_seen_idx ON sessions (user_id, last_seen_at DESC);

CREATE TABLE searches (
    id         BIGSERIAL PRIMARY KEY,
    session_id BIGINT NOT NULL REFERENCES sessions,
    user_id    BIGINT NOT NULL REFERENCES users,
    ts         TIMESTAMPTZ NOT NULL DEFAULT now(),
    query      TEXT NOT NULL
);

-- book:start events_table
-- One row per thing a user did to an item. Append-only: the recommender's
-- whole view of user behaviour comes from this table.
CREATE TABLE events (
    id         BIGSERIAL PRIMARY KEY,
    ts         TIMESTAMPTZ NOT NULL DEFAULT now(),
    user_id    BIGINT NOT NULL REFERENCES users,
    session_id BIGINT NOT NULL REFERENCES sessions,
    item_id    BIGINT NOT NULL REFERENCES items,
    event_type TEXT NOT NULL CHECK (event_type IN ('impression','click','add_to_cart','order','hide','return')),
    position   INT NULL,
    dwell_sec  DOUBLE PRECISION NULL,
    price_paid DOUBLE PRECISION NULL,
    surface    TEXT NULL
);
CREATE INDEX events_user_ts_idx ON events (user_id, ts DESC);
CREATE INDEX events_item_id_idx ON events (item_id);
-- book:end events_table

CREATE TABLE cart_items (
    user_id  BIGINT REFERENCES users,
    item_id  BIGINT REFERENCES items,
    quantity INT NOT NULL CHECK (quantity > 0),
    added_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, item_id)
);

CREATE TABLE orders (
    id               BIGSERIAL PRIMARY KEY,
    buyer_id         BIGINT NOT NULL REFERENCES users,
    shop_id          BIGINT NOT NULL REFERENCES shops,
    status           TEXT NOT NULL CHECK (status IN ('paid','shipped','completed')),
    shipping_address TEXT NOT NULL,
    subtotal         BIGINT NOT NULL,
    shipping_fee     BIGINT NOT NULL,
    total            BIGINT NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX orders_shop_id_idx ON orders (shop_id);

CREATE TABLE order_items (
    order_id   BIGINT REFERENCES orders ON DELETE CASCADE,
    item_id    BIGINT REFERENCES items,
    quantity   INT NOT NULL,
    unit_price BIGINT NOT NULL,
    PRIMARY KEY (order_id, item_id)
);
