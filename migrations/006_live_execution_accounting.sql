-- DARAMS live execution / accounting layer.
-- 將 recommendation 串到 account-aware orders / fills / PnL 快照。

BEGIN;

CREATE TABLE IF NOT EXISTS live_accounts (
    account_id       VARCHAR(50) PRIMARY KEY,
    account_type     VARCHAR(20) NOT NULL,
    broker           VARCHAR(40),
    base_currency    VARCHAR(10) NOT NULL DEFAULT 'TWD',
    initial_capital  DOUBLE PRECISION NOT NULL,
    status           VARCHAR(20) NOT NULL DEFAULT 'ACTIVE',
    metadata         JSONB NOT NULL DEFAULT '{}',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO live_accounts (
    account_id, account_type, broker, base_currency, initial_capital, status, metadata
) VALUES (
    'paper_main', 'paper', 'paper', 'TWD', 10000000, 'ACTIVE',
    '{"purpose": "default paper account"}'
)
ON CONFLICT (account_id) DO NOTHING;

CREATE TABLE IF NOT EXISTS live_market_prices (
    account_id       VARCHAR(50) NOT NULL REFERENCES live_accounts(account_id),
    as_of_date       DATE NOT NULL,
    run_id           UUID REFERENCES daily_live_runs(run_id) ON DELETE SET NULL,
    security_id      VARCHAR(20) NOT NULL,
    price_time       TIMESTAMPTZ NOT NULL,
    price            DOUBLE PRECISION NOT NULL,
    price_type       VARCHAR(20) NOT NULL,
    price_source     VARCHAR(40) NOT NULL,
    adjustment_mode  VARCHAR(40) NOT NULL,
    metadata         JSONB NOT NULL DEFAULT '{}',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (account_id, as_of_date, security_id, price_type, price_source)
);

CREATE INDEX IF NOT EXISTS ix_live_market_prices_run
    ON live_market_prices (run_id, as_of_date DESC);

CREATE TABLE IF NOT EXISTS live_position_snapshots (
    account_id       VARCHAR(50) NOT NULL REFERENCES live_accounts(account_id),
    as_of_date       DATE NOT NULL,
    run_id           UUID REFERENCES daily_live_runs(run_id) ON DELETE SET NULL,
    snapshot_time    TIMESTAMPTZ NOT NULL,
    security_id      VARCHAR(20) NOT NULL,
    quantity         INT NOT NULL DEFAULT 0,
    avg_cost         DOUBLE PRECISION NOT NULL DEFAULT 0,
    last_price       DOUBLE PRECISION,
    market_value     DOUBLE PRECISION NOT NULL DEFAULT 0,
    realized_pnl     DOUBLE PRECISION NOT NULL DEFAULT 0,
    unrealized_pnl   DOUBLE PRECISION NOT NULL DEFAULT 0,
    price_source     VARCHAR(40) NOT NULL,
    adjustment_mode  VARCHAR(40) NOT NULL,
    metadata         JSONB NOT NULL DEFAULT '{}',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (account_id, as_of_date, security_id)
);

CREATE INDEX IF NOT EXISTS ix_live_position_snapshots_run
    ON live_position_snapshots (run_id, as_of_date DESC);

CREATE TABLE IF NOT EXISTS live_account_snapshots (
    account_id          VARCHAR(50) NOT NULL REFERENCES live_accounts(account_id),
    as_of_date          DATE NOT NULL,
    run_id              UUID REFERENCES daily_live_runs(run_id) ON DELETE SET NULL,
    snapshot_time       TIMESTAMPTZ NOT NULL,
    cash                DOUBLE PRECISION NOT NULL,
    market_value        DOUBLE PRECISION NOT NULL,
    realized_pnl        DOUBLE PRECISION NOT NULL,
    unrealized_pnl      DOUBLE PRECISION NOT NULL,
    total_equity        DOUBLE PRECISION NOT NULL,
    daily_return        DOUBLE PRECISION,
    cumulative_return   DOUBLE PRECISION,
    price_source        VARCHAR(40) NOT NULL,
    adjustment_mode     VARCHAR(40) NOT NULL,
    metadata            JSONB NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (account_id, as_of_date)
);

CREATE INDEX IF NOT EXISTS ix_live_account_snapshots_run
    ON live_account_snapshots (run_id, as_of_date DESC);

ALTER TABLE orders
    ADD COLUMN IF NOT EXISTS account_id VARCHAR(50) REFERENCES live_accounts(account_id),
    ADD COLUMN IF NOT EXISTS run_id UUID REFERENCES daily_live_runs(run_id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS recommendation_id BIGINT REFERENCES trade_recommendations(recommendation_id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS execution_mode VARCHAR(20),
    ADD COLUMN IF NOT EXISTS broker_order_id VARCHAR(80),
    ADD COLUMN IF NOT EXISTS submitted_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS reject_reason TEXT,
    ADD COLUMN IF NOT EXISTS price_source VARCHAR(40),
    ADD COLUMN IF NOT EXISTS adjustment_mode VARCHAR(40),
    ADD COLUMN IF NOT EXISTS raw_payload JSONB NOT NULL DEFAULT '{}';

CREATE INDEX IF NOT EXISTS ix_orders_run_account
    ON orders (run_id, account_id, order_time DESC);
CREATE INDEX IF NOT EXISTS ix_orders_recommendation
    ON orders (recommendation_id);

ALTER TABLE fills
    ADD COLUMN IF NOT EXISTS account_id VARCHAR(50) REFERENCES live_accounts(account_id),
    ADD COLUMN IF NOT EXISTS run_id UUID REFERENCES daily_live_runs(run_id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS recommendation_id BIGINT REFERENCES trade_recommendations(recommendation_id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS broker_fill_id VARCHAR(80),
    ADD COLUMN IF NOT EXISTS side VARCHAR(4),
    ADD COLUMN IF NOT EXISTS gross_notional DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS tax DOUBLE PRECISION NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS fees_total DOUBLE PRECISION NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS price_source VARCHAR(40),
    ADD COLUMN IF NOT EXISTS adjustment_mode VARCHAR(40),
    ADD COLUMN IF NOT EXISTS source_file TEXT,
    ADD COLUMN IF NOT EXISTS raw_payload JSONB NOT NULL DEFAULT '{}';

CREATE INDEX IF NOT EXISTS ix_fills_run_account
    ON fills (run_id, account_id, fill_time DESC);
CREATE INDEX IF NOT EXISTS ix_fills_recommendation
    ON fills (recommendation_id);

COMMIT;
