-- DARAMS PostgreSQL Schema Migration v001
-- Tables for metadata, orders, monitoring, and model registry.
-- Time-series data (standardized_bars, alpha_features) lives in DolphinDB.

BEGIN;

-- ============================================================
-- 1. raw_market_events — append-only raw event log
-- ============================================================
CREATE TABLE IF NOT EXISTS raw_market_events (
    event_id      BIGSERIAL PRIMARY KEY,
    security_id   VARCHAR(20)  NOT NULL,
    event_type    VARCHAR(20)  NOT NULL,  -- tick / kbar_1m / kbar_5m / kbar_daily / bidask
    event_ts      TIMESTAMPTZ  NOT NULL,  -- when the event occurred in the market
    ingestion_ts  TIMESTAMPTZ  NOT NULL DEFAULT now(),  -- when we received it
    open          DOUBLE PRECISION,
    high          DOUBLE PRECISION,
    low           DOUBLE PRECISION,
    close         DOUBLE PRECISION,
    volume        DOUBLE PRECISION,
    vwap          DOUBLE PRECISION,
    bid_price     DOUBLE PRECISION,
    ask_price     DOUBLE PRECISION,
    bid_size      DOUBLE PRECISION,
    ask_size      DOUBLE PRECISION,
    raw_payload   JSONB
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_raw_event_dedup
    ON raw_market_events (security_id, event_ts, event_type);
CREATE INDEX IF NOT EXISTS ix_raw_event_ts
    ON raw_market_events (event_ts);

-- ============================================================
-- 2. security_master — stock universe metadata
-- ============================================================
CREATE TABLE IF NOT EXISTS security_master (
    security_id   VARCHAR(20) PRIMARY KEY,
    name          VARCHAR(100) NOT NULL,
    exchange      VARCHAR(10)  NOT NULL DEFAULT 'TWSE',
    industry_code INT,
    listing_date  DATE,
    is_active     BOOLEAN NOT NULL DEFAULT TRUE,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- 3. meta_signals — aggregated trading signals
-- ============================================================
CREATE TABLE IF NOT EXISTS meta_signals (
    signal_id        BIGSERIAL PRIMARY KEY,
    security_id      VARCHAR(20)  NOT NULL,
    signal_time      TIMESTAMPTZ  NOT NULL,
    bar_type         VARCHAR(10)  NOT NULL DEFAULT 'daily',
    signal_score     DOUBLE PRECISION NOT NULL,
    signal_direction SMALLINT     NOT NULL,  -- +1 / 0 / -1
    confidence       DOUBLE PRECISION,
    method           VARCHAR(20)  NOT NULL,  -- rule_based / ml_meta / regime_ensemble
    model_version_id VARCHAR(50)
);

CREATE INDEX IF NOT EXISTS ix_meta_signals_lookup
    ON meta_signals (security_id, signal_time);

-- ============================================================
-- 4. portfolio_targets — desired portfolio weights
-- ============================================================
CREATE TABLE IF NOT EXISTS portfolio_targets (
    target_id           BIGSERIAL PRIMARY KEY,
    rebalance_time      TIMESTAMPTZ  NOT NULL,
    security_id         VARCHAR(20)  NOT NULL,
    target_weight       DOUBLE PRECISION NOT NULL,
    target_shares       INT,
    construction_method VARCHAR(30)  NOT NULL,
    pre_risk            BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS ix_portfolio_rebalance
    ON portfolio_targets (rebalance_time);

-- ============================================================
-- 5. orders — order lifecycle tracking
-- ============================================================
CREATE TABLE IF NOT EXISTS orders (
    order_id       VARCHAR(50) PRIMARY KEY,
    security_id    VARCHAR(20)  NOT NULL,
    order_time     TIMESTAMPTZ  NOT NULL,
    side           VARCHAR(4)   NOT NULL,  -- BUY / SELL
    order_type     VARCHAR(10)  NOT NULL DEFAULT 'MARKET',
    quantity       INT          NOT NULL,
    limit_price    DOUBLE PRECISION,
    status         VARCHAR(20)  NOT NULL DEFAULT 'CREATED',
    expected_price DOUBLE PRECISION,
    updated_at     TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_orders_lookup
    ON orders (security_id, order_time);

-- ============================================================
-- 6. fills — execution / fill records
-- ============================================================
CREATE TABLE IF NOT EXISTS fills (
    fill_id       VARCHAR(50)  PRIMARY KEY,
    order_id      VARCHAR(50)  NOT NULL REFERENCES orders(order_id),
    security_id   VARCHAR(20)  NOT NULL,
    fill_time     TIMESTAMPTZ  NOT NULL,
    fill_price    DOUBLE PRECISION NOT NULL,
    fill_quantity INT          NOT NULL,
    commission    DOUBLE PRECISION NOT NULL DEFAULT 0,
    slippage_bps  DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS ix_fills_order
    ON fills (order_id);

-- ============================================================
-- 7. positions — point-in-time position snapshots
-- ============================================================
CREATE TABLE IF NOT EXISTS positions (
    snapshot_time   TIMESTAMPTZ  NOT NULL,
    security_id     VARCHAR(20)  NOT NULL,
    quantity        INT          NOT NULL DEFAULT 0,
    avg_cost        DOUBLE PRECISION NOT NULL DEFAULT 0,
    market_value    DOUBLE PRECISION NOT NULL DEFAULT 0,
    unrealized_pnl  DOUBLE PRECISION NOT NULL DEFAULT 0,
    PRIMARY KEY (snapshot_time, security_id)
);

-- ============================================================
-- 8. labels_outcomes — delayed forward-return labels
-- ============================================================
CREATE TABLE IF NOT EXISTS labels_outcomes (
    security_id       VARCHAR(20)  NOT NULL,
    signal_time       TIMESTAMPTZ  NOT NULL,
    horizon           INT          NOT NULL,  -- forward bars
    forward_return    DOUBLE PRECISION,
    forward_direction SMALLINT,               -- +1 / -1
    realized_pnl      DOUBLE PRECISION,
    label_available_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (security_id, signal_time, horizon)
);

CREATE INDEX IF NOT EXISTS ix_labels_available
    ON labels_outcomes (label_available_at);

-- ============================================================
-- 9. monitoring_metrics — unified metric store
-- ============================================================
CREATE TABLE IF NOT EXISTS monitoring_metrics (
    metric_id    BIGSERIAL PRIMARY KEY,
    metric_time  TIMESTAMPTZ  NOT NULL,
    monitor_type VARCHAR(20)  NOT NULL,  -- data / alpha / model / strategy
    metric_name  VARCHAR(50)  NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    dimension    VARCHAR(50),            -- e.g. security_id / alpha_id / model_id
    window_size  INT
);

CREATE INDEX IF NOT EXISTS ix_monitoring_lookup
    ON monitoring_metrics (monitor_type, metric_name, metric_time);

-- ============================================================
-- 10. alerts — triggered monitoring alerts
-- ============================================================
CREATE TABLE IF NOT EXISTS alerts (
    alert_id             BIGSERIAL PRIMARY KEY,
    alert_time           TIMESTAMPTZ  NOT NULL DEFAULT now(),
    monitor_type         VARCHAR(20)  NOT NULL,
    metric_name          VARCHAR(50)  NOT NULL,
    severity             VARCHAR(10)  NOT NULL,  -- WARNING / CRITICAL
    current_value        DOUBLE PRECISION NOT NULL,
    threshold            DOUBLE PRECISION NOT NULL,
    message              TEXT,
    is_acknowledged      BOOLEAN NOT NULL DEFAULT FALSE,
    triggered_adaptation BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS ix_alerts_time
    ON alerts (alert_time);
CREATE INDEX IF NOT EXISTS ix_alerts_severity
    ON alerts (severity, alert_time);

-- ============================================================
-- 11. model_registry — model version management
-- ============================================================
CREATE TABLE IF NOT EXISTS model_registry (
    model_id              VARCHAR(50) PRIMARY KEY,
    model_type            VARCHAR(30)  NOT NULL,
    trained_at            TIMESTAMPTZ  NOT NULL,
    training_window_start TIMESTAMPTZ  NOT NULL,
    training_window_end   TIMESTAMPTZ  NOT NULL,
    features_used         JSONB        NOT NULL DEFAULT '[]',
    hyperparams           JSONB        NOT NULL DEFAULT '{}',
    holdout_metrics       JSONB        NOT NULL DEFAULT '{}',
    status                VARCHAR(20)  NOT NULL DEFAULT 'shadow',
    regime_fingerprint    JSONB,
    parent_model_id       VARCHAR(50),
    artifact_path         VARCHAR(200)
);

CREATE INDEX IF NOT EXISTS ix_model_status
    ON model_registry (status);

-- ============================================================
-- 12. alpha_registry — alpha metadata catalogue
-- ============================================================
CREATE TABLE IF NOT EXISTS alpha_registry (
    alpha_id          VARCHAR(20) PRIMARY KEY,
    alpha_name        VARCHAR(50)  NOT NULL,
    category          VARCHAR(20)  NOT NULL,  -- price_volume / industry_aware / cap_aware
    requires_industry BOOLEAN NOT NULL DEFAULT FALSE,
    requires_cap      BOOLEAN NOT NULL DEFAULT FALSE,
    lookback_window   INT     NOT NULL DEFAULT 20,
    is_active         BOOLEAN NOT NULL DEFAULT TRUE,
    notes             TEXT
);

-- ============================================================
-- 13. regime_pool — recurring concept pool
-- ============================================================
CREATE TABLE IF NOT EXISTS regime_pool (
    regime_id               VARCHAR(50) PRIMARY KEY,
    detected_at             TIMESTAMPTZ  NOT NULL,
    fingerprint             JSONB        NOT NULL,
    associated_model_id     VARCHAR(50)  REFERENCES model_registry(model_id),
    associated_alpha_weights JSONB,
    performance_summary     JSONB,
    times_reused            INT NOT NULL DEFAULT 0,
    last_reused_at          TIMESTAMPTZ
);

-- ============================================================
-- Seed alpha_registry with MVP v1 alphas
-- ============================================================
INSERT INTO alpha_registry (alpha_id, alpha_name, category, requires_industry, requires_cap, lookback_window, is_active, notes) VALUES
    ('wq001', 'WQAlpha1',   'price_volume', FALSE, FALSE, 20,  TRUE, 'rank(Ts_ArgMax(SignedPower((returns<0 ? stddev(returns,20) : close), 2), 5)) - 0.5'),
    ('wq002', 'WQAlpha2',   'price_volume', FALSE, FALSE, 6,   TRUE, 'delta(log(volume), 2) * -1 * correlation(close, open, 10)'),
    ('wq003', 'WQAlpha3',   'price_volume', FALSE, FALSE, 10,  TRUE, '-1 * correlation(rank(open), rank(volume), 10)'),
    ('wq004', 'WQAlpha4',   'price_volume', FALSE, FALSE, 9,   TRUE, '-1 * Ts_Rank(rank(low), 9)'),
    ('wq006', 'WQAlpha6',   'price_volume', FALSE, FALSE, 10,  TRUE, '-1 * correlation(open, volume, 10)'),
    ('wq008', 'WQAlpha8',   'price_volume', FALSE, FALSE, 5,   TRUE, '-1 * rank(delta(((close-low)-(high-close))/(high-low), 1) * ...'),
    ('wq009', 'WQAlpha9',   'price_volume', FALSE, FALSE, 5,   TRUE, 'if 0 < ts_min(delta(close,1),5) then delta(close,1) else ...'),
    ('wq012', 'WQAlpha12',  'price_volume', FALSE, FALSE, 1,   TRUE, 'sign(delta(volume, 1)) * (-1 * delta(close, 1))'),
    ('wq014', 'WQAlpha14',  'price_volume', FALSE, FALSE, 10,  TRUE, '-1 * rank(delta(returns, 3)) * correlation(open, volume, 10)'),
    ('wq018', 'WQAlpha18',  'price_volume', FALSE, FALSE, 10,  TRUE, '-1 * rank(stddev(abs(close-open), 5) + (close-open) + correlation(close, open, 10))'),
    ('wq020', 'WQAlpha20',  'price_volume', FALSE, FALSE, 1,   TRUE, '-1 * rank(open - delay(high, 1)) * rank(open - delay(close, 1)) * rank(open - delay(low, 1))'),
    ('wq023', 'WQAlpha23',  'price_volume', FALSE, FALSE, 20,  TRUE, 'if sma(high, 20) < high then -1*delta(high,2) else 0'),
    ('wq026', 'WQAlpha26',  'price_volume', FALSE, FALSE, 5,   TRUE, '-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)'),
    ('wq028', 'WQAlpha28',  'price_volume', FALSE, FALSE, 20,  TRUE, 'scale(correlation(adv20, low, 5) + (high+low)/2 - close)'),
    ('wq041', 'WQAlpha41',  'price_volume', FALSE, FALSE, 5,   TRUE, 'power(high * low, 0.5) - vwap')
ON CONFLICT (alpha_id) DO NOTHING;

COMMIT;
