-- DARAMS live daily operating layer.
-- 用 run_id 串起每日 production prediction、alpha snapshot、持倉快照與買賣建議。

BEGIN;

CREATE TABLE IF NOT EXISTS daily_live_runs (
    run_id                         UUID PRIMARY KEY,
    as_of_date                     DATE NOT NULL,
    run_started_at                 TIMESTAMPTZ NOT NULL DEFAULT now(),
    run_finished_at                TIMESTAMPTZ,
    mode                           VARCHAR(20) NOT NULL,
    run_purpose                    VARCHAR(20) NOT NULL DEFAULT 'production',
    is_official                    BOOLEAN NOT NULL DEFAULT FALSE,
    status                         VARCHAR(20) NOT NULL,
    data_source                    VARCHAR(20) NOT NULL,
    data_max_date                  DATE,
    data_lag_days                  INT,
    data_freshness_status          VARCHAR(20) NOT NULL DEFAULT 'unknown',
    bars_path                      TEXT,
    bars_snapshot_hash             VARCHAR(64),
    alpha_cache_path               TEXT,
    alpha_cache_manifest_hash      VARCHAR(64),
    feature_store_version          VARCHAR(64),
    frozen_config_path             TEXT,
    frozen_config_hash             VARCHAR(64),
    frozen_selector_id             VARCHAR(120),
    selector_snapshot_hash         VARCHAR(64),
    diagnostic_selector_snapshot_hash VARCHAR(64),
    production_model_id            VARCHAR(50),
    production_model_artifact_path TEXT,
    feature_columns_hash           VARCHAR(64),
    n_feature_alphas               INT,
    retrain_action                 VARCHAR(40),
    message                        TEXT,
    metadata                       JSONB NOT NULL DEFAULT '{}',
    created_at                     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_daily_live_runs_asof
    ON daily_live_runs (as_of_date DESC, run_started_at DESC);
CREATE INDEX IF NOT EXISTS ix_daily_live_runs_official
    ON daily_live_runs (is_official, run_purpose, as_of_date DESC);
CREATE INDEX IF NOT EXISTS ix_daily_live_runs_model
    ON daily_live_runs (production_model_id, as_of_date DESC);

CREATE TABLE IF NOT EXISTS alpha_selection_snapshots (
    run_id                 UUID NOT NULL REFERENCES daily_live_runs(run_id) ON DELETE CASCADE,
    snapshot_hash          VARCHAR(64) NOT NULL,
    snapshot_role          VARCHAR(20) NOT NULL DEFAULT 'production',
    as_of_date             DATE NOT NULL,
    selector_name          VARCHAR(40) NOT NULL,
    selector_version       VARCHAR(60),
    selector_config_hash   VARCHAR(64),
    feature_columns_hash   VARCHAR(64),
    n_candidate_alphas     INT,
    n_selected_alphas      INT,
    event_metadata         JSONB NOT NULL DEFAULT '{}',
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, snapshot_hash, snapshot_role)
);

CREATE INDEX IF NOT EXISTS ix_alpha_selection_snapshots_asof
    ON alpha_selection_snapshots (as_of_date DESC, snapshot_role);

CREATE TABLE IF NOT EXISTS alpha_selection_scores (
    run_id                 UUID NOT NULL REFERENCES daily_live_runs(run_id) ON DELETE CASCADE,
    snapshot_hash          VARCHAR(64) NOT NULL,
    snapshot_role          VARCHAR(20) NOT NULL DEFAULT 'production',
    as_of_date             DATE NOT NULL,
    alpha_id               VARCHAR(20) NOT NULL,
    selected               BOOLEAN NOT NULL DEFAULT FALSE,
    weight                 DOUBLE PRECISION,
    raw_score              DOUBLE PRECISION,
    score                  DOUBLE PRECISION,
    n_observations         INT,
    coverage               DOUBLE PRECISION,
    rolling_rank_ic        DOUBLE PRECISION,
    stability              DOUBLE PRECISION,
    drift_score            DOUBLE PRECISION,
    turnover_penalty       DOUBLE PRECISION,
    alpha_pool             VARCHAR(30),
    admission_status       VARCHAR(30),
    admission_score        DOUBLE PRECISION,
    admission_reason       TEXT,
    excluded_reason        TEXT,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, snapshot_hash, snapshot_role, alpha_id)
);

CREATE INDEX IF NOT EXISTS ix_alpha_selection_scores_selected
    ON alpha_selection_scores (as_of_date DESC, selected, snapshot_role);

ALTER TABLE meta_signals
    ADD COLUMN IF NOT EXISTS run_id UUID REFERENCES daily_live_runs(run_id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS ix_meta_signals_run
    ON meta_signals (run_id, signal_time DESC);

ALTER TABLE portfolio_targets
    ADD COLUMN IF NOT EXISTS run_id UUID REFERENCES daily_live_runs(run_id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS ix_portfolio_targets_run
    ON portfolio_targets (run_id, rebalance_time DESC);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    run_id             UUID NOT NULL REFERENCES daily_live_runs(run_id) ON DELETE CASCADE,
    as_of_date         DATE NOT NULL,
    snapshot_time      TIMESTAMPTZ NOT NULL,
    security_id        VARCHAR(20) NOT NULL,
    current_weight     DOUBLE PRECISION NOT NULL DEFAULT 0,
    target_weight      DOUBLE PRECISION NOT NULL DEFAULT 0,
    target_shares      INT,
    last_price         DOUBLE PRECISION,
    market_value       DOUBLE PRECISION,
    unrealized_pnl     DOUBLE PRECISION,
    signal_score       DOUBLE PRECISION,
    rank               INT,
    holding_days       INT,
    reason             TEXT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, security_id)
);

CREATE INDEX IF NOT EXISTS ix_portfolio_snapshots_asof
    ON portfolio_snapshots (as_of_date DESC, security_id);

CREATE TABLE IF NOT EXISTS trade_recommendations (
    recommendation_id BIGSERIAL PRIMARY KEY,
    run_id            UUID NOT NULL REFERENCES daily_live_runs(run_id) ON DELETE CASCADE,
    as_of_date        DATE NOT NULL,
    security_id       VARCHAR(20) NOT NULL,
    action            VARCHAR(12) NOT NULL,
    current_weight    DOUBLE PRECISION NOT NULL DEFAULT 0,
    target_weight     DOUBLE PRECISION NOT NULL DEFAULT 0,
    delta_weight      DOUBLE PRECISION NOT NULL DEFAULT 0,
    current_shares    INT,
    target_shares     INT,
    delta_shares      INT,
    last_price        DOUBLE PRECISION,
    signal_score      DOUBLE PRECISION,
    rank              INT,
    reason            TEXT,
    status            VARCHAR(12) NOT NULL DEFAULT 'PENDING',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, security_id)
);

CREATE INDEX IF NOT EXISTS ix_trade_recommendations_asof
    ON trade_recommendations (as_of_date DESC, action, status);
CREATE INDEX IF NOT EXISTS ix_trade_recommendations_run
    ON trade_recommendations (run_id, action);

COMMIT;
