-- DARAMS PostgreSQL Schema Migration v003
-- Final robustness bundle reporting tables for Grafana.

BEGIN;

CREATE TABLE IF NOT EXISTS final_robustness_bundles (
    bundle_id            TEXT PRIMARY KEY,
    title                TEXT NOT NULL,
    is_official          BOOLEAN NOT NULL DEFAULT TRUE,
    status               TEXT NOT NULL DEFAULT 'completed',
    validation_start     DATE,
    validation_end       DATE,
    validation_status    TEXT NOT NULL,
    frozen_selector_id   TEXT NOT NULL,
    official_selector    TEXT NOT NULL,
    official_adaptation  TEXT NOT NULL,
    primary_execution    TEXT NOT NULL,
    secondary_execution  TEXT,
    summary_report       TEXT,
    manifest_path        TEXT,
    frozen_config        TEXT,
    config_json          JSONB NOT NULL DEFAULT '{}',
    created_at           TIMESTAMPTZ,
    ingested_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    notes                TEXT,
    CONSTRAINT ck_final_robustness_status
        CHECK (status IN ('completed', 'failed', 'partial'))
);

CREATE INDEX IF NOT EXISTS ix_final_robustness_bundle_lookup
    ON final_robustness_bundles (is_official, status, validation_end DESC, ingested_at DESC);

CREATE TABLE IF NOT EXISTS final_robustness_strategy_results (
    result_id                  BIGSERIAL PRIMARY KEY,
    bundle_id                  TEXT NOT NULL REFERENCES final_robustness_bundles(bundle_id) ON DELETE CASCADE,
    execution_price            TEXT NOT NULL,
    series                     TEXT NOT NULL,
    result_role                TEXT NOT NULL,
    is_official_strategy       BOOLEAN NOT NULL DEFAULT FALSE,
    is_benchmark               BOOLEAN NOT NULL DEFAULT FALSE,
    cumulative_return_pct      DOUBLE PRECISION,
    sharpe                     DOUBLE PRECISION,
    max_drawdown_pct           DOUBLE PRECISION,
    avg_turnover               DOUBLE PRECISION,
    avg_cost_bps               DOUBLE PRECISION,
    n_reuses                   INTEGER,
    n_misses                   INTEGER,
    sort_order                 INTEGER NOT NULL DEFAULT 0,
    UNIQUE (bundle_id, execution_price, series)
);

CREATE INDEX IF NOT EXISTS ix_final_robustness_strategy_lookup
    ON final_robustness_strategy_results (bundle_id, result_role, execution_price, sort_order);

CREATE TABLE IF NOT EXISTS final_robustness_checks (
    check_id        BIGSERIAL PRIMARY KEY,
    bundle_id       TEXT NOT NULL REFERENCES final_robustness_bundles(bundle_id) ON DELETE CASCADE,
    check_type      TEXT NOT NULL,
    execution_price TEXT,
    metric          TEXT NOT NULL,
    comparison      TEXT,
    real_value      DOUBLE PRECISION,
    reference_value DOUBLE PRECISION,
    p_value         DOUBLE PRECISION,
    percentile      DOUBLE PRECISION,
    ci05            DOUBLE PRECISION,
    ci95            DOUBLE PRECISION,
    n_samples       INTEGER,
    passed          BOOLEAN,
    sort_order      INTEGER NOT NULL DEFAULT 0,
    UNIQUE (bundle_id, check_type, execution_price, metric, comparison)
);

CREATE INDEX IF NOT EXISTS ix_final_robustness_checks_lookup
    ON final_robustness_checks (bundle_id, check_type, execution_price, sort_order);

CREATE TABLE IF NOT EXISTS final_robustness_regime_results (
    regime_result_id       BIGSERIAL PRIMARY KEY,
    bundle_id              TEXT NOT NULL REFERENCES final_robustness_bundles(bundle_id) ON DELETE CASCADE,
    execution_price        TEXT NOT NULL,
    regime                 TEXT NOT NULL,
    cumulative_return_pct  DOUBLE PRECISION,
    sharpe                 DOUBLE PRECISION,
    max_drawdown_pct       DOUBLE PRECISION,
    sort_order             INTEGER NOT NULL DEFAULT 0,
    UNIQUE (bundle_id, execution_price, regime)
);

CREATE INDEX IF NOT EXISTS ix_final_robustness_regime_lookup
    ON final_robustness_regime_results (bundle_id, execution_price, sort_order);

CREATE TABLE IF NOT EXISTS final_robustness_decisions (
    decision_id BIGSERIAL PRIMARY KEY,
    bundle_id   TEXT NOT NULL REFERENCES final_robustness_bundles(bundle_id) ON DELETE CASCADE,
    topic       TEXT NOT NULL,
    decision    TEXT NOT NULL,
    severity    TEXT NOT NULL DEFAULT 'info',
    evidence    TEXT NOT NULL,
    sort_order  INTEGER NOT NULL DEFAULT 0,
    UNIQUE (bundle_id, topic)
);

CREATE INDEX IF NOT EXISTS ix_final_robustness_decisions_lookup
    ON final_robustness_decisions (bundle_id, severity, sort_order);

CREATE TABLE IF NOT EXISTS final_robustness_artifacts (
    artifact_id   BIGSERIAL PRIMARY KEY,
    bundle_id     TEXT NOT NULL REFERENCES final_robustness_bundles(bundle_id) ON DELETE CASCADE,
    artifact_type TEXT NOT NULL,
    label         TEXT NOT NULL,
    path          TEXT NOT NULL,
    sort_order    INTEGER NOT NULL DEFAULT 0,
    UNIQUE (bundle_id, label)
);

CREATE INDEX IF NOT EXISTS ix_final_robustness_artifacts_lookup
    ON final_robustness_artifacts (bundle_id, artifact_type, sort_order);

COMMIT;
