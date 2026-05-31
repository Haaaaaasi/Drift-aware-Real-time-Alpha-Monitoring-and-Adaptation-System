-- DARAMS PostgreSQL Schema Migration v002
-- Experiment reporting schema for reproducible A/B, selector matrix, and diagnostics results.

BEGIN;

-- ============================================================
-- 1. experiment_runs -- experiment-level metadata
-- ============================================================
CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id            TEXT PRIMARY KEY,
    run_name          TEXT NOT NULL,
    run_type          TEXT NOT NULL,
    is_official       BOOLEAN NOT NULL DEFAULT FALSE,
    status            TEXT NOT NULL DEFAULT 'completed',
    started_at        TIMESTAMPTZ,
    completed_at      TIMESTAMPTZ,
    data_source       TEXT NOT NULL,
    start_date        DATE,
    end_date          DATE,
    config_json       JSONB NOT NULL DEFAULT '{}',
    report_path       TEXT,
    source_report_dir TEXT NOT NULL,
    ingested_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    git_sha           TEXT,
    notes             TEXT,
    CONSTRAINT ck_experiment_runs_run_type
        CHECK (run_type IN ('ab_experiment', 'simulate_recent', 'selector_matrix')),
    CONSTRAINT ck_experiment_runs_status
        CHECK (status IN ('completed', 'failed', 'partial'))
);

CREATE INDEX IF NOT EXISTS ix_experiment_runs_lookup
    ON experiment_runs (is_official, status, completed_at DESC, ingested_at DESC);
CREATE INDEX IF NOT EXISTS ix_experiment_runs_period
    ON experiment_runs (data_source, start_date, end_date);

-- ============================================================
-- 2. experiment_strategy_results -- strategy / variant / scenario summaries
-- ============================================================
CREATE TABLE IF NOT EXISTS experiment_strategy_results (
    result_id                  BIGSERIAL PRIMARY KEY,
    run_id                     TEXT NOT NULL REFERENCES experiment_runs(run_id) ON DELETE CASCADE,
    strategy                   TEXT NOT NULL,
    variant_name               TEXT NOT NULL DEFAULT '',
    scenario_name              TEXT NOT NULL DEFAULT 'baseline',
    round_trip_cost_pct        DOUBLE PRECISION,
    is_matrix_cell             BOOLEAN NOT NULL DEFAULT FALSE,
    is_benchmark               BOOLEAN NOT NULL DEFAULT FALSE,
    selection_metric           TEXT,
    similarity_threshold       DOUBLE PRECISION,
    cumulative_return_pct      DOUBLE PRECISION,
    annualized_return_pct      DOUBLE PRECISION,
    sharpe                     DOUBLE PRECISION,
    max_drawdown_pct           DOUBLE PRECISION,
    win_rate_pct               DOUBLE PRECISION,
    avg_turnover               DOUBLE PRECISION,
    avg_gross_return_bps       DOUBLE PRECISION,
    avg_total_cost_bps         DOUBLE PRECISION,
    avg_net_return_bps         DOUBLE PRECISION,
    final_value                DOUBLE PRECISION,
    n_retrains                 INTEGER NOT NULL DEFAULT 0,
    n_pool_reuses              INTEGER NOT NULL DEFAULT 0,
    n_pool_misses              INTEGER NOT NULL DEFAULT 0,
    rank_by_net_return         INTEGER,
    n_days                     INTEGER,
    avg_holdings               DOUBLE PRECISION,
    selected_current           INTEGER,
    selected_new               INTEGER,
    selected_reused            INTEGER,
    decision_rows              INTEGER,
    trigger_events             INTEGER,
    avg_selected_shadow_topk_net DOUBLE PRECISION,
    avg_selected_proxy_net     DOUBLE PRECISION,
    selected_proxy_rank_mean   DOUBLE PRECISION,
    UNIQUE (run_id, strategy, variant_name, scenario_name)
);

CREATE INDEX IF NOT EXISTS ix_experiment_strategy_run_rank
    ON experiment_strategy_results (run_id, is_benchmark, rank_by_net_return);
CREATE INDEX IF NOT EXISTS ix_experiment_strategy_matrix
    ON experiment_strategy_results (run_id, is_matrix_cell, variant_name);

-- ============================================================
-- 3. experiment_daily_pnl -- daily PnL curves
-- ============================================================
CREATE TABLE IF NOT EXISTS experiment_daily_pnl (
    pnl_id           BIGSERIAL PRIMARY KEY,
    run_id           TEXT NOT NULL REFERENCES experiment_runs(run_id) ON DELETE CASCADE,
    strategy         TEXT NOT NULL,
    variant_name     TEXT NOT NULL DEFAULT '',
    scenario_name    TEXT NOT NULL DEFAULT 'baseline',
    is_benchmark     BOOLEAN NOT NULL DEFAULT FALSE,
    trade_date       DATE NOT NULL,
    gross_return     DOUBLE PRECISION,
    commission_cost  DOUBLE PRECISION,
    tax_cost         DOUBLE PRECISION,
    slippage_cost    DOUBLE PRECISION,
    net_return       DOUBLE PRECISION,
    cumulative_value DOUBLE PRECISION,
    turnover         DOUBLE PRECISION,
    n_holdings       INTEGER,
    UNIQUE (run_id, strategy, variant_name, scenario_name, trade_date)
);

CREATE INDEX IF NOT EXISTS ix_experiment_daily_pnl_curve
    ON experiment_daily_pnl (run_id, strategy, variant_name, scenario_name, trade_date);
CREATE INDEX IF NOT EXISTS ix_experiment_daily_pnl_benchmark
    ON experiment_daily_pnl (run_id, is_benchmark, trade_date);

-- ============================================================
-- 4. experiment_model_pool_decisions -- candidate-level diagnostics
-- ============================================================
CREATE TABLE IF NOT EXISTS experiment_model_pool_decisions (
    decision_id                 BIGSERIAL PRIMARY KEY,
    run_id                      TEXT NOT NULL REFERENCES experiment_runs(run_id) ON DELETE CASCADE,
    strategy                    TEXT NOT NULL DEFAULT 'model_pool',
    variant_name                TEXT NOT NULL DEFAULT '',
    scenario_name               TEXT NOT NULL DEFAULT 'baseline',
    date                        DATE NOT NULL,
    day_idx                     INTEGER,
    current_model_id            TEXT,
    shadow_new_model_id         TEXT,
    live_model_id               TEXT,
    selected_candidate_model_id TEXT,
    applied_model_id            TEXT,
    candidate_model_id          TEXT,
    candidate_role              TEXT,
    selected                    BOOLEAN,
    selected_role               TEXT,
    decision_reason             TEXT,
    pool_hit                    BOOLEAN,
    candidate_similarity        DOUBLE PRECISION,
    selected_similarity         DOUBLE PRECISION,
    best_seen_similarity        DOUBLE PRECISION,
    n_reused_candidates         INTEGER,
    selection_metric            TEXT,
    selection_score             DOUBLE PRECISION,
    shadow_ic                   DOUBLE PRECISION,
    shadow_hit_rate             DOUBLE PRECISION,
    shadow_sharpe               DOUBLE PRECISION,
    shadow_n_samples            INTEGER,
    shadow_topk_gross_return    DOUBLE PRECISION,
    shadow_topk_net_return      DOUBLE PRECISION,
    shadow_topk_turnover        DOUBLE PRECISION,
    shadow_topk_n_days          INTEGER,
    proxy_n_days                INTEGER,
    proxy_gross_return          DOUBLE PRECISION,
    proxy_net_return            DOUBLE PRECISION,
    proxy_turnover              DOUBLE PRECISION,
    proxy_cost                  DOUBLE PRECISION,
    proxy_rank_by_net           DOUBLE PRECISION,
    raw_record                  JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_experiment_decisions_run_date
    ON experiment_model_pool_decisions (run_id, date);
CREATE INDEX IF NOT EXISTS ix_experiment_decisions_variant
    ON experiment_model_pool_decisions (run_id, variant_name, scenario_name);
CREATE INDEX IF NOT EXISTS ix_experiment_decisions_selected_role
    ON experiment_model_pool_decisions (run_id, selected, selected_role);

COMMIT;
