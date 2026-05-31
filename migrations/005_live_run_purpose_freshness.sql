-- Add official/smoke markers and data freshness fields to live runs.

BEGIN;

ALTER TABLE daily_live_runs
    ADD COLUMN IF NOT EXISTS run_purpose VARCHAR(20) NOT NULL DEFAULT 'production',
    ADD COLUMN IF NOT EXISTS is_official BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS data_max_date DATE,
    ADD COLUMN IF NOT EXISTS data_lag_days INT,
    ADD COLUMN IF NOT EXISTS data_freshness_status VARCHAR(20) NOT NULL DEFAULT 'unknown';

CREATE INDEX IF NOT EXISTS ix_daily_live_runs_official
    ON daily_live_runs (is_official, run_purpose, as_of_date DESC);

COMMIT;
