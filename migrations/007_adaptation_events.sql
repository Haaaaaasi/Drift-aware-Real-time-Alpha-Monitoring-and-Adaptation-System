-- DARAMS monitoring-scoped adaptation events.
-- 將 monitoring metrics / alerts 與 adaptation 決策串成可審計事件。

BEGIN;

ALTER TABLE monitoring_metrics
    ADD COLUMN IF NOT EXISTS run_id UUID REFERENCES daily_live_runs(run_id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS account_id VARCHAR(50) REFERENCES live_accounts(account_id),
    ADD COLUMN IF NOT EXISTS model_id VARCHAR(50),
    ADD COLUMN IF NOT EXISTS strategy_id VARCHAR(50),
    ADD COLUMN IF NOT EXISTS dimension_type VARCHAR(30),
    ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}';

CREATE INDEX IF NOT EXISTS ix_monitoring_scope_lookup
    ON monitoring_metrics (account_id, model_id, monitor_type, metric_name, metric_time DESC);

CREATE TABLE IF NOT EXISTS adaptation_events (
    event_id             UUID PRIMARY KEY,
    triggered_at         TIMESTAMPTZ NOT NULL,
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at         TIMESTAMPTZ,
    as_of_date           DATE,
    run_id               UUID REFERENCES daily_live_runs(run_id) ON DELETE SET NULL,
    account_id           VARCHAR(50) REFERENCES live_accounts(account_id),
    policy_name          VARCHAR(50) NOT NULL,
    trigger_type         VARCHAR(50) NOT NULL,
    severity             VARCHAR(20),
    production_model_id  VARCHAR(50),
    candidate_model_id   VARCHAR(50),
    status               VARCHAR(30) NOT NULL,
    reason               TEXT,
    metrics_snapshot     JSONB NOT NULL DEFAULT '{}',
    shadow_metrics       JSONB NOT NULL DEFAULT '{}',
    decision_metadata    JSONB NOT NULL DEFAULT '{}',
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_adaptation_events_scope
    ON adaptation_events (account_id, production_model_id, triggered_at DESC);
CREATE INDEX IF NOT EXISTS ix_adaptation_events_status
    ON adaptation_events (status, triggered_at DESC);

ALTER TABLE alerts
    ADD COLUMN IF NOT EXISTS run_id UUID REFERENCES daily_live_runs(run_id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS account_id VARCHAR(50) REFERENCES live_accounts(account_id),
    ADD COLUMN IF NOT EXISTS model_id VARCHAR(50),
    ADD COLUMN IF NOT EXISTS adaptation_event_id UUID REFERENCES adaptation_events(event_id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}';

CREATE INDEX IF NOT EXISTS ix_alerts_scope_unhandled
    ON alerts (account_id, model_id, severity, alert_time DESC)
    WHERE is_acknowledged = FALSE AND adaptation_event_id IS NULL;

COMMIT;
