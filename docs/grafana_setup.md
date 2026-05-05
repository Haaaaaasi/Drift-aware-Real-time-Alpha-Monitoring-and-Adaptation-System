# DARAMS Grafana Dashboard Setup

Grafana is provisioned automatically when you run `docker-compose up -d`. All datasources and dashboards are loaded from the `dashboards/` directory — no manual configuration required.

## Quick Start

```bash
# 1. Start all infrastructure
docker-compose up -d

# 2. Wait ~10 seconds for Grafana to initialize, then open:
#    http://localhost:3000
#    Username: admin
#    Password: admin  (or $GRAFANA_PASSWORD from .env)

# 3. (Optional) Generate monitoring data to see charts populated:
python -m pipelines.daily_batch_pipeline --csv data/tw_stocks_ohlcv.csv
```

## Dashboard Overview

Four dashboards are provisioned under the **DARAMS** folder:

| Dashboard | UID | Monitors |
|-----------|-----|---------|
| Data Monitor | `darams_data_monitor` | Missing bars, abnormal prices, KS p-value, PSI |
| Alpha Monitor | `darams_alpha_monitor` | Rolling IC, Rank-IC, turnover, PSI per alpha |
| Model Monitor | `darams_model_monitor` | Directional accuracy, ECE calibration, prediction drift |
| Strategy Monitor | `darams_strategy_monitor` | Sharpe, drawdown, realized vs expected, model registry |

The **Strategy Monitor** is set as the home dashboard.

## Directory Structure

```
dashboards/
├── provisioning/
│   ├── datasources/
│   │   └── datasource.yaml      # PostgreSQL connection (auto-provisioned)
│   └── dashboards/
│       └── dashboard.yaml       # Dashboard file provider config
├── data_monitor.json
├── alpha_monitor.json
├── model_monitor.json
└── strategy_monitor.json
```

## Data Prerequisites

Dashboards query two PostgreSQL tables populated by the monitoring pipeline:

| Table | Populated by |
|-------|-------------|
| `monitoring_metrics` | `AlertManager.persist_metrics()` in `alert_manager.py` |
| `alerts` | `AlertManager.fire_alerts()` in `alert_manager.py` |
| `model_registry` | `ModelRegistryManager.register_model()` in `model_registry.py` |

Run the batch pipeline to generate data:

```bash
# CSV mode (no DolphinDB needed):
python -m pipelines.daily_batch_pipeline --csv data/tw_stocks_ohlcv.csv

# Synthetic mode (no external dependencies):
python -m pipelines.daily_batch_pipeline --synthetic
```

> **Note**: The current pipeline runs monitoring in-memory and does not persist to PostgreSQL by default. To populate the dashboards, call `AlertManager.persist_metrics(metrics)` and `AlertManager.fire_alerts(metrics)` after running monitors. This is wired up in `pipelines/monitoring_pipeline.py`.

## Threshold Reference

### Data Monitor
| Metric | WARN | CRIT |
|--------|------|------|
| Missing ratio | ≥ 5% | ≥ 15% |
| Abnormal price ratio | ≥ 1% | ≥ 5% |
| KS p-value | < 0.05 | < 0.01 |
| PSI | ≥ 0.10 | ≥ 0.25 |

### Alpha Monitor
| Metric | WARN | CRIT |
|--------|------|------|
| Rolling IC | < 0.02 | < 0.00 |
| Alpha turnover | ≥ 80% | ≥ 95% |
| Alpha PSI | ≥ 0.10 | ≥ 0.25 |
| Correlation drift | ≥ 0.20 | ≥ 0.40 |

### Model Monitor
| Metric | WARN | CRIT |
|--------|------|------|
| Directional accuracy | < 52% | < 48% |
| Calibration ECE | ≥ 0.10 | ≥ 0.20 |
| Prediction KS p-value | < 0.05 | < 0.01 |

### Strategy Monitor
| Metric | WARN | CRIT |
|--------|------|------|
| Rolling Sharpe | < 0.5 | < 0.0 |
| Max drawdown | ≥ 10% | ≥ 20% |
| Realized vs expected | < 0.5 | < 0.2 |

## Troubleshooting

**Dashboards not appearing**
- Check Grafana logs: `docker logs darams-grafana`
- Confirm the `dashboards/` mount: `docker exec darams-grafana ls /var/lib/grafana/dashboards`

**"datasource not found" error**
- Confirm: `docker exec darams-grafana ls /etc/grafana/provisioning/datasources`
- Check PostgreSQL is running: `docker ps | grep darams-postgres`

**No data in panels**
- Run the pipeline first to populate `monitoring_metrics` and `alerts`
- Verify tables exist: connect to `psql -h localhost -p 5433 -U darams -d darams` and run `SELECT COUNT(*) FROM monitoring_metrics`
- Check the time range selector in Grafana (default: last 7 days)

**Password change**
Set `GRAFANA_PASSWORD` in `.env` before first startup. After Grafana has started, passwords cannot be changed via env var — use the Grafana UI instead.
