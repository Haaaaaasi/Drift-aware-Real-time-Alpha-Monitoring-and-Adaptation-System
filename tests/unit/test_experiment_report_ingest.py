import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import ingest_experiment_report as ing


def _write_daily(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {
            "date": "2024-01-02",
            "n_holdings": 2,
            "turnover": 1.0,
            "gross_return": 0.01,
            "commission_cost": 0.001,
            "tax_cost": 0.0,
            "slippage_cost": 0.0005,
            "net_return": 0.0085,
            "cumulative_value": 10_085_000,
        },
        {
            "date": "2024-01-03",
            "n_holdings": 2,
            "turnover": 0.0,
            "gross_return": 0.02,
            "commission_cost": 0.0,
            "tax_cost": 0.0,
            "slippage_cost": 0.0,
            "net_return": 0.02,
            "cumulative_value": 10_286_700,
        },
    ]).to_csv(path, index=False)


def _write_decisions(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {
            "date": "2024-01-03",
            "day_idx": 10,
            "current_model_id": "m0",
            "shadow_new_model_id": "m1",
            "live_model_id": "m2",
            "selected_candidate_model_id": "m1",
            "applied_model_id": "m2",
            "candidate_model_id": "m1",
            "candidate_role": "new",
            "selected": True,
            "selected_role": "new",
            "decision_reason": "shadow_selected_new",
            "pool_hit": False,
            "candidate_similarity": 0.6,
            "selected_similarity": 0.6,
            "best_seen_similarity": 0.7,
            "n_reused_candidates": 1,
            "selection_metric": "topk_net_return",
            "selection_score": 0.12,
            "shadow_ic": 0.03,
            "shadow_hit_rate": 0.55,
            "shadow_sharpe": 1.1,
            "shadow_n_samples": 100,
            "shadow_topk_gross_return": 0.05,
            "shadow_topk_net_return": 0.04,
            "shadow_topk_turnover": 0.8,
            "shadow_topk_n_days": 15,
            "proxy_n_days": 10,
            "proxy_gross_return": 0.02,
            "proxy_net_return": 0.015,
            "proxy_turnover": 0.9,
            "proxy_cost": 0.005,
            "proxy_rank_by_net": 1,
        }
    ]).to_csv(path, index=False)


def _options(report_dir: Path, run_type: str = "ab_experiment", **kwargs) -> ing.IngestOptions:
    return ing.IngestOptions(report_dir=report_dir, run_type=run_type, **kwargs)


def test_ab_report_imports_comparison_daily_benchmark_and_decisions(tmp_path: Path) -> None:
    report_dir = tmp_path / "ab_formal"
    model_pool_dir = report_dir / "simulations" / "model_pool"
    benchmark_path = report_dir / "benchmarks" / "ew_buy_hold_universe_daily_pnl.csv"
    _write_daily(model_pool_dir / "daily_pnl.csv")
    _write_daily(benchmark_path)
    _write_decisions(model_pool_dir / "model_pool_decisions.csv")
    (report_dir / "benchmarks").mkdir(parents=True, exist_ok=True)
    (report_dir / "experiment_summary.md").write_text("# summary\n", encoding="utf-8")
    (report_dir / "config.json").write_text(json.dumps({
        "run_id": "ab_formal",
        "start": "2024-01-02",
        "end": "2024-01-03",
        "benchmark": "ew_buy_hold_universe",
        "run_dirs": {"model_pool": str(model_pool_dir)},
        "benchmark_path": str(benchmark_path),
        "strategies": {
            "model_pool": {
                "strategy": "model_pool",
                "model_pool_selection_metric": "topk_net_return",
                "similarity_threshold": 0.5,
            }
        },
    }), encoding="utf-8")
    pd.DataFrame([
        {
            "strategy": "model_pool",
            "cumulative_return_pct": 12.0,
            "annualized_return_pct": 20.0,
            "sharpe": 1.2,
            "max_drawdown_pct": -5.0,
            "win_rate_pct": 60.0,
            "avg_turnover": 0.1,
            "avg_gross_return_bps": 12.0,
            "avg_total_cost_bps": 5.0,
            "avg_net_return_bps": 7.0,
            "n_retrains": 3,
            "n_pool_reuses": 1,
            "n_pool_misses": 2,
        },
        {
            "strategy": "ew_buy_hold_universe",
            "cumulative_return_pct": 8.0,
            "annualized_return_pct": 12.0,
            "sharpe": 0.9,
            "max_drawdown_pct": -4.0,
            "win_rate_pct": 55.0,
            "avg_turnover": 0.01,
            "avg_gross_return_bps": 8.0,
            "avg_total_cost_bps": 0.1,
            "avg_net_return_bps": 7.9,
            "n_retrains": 0,
            "n_pool_reuses": 0,
            "n_pool_misses": 0,
        },
    ]).to_csv(report_dir / "comparison.csv", index=False)

    payload = ing.build_payload(_options(
        report_dir,
        data_source="tej",
        is_official=True,
        run_name="formal ab",
    ))

    model_pool = next(r for r in payload["strategy_results"] if r["strategy"] == "model_pool")
    benchmark = next(r for r in payload["strategy_results"] if r["is_benchmark"])
    assert model_pool["selection_metric"] == "topk_net_return"
    assert model_pool["similarity_threshold"] == 0.5
    assert model_pool["rank_by_net_return"] == 1
    assert benchmark["rank_by_net_return"] is None
    assert any(r["is_benchmark"] for r in payload["daily_pnl"])
    assert payload["model_pool_decisions"][0]["raw_record"]["current_model_id"] == "m0"


def test_comparison_consolidated_fallback_is_supported(tmp_path: Path) -> None:
    report_dir = tmp_path / "ab_consolidated"
    report_dir.mkdir()
    pd.DataFrame([{"strategy": "none", "cumulative_return_pct": 1.0}]).to_csv(
        report_dir / "comparison_consolidated.csv", index=False
    )

    payload = ing.build_payload(_options(report_dir, data_source="tej"))

    assert payload["strategy_results"][0]["strategy"] == "none"


def test_official_run_requires_data_source(tmp_path: Path) -> None:
    report_dir = tmp_path / "ab_missing_ds"
    report_dir.mkdir()
    pd.DataFrame([{"strategy": "none", "cumulative_return_pct": 1.0}]).to_csv(
        report_dir / "comparison.csv", index=False
    )

    with pytest.raises(ValueError, match="--data-source"):
        ing.build_payload(_options(report_dir, is_official=True))


def test_selector_matrix_uses_variant_not_strategy(tmp_path: Path) -> None:
    report_dir = tmp_path / "selector_matrix"
    child_dir = report_dir / "sim_topknet"
    _write_daily(child_dir / "daily_pnl.csv")
    _write_decisions(child_dir / "model_pool_decisions.csv")
    (child_dir / "config.json").write_text(json.dumps({
        "start": "2024-01-02",
        "end": "2024-01-03",
        "csv_path": "data/tw_stocks_tej.parquet",
    }), encoding="utf-8")
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "label": "topknet_t05",
        "run_dir": str(child_dir),
        "selection_metric": "topk_net_return",
        "similarity_threshold": 0.5,
        "cum_return_pct": 49.2,
        "annualized_return_pct": 17.3,
        "sharpe": 0.77,
        "max_drawdown_pct": -30.7,
        "avg_gross_bps": 13.1,
        "avg_net_bps": 7.6,
        "avg_total_cost_bps": 5.5,
        "avg_turnover": 0.1,
        "n_retrains": 24,
        "n_pool_reuses": 5,
        "n_pool_misses": 7,
        "selected_current": 9,
        "selected_new": 9,
        "selected_reused": 5,
    }]).to_csv(report_dir / "selector_matrix_summary.csv", index=False)

    payload = ing.build_payload(_options(report_dir, run_type="selector_matrix", data_source="tej"))

    result = payload["strategy_results"][0]
    assert result["strategy"] == "model_pool"
    assert result["variant_name"] == "topknet_t05"
    assert result["is_matrix_cell"] is True
    assert payload["daily_pnl"][0]["variant_name"] == "topknet_t05"
    assert payload["model_pool_decisions"][0]["variant_name"] == "topknet_t05"


def test_cost_sweep_uses_scenario_and_round_trip_cost(tmp_path: Path) -> None:
    report_dir = tmp_path / "cost_sweep"
    sub_dir = report_dir / "sim_cost"
    _write_daily(sub_dir / "daily_pnl.csv")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "config.json").write_text(json.dumps({
        "mode": "cost_sweep",
        "sub_runs": {"cost_0.2": {"scheduled_20": str(sub_dir)}},
    }), encoding="utf-8")
    pd.DataFrame([{
        "cost_pct": 0.2,
        "strategy": "scheduled_20",
        "cumulative_return_pct": 10.0,
        "sharpe": 0.8,
        "avg_net_return_bps": 5.0,
        "avg_total_cost_bps": 2.0,
    }]).to_csv(report_dir / "cost_sensitivity.csv", index=False)

    payload = ing.build_payload(_options(report_dir, data_source="tej"))

    result = payload["strategy_results"][0]
    assert result["scenario_name"] == "cost_0.2pct"
    assert result["round_trip_cost_pct"] == 0.2
    assert payload["daily_pnl"][0]["scenario_name"] == "cost_0.2pct"


def test_dry_run_main_does_not_write_db(tmp_path: Path, monkeypatch, capsys) -> None:
    report_dir = tmp_path / "ab_dry"
    report_dir.mkdir()
    pd.DataFrame([{"strategy": "none", "cumulative_return_pct": 1.0}]).to_csv(
        report_dir / "comparison.csv", index=False
    )

    monkeypatch.setattr(ing, "write_payload", lambda payload: pytest.fail("dry-run 不應寫 DB"))

    code = ing.main([
        "--report-dir", str(report_dir),
        "--run-type", "ab_experiment",
        "--data-source", "tej",
        "--dry-run",
    ])

    assert code == 0
    assert "Strategy rows: 1" in capsys.readouterr().out


def test_write_payload_deletes_children_before_bulk_insert(monkeypatch) -> None:
    queries: list[str] = []
    bulk_tables: list[str] = []

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=None):
            queries.append(sql)

    class FakeConn:
        committed = False
        rolled_back = False

        def cursor(self):
            return FakeCursor()

        def commit(self):
            self.committed = True

        def rollback(self):
            self.rolled_back = True

    def fake_execute_values(cur, sql, rows, page_size=1000):
        bulk_tables.append(sql)

    monkeypatch.setattr(ing, "execute_values", fake_execute_values)
    payload = {
        "run": {
            "run_id": "run1",
            "run_name": "run1",
            "run_type": "ab_experiment",
            "is_official": False,
            "status": "completed",
            "started_at": None,
            "completed_at": None,
            "data_source": "tej",
            "start_date": None,
            "end_date": None,
            "config_json": {},
            "report_path": None,
            "source_report_dir": "reports/run1",
            "git_sha": None,
            "notes": None,
        },
        "strategy_results": [{"run_id": "run1", "strategy": "none"}],
        "daily_pnl": [{"run_id": "run1", "strategy": "none", "trade_date": "2024-01-02"}],
        "model_pool_decisions": [{
            "run_id": "run1",
            "strategy": "model_pool",
            "date": "2024-01-02",
            "raw_record": {},
        }],
    }
    conn = FakeConn()

    ing.write_payload(payload, conn=conn)

    joined = "\n".join(queries)
    assert "DELETE FROM experiment_model_pool_decisions" in joined
    assert "DELETE FROM experiment_daily_pnl" in joined
    assert "DELETE FROM experiment_strategy_results" in joined
    assert any("experiment_daily_pnl" in sql for sql in bulk_tables)
    assert conn.committed is True
    assert conn.rolled_back is False


def test_migration_and_dashboard_static_contracts() -> None:
    migration = (ing.PROJECT_ROOT / "migrations" / "002_experiment_reporting.sql").read_text(encoding="utf-8")
    dashboard = json.loads((ing.PROJECT_ROOT / "dashboards" / "experiment_results.json").read_text(encoding="utf-8"))

    assert "ON DELETE CASCADE" in migration
    assert "run_type IN ('ab_experiment', 'simulate_recent', 'selector_matrix')" in migration
    assert "variant_name     TEXT NOT NULL DEFAULT ''" in migration
    assert "scenario_name    TEXT NOT NULL DEFAULT 'baseline'" in migration
    assert dashboard["uid"] == "darams-experiment-results"
    assert "monitoring_metrics" not in json.dumps(dashboard)
