import json
from pathlib import Path

import pytest

from scripts import ingest_final_robustness_bundle as ing


def _write_bundle(report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "manifest.json").write_text(json.dumps({
        "package": "final_robustness_test",
        "frozen_selector_id": "selector_v1",
        "validation_period": {
            "start": "2024-07-01",
            "end": "2026-04-30",
            "status": "FROZEN_VALIDATION_NOT_UNTOUCHED_HOLDOUT",
        },
        "primary_execution": "next_vwap",
        "secondary_execution": "next_open",
        "frozen_config": "configs/frozen.yaml",
        "summary_report": "reports/final.md",
    }), encoding="utf-8")
    (report_dir / "grafana_tables.json").write_text(json.dumps({
        "bundle": {
            "bundle_id": "final_robustness_test",
            "title": "Final Test",
            "official_selector": "incumbent_55 + rolling_topk20",
            "official_adaptation": "scheduled_20",
        },
        "strategy_results": [{
            "execution_price": "next_vwap",
            "series": "rolling_topk20",
            "result_role": "incumbent",
            "is_official_strategy": True,
            "cumulative_return_pct": 62.12,
            "sharpe": 1.298,
            "sort_order": 1,
        }],
        "checks": [{
            "check_type": "placebo",
            "execution_price": "next_vwap",
            "metric": "sharpe",
            "real_value": 1.298,
            "reference_value": 0.173,
            "passed": True,
            "sort_order": 1,
        }],
        "regime_results": [{
            "execution_price": "next_vwap",
            "regime": "2024_H2",
            "cumulative_return_pct": -8.356,
            "sort_order": 1,
        }],
        "decisions": [{
            "topic": "model_pool",
            "decision": "CLOSED_AS_APPENDIX",
            "severity": "warning",
            "evidence": "輸給 scheduled_20",
            "sort_order": 1,
        }],
        "artifacts": [{
            "artifact_type": "summary",
            "label": "Summary",
            "path": "reports/final.md",
            "sort_order": 1,
        }],
    }), encoding="utf-8")


def test_build_payload_reads_manifest_and_grafana_tables(tmp_path: Path) -> None:
    report_dir = tmp_path / "bundle"
    _write_bundle(report_dir)

    payload = ing.build_payload(ing.IngestOptions(report_dir=report_dir))

    bundle = payload["bundle"]
    assert bundle["bundle_id"] == "final_robustness_test"
    assert bundle["validation_status"] == "FROZEN_VALIDATION_NOT_UNTOUCHED_HOLDOUT"
    assert bundle["primary_execution"] == "next_vwap"
    assert payload["strategy_results"][0]["bundle_id"] == "final_robustness_test"
    assert payload["checks"][0]["check_type"] == "placebo"
    assert payload["regime_results"][0]["regime"] == "2024_H2"
    assert payload["decisions"][0]["decision"] == "CLOSED_AS_APPENDIX"


def test_missing_grafana_tables_fails_cleanly(tmp_path: Path) -> None:
    report_dir = tmp_path / "bundle"
    report_dir.mkdir()
    (report_dir / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="grafana_tables.json"):
        ing.build_payload(ing.IngestOptions(report_dir=report_dir))


def test_write_payload_deletes_child_tables_before_insert(monkeypatch) -> None:
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
    conn = FakeConn()
    payload = {
        "bundle": {
            "bundle_id": "bundle1",
            "title": "bundle1",
            "is_official": True,
            "status": "completed",
            "validation_start": None,
            "validation_end": None,
            "validation_status": "FROZEN_VALIDATION_NOT_UNTOUCHED_HOLDOUT",
            "frozen_selector_id": "selector",
            "official_selector": "selector",
            "official_adaptation": "scheduled_20",
            "primary_execution": "next_vwap",
            "secondary_execution": "next_open",
            "summary_report": None,
            "manifest_path": None,
            "frozen_config": None,
            "config_json": {},
            "created_at": None,
            "notes": None,
        },
        "strategy_results": [{"bundle_id": "bundle1", "execution_price": "next_vwap", "series": "s", "result_role": "incumbent"}],
        "checks": [{"bundle_id": "bundle1", "check_type": "placebo", "metric": "sharpe"}],
        "regime_results": [{"bundle_id": "bundle1", "execution_price": "next_vwap", "regime": "2024_H2"}],
        "decisions": [{"bundle_id": "bundle1", "topic": "model_pool", "decision": "CLOSED", "severity": "warning", "evidence": "x"}],
        "artifacts": [{"bundle_id": "bundle1", "artifact_type": "summary", "label": "Summary", "path": "reports/final.md"}],
    }

    ing.write_payload(payload, conn=conn)

    joined = "\n".join(queries)
    assert "DELETE FROM final_robustness_artifacts" in joined
    assert "DELETE FROM final_robustness_strategy_results" in joined
    assert any("final_robustness_checks" in sql for sql in bulk_tables)
    assert conn.committed is True
    assert conn.rolled_back is False


def test_static_grafana_contracts() -> None:
    migration = (ing.PROJECT_ROOT / "migrations" / "003_final_robustness_reporting.sql").read_text(encoding="utf-8")
    dashboard = json.loads((ing.PROJECT_ROOT / "dashboards" / "final_robustness.json").read_text(encoding="utf-8"))
    tables = json.loads((ing.PROJECT_ROOT / "reports" / "adaptation_ab" / "final_robustness_20260518" / "grafana_tables.json").read_text(encoding="utf-8"))

    assert "final_robustness_bundles" in migration
    assert "ON DELETE CASCADE" in migration
    assert dashboard["uid"] == "darams-final-robustness"
    assert "final_robustness_strategy_results" in json.dumps(dashboard)
    assert tables["bundle"]["bundle_id"] == "final_robustness_20260518"
