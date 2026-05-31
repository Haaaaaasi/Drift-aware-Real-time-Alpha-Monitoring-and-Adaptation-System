from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from src.adaptation.performance_trigger import PerformanceTriggeredAdapter, TriggerDecision


def test_trigger_decision_supports_legacy_tuple_unpacking() -> None:
    decision = TriggerDecision(True, "Rolling IC < 0")

    triggered, reason = decision

    assert triggered is True
    assert "IC" in reason


def test_infer_trigger_type_handles_critical_without_ic_false_positive() -> None:
    adapter = PerformanceTriggeredAdapter()

    assert adapter._infer_trigger_type("Critical alerts (1) >= limit (1)") == "critical_alerts"
    assert adapter._infer_trigger_type("Rolling IC below threshold") == "rolling_ic"


def test_shadow_gate_auto_promotes_only_when_both_gates_pass() -> None:
    adapter = PerformanceTriggeredAdapter()
    adapter._registry = MagicMock()
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor

    passed = adapter.evaluate_shadow_gate_and_promote(
        conn=conn,
        event_id="11111111-1111-1111-1111-111111111111",
        current_model_id="current",
        candidate_model_id="candidate",
        shadow_metrics={
            "current": {"ic": 0.01, "topk_net_return": 0.02},
            "candidate": {"ic": 0.011, "topk_net_return": 0.03},
        },
        min_topk_net_improvement=0.005,
    )

    assert passed is True
    adapter._registry.promote_model.assert_called_once_with("candidate")
    assert cursor.execute.call_count == 1


def test_shadow_gate_rejects_when_net_improvement_is_too_small() -> None:
    adapter = PerformanceTriggeredAdapter()
    adapter._registry = MagicMock()
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor

    passed = adapter.evaluate_shadow_gate_and_promote(
        conn=conn,
        event_id="11111111-1111-1111-1111-111111111111",
        current_model_id="current",
        candidate_model_id="candidate",
        shadow_metrics={
            "current": {"ic": 0.01, "topk_net_return": 0.02},
            "candidate": {"ic": 0.02, "topk_net_return": 0.021},
        },
        min_topk_net_improvement=0.005,
    )

    assert passed is False
    adapter._registry.promote_model.assert_not_called()
    assert cursor.execute.call_count == 1


def test_cooldown_event_is_not_created_when_metrics_are_healthy() -> None:
    adapter = PerformanceTriggeredAdapter()
    adapter._latest_recent_event = MagicMock(return_value={"event_id": "old"})
    adapter.create_adaptation_event = MagicMock(return_value="skip")
    conn = MagicMock()
    healthy_ic = pd.DataFrame(
        {
            "metric_time": pd.date_range("2026-05-01", periods=5),
            "metric_value": [0.02, 0.01, 0.03, 0.02, 0.01],
        }
    )
    healthy_sharpe = pd.DataFrame(
        {
            "metric_time": pd.date_range("2026-05-01", periods=5),
            "metric_value": [0.5, 0.4, 0.6, 0.7, 0.8],
        }
    )
    with patch(
        "src.adaptation.performance_trigger.pd.read_sql",
        side_effect=[healthy_ic, healthy_sharpe, pd.DataFrame()],
    ):
        decision = adapter.check_trigger_from_db(
            conn=conn,
            window=5,
            account_id="paper_main",
            model_id="ml_prod",
            create_event=True,
        )

    assert decision.should_trigger is False
    assert decision.status == "NO_TRIGGER"
    adapter._latest_recent_event.assert_not_called()
    adapter.create_adaptation_event.assert_not_called()
