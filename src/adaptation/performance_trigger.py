"""Layer 10 — Policy 2: Performance-triggered adaptation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from uuid import uuid4

import pandas as pd
from psycopg2.extras import Json

from src.adaptation.model_registry import ModelRegistryManager
from src.common.logging import get_logger
from src.config.constants import AdaptationPolicy

logger = get_logger(__name__)


@dataclass(frozen=True)
class TriggerDecision:
    """DB-driven adaptation trigger decision."""

    should_trigger: bool
    reason: str
    trigger_type: str = "none"
    alert_ids: list[int] = field(default_factory=list)
    metrics_snapshot: dict = field(default_factory=dict)
    production_model_id: str | None = None
    account_id: str | None = None
    event_id: str | None = None
    status: str = "NO_TRIGGER"

    def __iter__(self):
        """保持舊呼叫端 ``triggered, reason = ...`` 的相容性。"""

        yield self.should_trigger
        yield self.reason


class PerformanceTriggeredAdapter:
    """Policy 2: Trigger adaptation when performance degrades beyond thresholds.

    Triggers when:
    - Rolling IC < threshold for N consecutive days
    - Rolling Sharpe < 0 for M consecutive days
    - Critical alerts accumulate beyond K count
    """

    def __init__(
        self,
        ic_threshold: float = 0.0,
        ic_consecutive_days: int = 5,
        sharpe_threshold: float = 0.0,
        sharpe_consecutive_days: int = 10,
        critical_alert_limit: int = 3,
    ) -> None:
        self._ic_thresh = ic_threshold
        self._ic_days = ic_consecutive_days
        self._sharpe_thresh = sharpe_threshold
        self._sharpe_days = sharpe_consecutive_days
        self._crit_limit = critical_alert_limit
        self._registry = ModelRegistryManager()

    def check_trigger(
        self,
        rolling_ic_series: pd.Series,
        rolling_sharpe_series: pd.Series,
        critical_alert_count: int,
    ) -> tuple[bool, str]:
        """Check if performance-triggered adaptation should fire.

        Returns:
            (should_trigger, reason)
        """
        # IC degradation
        if len(rolling_ic_series) >= self._ic_days:
            recent_ic = rolling_ic_series.tail(self._ic_days)
            if (recent_ic < self._ic_thresh).all():
                return True, f"Rolling IC < {self._ic_thresh} for {self._ic_days} consecutive days"

        # Sharpe degradation
        if len(rolling_sharpe_series) >= self._sharpe_days:
            recent_sharpe = rolling_sharpe_series.tail(self._sharpe_days)
            if (recent_sharpe < self._sharpe_thresh).all():
                return True, f"Rolling Sharpe < {self._sharpe_thresh} for {self._sharpe_days} days"

        # Critical alert accumulation
        if critical_alert_count >= self._crit_limit:
            return True, f"Critical alerts ({critical_alert_count}) >= limit ({self._crit_limit})"

        return False, ""

    def check_trigger_from_db(
        self,
        conn=None,
        window: int = 20,
        account_id: str | None = None,
        model_id: str | None = None,
        cooldown_days: int = 20,
        create_event: bool = False,
    ) -> TriggerDecision:
        """從 PostgreSQL monitoring_metrics 表讀取指標後呼叫 check_trigger()。

        DB 不可用或任何例外 → fallback 回傳 decision，不拋例外。

        Parameters
        ----------
        conn:
            可選的 psycopg2 connection；若 None 則自行呼叫 get_pg_connection()
        window:
            從 monitoring_metrics 取最近幾筆資料（預設 20）
        """
        _owns_conn = False
        try:
            if conn is None:
                from src.common.db import get_pg_connection
                conn = get_pg_connection()
                _owns_conn = True
        except Exception as exc:
            logger.warning("check_trigger_from_db_conn_failed", error=str(exc))
            return TriggerDecision(False, "db_unavailable", status="DB_UNAVAILABLE")

        try:
            if model_id is None:
                model_id = self._lookup_production_model_id(conn)
            if account_id is None:
                account_id = self._lookup_latest_account_id(conn) or "paper_main"

            ic_df = pd.read_sql(
                "SELECT metric_time, metric_value FROM monitoring_metrics "
                "WHERE metric_name = 'rolling_ic' "
                "AND (%(model_id)s IS NULL OR model_id = %(model_id)s) "
                "AND (%(account_id)s IS NULL OR account_id = %(account_id)s) "
                "ORDER BY metric_time DESC LIMIT %(window)s",
                conn,
                params={"window": window, "model_id": model_id, "account_id": account_id},
            )
            sharpe_df = pd.read_sql(
                "SELECT metric_time, metric_value FROM monitoring_metrics "
                "WHERE metric_name = 'rolling_sharpe' "
                "AND (%(model_id)s IS NULL OR model_id = %(model_id)s) "
                "AND (%(account_id)s IS NULL OR account_id = %(account_id)s) "
                "ORDER BY metric_time DESC LIMIT %(window)s",
                conn,
                params={"window": window, "model_id": model_id, "account_id": account_id},
            )

            # DESC 查詢需 reverse 還原時間正序，使 .tail(N) 取到最新 N 天
            rolling_ic = pd.Series(
                ic_df["metric_value"].iloc[::-1].values if not ic_df.empty else [],
                dtype=float,
            )
            rolling_sharpe = pd.Series(
                sharpe_df["metric_value"].iloc[::-1].values if not sharpe_df.empty else [],
                dtype=float,
            )

            alert_df = self._load_unhandled_alerts(
                conn,
                account_id=account_id,
                model_id=model_id,
                limit=window,
            )
            critical_count = len(alert_df)
            metrics_snapshot = {
                "rolling_ic": ic_df.iloc[::-1].to_dict(orient="records"),
                "rolling_sharpe": sharpe_df.iloc[::-1].to_dict(orient="records"),
                "critical_alerts": alert_df.to_dict(orient="records"),
            }
            alert_ids = (
                alert_df["alert_id"].dropna().astype(int).tolist()
                if not alert_df.empty and "alert_id" in alert_df.columns
                else []
            )

            logger.info(
                "check_trigger_from_db_loaded",
                ic_records=len(rolling_ic),
                sharpe_records=len(rolling_sharpe),
                critical_count=critical_count,
                window=window,
            )
            triggered, reason = self.check_trigger(rolling_ic, rolling_sharpe, critical_count)
            trigger_type = self._infer_trigger_type(reason)
            if not triggered:
                return TriggerDecision(
                    False,
                    reason,
                    trigger_type=trigger_type,
                    alert_ids=alert_ids,
                    metrics_snapshot=metrics_snapshot,
                    production_model_id=model_id,
                    account_id=account_id,
                    status="NO_TRIGGER",
                )

            cooldown_event = self._latest_recent_event(
                conn,
                account_id=account_id,
                model_id=model_id,
                cooldown_days=cooldown_days,
            )
            if cooldown_event is not None:
                decision = TriggerDecision(
                    False,
                    f"cooldown_active_{cooldown_days}_business_days; trigger_suppressed={reason}",
                    trigger_type=trigger_type,
                    alert_ids=alert_ids,
                    metrics_snapshot=metrics_snapshot,
                    production_model_id=model_id,
                    account_id=account_id,
                    status="SKIPPED_COOLDOWN",
                )
                if create_event:
                    event_id = self.create_adaptation_event(
                        conn,
                        decision=decision,
                        status="SKIPPED_COOLDOWN",
                    )
                    decision = TriggerDecision(
                        **{**decision.__dict__, "event_id": event_id}
                    )
                return decision

            decision = TriggerDecision(
                triggered,
                reason,
                trigger_type=trigger_type,
                alert_ids=alert_ids,
                metrics_snapshot=metrics_snapshot,
                production_model_id=model_id,
                account_id=account_id,
                status="TRIGGERED" if triggered else "NO_TRIGGER",
            )
            if create_event and triggered:
                event_id = self.create_adaptation_event(
                    conn,
                    decision=decision,
                    status="TRIGGERED",
                )
                self.mark_alerts_for_event(conn, alert_ids, event_id)
                decision = TriggerDecision(**{**decision.__dict__, "event_id": event_id})
            return decision

        except Exception as exc:
            logger.warning(
                "check_trigger_from_db_failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return TriggerDecision(False, "db_unavailable", status="DB_UNAVAILABLE")

        finally:
            if _owns_conn and conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def create_adaptation_event(
        self,
        conn,
        *,
        decision: TriggerDecision,
        status: str,
    ) -> str:
        """Insert an adaptation event for a trigger decision."""

        event_id = str(uuid4())
        now = datetime.utcnow()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO adaptation_events
                    (event_id, triggered_at, updated_at, as_of_date, run_id,
                     account_id, policy_name, trigger_type, severity,
                     production_model_id, status, reason, metrics_snapshot,
                     decision_metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    event_id,
                    now,
                    now,
                    self._latest_metric_date(decision.metrics_snapshot),
                    self._latest_run_id(decision.metrics_snapshot),
                    decision.account_id,
                    AdaptationPolicy.PERFORMANCE_TRIGGERED.value,
                    decision.trigger_type,
                    "CRITICAL" if decision.should_trigger else None,
                    decision.production_model_id,
                    status,
                    decision.reason,
                    Json(_jsonable(decision.metrics_snapshot)),
                    Json({"alert_ids": decision.alert_ids}),
                ),
            )
        conn.commit()
        return event_id

    def mark_alerts_for_event(self, conn, alert_ids: list[int], event_id: str) -> int:
        """Mark alerts as consumed by an adaptation event."""

        if not alert_ids:
            return 0
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE alerts
                SET adaptation_event_id = %s,
                    triggered_adaptation = TRUE
                WHERE alert_id = ANY(%s)
                """,
                (event_id, alert_ids),
            )
            count = cur.rowcount
        conn.commit()
        return count

    def complete_adaptation_event(
        self,
        conn,
        *,
        event_id: str,
        status: str,
        candidate_model_id: str | None = None,
        shadow_metrics: dict | None = None,
        decision_metadata: dict | None = None,
    ) -> None:
        """Complete an adaptation event with final promote/reject status."""

        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE adaptation_events
                SET status = %s,
                    candidate_model_id = %s,
                    shadow_metrics = %s,
                    decision_metadata = %s,
                    completed_at = now(),
                    updated_at = now()
                WHERE event_id = %s
                """,
                (
                    status,
                    candidate_model_id,
                    Json(_jsonable(shadow_metrics or {})),
                    Json(_jsonable(decision_metadata or {})),
                    event_id,
                ),
            )
        conn.commit()

    def evaluate_shadow_gate_and_promote(
        self,
        *,
        conn,
        event_id: str,
        current_model_id: str,
        candidate_model_id: str,
        shadow_metrics: dict[str, dict[str, float]],
        min_topk_net_improvement: float = 0.005,
    ) -> bool:
        """Auto promote a candidate only when IC and net-return gates both pass."""

        current = shadow_metrics.get(current_model_id, {})
        candidate = shadow_metrics.get(candidate_model_id, {})
        current_ic = float(current.get("ic", 0.0))
        candidate_ic = float(candidate.get("ic", 0.0))
        current_net = float(current.get("topk_net_return", 0.0))
        candidate_net = float(candidate.get("topk_net_return", 0.0))
        passed = (
            candidate_ic >= current_ic
            and candidate_net - current_net >= min_topk_net_improvement
        )
        status = "PROMOTED" if passed else "REJECTED"
        if passed:
            self._registry.promote_model(candidate_model_id)
        self.complete_adaptation_event(
            conn,
            event_id=event_id,
            status=status,
            candidate_model_id=candidate_model_id,
            shadow_metrics=shadow_metrics,
            decision_metadata={
                "gate": "ic_non_degradation_and_topk_net_improvement",
                "current_ic": current_ic,
                "candidate_ic": candidate_ic,
                "current_topk_net_return": current_net,
                "candidate_topk_net_return": candidate_net,
                "min_topk_net_improvement": min_topk_net_improvement,
            },
        )
        return passed

    @staticmethod
    def _lookup_production_model_id(conn) -> str | None:
        try:
            df = pd.read_sql(
                "SELECT model_id FROM model_registry WHERE status = 'production' "
                "ORDER BY trained_at DESC LIMIT 1",
                conn,
            )
            return None if df.empty else str(df.iloc[0]["model_id"])
        except Exception:
            return None

    @staticmethod
    def _lookup_latest_account_id(conn) -> str | None:
        try:
            df = pd.read_sql(
                "SELECT account_id FROM live_account_snapshots "
                "ORDER BY as_of_date DESC LIMIT 1",
                conn,
            )
            return None if df.empty else str(df.iloc[0]["account_id"])
        except Exception:
            return None

    @staticmethod
    def _load_unhandled_alerts(
        conn,
        *,
        account_id: str | None,
        model_id: str | None,
        limit: int,
    ) -> pd.DataFrame:
        return pd.read_sql(
            """
            SELECT *
            FROM alerts
            WHERE severity = 'CRITICAL'
              AND is_acknowledged = FALSE
              AND adaptation_event_id IS NULL
              AND monitor_type IN ('model', 'strategy')
              AND (%(account_id)s IS NULL OR account_id = %(account_id)s)
              AND (%(model_id)s IS NULL OR model_id = %(model_id)s)
            ORDER BY alert_time DESC
            LIMIT %(limit)s
            """,
            conn,
            params={"account_id": account_id, "model_id": model_id, "limit": limit},
        )

    @staticmethod
    def _latest_recent_event(
        conn,
        *,
        account_id: str | None,
        model_id: str | None,
        cooldown_days: int,
    ) -> dict | None:
        df = pd.read_sql(
            """
            SELECT *
            FROM adaptation_events
            WHERE status IN ('TRIGGERED', 'PROMOTED', 'REJECTED')
              AND (%(account_id)s IS NULL OR account_id = %(account_id)s)
              AND (%(model_id)s IS NULL OR production_model_id = %(model_id)s)
            ORDER BY triggered_at DESC
            LIMIT 1
            """,
            conn,
            params={"account_id": account_id, "model_id": model_id},
        )
        if df.empty:
            return None
        last = pd.Timestamp(df.iloc[0]["triggered_at"]).normalize()
        today = pd.Timestamp(datetime.utcnow()).normalize()
        bdays = len(pd.bdate_range(last, today)) - 1
        return df.iloc[0].to_dict() if bdays < cooldown_days else None

    @staticmethod
    def _infer_trigger_type(reason: str) -> str:
        text = reason.lower()
        if "critical alerts" in text:
            return "critical_alerts"
        if "sharpe" in text:
            return "rolling_sharpe"
        if "rolling ic" in text or " ic " in f" {text} ":
            return "rolling_ic"
        return "none"

    @staticmethod
    def _latest_metric_date(metrics_snapshot: dict) -> object | None:
        for key in ("rolling_ic", "rolling_sharpe"):
            rows = metrics_snapshot.get(key) or []
            if rows:
                return pd.Timestamp(rows[-1]["metric_time"]).date()
        alerts = metrics_snapshot.get("critical_alerts") or []
        if alerts:
            return pd.Timestamp(alerts[0]["alert_time"]).date()
        return None

    @staticmethod
    def _latest_run_id(metrics_snapshot: dict) -> str | None:
        for rows in metrics_snapshot.values():
            for row in rows or []:
                run_id = row.get("run_id")
                if run_id:
                    return str(run_id)
        return None

    def adapt(
        self,
        alpha_panel: pd.DataFrame,
        forward_returns: pd.Series,
        current_time: datetime,
        reason: str,
    ) -> dict[str, float]:
        """Execute performance-triggered adaptation: retrain and compare with current.

        Returns new IC weights if improvement is found, else current weights.
        """
        from src.meta_signal.rule_based import RuleBasedSignalGenerator
        from src.common.metrics import information_coefficient

        generator = RuleBasedSignalGenerator()
        new_weights = generator.compute_ic_weights(alpha_panel, forward_returns)

        model_id = f"perf_{current_time.strftime('%Y%m%d')}_{uuid4().hex[:6]}"

        # Evaluate new weights
        signals = generator.generate_signal(alpha_panel, new_weights)
        if not signals.empty:
            sig = signals.set_index(["security_id", "tradetime"])["signal_score"]
            common = sig.index.intersection(forward_returns.index)
            new_ic = information_coefficient(sig.loc[common], forward_returns.loc[common])
        else:
            new_ic = 0.0

        dates = alpha_panel["tradetime"].agg(["min", "max"])
        self._registry.register_model(
            model_id=model_id,
            model_type="rule_based",
            trained_at=current_time,
            training_window=(dates["min"], dates["max"]),
            features_used=list(new_weights.keys()),
            hyperparams={"trigger_reason": reason},
            holdout_metrics={"ic": float(new_ic) if not pd.isna(new_ic) else 0.0},
        )

        logger.info(
            "performance_triggered_adaptation",
            model_id=model_id,
            policy=AdaptationPolicy.PERFORMANCE_TRIGGERED.value,
            reason=reason,
            new_ic=new_ic,
        )
        return new_weights


def _jsonable(payload: dict) -> dict:
    return json.loads(json.dumps(payload, ensure_ascii=False, default=str))
