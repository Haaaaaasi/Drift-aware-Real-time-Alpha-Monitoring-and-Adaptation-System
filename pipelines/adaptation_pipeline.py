"""Adaptation Pipeline — Check triggers and execute adaptation policies.

Implements the full adaptation loop:
    1. Load recent monitoring metrics from PostgreSQL
    2. Check Policy 1 (scheduled retrain) and Policy 2 (performance-triggered)
    3. If triggered: retrain XGBoost meta model on recent data
    4. Shadow-evaluate retrained model against current production model
    5. Promote if improvement exceeds threshold

Can also run in offline TEJ/parquet mode for testing without Docker infrastructure.

Usage:
    # Offline TEJ mode (default, no Docker needed):
    python -m pipelines.adaptation_pipeline

    # With Docker infrastructure (reads from PostgreSQL):
    python -m pipelines.adaptation_pipeline --online
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.adaptation.performance_trigger import PerformanceTriggeredAdapter
from src.adaptation.scheduler import ScheduledRetrainer
from src.adaptation.shadow_evaluator import ShadowEvaluator
from src.alpha_engine.alpha_cache import cache_path_for_data_path
from src.common.logging import get_logger, setup_logging
from src.common.metrics import information_coefficient
from src.config.alpha_selection import load_effective_alpha_ids
from src.config.constants import DATA_SOURCE_DEFAULT_PATHS, DEFAULT_DATA_SOURCE
from src.meta_signal.ml_meta_model import MLMetaModel
from src.meta_signal.rule_based import RuleBasedSignalGenerator

setup_logging()
logger = get_logger("adaptation_pipeline")


def run_adaptation_offline(
    csv_path: str | Path,
    start: date = date(2022, 1, 1),
    end: date = date(2024, 12, 31),
    allow_yfinance: bool = False,
) -> dict:
    """Run the full adaptation loop in offline mode using CSV data.

    This is the testable, self-contained version that doesn't require
    PostgreSQL / Redis / DolphinDB. It demonstrates the complete chain:
        monitor → trigger → retrain → shadow eval → promote decision
    """
    from pipelines.daily_batch_pipeline import compute_python_alphas, load_csv_data
    from src.labeling.label_generator import LabelGenerator
    from src.monitoring.alpha_monitor import AlphaMonitor
    from src.monitoring.strategy_monitor import StrategyMonitor

    # Load effective alphas from WP2 if available
    effective_alphas = load_effective_alpha_ids(required=True)

    # --- Step 1: Load data ---
    logger.info("adaptation_loading_data", csv=str(csv_path))
    bars = load_csv_data(csv_path, start=start, end=end, allow_yfinance=allow_yfinance)
    alpha_panel = compute_python_alphas(
        bars,
        cache_path=cache_path_for_data_path(csv_path),
    )
    if effective_alphas:
        alpha_panel = alpha_panel[alpha_panel["alpha_id"].isin(effective_alphas)]

    labels = LabelGenerator(horizons=[5], bar_type="daily").generate_labels(
        bars[["security_id", "tradetime", "close"]]
    )
    fwd = (
        labels[labels["horizon"] == 5]
        .set_index(["security_id", "signal_time"])["forward_return"]
    )
    fwd.index = fwd.index.set_names(["security_id", "tradetime"])

    # --- Step 2: Split into reference / recent windows ---
    dates = np.sort(alpha_panel["tradetime"].unique())
    split_idx = int(len(dates) * 0.6)
    split_date = dates[split_idx]

    ref_panel = alpha_panel[alpha_panel["tradetime"] <= split_date]
    recent_panel = alpha_panel[alpha_panel["tradetime"] > split_date]
    ref_fwd = fwd[fwd.index.get_level_values("tradetime") <= split_date]
    recent_fwd = fwd[fwd.index.get_level_values("tradetime") > split_date]

    alpha_ids = alpha_panel["alpha_id"].unique().tolist()
    logger.info("adaptation_data_split",
                ref_dates=split_idx, recent_dates=len(dates) - split_idx,
                split=str(split_date))

    # --- Step 3: Train current production model on reference data ---
    current_model = MLMetaModel(feature_columns=alpha_ids)
    current_result = current_model.train(ref_panel, ref_fwd)
    logger.info("current_model_trained",
                model_id=current_result["model_id"],
                ic=current_result["holdout_metrics"].get("ic", 0))
    # Register to model_registry (graceful failure when DB unavailable)
    current_model.register_to_registry()

    # Generate current model predictions on recent data
    current_pred = current_model.predict(recent_panel)
    current_scores = current_pred.set_index(["security_id", "tradetime"])["signal_score"]

    # --- Step 4: Monitor recent performance ---
    # Alpha monitor: check for PSI drift
    ref_dists = {}
    for aid in alpha_ids:
        ref_dists[aid] = (
            ref_panel[ref_panel["alpha_id"] == aid]["alpha_value"]
            .dropna().values
        )
    alpha_mon = AlphaMonitor(psi_warn=0.10, psi_crit=0.25)
    alpha_metrics = alpha_mon.run(
        recent_panel, recent_fwd, reference_alpha_values=ref_dists,
    )
    critical_count = sum(1 for m in alpha_metrics if m.get("severity") == "CRITICAL")

    # Compute rolling IC on recent predictions
    common_idx = current_scores.index.intersection(recent_fwd.index)
    if len(common_idx) >= 50:
        rolling_ic_vals = []
        sorted_idx = common_idx.sortlevel("tradetime")[0]
        chunk_size = max(len(sorted_idx) // 10, 20)
        for i in range(0, len(sorted_idx) - chunk_size, chunk_size):
            chunk = sorted_idx[i:i + chunk_size]
            ic = information_coefficient(
                current_scores.loc[chunk], recent_fwd.loc[chunk]
            )
            rolling_ic_vals.append(ic if not np.isnan(ic) else 0.0)
        rolling_ic = pd.Series(rolling_ic_vals)
    else:
        rolling_ic = pd.Series([0.0])

    logger.info("monitoring_complete",
                alpha_metrics=len(alpha_metrics),
                critical_alerts=critical_count,
                rolling_ic_mean=rolling_ic.mean())

    # --- Step 5: Check adaptation triggers ---
    # Policy 1: Scheduled (30-day cadence for offline demo)
    scheduler = ScheduledRetrainer(retrain_interval_days=30)
    scheduled_due = scheduler.should_retrain(datetime.utcnow())

    # Policy 2: Performance-triggered
    adapter = PerformanceTriggeredAdapter(
        ic_threshold=0.02,
        ic_consecutive_days=3,
        sharpe_threshold=0.0,
        sharpe_consecutive_days=10,
        critical_alert_limit=3,
    )
    triggered, reason = adapter.check_trigger(
        rolling_ic,
        pd.Series(dtype=float),   # offline 無 portfolio returns，Sharpe 條件略過
        critical_count,
    )

    logger.info("trigger_check",
                scheduled_due=scheduled_due,
                performance_triggered=triggered,
                reason=reason)

    # --- Step 6: Retrain if triggered ---
    retrain_result = None
    shadow_result = None
    promote_decision = None

    if triggered or scheduled_due:
        logger.info("adaptation_executing", reason=reason or "scheduled")

        # Retrain on full data (ref + recent)
        new_model = MLMetaModel(feature_columns=alpha_ids)
        retrain_result = new_model.train(alpha_panel, fwd)
        logger.info("new_model_trained",
                    model_id=retrain_result["model_id"],
                    ic=retrain_result["holdout_metrics"].get("ic", 0))
        new_model.register_to_registry()

        # Shadow evaluate
        new_pred = new_model.predict(recent_panel)
        evaluator = ShadowEvaluator(min_improvement_ic=0.005)
        shadow_result = evaluator.evaluate_candidates(
            {
                current_model.model_id: current_pred,
                new_model.model_id: new_pred,
            },
            recent_fwd,
        )

        promote_decision = evaluator.select_best(
            shadow_result, current_model_id=current_model.model_id
        )

        if promote_decision == new_model.model_id:
            logger.info("promote_new_model",
                        model_id=new_model.model_id,
                        old_ic=shadow_result[current_model.model_id].get("ic", 0),
                        new_ic=shadow_result[new_model.model_id].get("ic", 0))
            try:
                from src.adaptation.model_registry import ModelRegistryManager
                ModelRegistryManager().promote_model(promote_decision)
            except Exception as exc:
                logger.warning("promote_model_db_unavailable", error=str(exc))
        else:
            logger.info("keep_current_model",
                        reason="New model does not improve sufficiently")

    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "data_range": f"{start} -> {end}",
        "split_date": str(split_date),
        "current_model_id": current_model.model_id,
        "current_holdout_ic": current_result["holdout_metrics"].get("ic", 0),
        "monitoring": {
            "alpha_metrics": len(alpha_metrics),
            "critical_alerts": critical_count,
            "rolling_ic_mean": float(rolling_ic.mean()),
        },
        "triggers": {
            "scheduled_due": scheduled_due,
            "performance_triggered": triggered,
            "trigger_reason": reason,
        },
        "adaptation": {
            "retrained": retrain_result is not None,
            "new_model_id": retrain_result["model_id"] if retrain_result else None,
            "new_holdout_ic": retrain_result["holdout_metrics"].get("ic", 0) if retrain_result else None,
            "shadow_result": {
                k: {mk: round(mv, 4) for mk, mv in v.items()}
                for k, v in (shadow_result or {}).items()
            },
            "promote_decision": promote_decision,
        },
    }

    logger.info("adaptation_pipeline_complete", **{
        k: v for k, v in summary.items() if k not in ("adaptation",)
    })
    return summary


def run_adaptation_online() -> dict:
    """Run adaptation using live PostgreSQL monitoring data.

    Requires Docker infrastructure (PostgreSQL + Redis) to be running.
    """
    from src.adaptation.model_registry import ModelRegistryManager
    from src.monitoring.alert_manager import AlertManager

    now = datetime.utcnow()
    registry = ModelRegistryManager()
    alert_mgr = AlertManager()

    # Policy 1: Scheduled retrain
    scheduler = ScheduledRetrainer(retrain_interval_days=7)
    scheduled_due = scheduler.should_retrain(now)
    if scheduled_due:
        logger.info("scheduled_retrain_due")

    # Policy 2: Performance-triggered（DB-driven 路徑，DB 不可用時自動 fallback）
    adapter = PerformanceTriggeredAdapter()
    triggered, reason = adapter.check_trigger_from_db()

    # critical_count 僅用於 summary 記錄，獨立查詢，失敗 fallback -1
    try:
        critical_count = alert_mgr.get_unacknowledged_critical_count()
    except Exception:
        critical_count = -1

    if triggered:
        logger.info("performance_trigger_fired", reason=reason)

    prod_model = registry.get_production_model()
    summary = {
        "timestamp": now.isoformat(),
        "scheduled_retrain_due": scheduled_due,
        "performance_trigger": triggered,
        "trigger_reason": reason,
        "critical_alerts": critical_count,
        "production_model": prod_model.get("model_id") if prod_model else None,
    }
    logger.info("adaptation_pipeline_complete", **summary)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="DARAMS Adaptation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Offline TEJ mode (default, no Docker needed):
  python -m pipelines.adaptation_pipeline

  # Online mode (requires Docker infrastructure):
  python -m pipelines.adaptation_pipeline --online
        """,
    )
    parser.add_argument("--online", action="store_true",
                        help="Use PostgreSQL/Redis/DolphinDB online mode")
    parser.add_argument(
        "--data-source",
        choices=["csv", "tej"],
        default=DEFAULT_DATA_SOURCE,
        help=(
            "tej = TEJ survivorship-correct parquet（預設）; "
            "csv = yfinance demo 路徑（已知 8476 資料污染）"
        ),
    )
    parser.add_argument("--csv", type=str, default=None, metavar="PATH",
                        help="Run offline with an explicit OHLCV CSV/parquet path")
    parser.add_argument(
        "--allow-yfinance",
        action="store_true",
        help="明確允許使用已知污染的 yfinance CSV（僅 demo/反例使用）",
    )
    parser.add_argument("--start", type=str, default="2022-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    args = parser.parse_args()

    if not args.online:
        data_path = args.csv or DATA_SOURCE_DEFAULT_PATHS[args.data_source]
        result = run_adaptation_offline(
            data_path,
            start=date.fromisoformat(args.start),
            end=date.fromisoformat(args.end),
            allow_yfinance=args.allow_yfinance,
        )
        print(f"\n=== DARAMS Adaptation Pipeline [Offline] ===")
        print(f"Data: {result['data_range']}")
        print(f"Split: {result['split_date']}")
        print(f"\n--- Monitoring ---")
        mon = result["monitoring"]
        print(f"Alpha metrics: {mon['alpha_metrics']}")
        print(f"Critical alerts: {mon['critical_alerts']}")
        print(f"Rolling IC mean: {mon['rolling_ic_mean']:.4f}")
        print(f"\n--- Triggers ---")
        trig = result["triggers"]
        print(f"Scheduled due: {trig['scheduled_due']}")
        print(f"Performance triggered: {trig['performance_triggered']}")
        if trig["trigger_reason"]:
            print(f"Reason: {trig['trigger_reason']}")
        print(f"\n--- Adaptation ---")
        adapt = result["adaptation"]
        print(f"Retrained: {adapt['retrained']}")
        if adapt["retrained"]:
            print(f"New model: {adapt['new_model_id']}")
            print(f"New holdout IC: {adapt['new_holdout_ic']:.4f}")
            print(f"Promote decision: {adapt['promote_decision'] or 'Keep current'}")
            if adapt["shadow_result"]:
                print(f"Shadow evaluation:")
                for mid, metrics in adapt["shadow_result"].items():
                    print(f"  {mid}: {metrics}")
    else:
        result = run_adaptation_online()
        print(f"Adaptation check complete: {result}")


if __name__ == "__main__":
    main()
