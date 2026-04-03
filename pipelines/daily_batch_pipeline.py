"""MVP v1 — Daily Batch Pipeline: end-to-end from data to signals to paper trading.

This is the core research pipeline. Run after market close:
1. Standardize today's bars
2. Compute alpha features (DolphinDB batch)
3. Generate delayed labels (for past signals)
4. Generate meta signals (IC-weighted composite)
5. Construct portfolio targets
6. Apply risk constraints
7. Execute paper trading
8. Run monitoring
9. Check adaptation triggers

Usage:
    python -m pipelines.daily_batch_pipeline --start 2023-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from src.common.logging import get_logger, setup_logging
from src.config.constants import MVP_V1_ALPHA_IDS, LABEL_HORIZONS
from src.execution.order_manager import OrderManager
from src.execution.paper_engine import PaperTradingEngine
from src.labeling.evaluator import Evaluator
from src.labeling.label_generator import LabelGenerator
from src.meta_signal.signal_generator import SignalGenerator
from src.monitoring.alert_manager import AlertManager
from src.monitoring.alpha_monitor import AlphaMonitor
from src.monitoring.data_monitor import DataMonitor
from src.monitoring.strategy_monitor import StrategyMonitor
from src.portfolio.constructor import PortfolioConstructor
from src.risk.risk_manager import RiskManager

setup_logging()
logger = get_logger("daily_batch_pipeline")


def generate_synthetic_data(
    start: date, end: date, n_symbols: int = 20
) -> pd.DataFrame:
    """Generate synthetic OHLCV data for pipeline testing when no real data is available."""
    np.random.seed(42)
    symbols = [f"TW{str(i).zfill(4)}" for i in range(1, n_symbols + 1)]
    dates = pd.bdate_range(start=start, end=end, freq="B")

    rows = []
    for sym in symbols:
        price = 100.0 + np.random.randn() * 20
        for d in dates:
            ret = np.random.randn() * 0.02
            price *= 1 + ret
            o = price * (1 + np.random.randn() * 0.005)
            h = max(o, price) * (1 + abs(np.random.randn()) * 0.005)
            l = min(o, price) * (1 - abs(np.random.randn()) * 0.005)
            vol = max(1000, int(np.random.exponential(500000)))
            rows.append({
                "security_id": sym,
                "tradetime": d,
                "bar_type": "daily",
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(l, 2),
                "close": round(price, 2),
                "vol": float(vol),
                "vwap": round((h + l + price) / 3, 2),
                "cap": round(price * 1e6, 2),
                "indclass": (hash(sym) % 5) + 1,
                "is_tradable": True,
                "missing_flags": 0,
            })

    return pd.DataFrame(rows)


def generate_synthetic_alphas(bars: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic alpha values (mock DolphinDB output for testing)."""
    np.random.seed(123)
    rows = []
    for (sec, dt), group in bars.groupby(["security_id", "tradetime"]):
        for alpha_id in MVP_V1_ALPHA_IDS:
            rows.append({
                "security_id": sec,
                "tradetime": dt,
                "alpha_id": alpha_id,
                "alpha_value": np.random.randn(),
            })
    return pd.DataFrame(rows)


def run_backtest(start: date, end: date, use_synthetic: bool = True) -> dict:
    """Run the full MVP v1 backtest pipeline."""
    logger.info("pipeline_start", start=str(start), end=str(end))

    # --- Step 1: Load / generate data ---
    if use_synthetic:
        bars = generate_synthetic_data(start, end)
        alpha_panel = generate_synthetic_alphas(bars)
        logger.info("synthetic_data_generated", bars=len(bars), alphas=len(alpha_panel))
    else:
        # TODO: load from DolphinDB
        raise NotImplementedError("Real data pipeline requires DolphinDB connection")

    # --- Step 2: Generate labels ---
    label_gen = LabelGenerator(horizons=[1, 5, 10], bar_type="daily")
    price_for_labels = bars[["security_id", "tradetime", "close"]].copy()
    labels = label_gen.generate_labels(price_for_labels)

    # Build forward return series for IC weighting (horizon=5)
    fwd_5 = labels[labels["horizon"] == 5].set_index(
        ["security_id", "signal_time"]
    )["forward_return"]

    # --- Step 3: Compute IC weights and generate signals ---
    signal_gen = SignalGenerator()
    signals = signal_gen.generate(
        alpha_panel=alpha_panel,
        forward_returns=fwd_5,
    )

    logger.info("signals_generated", count=len(signals))

    # --- Step 4-6: Portfolio → Risk → Execution ---
    portfolio = PortfolioConstructor(method="equal_weight_topk", top_k=10, long_only=True)
    risk_mgr = RiskManager(max_position_weight=0.10, max_gross_exposure=1.0)
    engine = PaperTradingEngine(initial_capital=10_000_000.0, slippage_bps=5.0)
    order_mgr = OrderManager()

    all_orders = []
    all_fills = []
    portfolio_values = []

    rebalance_dates = signals["signal_time"].unique()

    for rb_time in sorted(rebalance_dates):
        day_signals = signals[signals["signal_time"] == rb_time]
        day_bars = bars[bars["tradetime"] == rb_time][["security_id", "close"]]

        if day_bars.empty:
            continue

        targets = portfolio.construct(day_signals)
        if targets.empty:
            continue

        adj_targets = risk_mgr.apply_constraints(targets)
        orders, fills = engine.execute_rebalance(adj_targets, day_bars, rb_time)

        all_orders.append(orders)
        all_fills.append(fills)
        portfolio_values.append({
            "date": rb_time,
            "value": engine.portfolio_value,
        })

    # --- Step 7: Compute portfolio returns ---
    pv_df = pd.DataFrame(portfolio_values)
    if len(pv_df) > 1:
        pv_df["return"] = pv_df["value"].pct_change()
        portfolio_returns = pv_df["return"].dropna()
    else:
        portfolio_returns = pd.Series(dtype=float)

    # --- Step 8: Evaluation ---
    evaluator = Evaluator()
    strategy_metrics = evaluator.evaluate_strategy(portfolio_returns)
    alpha_metrics = evaluator.evaluate_all_alphas(alpha_panel, fwd_5)

    logger.info("strategy_metrics", **strategy_metrics)

    # --- Step 9: Monitoring ---
    data_mon = DataMonitor()
    alpha_mon = AlphaMonitor()
    strat_mon = StrategyMonitor()
    alert_mgr = AlertManager()

    all_monitoring_metrics = []
    all_monitoring_metrics.extend(data_mon.run(bars))
    all_monitoring_metrics.extend(alpha_mon.run(alpha_panel, fwd_5))
    all_monitoring_metrics.extend(strat_mon.run(portfolio_returns))

    alert_count = sum(1 for m in all_monitoring_metrics if m.get("severity"))
    logger.info(
        "monitoring_complete",
        total_metrics=len(all_monitoring_metrics),
        alerts=alert_count,
    )

    # --- Summary ---
    summary = {
        "start": str(start),
        "end": str(end),
        "n_rebalances": len(rebalance_dates),
        "n_orders": sum(len(o) for o in all_orders if not o.empty),
        "strategy_metrics": strategy_metrics,
        "alpha_count": len(alpha_metrics),
        "monitoring_metrics": len(all_monitoring_metrics),
        "alerts_triggered": alert_count,
    }

    logger.info("pipeline_complete", **summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="DARAMS Daily Batch Pipeline")
    parser.add_argument("--start", type=str, default="2023-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--synthetic", action="store_true", default=True)
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    result = run_backtest(start, end, use_synthetic=args.synthetic)

    print("\n=== DARAMS MVP v1 Backtest Summary ===")
    print(f"Period: {result['start']} to {result['end']}")
    print(f"Rebalances: {result['n_rebalances']}")
    print(f"Total orders: {result['n_orders']}")
    print(f"Alphas analyzed: {result['alpha_count']}")
    print(f"Monitoring metrics: {result['monitoring_metrics']}")
    print(f"Alerts triggered: {result['alerts_triggered']}")
    sm = result["strategy_metrics"]
    print(f"\n--- Strategy Performance ---")
    print(f"Total Return:      {sm['total_return']:.4f}")
    print(f"Annualized Return: {sm['annualized_return']:.4f}")
    print(f"Sharpe Ratio:      {sm['sharpe']:.4f}")
    print(f"Max Drawdown:      {sm['max_drawdown']:.4f}")
    print(f"Win Rate:          {sm['win_rate']:.4f}")
    print(f"Profit Factor:     {sm['profit_factor']:.4f}")


if __name__ == "__main__":
    main()
