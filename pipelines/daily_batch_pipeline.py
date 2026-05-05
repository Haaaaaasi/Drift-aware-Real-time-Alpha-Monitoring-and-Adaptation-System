"""Daily Batch Pipeline: end-to-end from data to signals to paper trading.

This is the core research pipeline. Run after market close:
1. Standardize today's bars
2. Compute alpha features (DolphinDB batch or Python fallback)
3. Generate delayed labels (for past signals)
4. Generate meta signals (rule-based IC-weighted OR XGBoost ML meta model)
5. Construct portfolio targets
6. Apply risk constraints
7. Execute paper trading
8. Run monitoring + persist metrics/alerts to PostgreSQL (graceful failure)
9. Check adaptation triggers

Run modes
---------
Synthetic (no external dependencies):
    python -m pipelines.daily_batch_pipeline --synthetic

TEJ survivorship-correct + rule-based IC signal (default):
    python -m pipelines.daily_batch_pipeline

TEJ survivorship-correct + XGBoost ML signal (MVP v2):
    python -m pipelines.daily_batch_pipeline --signal-method ml_meta

Full real pipeline — DolphinDB alpha_features + TEJ effective_alphas.json（requires Docker）:
    python -m pipelines.daily_batch_pipeline --start 2022-01-01 --end 2024-12-31
    python -m pipelines.daily_batch_pipeline --start 2022-01-01 --end 2024-12-31 --signal-method ml_meta

Full pipeline：使用預計算的 alpha_features（dfs://darams_alpha），不再重算 DolphinDB alpha。
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.alpha_engine.alpha_cache import CACHE_PATH, cache_path_for_data_path, compute_with_cache
from src.alpha_engine.wq101_python import compute_wq101_alphas
from src.common.logging import get_logger, setup_logging
from src.config.constants import (
    DATA_SOURCE_DEFAULT_PATHS,
    DEFAULT_DATA_SOURCE,
    MVP_V1_ALPHA_IDS,
    LABEL_HORIZONS,
    MetaSignalMethod,
)
from src.config.alpha_selection import load_effective_alpha_ids
from src.config.data_sources import assert_yfinance_allowed
from src.execution.order_manager import OrderManager
from src.execution.paper_engine import PaperTradingEngine
from src.labeling.evaluator import Evaluator
from src.labeling.label_generator import LabelGenerator
from src.meta_signal.signal_generator import SignalGenerator
from src.monitoring.alert_manager import AlertManager
from src.monitoring.alpha_monitor import AlphaMonitor
from src.monitoring.data_monitor import DataMonitor
from src.monitoring.model_monitor import ModelMonitor
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


def load_csv_data(
    path: str | Path,
    start: date | None = None,
    end: date | None = None,
    *,
    allow_yfinance: bool = False,
) -> pd.DataFrame:
    """Load real OHLCV data from CSV (yfinance) or parquet (TEJ ingestion output).

    Auto-detects format by file extension. Both sources share the columns
    ``security_id, datetime, open, high, low, close, volume`` and are converted
    to the standardized bars format expected by the pipeline:
        security_id, tradetime, bar_type, open, high, low, close, vol, vwap,
        cap, indclass, is_tradable, missing_flags

    Source-specific notes:
        * CSV (yfinance, ``data/tw_stocks_ohlcv.csv``): 1083 上市檔 + ETF 混入；
          無下市股（survivorship-biased）。
        * Parquet (TEJ, ``data/tw_stocks_tej.parquet``): 1105 上市檔含期間下市；
          ETF/權證/TDR 已過濾。下市股的 OHLCV 保留到下市日，再下一日因為沒
          資料而自然退出 universe，符合「下市日當日報酬計 0、隔日退出」規則。
    """
    path = Path(path)
    assert_yfinance_allowed(path, allow_yfinance=allow_yfinance)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
        df["datetime"] = pd.to_datetime(df["datetime"])
    else:
        df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.rename(columns={"datetime": "tradetime", "volume": "vol"})
    df["tradetime"] = pd.to_datetime(df["tradetime"])

    if start:
        df = df[df["tradetime"].dt.date >= start]
    if end:
        df = df[df["tradetime"].dt.date <= end]

    if df.empty:
        raise ValueError(f"No data in {path} for range {start} → {end}")

    df["bar_type"] = "daily"
    df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0
    df["cap"] = df["close"] * 1_000_000
    df["indclass"] = df["security_id"].apply(lambda s: (hash(s) % 10) + 1)
    df["is_tradable"] = (df["vol"] > 0) & df["close"].notna()
    df["missing_flags"] = 0

    cols = ["security_id", "tradetime", "bar_type", "open", "high", "low",
            "close", "vol", "vwap", "cap", "indclass", "is_tradable", "missing_flags"]
    return df[cols].sort_values(["security_id", "tradetime"]).reset_index(drop=True)


def compute_python_alphas(
    bars: pd.DataFrame,
    alpha_ids: list[str] | None = None,
    use_cache: bool = True,
    cache_path: Path = CACHE_PATH,
) -> pd.DataFrame:
    """Compute WQ101 alpha factors via the Python engine (no DolphinDB needed).

    Args:
        bars: Standardised OHLCV bars from load_csv_data() or generate_synthetic_data().
        alpha_ids: Subset of alpha IDs to return. None = all 101.
        use_cache: If True, read/update the parquet cache at data/alpha_cache/.
                   Set False to force a fresh in-memory computation (e.g. for tests).

    Returns:
        Long-format DataFrame: (security_id, tradetime, alpha_id, alpha_value).
        Values are cross-sectionally z-scored per date, NaN rows dropped.
    """
    if use_cache:
        return compute_with_cache(bars, alpha_ids=alpha_ids, cache_path=cache_path)
    return compute_wq101_alphas(bars, alpha_ids=alpha_ids)


def _load_effective_alphas(
    path: str | Path | None = None,
    *,
    required: bool = True,
) -> list[str] | None:
    """Load the effective alpha list produced by WP2 IC analysis.

    Returns None if the file doesn't exist (pipeline falls back to all alphas).
    """
    target = path or "reports/alpha_ic_analysis/effective_alphas.json"
    alphas = load_effective_alpha_ids(target, required=required)
    if alphas:
        logger.info("effective_alphas_loaded", count=len(alphas), path=str(target))
    return alphas


def _walk_forward_monitor(
    bars: pd.DataFrame,
    alpha_panel: pd.DataFrame,
    fwd_5: pd.Series,
    signals: pd.DataFrame | None,
    portfolio_values: list[dict],
    ref_ratio: float = 0.25,
    checkpoint_every: int = 5,
    window_days: int = 60,
) -> list[dict]:
    """Walk-forward monitoring: emit one metrics snapshot per checkpoint.

    Splits trading dates into a reference window (first `ref_ratio` chronologically)
    and a production window. At each checkpoint in the production window, computes
    the four monitors against the trailing `window_days` of data, with reference
    distributions for PSI / KS / prediction-drift baselines.

    Each metric's `metric_time` is overridden with the checkpoint date so Grafana
    sees a real time-series rather than a single end-of-run snapshot.
    """
    all_dates = sorted(pd.to_datetime(bars["tradetime"]).dt.normalize().unique())
    if len(all_dates) < 30:
        logger.warning("walk_forward_skipped", reason="insufficient_dates", n=len(all_dates))
        return []

    ref_cutoff = all_dates[int(len(all_dates) * ref_ratio)]
    production_dates = [d for d in all_dates if d > ref_cutoff]
    checkpoints = production_dates[checkpoint_every - 1::checkpoint_every]

    # --- Build reference baselines from data ≤ ref_cutoff ---
    ref_bars = bars[bars["tradetime"] <= ref_cutoff]
    ref_close = ref_bars["close"].dropna().to_numpy()

    ref_alpha = alpha_panel[alpha_panel["tradetime"] <= ref_cutoff]
    ref_alpha_values = {
        aid: g["alpha_value"].dropna().to_numpy()
        for aid, g in ref_alpha.groupby("alpha_id")
        if len(g) >= 30
    }

    ref_predictions = None
    if signals is not None and not signals.empty:
        ref_sig = signals[signals["signal_time"] <= ref_cutoff]
        if not ref_sig.empty:
            ref_predictions = ref_sig["signal_score"].dropna().to_numpy()

    logger.info(
        "walk_forward_setup",
        ref_cutoff=str(ref_cutoff.date()),
        n_checkpoints=len(checkpoints),
        ref_close_n=len(ref_close),
        ref_alphas=len(ref_alpha_values),
        ref_predictions_n=len(ref_predictions) if ref_predictions is not None else 0,
    )

    data_mon = DataMonitor()
    alpha_mon = AlphaMonitor()
    model_mon = ModelMonitor()
    strat_mon = StrategyMonitor()

    # Pre-compute per-day portfolio returns for strategy monitor
    pv_df = pd.DataFrame(portfolio_values)
    if not pv_df.empty:
        pv_df["date"] = pd.to_datetime(pv_df["date"]).dt.normalize()
        pv_df = pv_df.sort_values("date").set_index("date")
        pv_df["return"] = pv_df["value"].pct_change()

    all_metrics: list[dict] = []
    for ckpt in checkpoints:
        win_start = ckpt - pd.Timedelta(days=window_days)

        win_bars = bars[(bars["tradetime"] > win_start) & (bars["tradetime"] <= ckpt)]
        win_alpha = alpha_panel[
            (alpha_panel["tradetime"] > win_start) & (alpha_panel["tradetime"] <= ckpt)
        ]
        if win_bars.empty or win_alpha.empty:
            continue

        # forward_returns index: (security_id, signal_time)
        sig_idx = fwd_5.index.get_level_values("signal_time")
        win_fwd = fwd_5[(sig_idx > win_start) & (sig_idx <= ckpt)]

        ckpt_metrics: list[dict] = []
        ckpt_metrics.extend(
            data_mon.run(win_bars, reference_features=ref_close)
        )
        ckpt_metrics.extend(
            alpha_mon.run(win_alpha, win_fwd, reference_alpha_values=ref_alpha_values)
        )

        # Model monitor — only if we have predictions
        if signals is not None and not signals.empty:
            win_sig = signals[
                (signals["signal_time"] > win_start) & (signals["signal_time"] <= ckpt)
            ]
            if not win_sig.empty:
                pred = win_sig.set_index(["security_id", "signal_time"])["signal_score"]
                common = pred.index.intersection(win_fwd.index)
                if len(common) >= 20:
                    ckpt_metrics.extend(
                        model_mon.run(
                            pred.loc[common],
                            win_fwd.loc[common],
                            reference_predictions=ref_predictions,
                        )
                    )

        # Strategy monitor — needs cumulative returns up to ckpt
        if not pv_df.empty:
            recent_returns = pv_df[pv_df.index <= ckpt]["return"].dropna()
            if len(recent_returns) >= 20:
                ckpt_metrics.extend(strat_mon.run(recent_returns))

        # Override metric_time so Grafana gets a real time-series
        for m in ckpt_metrics:
            m["metric_time"] = ckpt
        all_metrics.extend(ckpt_metrics)

    logger.info(
        "walk_forward_complete",
        checkpoints=len(checkpoints),
        total_metrics=len(all_metrics),
        alerts=sum(1 for m in all_metrics if m.get("severity")),
    )
    return all_metrics


def run_backtest(
    start: date,
    end: date,
    use_synthetic: bool = True,
    csv_path: str | Path | None = None,
    signal_method: str = "rule_based",
    effective_alphas: list[str] | None = None,
    symbols: list[str] | None = None,
    allow_yfinance: bool = False,
) -> dict:
    """Run the full backtest pipeline.

    Args:
        start: Backtest start date.
        end: Backtest end date.
        use_synthetic: Use purely synthetic data (no external dependencies).
        csv_path: Path to real OHLCV CSV (from download_tw_stocks.py).
        signal_method: "rule_based" (IC-weighted, default) or "ml_meta" (XGBoost).
        effective_alphas: Explicit alpha subset. In real/DolphinDB mode, controls which
            alphas are loaded from alpha_features. In ml_meta mode, also filters the feature panel; if None, loads from
            reports/alpha_ic_analysis/effective_alphas.json.
    """
    logger.info("pipeline_start", start=str(start), end=str(end))

    # --- Step 1: Load / generate data ---
    if csv_path is not None:
        bars = load_csv_data(csv_path, start=start, end=end, allow_yfinance=allow_yfinance)
        if symbols:
            sym_set = {str(s) for s in symbols}
            bars = bars[bars["security_id"].astype(str).isin(sym_set)].reset_index(drop=True)
            logger.info("symbols_filtered", kept=bars["security_id"].nunique(), requested=len(sym_set))
        alpha_panel = compute_python_alphas(
            bars,
            cache_path=cache_path_for_data_path(csv_path),
        )
        logger.info("csv_data_loaded", bars=len(bars), alphas=len(alpha_panel),
                    symbols=bars["security_id"].nunique())
    elif use_synthetic:
        bars = generate_synthetic_data(start, end)
        alpha_panel = generate_synthetic_alphas(bars)
        logger.info("synthetic_data_generated", bars=len(bars), alphas=len(alpha_panel))
    else:
        # Full real pipeline: DolphinDB alpha computation
        from src.alpha_engine.dolphindb_client import DolphinDBClient
        client = DolphinDBClient()

        # Load bars from DolphinDB
        bars = client.run(f'''
            select * from loadTable("dfs://darams_market", "standardized_bars")
            where tradetime between timestamp({start.strftime("%Y.%m.%d")})
                : timestamp({end.strftime("%Y.%m.%d")})
            and bar_type = "daily"
            and is_tradable = true
        ''')
        if bars is None or bars.empty:
            raise RuntimeError(
                "No data in DolphinDB standardized_bars. "
                "Ingest data first — see scripts/validate_infrastructure.py"
            )
        bars["bar_type"] = "daily"
        bars["is_tradable"] = True
        bars["missing_flags"] = 0
        if symbols:
            sym_set = {str(s) for s in symbols}
            bars = bars[bars["security_id"].astype(str).isin(sym_set)].reset_index(drop=True)
        logger.info("dolphindb_bars_loaded", rows=len(bars),
                     symbols=bars["security_id"].nunique())

        # 從 alpha_features 讀取預計算 alpha，再套用 TEJ effective_alphas.json。
        alpha_ids_to_load = effective_alphas or _load_effective_alphas(required=True)
        ids_filter = ",".join(f'"{a}"' for a in alpha_ids_to_load)
        start_str = start.strftime("%Y.%m.%d")
        end_str = end.strftime("%Y.%m.%d")
        alpha_panel = client.run(
            f'select security_id, tradetime, alpha_id, alpha_value '
            f'from loadTable("dfs://darams_alpha", "alpha_features") '
            f'where tradetime between timestamp({start_str}) : timestamp({end_str}) '
            f'and bar_type = "daily" '
            f'and alpha_id in [{ids_filter}]'
        )
        client.close()

        if alpha_panel is None or len(alpha_panel) == 0:
            raise RuntimeError(
                f"DolphinDB alpha_features 在 {start_str}~{end_str} 無資料；"
                "請先執行：python scripts/backfill_alpha.py --alpha-set all"
            )

        alpha_panel["security_id"] = alpha_panel["security_id"].astype(str)
        alpha_panel["tradetime"] = pd.to_datetime(alpha_panel["tradetime"])
        alpha_panel = alpha_panel.dropna(subset=["alpha_value"])

        # Cross-sectional z-score normalization
        for aid in alpha_panel["alpha_id"].unique():
            mask = alpha_panel["alpha_id"] == aid
            grp = alpha_panel.loc[mask, "alpha_value"].groupby(
                alpha_panel.loc[mask, "tradetime"]
            )
            mu = grp.transform("mean")
            sigma = grp.transform("std").replace(0, np.nan)
            alpha_panel.loc[mask, "alpha_value"] = (
                (alpha_panel.loc[mask, "alpha_value"] - mu) / sigma
            )
        alpha_panel = alpha_panel.dropna(subset=["alpha_value"])

        logger.info("dolphindb_alphas_loaded", rows=len(alpha_panel),
                     alphas=alpha_panel["alpha_id"].nunique(),
                     symbols=alpha_panel["security_id"].nunique())

    # --- Step 2: Generate labels ---
    label_gen = LabelGenerator(horizons=[1, 5, 10], bar_type="daily")
    price_for_labels = bars[["security_id", "tradetime", "close"]].copy()
    labels = label_gen.generate_labels(price_for_labels)

    # Build forward return series for IC weighting (horizon=5)
    fwd_5 = labels[labels["horizon"] == 5].set_index(
        ["security_id", "signal_time"]
    )["forward_return"]

    # --- Step 3: Generate signals ---
    # fwd_5 index uses level name "signal_time"; rename for ML alignment
    fwd_for_ml = fwd_5.copy()
    fwd_for_ml.index = fwd_for_ml.index.set_names(["security_id", "tradetime"])

    ml_model_id: str | None = None

    if signal_method == "ml_meta":
        from src.meta_signal.ml_meta_model import MLMetaModel

        # Load effective alpha subset (WP2 output) if not provided
        eff_alphas = effective_alphas or _load_effective_alphas(required=True)
        ml_panel = (
            alpha_panel[alpha_panel["alpha_id"].isin(eff_alphas)]
            if eff_alphas else alpha_panel
        )

        model = MLMetaModel(feature_columns=eff_alphas)
        train_result = model.train(ml_panel, fwd_for_ml)
        ml_model_id = train_result["model_id"]
        logger.info(
            "ml_meta_signal_trained",
            model_id=ml_model_id,
            ic=train_result["holdout_metrics"].get("ic", 0),
            rank_ic=train_result["holdout_metrics"].get("rank_ic", 0),
            n_features=train_result["n_features"],
        )
        # Register model (graceful failure when DB unavailable)
        model.register_to_registry()

        signals = model.predict(ml_panel)
        signals = signals.rename(columns={"tradetime": "signal_time"})
        signals["method"] = MetaSignalMethod.ML_META.value
        signals["model_version_id"] = ml_model_id
        signals["bar_type"] = "daily"
    else:
        signal_gen = SignalGenerator()
        signals = signal_gen.generate(
            alpha_panel=alpha_panel,
            forward_returns=fwd_5,
        )

    logger.info("signals_generated", count=len(signals), method=signal_method)

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

    # --- Step 9: Walk-forward Monitoring (4 layers, time-series) ---
    alert_mgr = AlertManager()
    all_monitoring_metrics = _walk_forward_monitor(
        bars=bars,
        alpha_panel=alpha_panel,
        fwd_5=fwd_5,
        signals=signals,
        portfolio_values=portfolio_values,
    )
    alert_count = sum(1 for m in all_monitoring_metrics if m.get("severity"))
    logger.info(
        "monitoring_complete",
        total_metrics=len(all_monitoring_metrics),
        alerts=alert_count,
    )

    # Persist to PostgreSQL (graceful failure — pipeline still succeeds without DB)
    persisted, fired = 0, 0
    try:
        persisted = alert_mgr.persist_metrics(all_monitoring_metrics)
        fired = alert_mgr.fire_alerts(all_monitoring_metrics)
        logger.info("monitoring_persisted", metrics=persisted, alerts_fired=fired)
    except Exception as exc:
        logger.warning("monitoring_db_unavailable", error=str(exc))

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
        "metrics_persisted": persisted,
        "signal_method": signal_method,
        "ml_model_id": ml_model_id,
    }

    logger.info("pipeline_complete", **summary)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="DARAMS Daily Batch Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pure synthetic data (no dependencies):
  python -m pipelines.daily_batch_pipeline --synthetic

  # TEJ survivorship-correct + rule-based IC signal (default):
  python -m pipelines.daily_batch_pipeline

  # TEJ survivorship-correct + XGBoost ML signal (MVP v2):
  python -m pipelines.daily_batch_pipeline --signal-method ml_meta

  # Docker/DolphinDB real mode:
  python -m pipelines.daily_batch_pipeline --real

  # Date filter:
  python -m pipelines.daily_batch_pipeline --start 2023-01-01 --end 2024-06-30
        """,
    )
    parser.add_argument("--start", type=str, default="2022-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--synthetic", action="store_true", default=False,
                        help="Use purely synthetic data (no external dependencies)")
    parser.add_argument("--real", action="store_true", default=False,
                        help="Use DolphinDB real mode instead of the default TEJ offline path")
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
                        help="Path to real OHLCV CSV/parquet; provided path overrides --data-source")
    parser.add_argument(
        "--allow-yfinance",
        action="store_true",
        help="明確允許使用已知污染的 yfinance CSV（僅 demo/反例使用）",
    )
    parser.add_argument(
        "--signal-method",
        choices=["rule_based", "ml_meta"],
        default="rule_based",
        help="Signal generation method: rule_based (IC-weighted) or ml_meta (XGBoost)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Whitelist of security IDs to include (e.g. --symbols 2330 2317 2454)",
    )
    args = parser.parse_args()

    use_synthetic = args.synthetic
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    csv_path = None if args.real or args.synthetic else (
        args.csv or DATA_SOURCE_DEFAULT_PATHS[args.data_source]
    )
    result = run_backtest(
        start, end,
        use_synthetic=use_synthetic,
        csv_path=csv_path,
        signal_method=args.signal_method,
        symbols=args.symbols,
        allow_yfinance=args.allow_yfinance,
    )

    mode = "Offline+Python-Alpha" if csv_path else ("Synthetic" if use_synthetic else "Real")
    sig_label = result.get("signal_method", "rule_based")
    print(f"\n=== DARAMS Backtest Summary [{mode} / {sig_label}] ===")
    print(f"Period: {result['start']} to {result['end']}")
    print(f"Rebalances: {result['n_rebalances']}")
    print(f"Total orders: {result['n_orders']}")
    print(f"Alphas analyzed: {result['alpha_count']}")
    print(f"Monitoring metrics: {result['monitoring_metrics']}")
    print(f"Alerts triggered: {result['alerts_triggered']}")
    if result.get("ml_model_id"):
        print(f"ML model id: {result['ml_model_id']}")
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
