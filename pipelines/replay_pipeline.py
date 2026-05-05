"""Replay Pipeline — Validate streaming alpha consistency against batch mode.

WP7: Demonstrates and validates the DolphinDB streamEngineParser integration
by replaying historical TEJ/parquet data bar-by-bar and comparing streaming output
against single-pass batch computation.

Core invariant being tested
----------------------------
For any rolling-window alpha function f and date D:
    f(data[0..D])[-1]  ==  f(data[0..T])  evaluated at row D
i.e. incremental/streaming computation equals batch computation.

Two modes
---------
Offline (--csv):
    Pure-Python simulation.  Replays data on an expanding window at N
    checkpoint dates and verifies alpha values match the single-pass batch.
    No Docker or DolphinDB required.

Online (default):
    Pushes bars to the live DolphinDB ``standardized_stream``, subscribes
    to ``alpha_output_stream``, and compares with DolphinDB batch output.
    Requires ``docker-compose up -d``.

Usage
-----
    # Offline mode (no Docker needed):
    python -m pipelines.replay_pipeline --csv data/tw_stocks_tej.parquet

    # Online mode (requires Docker + DolphinDB):
    python -m pipelines.replay_pipeline
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.logging import get_logger, setup_logging

setup_logging()
logger = get_logger("replay_pipeline")


# ---------------------------------------------------------------------------
# Offline: Python-side streaming simulation
# ---------------------------------------------------------------------------

def _expanding_alpha_panel(
    bars: pd.DataFrame,
    checkpoint_dates: list,
) -> pd.DataFrame:
    """For each checkpoint date, compute alphas on expanding history and keep
    only the last row (most recent date).  Returns a long-format panel with
    the same columns as the batch panel."""
    from pipelines.daily_batch_pipeline import compute_python_alphas

    rows = []
    for ckpt in checkpoint_dates:
        history = bars[bars["tradetime"] <= ckpt]
        if history.empty:
            continue
        panel = compute_python_alphas(history, use_cache=False)
        latest = panel[panel["tradetime"] == panel["tradetime"].max()]
        rows.append(latest)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _compute_match_rates(
    batch: pd.DataFrame,
    streaming: pd.DataFrame,
    eval_dates: list,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Per-alpha match rate (fraction of (security, date) pairs whose values
    differ by less than ``tol``)."""
    alpha_ids = batch["alpha_id"].unique().tolist()
    match_rates: dict[str, float] = {}

    # Normalise to pandas Timestamps for reliable comparison
    eval_ts = set(pd.Timestamp(d) for d in eval_dates)

    for aid in alpha_ids:
        b = (
            batch[
                (batch["alpha_id"] == aid)
                & batch["tradetime"].apply(pd.Timestamp).isin(eval_ts)
            ]
            .set_index(["security_id", "tradetime"])["alpha_value"]
            .dropna()
        )
        s = (
            streaming[streaming["alpha_id"] == aid]
            .set_index(["security_id", "tradetime"])["alpha_value"]
            .dropna()
        )
        # Normalise index dtypes to Timestamp so intersection works
        b.index = pd.MultiIndex.from_arrays([
            b.index.get_level_values("security_id"),
            pd.to_datetime(b.index.get_level_values("tradetime")),
        ], names=["security_id", "tradetime"])
        s.index = pd.MultiIndex.from_arrays([
            s.index.get_level_values("security_id"),
            pd.to_datetime(s.index.get_level_values("tradetime")),
        ], names=["security_id", "tradetime"])

        common = b.index.intersection(s.index)
        if len(common) == 0:
            match_rates[aid] = float("nan")
            continue
        abs_diff = (b.loc[common] - s.loc[common]).abs()
        match_rates[aid] = float((abs_diff < tol).mean())

    return match_rates


def run_replay_offline(
    csv_path: str | Path,
    start: date = date(2022, 1, 1),
    end: date = date(2024, 12, 31),
    n_checkpoints: int = 10,
    warm_up_days: int = 30,
    allow_yfinance: bool = False,
) -> dict:
    """Run streaming replay simulation in offline mode.

    Parameters
    ----------
    csv_path:       Path to OHLCV CSV file.
    start / end:    Date range for the backtest.
    n_checkpoints:  How many expanding-window checkpoints to evaluate.
                    Higher values are more thorough but slower.
    warm_up_days:   Skip the first N dates to allow rolling windows to stabilise.
    """
    from pipelines.daily_batch_pipeline import compute_python_alphas, load_csv_data

    logger.info("replay_loading_data", csv=str(csv_path))
    bars = load_csv_data(csv_path, start=start, end=end, allow_yfinance=allow_yfinance)

    dates = sorted(bars["tradetime"].unique())
    if len(dates) <= warm_up_days:
        raise ValueError(
            f"Need more than {warm_up_days} trading dates; got {len(dates)}"
        )

    # Single-pass batch reference (no cache: replay uses synthetic/small data)
    logger.info("replay_batch_compute_start")
    batch_panel = compute_python_alphas(bars, use_cache=False)
    logger.info("replay_batch_compute_done", rows=len(batch_panel))

    # Select evenly-spaced checkpoints after warm-up
    eval_pool = dates[warm_up_days:]
    step = max(1, len(eval_pool) // n_checkpoints)
    checkpoint_dates = eval_pool[::step][:n_checkpoints]

    logger.info(
        "replay_simulation_start",
        checkpoints=len(checkpoint_dates),
        first=str(checkpoint_dates[0]),
        last=str(checkpoint_dates[-1]),
    )
    streaming_panel = _expanding_alpha_panel(bars, checkpoint_dates)

    match_rates = _compute_match_rates(batch_panel, streaming_panel, checkpoint_dates)

    valid_rates = [r for r in match_rates.values() if not np.isnan(r)]
    overall = float(np.mean(valid_rates)) if valid_rates else 0.0

    n_perfect = sum(1 for r in valid_rates if r >= 1.0 - 1e-9)

    summary = {
        "mode": "offline",
        "data_range": f"{start} -> {end}",
        "total_dates": len(dates),
        "warm_up_days": warm_up_days,
        "checkpoints_evaluated": len(checkpoint_dates),
        "batch_rows": len(batch_panel),
        "streaming_rows": len(streaming_panel),
        "alphas_checked": len(match_rates),
        "overall_match_rate": overall,
        "perfect_match_alphas": n_perfect,
        "per_alpha_match_rate": {k: round(v, 4) for k, v in match_rates.items()},
    }
    logger.info("replay_complete", **{k: v for k, v in summary.items()
                                      if k != "per_alpha_match_rate"})
    return summary


# ---------------------------------------------------------------------------
# Online: DolphinDB streamEngineParser validation
# ---------------------------------------------------------------------------

def run_replay_online(
    csv_path: str | Path | None = None,
    start: date = date(2022, 1, 1),
    end: date = date(2024, 12, 31),
    n_validate_dates: int = 5,
    allow_yfinance: bool = False,
) -> dict:
    """Replay bars through DolphinDB streaming engine and validate output.

    Requires Docker infrastructure (DolphinDB) to be running.
    If csv_path is provided, uses CSV data; otherwise reads from DolphinDB
    standardized_bars table.
    """
    from src.alpha_engine.batch_compute import BatchAlphaComputer
    from src.alpha_engine.stream_compute import StreamAlphaComputer

    computer = StreamAlphaComputer()
    try:
        computer.setup_engine()
        computer.subscribe_output()

        # Load bars to replay
        if csv_path is not None:
            from pipelines.daily_batch_pipeline import load_csv_data
            bars = load_csv_data(
                csv_path,
                start=start,
                end=end,
                allow_yfinance=allow_yfinance,
            )
        else:
            batch_computer = BatchAlphaComputer()
            bars = batch_computer.load_standardized_bars(
                start=str(start), end=str(end)
            )

        dates = sorted(bars["tradetime"].unique())

        # Replay all bars through DolphinDB stream engine
        streaming_output = computer.replay_dataframe(bars, delay_ms=0)

        # Get DolphinDB batch output for comparison
        batch_computer = BatchAlphaComputer()
        batch_df = batch_computer.compute(
            start=str(start),
            end=str(end),
        )

        # Compare on n_validate_dates spot-check dates
        if len(dates) > n_validate_dates:
            step = len(dates) // n_validate_dates
            validate_dates = dates[step::step][:n_validate_dates]
        else:
            validate_dates = dates

        streaming_df = pd.DataFrame(streaming_output) if streaming_output else pd.DataFrame()
        if not streaming_df.empty and not batch_df.empty:
            match_rates = _compute_match_rates(batch_df, streaming_df, validate_dates)
            valid = [r for r in match_rates.values() if not np.isnan(r)]
            overall = float(np.mean(valid)) if valid else 0.0
        else:
            match_rates = {}
            overall = 0.0

        summary = {
            "mode": "online",
            "data_range": f"{start} -> {end}",
            "dates_replayed": len(dates),
            "streaming_output_rows": len(streaming_output),
            "batch_output_rows": len(batch_df) if not batch_df.empty else 0,
            "validate_dates": n_validate_dates,
            "overall_match_rate": overall,
            "per_alpha_match_rate": {k: round(v, 4) for k, v in match_rates.items()},
        }
        logger.info("replay_online_complete", **{
            k: v for k, v in summary.items() if k != "per_alpha_match_rate"
        })
        return summary

    finally:
        computer.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DARAMS Replay Pipeline — validate streaming vs batch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Offline mode (no Docker needed):
  python -m pipelines.replay_pipeline --csv data/tw_stocks_tej.parquet

  # Offline with more checkpoints for a thorough test:
  python -m pipelines.replay_pipeline --csv data/tw_stocks_tej.parquet --checkpoints 20

  # Online mode (requires docker-compose up -d):
  python -m pipelines.replay_pipeline
        """,
    )
    parser.add_argument("--csv", type=str, default=None, metavar="PATH",
                        help="Run in offline mode with OHLCV CSV/parquet data")
    parser.add_argument(
        "--allow-yfinance",
        action="store_true",
        help="明確允許使用已知污染的 yfinance CSV（僅 demo/反例使用）",
    )
    parser.add_argument("--start", type=str, default="2022-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--checkpoints", type=int, default=10,
                        help="Number of expanding-window checkpoints (offline mode)")
    parser.add_argument("--warm-up", type=int, default=30,
                        help="Warm-up days before evaluating (offline mode)")
    args = parser.parse_args()

    if args.csv:
        result = run_replay_offline(
            args.csv,
            start=date.fromisoformat(args.start),
            end=date.fromisoformat(args.end),
            n_checkpoints=args.checkpoints,
            warm_up_days=args.warm_up,
            allow_yfinance=args.allow_yfinance,
        )
        print(f"\n=== DARAMS Replay Pipeline [Offline] ===")
        print(f"Data: {result['data_range']}")
        print(f"Trading dates: {result['total_dates']}  (warm-up: {result['warm_up_days']})")
        print(f"Checkpoints evaluated: {result['checkpoints_evaluated']}")
        print(f"Batch rows: {result['batch_rows']}")
        print(f"Streaming rows: {result['streaming_rows']}")
        print(f"\n--- Consistency Check ---")
        print(f"Alphas checked: {result['alphas_checked']}")
        print(f"Overall match rate: {result['overall_match_rate']:.4f}")
        print(f"Perfect-match alphas: {result['perfect_match_alphas']} / {result['alphas_checked']}")
        print(f"\nPer-alpha match rates:")
        for alpha, rate in sorted(result["per_alpha_match_rate"].items()):
            flag = "OK" if rate >= 1.0 - 1e-4 else ("~" if rate >= 0.95 else "!!")
            print(f"  {flag} {alpha}: {rate:.4f}")
    else:
        result = run_replay_online(
            csv_path=None,
            start=date.fromisoformat(args.start),
            end=date.fromisoformat(args.end),
        )
        print(f"\n=== DARAMS Replay Pipeline [Online] ===")
        print(f"Dates replayed: {result['dates_replayed']}")
        print(f"Streaming output rows: {result['streaming_output_rows']}")
        print(f"Batch output rows: {result['batch_output_rows']}")
        print(f"Overall match rate: {result['overall_match_rate']:.4f}")


if __name__ == "__main__":
    main()
