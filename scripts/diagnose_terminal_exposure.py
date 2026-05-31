"""診斷 OOS 持倉遇到 missing next execution return 的曝險。

這個診斷用來支持「最後可觀測價格結清」假設：若 true terminal/missing
exposure 很小，就不需要把所有下市事件硬套 -50% / -100% shock。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipelines.daily_batch_pipeline import load_csv_data
from pipelines.simulate_recent import (
    DATA_SOURCE_DEFAULTS,
    _filter_universe,
    _next_day_returns,
    _trading_days,
)
from src.config.constants import DEFAULT_DATA_SOURCE


DEFAULT_OUT_DIR = Path("reports/adaptation_ab/terminal_exposure_20260512")
DEFAULT_RUN_DIRS = [
    Path("reports/adaptation_ab/ab_20240701_20260430_top10_oos_no_indcap_cachealign_tail25_e20x60_t25_h10_nextvwap_20260512"),
    Path("reports/adaptation_ab/ab_20240701_20260430_top10_oos_no_indcap_cachealign_tail25_e20x60_t25_h10_nextopen_20260512"),
]


def _execution_from_run_dir(run_dir: Path) -> str:
    name = run_dir.name.lower()
    if "nextopen" in name or "next_open" in name:
        return "next_open"
    if "nextvwap" in name or "next_vwap" in name:
        return "next_vwap"
    raise ValueError(f"Cannot infer execution price from {run_dir}")


def _strategy_from_sim_dir(path: Path) -> str:
    name = path.name
    if "sched20" in name or "scheduled_20" in name:
        return "scheduled_20"
    if "none" in name:
        return "none"
    return name


def _complete_cutoff(days: list[pd.Timestamp], execution_price: str) -> pd.Timestamp:
    offset = 1 if execution_price == "close" else 2
    if len(days) <= offset:
        return days[0]
    return days[-(offset + 1)]


def _terminal_rows(
    holdings: pd.DataFrame,
    *,
    run_name: str,
    execution_price: str,
    strategy: str,
    next_ret: pd.DataFrame,
    global_complete_cutoff: pd.Timestamp,
) -> pd.DataFrame:
    if holdings.empty:
        return pd.DataFrame()

    h = holdings.copy()
    h["date"] = pd.to_datetime(h["date"])
    h["security_id"] = h["security_id"].astype(str)
    h["weight_abs"] = h["target_weight"].astype(float).abs()

    idx = pd.MultiIndex.from_frame(h[["security_id", "date"]].rename(columns={"date": "tradetime"}))
    returns = next_ret["next_return"].reindex(idx).to_numpy()
    h["has_next_return"] = pd.notna(returns)
    missing = h[~h["has_next_return"]].copy()
    if missing.empty:
        return missing

    missing["missing_type"] = np.where(
        missing["date"] > global_complete_cutoff,
        "dataset_end_boundary",
        "true_terminal_or_missing",
    )
    missing.insert(0, "strategy", strategy)
    missing.insert(0, "execution_price", execution_price)
    missing.insert(0, "run_name", run_name)
    return missing


def _benchmark_holdings(
    bars: pd.DataFrame,
    *,
    start: str,
    end: str,
    execution_price: str,
) -> pd.DataFrame:
    days = _trading_days(bars, pd.to_datetime(start).date(), pd.to_datetime(end).date())
    if not days:
        return pd.DataFrame()
    first_day = days[0]
    initial = bars[bars["tradetime"] == first_day]["security_id"].astype(str).drop_duplicates()
    if initial.empty:
        return pd.DataFrame()
    weight = 1.0 / len(initial)
    records: list[dict] = []
    for day in days:
        for sec in initial:
            records.append({
                "date": day,
                "security_id": str(sec),
                "target_weight": weight,
                "signal_score": np.nan,
            })
    return pd.DataFrame(records)


def _summarise(events: pd.DataFrame, *, run_name: str, execution_price: str, strategy: str) -> dict:
    base = {
        "run_name": run_name,
        "execution_price": execution_price,
        "strategy": strategy,
    }
    if events.empty:
        return {
            **base,
            "missing_rows": 0,
            "true_terminal_rows": 0,
            "dataset_boundary_rows": 0,
            "unique_terminal_stocks": 0,
            "sum_true_terminal_exposure": 0.0,
            "max_daily_true_terminal_exposure": 0.0,
            "mean_daily_true_terminal_exposure": 0.0,
        }

    true_events = events[events["missing_type"] == "true_terminal_or_missing"]
    daily = true_events.groupby("date")["weight_abs"].sum() if not true_events.empty else pd.Series(dtype=float)
    return {
        **base,
        "missing_rows": int(len(events)),
        "true_terminal_rows": int(len(true_events)),
        "dataset_boundary_rows": int((events["missing_type"] == "dataset_end_boundary").sum()),
        "unique_terminal_stocks": int(true_events["security_id"].nunique()) if not true_events.empty else 0,
        "sum_true_terminal_exposure": float(true_events["weight_abs"].sum()) if not true_events.empty else 0.0,
        "max_daily_true_terminal_exposure": float(daily.max()) if not daily.empty else 0.0,
        "mean_daily_true_terminal_exposure": float(daily.mean()) if not daily.empty else 0.0,
    }


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        vals = []
        for col in headers:
            val = row[col]
            if pd.isna(val):
                vals.append("")
            elif isinstance(val, (float, np.floating)):
                vals.append(f"{float(val):.6f}".rstrip("0").rstrip("."))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_summary(summary: pd.DataFrame, path: Path) -> None:
    display = summary.copy()
    lines = [
        "# Terminal exposure diagnostic",
        "",
        "目的：量化 `next_return = NaN -> 0` 所對應的 terminal/missing next execution exposure。",
        "",
        _markdown_table(display),
        "",
        "判讀：`dataset_end_boundary` 是樣本結束造成的缺下一日或下兩日價格；",
        "`true_terminal_or_missing` 才是下市、換股、停牌或資料缺口風險的近似曝險。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def analyse(args: argparse.Namespace) -> None:
    csv_path = Path(args.csv or DATA_SOURCE_DEFAULTS[args.data_source])
    bars = load_csv_data(csv_path, allow_yfinance=args.allow_yfinance)
    bars["security_id"] = bars["security_id"].astype(str)
    bars, _ = _filter_universe(bars, args.symbols, args.min_turnover_ntd, pd.to_datetime(args.start).date())
    all_days = _trading_days(bars, pd.to_datetime(args.start).date(), pd.to_datetime(args.end).date())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    all_events: list[pd.DataFrame] = []
    for run_dir in args.run_dir:
        execution_price = _execution_from_run_dir(run_dir)
        next_ret = _next_day_returns(bars, execution_price=execution_price)
        cutoff = _complete_cutoff(all_days, execution_price)

        for holdings_path in sorted((run_dir / "simulations").glob("*/holdings.csv")):
            strategy = _strategy_from_sim_dir(holdings_path.parent)
            holdings = pd.read_csv(holdings_path)
            events = _terminal_rows(
                holdings,
                run_name=run_dir.name,
                execution_price=execution_price,
                strategy=strategy,
                next_ret=next_ret,
                global_complete_cutoff=cutoff,
            )
            summary_rows.append(_summarise(
                events,
                run_name=run_dir.name,
                execution_price=execution_price,
                strategy=strategy,
            ))
            all_events.append(events)

        benchmark = _benchmark_holdings(
            bars,
            start=args.start,
            end=args.end,
            execution_price=execution_price,
        )
        events = _terminal_rows(
            benchmark,
            run_name=run_dir.name,
            execution_price=execution_price,
            strategy="ew_buy_hold_universe",
            next_ret=next_ret,
            global_complete_cutoff=cutoff,
        )
        summary_rows.append(_summarise(
            events,
            run_name=run_dir.name,
            execution_price=execution_price,
            strategy="ew_buy_hold_universe",
        ))
        all_events.append(events)

    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    events_df.to_csv(args.out_dir / "terminal_exposure_events.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.out_dir / "terminal_exposure_summary.csv", index=False, encoding="utf-8-sig")
    write_summary(summary, args.out_dir / "terminal_exposure_summary.md")
    print(f"wrote {args.out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-source", choices=["csv", "tej"], default=DEFAULT_DATA_SOURCE)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--allow-yfinance", action="store_true")
    parser.add_argument("--start", default="2024-07-01")
    parser.add_argument("--end", default="2026-04-30")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--min-turnover-ntd", type=float, default=0.0)
    parser.add_argument("--run-dir", action="append", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    if args.run_dir is None:
        args.run_dir = DEFAULT_RUN_DIRS
    analyse(args)


if __name__ == "__main__":
    main()
