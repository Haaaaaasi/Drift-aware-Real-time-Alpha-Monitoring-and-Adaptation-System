"""Generate a visual backtest report from the DARAMS MVP v1 pipeline.

Runs the pipeline (synthetic or TEJ/parquet mode) and saves a 4-panel matplotlib
figure to reports/<timestamp>_backtest_report.png plus a text summary.

Usage:
    # Synthetic data
    python scripts/generate_report.py

    # TEJ survivorship-correct data
    python scripts/generate_report.py --csv data/tw_stocks_tej.parquet

    # Custom date range
    python scripts/generate_report.py --csv data/tw_stocks_tej.parquet \\
        --start 2022-01-01 --end 2024-06-30 --output reports/
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ── make sure the project root is on sys.path ────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.daily_batch_pipeline import (
    generate_synthetic_data,
    generate_synthetic_alphas,
    load_csv_data,
    compute_python_alphas,
    run_backtest,
)
from src.common.logging import setup_logging
from src.labeling.evaluator import Evaluator
from src.labeling.label_generator import LabelGenerator
from src.config.constants import MVP_V1_ALPHA_IDS

setup_logging()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _portfolio_series(bars: pd.DataFrame, alpha_panel: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Re-run pipeline steps and collect per-date portfolio value series."""
    from src.labeling.label_generator import LabelGenerator
    from src.meta_signal.signal_generator import SignalGenerator
    from src.portfolio.constructor import PortfolioConstructor
    from src.risk.risk_manager import RiskManager
    from src.execution.paper_engine import PaperTradingEngine

    label_gen = LabelGenerator(horizons=[1, 5, 10], bar_type="daily")
    price_data = bars[["security_id", "tradetime", "close"]].copy()
    labels = label_gen.generate_labels(price_data)
    fwd_5 = labels[labels["horizon"] == 5].set_index(["security_id", "signal_time"])["forward_return"]

    signal_gen = SignalGenerator()
    signals = signal_gen.generate(alpha_panel=alpha_panel, forward_returns=fwd_5)

    portfolio = PortfolioConstructor(method="equal_weight_topk", top_k=10, long_only=True)
    risk_mgr = RiskManager(max_position_weight=0.10, max_gross_exposure=1.0)
    engine = PaperTradingEngine(initial_capital=10_000_000.0, slippage_bps=5.0)

    portfolio_values: list[dict] = []
    for rb_time in sorted(signals["signal_time"].unique()):
        day_signals = signals[signals["signal_time"] == rb_time]
        day_bars = bars[bars["tradetime"] == rb_time][["security_id", "close"]]
        if day_bars.empty:
            continue
        targets = portfolio.construct(day_signals)
        if targets.empty:
            continue
        adj_targets = risk_mgr.apply_constraints(targets)
        engine.execute_rebalance(adj_targets, day_bars, rb_time)
        portfolio_values.append({"date": rb_time, "value": engine.portfolio_value})

    pv_df = pd.DataFrame(portfolio_values)
    if len(pv_df) < 2:
        return pd.Series(dtype=float), pd.DataFrame()

    pv_df["return"] = pv_df["value"].pct_change()
    rets = pv_df.set_index("date")["return"].dropna()
    return rets, fwd_5


def _alpha_ic_table(alpha_panel: pd.DataFrame, fwd_5: pd.Series) -> pd.DataFrame:
    """Compute per-alpha IC and rank-IC."""
    from src.common.metrics import information_coefficient, rank_information_coefficient
    rows = []
    for alpha_id, g in alpha_panel.groupby("alpha_id"):
        vals = g.set_index(["security_id", "tradetime"])["alpha_value"]
        common = vals.index.intersection(fwd_5.index)
        if len(common) < 10:
            continue
        ic = information_coefficient(vals.loc[common], fwd_5.loc[common])
        ric = rank_information_coefficient(vals.loc[common], fwd_5.loc[common])
        rows.append({"alpha_id": alpha_id, "IC": ic, "Rank-IC": ric})
    return pd.DataFrame(rows).set_index("alpha_id").sort_values("IC")


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    start: date,
    end: date,
    csv_path: str | Path | None,
    output_dir: Path,
    allow_yfinance: bool = False,
) -> Path:
    """Build the 4-panel report and return the saved PNG path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Load data ─────────────────────────────────────────────────────────────
    if csv_path is not None:
        bars = load_csv_data(csv_path, start=start, end=end, allow_yfinance=allow_yfinance)
        alpha_panel = compute_python_alphas(bars)
        mode_label = f"CSV: {Path(csv_path).name}"
    else:
        bars = generate_synthetic_data(start, end)
        alpha_panel = generate_synthetic_alphas(bars)
        mode_label = "Synthetic data"

    rets, fwd_5 = _portfolio_series(bars, alpha_panel)
    if rets.empty:
        print("ERROR: not enough data to build portfolio return series.")
        sys.exit(1)

    cumret = (1 + rets).cumprod()
    drawdown = cumret / cumret.cummax() - 1
    ic_df = _alpha_ic_table(alpha_panel, fwd_5) if not fwd_5.empty else pd.DataFrame()

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"DARAMS MVP v1 — Backtest Report\n"
        f"{start} → {end}  |  {mode_label}  |  "
        f"{bars['security_id'].nunique()} symbols",
        fontsize=13, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.32,
                           left=0.07, right=0.97, top=0.91, bottom=0.06)

    # ── Panel 1: Cumulative return ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(cumret.index, cumret.values, color="#1f77b4", linewidth=1.5, label="Strategy")
    ax1.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax1.fill_between(cumret.index, 1.0, cumret.values,
                     where=cumret.values >= 1.0, alpha=0.15, color="#1f77b4")
    ax1.fill_between(cumret.index, 1.0, cumret.values,
                     where=cumret.values < 1.0, alpha=0.15, color="red")
    total_ret = float(cumret.iloc[-1] - 1)
    ax1.set_title(f"Cumulative Return  ({total_ret:+.1%})", fontsize=11)
    ax1.set_ylabel("NAV (start = 1)")
    ax1.tick_params(axis="x", rotation=30)
    ax1.grid(alpha=0.3)

    # ── Panel 2: Drawdown ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.fill_between(drawdown.index, drawdown.values, 0, color="crimson", alpha=0.5)
    ax2.axhline(0, color="gray", linewidth=0.8)
    max_dd = float(drawdown.min())
    ax2.set_title(f"Drawdown  (Max: {max_dd:.1%})", fontsize=11)
    ax2.set_ylabel("Drawdown (%)")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax2.tick_params(axis="x", rotation=30)
    ax2.grid(alpha=0.3)

    # ── Panel 3: Per-alpha IC bar chart ───────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    if not ic_df.empty:
        colors = ["#d62728" if v < 0 else "#2ca02c" for v in ic_df["IC"]]
        ax3.barh(ic_df.index, ic_df["IC"], color=colors, height=0.6)
        ax3.axvline(0, color="black", linewidth=0.8)
        ax3.set_title("Per-Alpha IC (vs 5-day fwd return)", fontsize=11)
        ax3.set_xlabel("Information Coefficient")
        ax3.grid(axis="x", alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No IC data", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Per-Alpha IC", fontsize=11)

    # ── Panel 4: Performance metrics summary table ────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    from src.common.metrics import sharpe_ratio, max_drawdown as mdd_fn, profit_factor
    ann_ret = float((cumret.iloc[-1]) ** (252 / max(len(cumret), 1)) - 1)
    sharpe = sharpe_ratio(rets)
    vol = float(rets.std() * np.sqrt(252))
    win_rate = float((rets > 0).mean())
    pf = profit_factor(rets)

    metrics = [
        ("Period",           f"{start} → {end}"),
        ("Symbols",          f"{bars['security_id'].nunique()}"),
        ("Rebalances",       f"{len(rets)}"),
        ("Total Return",     f"{total_ret:+.2%}"),
        ("Ann. Return",      f"{ann_ret:+.2%}"),
        ("Sharpe Ratio",     f"{sharpe:.3f}"),
        ("Max Drawdown",     f"{max_dd:.2%}"),
        ("Volatility (ann)", f"{vol:.2%}"),
        ("Win Rate",         f"{win_rate:.1%}"),
        ("Profit Factor",    f"{pf:.2f}"),
        ("Alphas",           f"{alpha_panel['alpha_id'].nunique()}"),
        ("Data mode",        mode_label[:40]),
    ]

    col_labels = ["Metric", "Value"]
    table = ax4.table(
        cellText=metrics,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # Style header
    for j in range(2):
        table[0, j].set_facecolor("#1f77b4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # Alternate row colours
    for i in range(1, len(metrics) + 1):
        bg = "#f0f4f8" if i % 2 == 0 else "white"
        for j in range(2):
            table[i, j].set_facecolor(bg)

    ax4.set_title("Performance Summary", fontsize=11, pad=12)

    # ── Save ─────────────────────────────────────────────────────────────────
    png_path = output_dir / f"{timestamp}_backtest_report.png"
    txt_path = output_dir / f"{timestamp}_backtest_report.txt"

    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plain-text summary
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"DARAMS MVP v1 — Backtest Report\n")
        f.write(f"Generated : {datetime.now()}\n")
        f.write(f"Mode      : {mode_label}\n")
        f.write(f"Period    : {start} → {end}\n")
        f.write(f"{'─'*40}\n")
        for k, v in metrics:
            f.write(f"{k:<20} {v}\n")
        if not ic_df.empty:
            f.write(f"\n{'─'*40}\nPer-Alpha IC\n{'─'*40}\n")
            f.write(ic_df.to_string())
            f.write("\n")

    print(f"\nReport saved:")
    print(f"  PNG : {png_path}")
    print(f"  TXT : {txt_path}")

    # Print summary to terminal
    print(f"\n{'='*50}")
    print(f"  DARAMS MVP v1 Backtest — {mode_label}")
    print(f"{'='*50}")
    for k, v in metrics:
        print(f"  {k:<20} {v}")

    return png_path


def main() -> None:
    parser = argparse.ArgumentParser(description="DARAMS MVP v1 Backtest Report Generator")
    parser.add_argument("--csv", type=str, default=None, metavar="PATH",
                        help="OHLCV CSV/parquet path (omit for synthetic)")
    parser.add_argument(
        "--allow-yfinance",
        action="store_true",
        help="明確允許使用已知污染的 yfinance CSV（僅 demo/反例使用）",
    )
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--output", default="reports/", metavar="DIR",
                        help="Output directory (default: reports/)")
    args = parser.parse_args()

    generate_report(
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
        csv_path=args.csv,
        output_dir=Path(args.output),
        allow_yfinance=args.allow_yfinance,
    )


if __name__ == "__main__":
    main()
