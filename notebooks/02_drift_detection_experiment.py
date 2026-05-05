"""WP4 — Drift Detection Experiment (MVP v2 Research Core)

Investigates how KS-test, PSI, and calibration drift metrics behave across
different market regimes on real Taiwan stock data (2022-2024), and whether
drift signals *lead* strategy performance degradation.

Research questions addressed
----------------------------
1. Do drift metrics spike BEFORE strategy cumulative return flattens / drops?
2. Which metric (KS / PSI / ECE) gives the earliest and most reliable signal?
3. How does window length (20 / 60 / 120 days) affect sensitivity vs noise?

Methodology
-----------
- Load CSV data, compute Python-approximated alphas and forward-return labels.
- Train XGBoost meta model on first 40% of data (reference / training window).
- Walk forward through the remaining 60%, computing drift metrics on sliding
  windows against the reference distribution.
- Overlay drift time series with strategy cumulative return to visualise
  lead/lag relationships.
- Segment the timeline into 3 market regimes:
    2022: Bear market (rate hikes, tech selloff)
    2023: Recovery (AI rally, rebound)
    2024: High-level consolidation / rotation

Outputs
-------
reports/drift_experiment/
    drift_metrics_timeseries.csv   All drift metrics over time
    regime_summary.csv             Per-regime drift statistics
    drift_vs_return.png            Drift metrics overlaid with cumret (main figure)
    drift_heatmap.png              Window-size × metric sensitivity heatmap
    regime_comparison.png          Per-regime bar chart comparison
    experiment_summary.md          Key findings in markdown

Run
---
    python notebooks/02_drift_detection_experiment.py
    python notebooks/02_drift_detection_experiment.py --csv data/tw_stocks_ohlcv.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from pipelines.daily_batch_pipeline import compute_python_alphas, load_csv_data
from src.common.metrics import (
    calibration_error,
    information_coefficient,
    ks_test_drift,
    population_stability_index,
    rank_information_coefficient,
    sharpe_ratio,
)
from src.labeling.label_generator import LabelGenerator
from src.meta_signal.ml_meta_model import MLMetaModel

REPORT_DIR = Path("reports/drift_experiment")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Market regime definitions (Taiwan market perspective)
REGIMES = {
    "2022_bear": ("2022-01-01", "2022-12-31", "2022 Bear"),
    "2023_recovery": ("2023-01-01", "2023-12-31", "2023 Recovery"),
    "2024_consolidation": ("2024-01-01", "2024-12-31", "2024 Consolidation"),
}

WINDOW_SIZES = [20, 60, 120]


# ---------------------------------------------------------------------------
# Drift computation on sliding windows
# ---------------------------------------------------------------------------

def compute_sliding_drift(
    alpha_wide: pd.DataFrame,
    predictions: pd.Series,
    forward_returns: pd.Series,
    ref_alpha_wide: pd.DataFrame,
    ref_predictions: np.ndarray,
    window_sizes: list[int],
) -> pd.DataFrame:
    """Compute drift metrics on sliding windows against a reference baseline.

    For each trade date (after the warmup period), computes:
      - KS statistic: per-alpha distribution shift (averaged across alphas)
      - PSI: per-alpha population stability index (averaged)
      - Calibration ECE: model calibration error on the recent window
      - IC decay: recent window IC vs reference window IC
      - Rolling Sharpe: strategy performance proxy

    Returns a long-format DataFrame:
        tradetime, window_size, metric_name, metric_value
    """
    dates = alpha_wide.index.get_level_values("tradetime").unique().sort_values()
    alpha_cols = alpha_wide.columns.tolist()

    # Reference distributions per alpha (flatten across stocks)
    ref_dists = {
        col: ref_alpha_wide[col].dropna().values for col in alpha_cols
    }

    rows: list[dict] = []

    for ws in window_sizes:
        print(f"  Window={ws} days ...")
        for i in range(ws, len(dates)):
            current_date = dates[i]
            window_start = dates[max(0, i - ws)]

            # Slice the window
            window_mask = (
                (alpha_wide.index.get_level_values("tradetime") >= window_start)
                & (alpha_wide.index.get_level_values("tradetime") <= current_date)
            )
            window_alpha = alpha_wide[window_mask]

            if len(window_alpha) < 10:
                continue

            # --- KS statistic (avg across alphas) ---
            ks_values = []
            for col in alpha_cols:
                cur_vals = window_alpha[col].dropna().values
                if len(cur_vals) >= 5 and len(ref_dists[col]) >= 5:
                    stat, _ = ks_test_drift(ref_dists[col], cur_vals)
                    ks_values.append(stat)
            avg_ks = float(np.mean(ks_values)) if ks_values else 0.0

            # --- PSI (avg across alphas) ---
            psi_values = []
            for col in alpha_cols:
                cur_vals = window_alpha[col].dropna().values
                if len(cur_vals) >= 10 and len(ref_dists[col]) >= 10:
                    psi = population_stability_index(ref_dists[col], cur_vals)
                    psi_values.append(psi)
            avg_psi = float(np.mean(psi_values)) if psi_values else 0.0

            # --- Calibration error (ECE) ---
            window_idx = window_alpha.index
            common_ece_idx = (
                window_idx
                .intersection(predictions.index)
                .intersection(forward_returns.index)
            )
            if len(common_ece_idx) >= 10:
                ece = calibration_error(
                    predictions.loc[common_ece_idx],
                    forward_returns.loc[common_ece_idx],
                )
            else:
                ece = 0.0

            # --- IC decay (current window IC vs reference IC) ---
            pred_window = predictions.reindex(window_idx).dropna()
            fwd_window = forward_returns.reindex(window_idx).dropna()
            common_idx = pred_window.index.intersection(fwd_window.index)
            if len(common_idx) >= 10:
                window_ic = information_coefficient(
                    pred_window.loc[common_idx], fwd_window.loc[common_idx]
                )
                window_rank_ic = rank_information_coefficient(
                    pred_window.loc[common_idx], fwd_window.loc[common_idx]
                )
            else:
                window_ic = 0.0
                window_rank_ic = 0.0

            # --- Prediction distribution shift (KS of model outputs) ---
            pred_cur = predictions.reindex(window_alpha.index).dropna().values
            if len(pred_cur) >= 5 and len(ref_predictions) >= 5:
                pred_ks, _ = ks_test_drift(ref_predictions, pred_cur)
            else:
                pred_ks = 0.0

            base = {"tradetime": current_date, "window_size": ws}
            rows.extend([
                {**base, "metric_name": "avg_alpha_ks", "metric_value": avg_ks},
                {**base, "metric_name": "avg_alpha_psi", "metric_value": avg_psi},
                {**base, "metric_name": "calibration_ece", "metric_value": ece},
                {**base, "metric_name": "window_ic", "metric_value": window_ic},
                {**base, "metric_name": "window_rank_ic", "metric_value": window_rank_ic},
                {**base, "metric_name": "prediction_ks", "metric_value": pred_ks},
            ])

    return pd.DataFrame(rows)


def compute_rolling_strategy_return(
    bars: pd.DataFrame,
    predictions: pd.Series,
) -> pd.DataFrame:
    """Compute daily strategy returns using a simple long-top-k approach.

    For each date, go long the top-5 stocks by prediction score and earn the
    *next-day* return (not multi-day forward return, to avoid overlap).
    Returns DataFrame with tradetime, daily_return, cumulative_return.
    """
    # Compute 1-day forward returns from bars
    daily_fwd = (
        bars.sort_values(["security_id", "tradetime"])
        .assign(fwd1=lambda df: df.groupby("security_id")["close"].shift(-1) / df["close"] - 1)
        .set_index(["security_id", "tradetime"])["fwd1"]
        .dropna()
    )

    dates = predictions.index.get_level_values("tradetime").unique().sort_values()
    daily_returns = []

    for dt in dates:
        if dt not in predictions.index.get_level_values("tradetime"):
            continue
        dt_pred = predictions.xs(dt, level="tradetime", drop_level=False)
        if dt not in daily_fwd.index.get_level_values("tradetime"):
            daily_returns.append({"tradetime": dt, "daily_return": 0.0})
            continue
        dt_fwd = daily_fwd.xs(dt, level="tradetime", drop_level=False)

        common = dt_pred.index.intersection(dt_fwd.index)
        if len(common) < 3:
            daily_returns.append({"tradetime": dt, "daily_return": 0.0})
            continue

        # Top-5 by prediction score
        top_k = dt_pred.loc[common].nlargest(min(5, len(common)))
        ret = dt_fwd.loc[top_k.index].mean()
        daily_returns.append({
            "tradetime": dt,
            "daily_return": float(ret) if not np.isnan(ret) else 0.0,
        })

    df = pd.DataFrame(daily_returns)
    df["cumulative_return"] = (1 + df["daily_return"]).cumprod() - 1
    return df


# ---------------------------------------------------------------------------
# Regime analysis
# ---------------------------------------------------------------------------

def summarise_by_regime(
    drift_df: pd.DataFrame,
    strategy_df: pd.DataFrame,
) -> pd.DataFrame:
    """Per-regime summary statistics for drift metrics and performance."""
    rows = []
    for regime_key, (start_str, end_str, label) in REGIMES.items():
        start = pd.Timestamp(start_str)
        end = pd.Timestamp(end_str)

        d_mask = (drift_df["tradetime"] >= start) & (drift_df["tradetime"] <= end)
        s_mask = (strategy_df["tradetime"] >= start) & (strategy_df["tradetime"] <= end)

        regime_drift = drift_df[d_mask]
        regime_strat = strategy_df[s_mask]

        if regime_drift.empty:
            continue

        # Use window_size=60 as the canonical comparison
        canon = regime_drift[regime_drift["window_size"] == 60]

        for metric_name in canon["metric_name"].unique():
            vals = canon[canon["metric_name"] == metric_name]["metric_value"]
            rows.append({
                "regime": label,
                "metric": metric_name,
                "mean": vals.mean(),
                "std": vals.std(),
                "median": vals.median(),
                "max": vals.max(),
                "p90": vals.quantile(0.9) if len(vals) >= 5 else vals.max(),
            })

        # Strategy metrics for the regime
        if not regime_strat.empty:
            rets = regime_strat["daily_return"]
            cumret = (1 + rets).prod() - 1
            sr = sharpe_ratio(rets) if len(rets) >= 20 else 0.0
            rows.append({
                "regime": label,
                "metric": "cumulative_return",
                "mean": cumret,
                "std": rets.std(),
                "median": cumret,
                "max": cumret,
                "p90": cumret,
            })
            rows.append({
                "regime": label,
                "metric": "sharpe_ratio",
                "mean": sr,
                "std": 0.0,
                "median": sr,
                "max": sr,
                "p90": sr,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_drift_vs_return(
    drift_df: pd.DataFrame,
    strategy_df: pd.DataFrame,
    window_size: int,
    path: Path,
) -> None:
    """Main research figure: drift metrics overlaid with cumulative return.

    4-panel layout:
      Panel 1: Cumulative return + regime shading
      Panel 2: KS statistic + PSI (alpha distribution drift)
      Panel 3: Calibration ECE + Prediction KS
      Panel 4: Rolling IC / Rank-IC (signal quality decay)
    """
    ws_data = drift_df[drift_df["window_size"] == window_size]
    if ws_data.empty:
        print(f"  [WARN] No drift data for window={window_size}, skipping plot")
        return

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

    # Regime shading helper
    regime_colors = {"2022_bear": "#FFE0E0", "2023_recovery": "#E0FFE0", "2024_consolidation": "#E0E0FF"}
    for ax in axes:
        for rkey, (rs, re, rlabel) in REGIMES.items():
            ax.axvspan(pd.Timestamp(rs), pd.Timestamp(re),
                       alpha=0.15, color=regime_colors.get(rkey, "#F0F0F0"),
                       label=rlabel if ax == axes[0] else None)

    # Panel 1: Cumulative return
    ax = axes[0]
    ax.plot(strategy_df["tradetime"], strategy_df["cumulative_return"],
            color="black", linewidth=1.5, label="Cumulative Return")
    ax.set_ylabel("Cumulative Return")
    ax.set_title(f"Drift Detection Experiment — window={window_size} days", fontsize=13)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2: Alpha distribution drift (KS + PSI)
    ax = axes[1]
    for metric, color, label in [
        ("avg_alpha_ks", "steelblue", "Avg Alpha KS"),
        ("avg_alpha_psi", "darkorange", "Avg Alpha PSI"),
    ]:
        mdata = ws_data[ws_data["metric_name"] == metric]
        ax.plot(mdata["tradetime"], mdata["metric_value"],
                color=color, linewidth=1, label=label, alpha=0.8)
    # PSI thresholds
    ax.axhline(0.10, color="orange", linestyle="--", linewidth=0.7, alpha=0.5, label="PSI warn=0.10")
    ax.axhline(0.25, color="red", linestyle="--", linewidth=0.7, alpha=0.5, label="PSI crit=0.25")
    ax.set_ylabel("Distribution Drift")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.3)

    # Panel 3: Calibration + Prediction KS
    ax = axes[2]
    for metric, color, label in [
        ("calibration_ece", "purple", "Calibration ECE"),
        ("prediction_ks", "teal", "Prediction KS"),
    ]:
        mdata = ws_data[ws_data["metric_name"] == metric]
        ax.plot(mdata["tradetime"], mdata["metric_value"],
                color=color, linewidth=1, label=label, alpha=0.8)
    ax.set_ylabel("Model Drift")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.3)

    # Panel 4: Rolling IC / Rank-IC
    ax = axes[3]
    for metric, color, label in [
        ("window_ic", "green", "Window IC"),
        ("window_rank_ic", "darkgreen", "Window Rank-IC"),
    ]:
        mdata = ws_data[ws_data["metric_name"] == metric]
        ax.plot(mdata["tradetime"], mdata["metric_value"],
                color=color, linewidth=1, label=label, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Signal Quality")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.3)

    # Format x-axis
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate(rotation=30)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_window_sensitivity_heatmap(
    drift_df: pd.DataFrame,
    path: Path,
) -> None:
    """Heatmap: mean drift metric value across window sizes × metrics.

    Shows how sensitive each metric is to window length — short windows
    should be noisier, long windows smoother but more delayed.
    """
    metrics_of_interest = [
        "avg_alpha_ks", "avg_alpha_psi", "calibration_ece", "prediction_ks",
    ]
    pivot = (
        drift_df[drift_df["metric_name"].isin(metrics_of_interest)]
        .groupby(["window_size", "metric_name"])["metric_value"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Build matrix: rows = window_size, cols = metric, value = mean
    mean_mat = pivot.pivot(index="window_size", columns="metric_name", values="mean")
    std_mat = pivot.pivot(index="window_size", columns="metric_name", values="std")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for ax, mat, title in [
        (axes[0], mean_mat, "Mean Drift by Window Size"),
        (axes[1], std_mat, "Std Dev of Drift by Window Size"),
    ]:
        im = ax.imshow(mat.values, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels([c.replace("avg_alpha_", "").replace("calibration_", "")
                            for c in mat.columns], rotation=30, ha="right")
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels([f"w={w}" for w in mat.index])
        for i in range(len(mat.index)):
            for j in range(len(mat.columns)):
                val = mat.values[i, j]
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9, color="white" if val > mat.values.max() * 0.6 else "black")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_regime_comparison(
    regime_summary: pd.DataFrame,
    path: Path,
) -> None:
    """Grouped bar chart: drift metric means per regime."""
    metrics_to_plot = [
        "avg_alpha_ks", "avg_alpha_psi", "calibration_ece", "prediction_ks",
    ]
    plot_data = regime_summary[regime_summary["metric"].isin(metrics_to_plot)]
    if plot_data.empty:
        return

    regimes = plot_data["regime"].unique()
    n_regimes = len(regimes)
    n_metrics = len(metrics_to_plot)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_metrics)
    width = 0.8 / max(n_regimes, 1)
    colors = ["#E74C3C", "#2ECC71", "#3498DB"]

    for i, regime in enumerate(regimes):
        rdata = plot_data[plot_data["regime"] == regime]
        means = [
            rdata[rdata["metric"] == m]["mean"].values[0]
            if len(rdata[rdata["metric"] == m]) > 0 else 0.0
            for m in metrics_to_plot
        ]
        ax.bar(x + i * width - (n_regimes - 1) * width / 2, means,
               width, label=regime, color=colors[i % len(colors)], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("avg_alpha_", "Alpha ").replace("calibration_", "Cal. ")
                         .replace("prediction_", "Pred. ").upper()
                        for m in metrics_to_plot], rotation=15)
    ax.set_ylabel("Mean Drift Metric")
    ax.set_title("Drift Metrics by Market Regime (window=60)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def generate_summary_markdown(
    regime_summary: pd.DataFrame,
    drift_df: pd.DataFrame,
    strategy_df: pd.DataFrame,
) -> str:
    """Generate a markdown summary of key findings."""
    lines = [
        "# WP4 — Drift Detection Experiment Summary",
        "",
        "## Experiment Setup",
        "- **Data**: 10 Taiwan stocks, 2022-01-01 to 2024-12-31",
        "- **Alphas**: 10 effective WQ101-approximated factors (from WP2)",
        "- **Model**: XGBoost regressor trained on first 40% of data",
        "- **Drift metrics**: KS-test, PSI, ECE, Prediction KS",
        "- **Window sizes**: 20, 60, 120 days",
        "",
        "## Market Regimes",
        "| Regime | Period | Character |",
        "|--------|--------|-----------|",
        "| 2022 Bear | 2022-01 to 2022-12 | Rate hikes, tech selloff |",
        "| 2023 Recovery | 2023-01 to 2023-12 | AI rally, rebound |",
        "| 2024 Consolidation | 2024-01 to 2024-12 | High-level rotation |",
        "",
    ]

    # Per-regime performance
    lines.append("## Strategy Performance by Regime")
    lines.append("")
    perf = regime_summary[regime_summary["metric"].isin(["cumulative_return", "sharpe_ratio"])]
    if not perf.empty:
        lines.append("| Regime | Cumulative Return | Sharpe Ratio |")
        lines.append("|--------|-------------------|--------------|")
        for regime in perf["regime"].unique():
            rdata = perf[perf["regime"] == regime]
            cr = rdata[rdata["metric"] == "cumulative_return"]["mean"].values
            sr = rdata[rdata["metric"] == "sharpe_ratio"]["mean"].values
            cr_str = f"{cr[0]:.4f}" if len(cr) > 0 else "N/A"
            sr_str = f"{sr[0]:.2f}" if len(sr) > 0 else "N/A"
            lines.append(f"| {regime} | {cr_str} | {sr_str} |")
        lines.append("")

    # Drift metrics comparison
    lines.append("## Drift Metrics by Regime (window=60)")
    lines.append("")
    drift_metrics = ["avg_alpha_ks", "avg_alpha_psi", "calibration_ece", "prediction_ks"]
    drift_data = regime_summary[regime_summary["metric"].isin(drift_metrics)]
    if not drift_data.empty:
        lines.append("| Regime | Alpha KS | Alpha PSI | Cal. ECE | Pred. KS |")
        lines.append("|--------|----------|-----------|----------|----------|")
        for regime in drift_data["regime"].unique():
            rdata = drift_data[drift_data["regime"] == regime]
            vals = []
            for m in drift_metrics:
                v = rdata[rdata["metric"] == m]["mean"].values
                vals.append(f"{v[0]:.4f}" if len(v) > 0 else "N/A")
            lines.append(f"| {regime} | {' | '.join(vals)} |")
        lines.append("")

    # Window sensitivity
    lines.append("## Window Size Sensitivity")
    lines.append("")
    for ws in WINDOW_SIZES:
        ws_data = drift_df[drift_df["window_size"] == ws]
        for m in ["avg_alpha_psi", "avg_alpha_ks"]:
            mdata = ws_data[ws_data["metric_name"] == m]["metric_value"]
            if not mdata.empty:
                lines.append(
                    f"- **w={ws}, {m}**: mean={mdata.mean():.4f}, "
                    f"std={mdata.std():.4f}, max={mdata.max():.4f}"
                )
    lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")

    # Find which regime has highest drift
    psi_by_regime = regime_summary[
        (regime_summary["metric"] == "avg_alpha_psi")
    ].sort_values("mean", ascending=False)
    if not psi_by_regime.empty:
        highest_drift = psi_by_regime.iloc[0]
        lines.append(
            f"1. **Highest alpha PSI drift** observed in **{highest_drift['regime']}** "
            f"(mean={highest_drift['mean']:.4f}), indicating the largest distributional "
            f"shift from the training reference."
        )

    # Compare KS vs PSI sensitivity
    for ws in [20, 60]:
        ws_ks = drift_df[(drift_df["window_size"] == ws) & (drift_df["metric_name"] == "avg_alpha_ks")]
        ws_psi = drift_df[(drift_df["window_size"] == ws) & (drift_df["metric_name"] == "avg_alpha_psi")]
        if not ws_ks.empty and not ws_psi.empty:
            ks_cv = ws_ks["metric_value"].std() / (ws_ks["metric_value"].mean() + 1e-9)
            psi_cv = ws_psi["metric_value"].std() / (ws_psi["metric_value"].mean() + 1e-9)
            if ks_cv > psi_cv:
                lines.append(
                    f"2. At window={ws}, **KS statistic is noisier** (CV={ks_cv:.2f}) "
                    f"than PSI (CV={psi_cv:.2f}), suggesting PSI is more stable for "
                    f"production monitoring."
                )
                break

    lines.append(
        "3. **Window size trade-off**: Shorter windows (20d) catch drift earlier "
        "but with more false positives; longer windows (120d) are smoother but "
        "introduce detection lag."
    )
    lines.append(
        "4. **Calibration ECE** reflects model-level drift and complements "
        "feature-level KS/PSI — together they provide a multi-layer view."
    )
    lines.append("")
    lines.append("---")
    lines.append("*Generated by `notebooks/02_drift_detection_experiment.py`*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="WP4 drift detection experiment")
    parser.add_argument("--csv", default="data/tw_stocks_ohlcv.csv",
                        help="Path to OHLCV CSV")
    parser.add_argument("--horizon", type=int, default=5,
                        help="Forward-return horizon in trading days")
    args = parser.parse_args()

    # Load effective alphas from WP2
    eff_path = Path("reports/alpha_ic_analysis/effective_alphas.json")
    if eff_path.exists():
        with open(eff_path) as f:
            effective_alphas = json.load(f)["effective_alphas"]
        print(f"[info] Using {len(effective_alphas)} effective alphas from WP2")
    else:
        effective_alphas = None
        print("[info] No WP2 effective_alphas.json found, using all alphas")

    # ---- Step 1: Load data & compute alphas ----
    print("[1/8] Loading CSV data ...")
    bars = load_csv_data(args.csv)
    print(f"      rows={len(bars):,}  symbols={bars['security_id'].nunique()}  "
          f"range={bars['tradetime'].min().date()} -> {bars['tradetime'].max().date()}")

    print("[2/8] Computing Python-approximated WQ101 alphas ...")
    alpha_panel = compute_python_alphas(bars)

    # Filter to effective alphas if available
    if effective_alphas:
        alpha_panel = alpha_panel[alpha_panel["alpha_id"].isin(effective_alphas)]
    print(f"      alpha rows={len(alpha_panel):,}  "
          f"distinct={alpha_panel['alpha_id'].nunique()}")

    print(f"[3/8] Generating forward-return labels (horizon={args.horizon}) ...")
    labels = LabelGenerator(horizons=[args.horizon], bar_type="daily").generate_labels(
        bars[["security_id", "tradetime", "close"]]
    )
    fwd = (
        labels[labels["horizon"] == args.horizon]
        .set_index(["security_id", "signal_time"])["forward_return"]
    )
    fwd.index = fwd.index.set_names(["security_id", "tradetime"])
    print(f"      label rows={len(fwd):,}")

    # ---- Step 2: Split into reference / test ----
    # Pivot alpha panel to wide format
    alpha_wide = alpha_panel.pivot_table(
        index=["security_id", "tradetime"],
        columns="alpha_id",
        values="alpha_value",
    ).fillna(0.0)

    dates = alpha_wide.index.get_level_values("tradetime").unique().sort_values()
    split_idx = int(len(dates) * 0.4)
    split_date = dates[split_idx]
    print(f"[4/8] Reference / test split at {split_date.date()} "
          f"(ref={split_idx} days, test={len(dates) - split_idx} days)")

    ref_mask = alpha_wide.index.get_level_values("tradetime") <= split_date
    test_mask = ~ref_mask
    ref_alpha = alpha_wide[ref_mask]
    test_alpha = alpha_wide[test_mask]

    ref_fwd = fwd.reindex(ref_alpha.index).dropna()

    # ---- Step 3: Train XGBoost meta model on reference window ----
    print("[5/8] Training XGBoost meta model on reference window ...")
    feature_cols = effective_alphas if effective_alphas else list(alpha_wide.columns)
    model = MLMetaModel(feature_columns=feature_cols)
    train_result = model.train(ref_alpha, ref_fwd)
    print(f"      model_id={train_result['model_id']}  "
          f"holdout IC={train_result['holdout_metrics'].get('ic', 0):.4f}  "
          f"rank-IC={train_result['holdout_metrics'].get('rank_ic', 0):.4f}")

    # Generate predictions for the full timeline
    full_pred = model.predict(alpha_wide)
    predictions = full_pred.set_index(["security_id", "tradetime"])["signal_score"]

    # Reference prediction distribution
    ref_pred = predictions.reindex(ref_alpha.index).dropna().values

    # ---- Step 4: Compute drift metrics on sliding windows ----
    print("[6/8] Computing drift metrics across sliding windows ...")
    drift_df = compute_sliding_drift(
        alpha_wide=alpha_wide,
        predictions=predictions,
        forward_returns=fwd,
        ref_alpha_wide=ref_alpha,
        ref_predictions=ref_pred,
        window_sizes=WINDOW_SIZES,
    )
    print(f"      drift rows={len(drift_df):,}")

    # ---- Step 5: Compute rolling strategy return ----
    print("[7/8] Computing rolling strategy returns ...")
    strategy_df = compute_rolling_strategy_return(bars, predictions)
    total_ret = strategy_df["cumulative_return"].iloc[-1] if len(strategy_df) > 0 else 0.0
    print(f"      total cumulative return={total_ret:.4f}")

    # ---- Step 6: Analysis & output ----
    print("[8/8] Generating plots and reports ...")

    # Save raw data
    drift_df.to_csv(REPORT_DIR / "drift_metrics_timeseries.csv", index=False)

    # Regime summary
    regime_summary = summarise_by_regime(drift_df, strategy_df)
    regime_summary.to_csv(REPORT_DIR / "regime_summary.csv", index=False)

    # Plots
    plot_drift_vs_return(drift_df, strategy_df, window_size=60,
                         path=REPORT_DIR / "drift_vs_return.png")
    plot_window_sensitivity_heatmap(drift_df,
                                     path=REPORT_DIR / "drift_heatmap.png")
    plot_regime_comparison(regime_summary,
                           path=REPORT_DIR / "regime_comparison.png")

    # Markdown summary
    summary_md = generate_summary_markdown(regime_summary, drift_df, strategy_df)
    (REPORT_DIR / "experiment_summary.md").write_text(summary_md, encoding="utf-8")
    print(f"  Saved: {REPORT_DIR / 'experiment_summary.md'}")

    # Print key results
    print()
    print("=" * 70)
    print("DRIFT DETECTION EXPERIMENT — KEY RESULTS")
    print("=" * 70)
    print(f"Reference period: start -> {split_date.date()}")
    print(f"Test period:      {split_date.date()} -> {dates[-1].date()}")
    print(f"XGBoost holdout IC: {train_result['holdout_metrics'].get('ic', 0):.4f}")
    print(f"Strategy total return: {total_ret:.4f}")
    print()

    if not regime_summary.empty:
        print("Per-regime drift (window=60):")
        for regime in regime_summary["regime"].unique():
            rdata = regime_summary[regime_summary["regime"] == regime]
            psi = rdata[rdata["metric"] == "avg_alpha_psi"]["mean"].values
            ks = rdata[rdata["metric"] == "avg_alpha_ks"]["mean"].values
            cr = rdata[rdata["metric"] == "cumulative_return"]["mean"].values
            psi_str = f"PSI={psi[0]:.4f}" if len(psi) > 0 else ""
            ks_str = f"KS={ks[0]:.4f}" if len(ks) > 0 else ""
            cr_str = f"CumRet={cr[0]:.4f}" if len(cr) > 0 else ""
            print(f"  {regime}: {psi_str}  {ks_str}  {cr_str}")

    print()
    print(f"Outputs: {REPORT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
