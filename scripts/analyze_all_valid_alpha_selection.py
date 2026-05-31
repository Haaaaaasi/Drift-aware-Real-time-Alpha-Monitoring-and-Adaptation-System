"""彙整 all_valid alpha selection 實驗結果。

用途：
1. 比較 rolling_topk over all_valid 與既有 55-alpha incumbent。
2. 產出 regime 分段、paired t-test、circular block bootstrap。
3. 檢查新增 alpha 候選是否大量擠入 selector snapshot。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path("reports/adaptation_ab")
OUT_DIR = BASE_DIR / "rolling_topk_all_valid_oos_20260516"
BENCHMARK_DIR = BASE_DIR / "rolling_topk_validation_20260514" / "benchmark_sensitivity"
ALPHA_AUDIT_MANIFEST = Path("reports/alpha_audit/all_valid_alpha_audit_20260516/alpha_audit_manifest.json")

RUN_DIRS = {
    "all_valid_82": {
        "next_vwap": OUT_DIR / "sim_20240701_20260430_top10_sched20_allvalid_rtop20_w126_pen10_nextvwap",
        "next_open": OUT_DIR / "sim_20240701_20260430_top10_sched20_allvalid_rtop20_w126_pen10_nextopen",
    },
    "incumbent_55": {
        "next_vwap": BASE_DIR
        / "rolling_topk_stability_matrix_20260514"
        / "sim_20240701_20260430_top10_sched20_rtop20_w126_pen10_nextvwap",
        "next_open": BASE_DIR
        / "rolling_topk_best_execution_check_20260514"
        / "sim_20240701_20260430_top10_sched20_rtop20_w126_pen10_nextopen",
    },
    "static_is_55": {
        "next_vwap": BASE_DIR
        / "selector_equivalence_full_20260514"
        / "sim_20240701_20260430_top10_sched20_static_is_nextvwap",
        "next_open": BASE_DIR
        / "selector_equivalence_full_20260514"
        / "sim_20240701_20260430_top10_sched20_static_is_nextopen",
    },
}

REGIMES = [
    ("2024_H2", "2024-07-01", "2024-12-31"),
    ("2025_H1", "2025-01-01", "2025-06-30"),
    ("2025_H2", "2025-07-01", "2025-12-31"),
    ("2026_YTD", "2026-01-01", "2026-04-30"),
]


def read_pnl(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    if "net_return" not in df.columns:
        raise ValueError(f"{path} 缺少 net_return 欄位")
    return df


def summarize_pnl(df: pd.DataFrame) -> dict[str, float | int]:
    returns = df["net_return"].astype(float)
    n_days = int(len(returns))
    cumulative = float(np.prod(1.0 + returns) - 1.0)
    annualized = float((1.0 + cumulative) ** (252.0 / n_days) - 1.0) if n_days else np.nan
    std = float(returns.std(ddof=1))
    sharpe = 0.0 if std == 0.0 else float(returns.mean() / std * math.sqrt(252.0))
    equity = (1.0 + returns).cumprod()
    max_drawdown = float((equity / equity.cummax() - 1.0).min())
    out: dict[str, float | int] = {
        "n_days": n_days,
        "cumulative_return_pct": cumulative * 100.0,
        "annualized_return_pct": annualized * 100.0,
        "sharpe": sharpe,
        "max_drawdown_pct": max_drawdown * 100.0,
        "win_rate_pct": float((returns > 0.0).mean() * 100.0),
        "avg_net_return_bps": float(returns.mean() * 10_000.0),
    }
    if "turnover" in df.columns:
        out["avg_turnover"] = float(df["turnover"].mean())
    if "n_holdings" in df.columns:
        out["avg_holdings"] = float(df["n_holdings"].mean())
    return out


def benchmark_path(execution_price: str, benchmark: str) -> Path:
    return BENCHMARK_DIR / f"{execution_price}_{benchmark}_daily_pnl.csv"


def load_series() -> dict[tuple[str, str], pd.DataFrame]:
    series: dict[tuple[str, str], pd.DataFrame] = {}
    for label, by_execution in RUN_DIRS.items():
        for execution_price, run_dir in by_execution.items():
            series[(execution_price, label)] = read_pnl(run_dir / "daily_pnl.csv")
    for execution_price in ["next_vwap", "next_open"]:
        for benchmark in ["ew_same_cadence_liq100m", "ew_same_cadence_liq200m"]:
            series[(execution_price, benchmark)] = read_pnl(benchmark_path(execution_price, benchmark))
    return series


def write_comparison_summary(series: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for (execution_price, label), df in sorted(series.items()):
        rows.append({"execution_price": execution_price, "series": label, **summarize_pnl(df)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "all_valid_comparison_summary.csv", index=False, encoding="utf-8-sig")
    return out


def segment_summary(df: pd.DataFrame, start: str, end: str) -> dict[str, float | int]:
    mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    seg = df.loc[mask].copy()
    if seg.empty:
        return {"n_days": 0}
    return summarize_pnl(seg)


def write_regime_breakdown(series: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for (execution_price, label), df in sorted(series.items()):
        for regime, start, end in REGIMES:
            rows.append(
                {
                    "execution_price": execution_price,
                    "series": label,
                    "regime": regime,
                    "start": start,
                    "end": end,
                    **segment_summary(df, start, end),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "all_valid_regime_breakdown.csv", index=False, encoding="utf-8-sig")
    return out


def paired_t(diff: pd.Series) -> tuple[float, float]:
    x = diff.dropna().astype(float)
    if len(x) < 3:
        return float("nan"), float("nan")
    std = float(x.std(ddof=1))
    mean = float(x.mean())
    if std == 0.0:
        return (float("inf"), 0.0) if mean > 0 else (float("-inf"), 1.0)
    t_stat = mean / (std / math.sqrt(len(x)))
    try:
        from scipy import stats

        p_one_sided = float(1.0 - stats.t.cdf(t_stat, df=len(x) - 1))
    except Exception:
        p_one_sided = float("nan")
    return float(t_stat), p_one_sided


def block_bootstrap(
    diff: pd.Series,
    *,
    block_len: int = 20,
    n_boot: int = 3000,
    seed: int = 20260516,
) -> dict[str, float | int]:
    x = diff.dropna().astype(float).to_numpy()
    n = len(x)
    if n == 0:
        return {
            "block_len": block_len,
            "n_boot": n_boot,
            "bootstrap_mean_bps": np.nan,
            "bootstrap_p_one_sided": np.nan,
            "bootstrap_ci05_bps": np.nan,
            "bootstrap_ci50_bps": np.nan,
            "bootstrap_ci95_bps": np.nan,
        }
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=math.ceil(n / block_len))
        sample: list[float] = []
        for start in starts:
            idx = (np.arange(block_len) + start) % n
            sample.extend(x[idx])
            if len(sample) >= n:
                break
        means[i] = np.mean(sample[:n])
    return {
        "block_len": block_len,
        "n_boot": n_boot,
        "bootstrap_mean_bps": float(means.mean() * 10_000.0),
        "bootstrap_p_one_sided": float((means <= 0.0).mean()),
        "bootstrap_ci05_bps": float(np.quantile(means, 0.05) * 10_000.0),
        "bootstrap_ci50_bps": float(np.quantile(means, 0.50) * 10_000.0),
        "bootstrap_ci95_bps": float(np.quantile(means, 0.95) * 10_000.0),
    }


def write_bootstrap(series: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    comparisons = [
        ("all_valid_82", "incumbent_55"),
        ("incumbent_55", "all_valid_82"),
        ("all_valid_82", "static_is_55"),
        ("all_valid_82", "ew_same_cadence_liq100m"),
        ("all_valid_82", "ew_same_cadence_liq200m"),
    ]
    for execution_price in ["next_vwap", "next_open"]:
        for contender, baseline in comparisons:
            left = series[(execution_price, contender)][["date", "net_return"]].rename(
                columns={"net_return": "contender_return"}
            )
            right = series[(execution_price, baseline)][["date", "net_return"]].rename(
                columns={"net_return": "baseline_return"}
            )
            merged = left.merge(right, on="date", how="inner")
            diff = merged["contender_return"] - merged["baseline_return"]
            t_stat, p_one_sided = paired_t(diff)
            rows.append(
                {
                    "execution_price": execution_price,
                    "comparison": f"{contender}_vs_{baseline}",
                    "n_days": int(len(diff)),
                    "mean_daily_excess_bps": float(diff.mean() * 10_000.0),
                    "paired_t_stat": t_stat,
                    "paired_t_p_one_sided": p_one_sided,
                    **block_bootstrap(diff),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "all_valid_bootstrap_paired_results.csv", index=False, encoding="utf-8-sig")
    return out


def read_manifest_lists() -> tuple[set[str], set[str]]:
    import json

    manifest = json.loads(ALPHA_AUDIT_MANIFEST.read_text(encoding="utf-8"))
    all_valid = set(manifest["all_valid_alpha_ids"])
    effective = set(manifest["effective_no_indcap_alpha_ids"])
    return all_valid, effective


def write_alpha_selection_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame]:
    all_valid, effective = read_manifest_lists()
    added = all_valid - effective
    all_weights = pd.read_csv(RUN_DIRS["all_valid_82"]["next_vwap"] / "alpha_weights_by_date.csv")
    incumbent_weights = pd.read_csv(RUN_DIRS["incumbent_55"]["next_vwap"] / "alpha_weights_by_date.csv")

    all_counts = all_weights.groupby("alpha_id").size().rename("all_valid_selected_count")
    incumbent_counts = incumbent_weights.groupby("alpha_id").size().rename("incumbent_selected_count")
    freq = pd.concat([all_counts, incumbent_counts], axis=1).fillna(0).reset_index()
    freq["is_added_candidate"] = freq["alpha_id"].isin(added)
    freq["all_valid_selected_count"] = freq["all_valid_selected_count"].astype(int)
    freq["incumbent_selected_count"] = freq["incumbent_selected_count"].astype(int)
    freq["selection_count_delta"] = freq["all_valid_selected_count"] - freq["incumbent_selected_count"]
    freq = freq.sort_values(
        ["all_valid_selected_count", "selection_count_delta", "alpha_id"],
        ascending=[False, False, True],
    )
    freq.to_csv(OUT_DIR / "all_valid_selected_alpha_frequency.csv", index=False, encoding="utf-8-sig")

    per_snapshot = (
        all_weights.assign(is_added_candidate=all_weights["alpha_id"].isin(added))
        .groupby("as_of_date")
        .agg(
            n_selected=("alpha_id", "size"),
            n_added_selected=("is_added_candidate", "sum"),
            added_weight=("weight", lambda x: float(x[all_weights.loc[x.index, "alpha_id"].isin(added)].sum())),
        )
        .reset_index()
    )
    per_snapshot["added_selected_share"] = per_snapshot["n_added_selected"] / per_snapshot["n_selected"]
    summary = pd.DataFrame(
        [
            {
                "n_all_valid_alphas": len(all_valid),
                "n_effective_no_indcap_alphas": len(effective),
                "n_added_candidates": len(added),
                "n_snapshots": int(per_snapshot["as_of_date"].nunique()),
                "avg_added_selected_count": float(per_snapshot["n_added_selected"].mean()),
                "max_added_selected_count": int(per_snapshot["n_added_selected"].max()),
                "avg_added_selected_share": float(per_snapshot["added_selected_share"].mean()),
                "avg_added_weight": float(per_snapshot["added_weight"].mean()),
            }
        ]
    )
    summary.to_csv(OUT_DIR / "all_valid_selection_pool_shift_summary.csv", index=False, encoding="utf-8-sig")
    return freq, summary


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None, limit: int | None = None) -> str:
    df = frame.copy()
    if columns is not None:
        df = df[columns]
    if limit is not None:
        df = df.head(limit)
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in headers:
            val = row[col]
            if pd.isna(val):
                vals.append("")
            elif isinstance(val, (float, np.floating)):
                vals.append(f"{float(val):.3f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_markdown(
    comparison: pd.DataFrame,
    regime: pd.DataFrame,
    bootstrap: pd.DataFrame,
    freq: pd.DataFrame,
    pool_shift: pd.DataFrame,
) -> None:
    top_new = freq[freq["is_added_candidate"]].head(15)
    compact_comparison = comparison[
        comparison["series"].isin(
            [
                "all_valid_82",
                "incumbent_55",
                "static_is_55",
                "ew_same_cadence_liq100m",
                "ew_same_cadence_liq200m",
            ]
        )
    ].sort_values(["execution_price", "series"])
    key_boot = bootstrap[
        bootstrap["comparison"].isin(
            [
                "all_valid_82_vs_incumbent_55",
                "incumbent_55_vs_all_valid_82",
                "all_valid_82_vs_static_is_55",
                "all_valid_82_vs_ew_same_cadence_liq100m",
                "all_valid_82_vs_ew_same_cadence_liq200m",
            ]
        )
    ]
    lines = [
        "# rolling_topk all_valid alpha 實驗彙整",
        "",
        "- OOS: 2024-07-01 -> 2026-04-30",
        "- Selector: rolling_topk20_w126_pen10",
        "- all_valid 定義：101 WQ alpha 中排除需要真實 indclass / cap 的 alpha，未再用 IS IC 預篩。",
        "- 結論：82-alpha all_valid 候選池可跑通，但明顯輸給 55-alpha incumbent；盲目擴 alpha 會引入 selector noise。",
        "",
        "## Summary",
        "",
        markdown_table(
            compact_comparison,
            [
                "execution_price",
                "series",
                "cumulative_return_pct",
                "sharpe",
                "max_drawdown_pct",
                "avg_net_return_bps",
                "avg_turnover",
                "avg_holdings",
            ],
        ),
        "",
        "## Paired / Block Bootstrap",
        "",
        "p 值方向為 contender - baseline > 0；因此 all_valid_vs_incumbent 的 p 接近 1 表示 all_valid 沒有勝出，incumbent_vs_all_valid 的 p 接近 0 表示 incumbent 勝出。",
        "",
        markdown_table(
            key_boot,
            [
                "execution_price",
                "comparison",
                "mean_daily_excess_bps",
                "paired_t_p_one_sided",
                "bootstrap_p_one_sided",
                "bootstrap_ci05_bps",
                "bootstrap_ci95_bps",
            ],
        ),
        "",
        "## Selection Pool Shift",
        "",
        markdown_table(pool_shift),
        "",
        "## Top Added Candidates Selected By all_valid",
        "",
        markdown_table(
            top_new,
            [
                "alpha_id",
                "all_valid_selected_count",
                "incumbent_selected_count",
                "selection_count_delta",
                "is_added_candidate",
            ],
        ),
        "",
        "## Regime Breakdown",
        "",
        markdown_table(
            regime[regime["series"].isin(["all_valid_82", "incumbent_55", "static_is_55"])],
            [
                "execution_price",
                "series",
                "regime",
                "cumulative_return_pct",
                "sharpe",
                "max_drawdown_pct",
                "avg_net_return_bps",
            ],
        ),
        "",
    ]
    (OUT_DIR / "all_valid_experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    series = load_series()
    comparison = write_comparison_summary(series)
    regime = write_regime_breakdown(series)
    bootstrap = write_bootstrap(series)
    freq, pool_shift = write_alpha_selection_diagnostics()
    write_markdown(comparison, regime, bootstrap, freq, pool_shift)
    print(f"[done] wrote all_valid diagnostics to {OUT_DIR}")


if __name__ == "__main__":
    main()
