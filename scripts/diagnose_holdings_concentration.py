"""診斷 turnover-aware 實驗的持倉集中度。

輸入一個或多個 A/B run 目錄，讀取各策略的 holdings.csv / daily_pnl.csv，
輸出每日集中度與策略層級摘要。此腳本只讀既有報告，不重新跑回測。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RUN_DIRS = [
    Path("reports/adaptation_ab/ab_20240701_20260430_top10_oos_no_indcap_turnoveraware_e20x40_t25_h5_nextopen_20260511"),
    Path("reports/adaptation_ab/ab_20240701_20260430_top10_oos_no_indcap_turnoveraware_e20x40_t25_h5_nextvwap_20260511"),
]
DEFAULT_OUT_DIR = Path("reports/adaptation_ab/holdings_concentration_20260512")


def _strategy_from_sim_dir(path: Path) -> str:
    name = path.name
    if "sched20" in name or "scheduled_20" in name:
        return "scheduled_20"
    if "none" in name:
        return "none"
    return name


def _execution_from_run_dir(path: Path) -> str:
    name = path.name.lower()
    if "nextopen" in name or "next_open" in name:
        return "next_open"
    if "nextvwap" in name or "next_vwap" in name:
        return "next_vwap"
    return "unknown"


def _add_holding_age(holdings: pd.DataFrame) -> pd.DataFrame:
    holdings = holdings.sort_values(["security_id", "date"]).copy()
    holdings["date"] = pd.to_datetime(holdings["date"])
    age_chunks: list[pd.Series] = []
    for _, group in holdings.groupby("security_id", sort=False):
        dates = group["date"]
        breaks = dates.diff().dt.days.fillna(999).gt(7).cumsum()
        age = group.groupby(breaks).cumcount() + 1
        age_chunks.append(age)
    holdings["holding_age_days"] = pd.concat(age_chunks).sort_index() if age_chunks else []
    return holdings


def _daily_concentration(holdings: pd.DataFrame, daily_pnl: pd.DataFrame) -> pd.DataFrame:
    if holdings.empty:
        return pd.DataFrame()

    holdings = _add_holding_age(holdings)
    rows: list[dict] = []
    for date, group in holdings.groupby("date", sort=True):
        weights = group["target_weight"].astype(float)
        abs_w = weights.abs().sort_values(ascending=False)
        gross = float(abs_w.sum())
        if gross <= 0:
            continue
        norm_w = abs_w / gross
        effective = float(1.0 / np.square(norm_w).sum()) if len(norm_w) else 0.0
        score = group["signal_score"].astype(float)
        neg_mask = score <= 0
        rows.append({
            "date": date.date().isoformat(),
            "n_positions": int(len(group)),
            "gross_exposure_from_holdings": gross,
            "effective_holdings": effective,
            "top1_weight_share": float(norm_w.head(1).sum()),
            "top5_weight_share": float(norm_w.head(5).sum()),
            "top10_weight_share": float(norm_w.head(10).sum()),
            "top20_weight_share": float(norm_w.head(20).sum()),
            "tail_count_lt_25bps": int((abs_w < 0.0025).sum()),
            "tail_weight_lt_25bps": float(abs_w[abs_w < 0.0025].sum()),
            "tail_count_lt_50bps": int((abs_w < 0.005).sum()),
            "tail_weight_lt_50bps": float(abs_w[abs_w < 0.005].sum()),
            "tail_count_lt_1pct": int((abs_w < 0.01).sum()),
            "tail_weight_lt_1pct": float(abs_w[abs_w < 0.01].sum()),
            "max_weight": float(abs_w.max()),
            "median_weight": float(abs_w.median()),
            "min_weight": float(abs_w.min()),
            "negative_score_count": int(neg_mask.sum()),
            "negative_score_weight": float(group.loc[neg_mask, "target_weight"].abs().sum()),
            "mean_holding_age_days": float(group["holding_age_days"].mean()),
            "max_holding_age_days": int(group["holding_age_days"].max()),
        })

    daily = pd.DataFrame(rows)
    if daily.empty or daily_pnl.empty:
        return daily

    pnl_cols = [
        "date",
        "turnover",
        "rebalance_flag",
        "held_from_prev_count",
        "forced_sells_count",
        "turnover_cap_applied",
        "net_return",
        "gross_return",
    ]
    available = [c for c in pnl_cols if c in daily_pnl.columns]
    pnl = daily_pnl[available].copy()
    pnl["date"] = pd.to_datetime(pnl["date"]).dt.date.astype(str)
    return daily.merge(pnl, on="date", how="left")


def _summarise_daily(run_name: str, execution: str, strategy: str, daily: pd.DataFrame) -> dict:
    if daily.empty:
        return {
            "run_name": run_name,
            "execution": execution,
            "strategy": strategy,
            "n_days": 0,
        }

    cap_days = (
        daily.get("turnover_cap_applied", pd.Series([False] * len(daily)))
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    )
    return {
        "run_name": run_name,
        "execution": execution,
        "strategy": strategy,
        "n_days": int(len(daily)),
        "avg_n_positions": float(daily["n_positions"].mean()),
        "p95_n_positions": float(daily["n_positions"].quantile(0.95)),
        "avg_effective_holdings": float(daily["effective_holdings"].mean()),
        "p95_effective_holdings": float(daily["effective_holdings"].quantile(0.95)),
        "avg_top10_weight_share": float(daily["top10_weight_share"].mean()),
        "avg_top20_weight_share": float(daily["top20_weight_share"].mean()),
        "avg_tail_count_lt_50bps": float(daily["tail_count_lt_50bps"].mean()),
        "avg_tail_weight_lt_50bps": float(daily["tail_weight_lt_50bps"].mean()),
        "avg_negative_score_count": float(daily["negative_score_count"].mean()),
        "avg_negative_score_weight": float(daily["negative_score_weight"].mean()),
        "avg_holding_age_days": float(daily["mean_holding_age_days"].mean()),
        "p95_max_holding_age_days": float(daily["max_holding_age_days"].quantile(0.95)),
        "turnover_cap_applied_pct": float(cap_days.mean() * 100.0),
        "avg_turnover": float(daily["turnover"].mean()) if "turnover" in daily else np.nan,
        "avg_net_return_bps": float(daily["net_return"].mean() * 1e4) if "net_return" in daily else np.nan,
    }


def analyse_run_dir(run_dir: Path, out_dir: Path) -> list[dict]:
    run_dir = Path(run_dir)
    execution = _execution_from_run_dir(run_dir)
    summaries: list[dict] = []
    for holdings_path in sorted((run_dir / "simulations").glob("*/holdings.csv")):
        sim_dir = holdings_path.parent
        strategy = _strategy_from_sim_dir(sim_dir)
        daily_pnl_path = sim_dir / "daily_pnl.csv"
        holdings = pd.read_csv(holdings_path)
        daily_pnl = pd.read_csv(daily_pnl_path) if daily_pnl_path.exists() else pd.DataFrame()
        daily = _daily_concentration(holdings, daily_pnl)
        daily.insert(0, "strategy", strategy)
        daily.insert(0, "execution", execution)
        daily.insert(0, "run_name", run_dir.name)
        out_csv = out_dir / f"{run_dir.name}_{strategy}_daily_concentration.csv"
        daily.to_csv(out_csv, index=False, encoding="utf-8-sig")
        summaries.append(_summarise_daily(run_dir.name, execution, strategy, daily))
    return summaries


def write_markdown(summary: pd.DataFrame, out_path: Path) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        headers = list(frame.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in frame.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
        return "\n".join(lines)

    lines = [
        "# Holdings concentration 診斷",
        "",
        "## 策略層級摘要",
        "",
    ]
    display_cols = [
        "execution",
        "strategy",
        "avg_n_positions",
        "avg_effective_holdings",
        "avg_top10_weight_share",
        "avg_tail_count_lt_50bps",
        "avg_negative_score_weight",
        "turnover_cap_applied_pct",
        "avg_turnover",
        "avg_net_return_bps",
    ]
    md = summary[display_cols].copy()
    for col in md.columns:
        if col not in {"execution", "strategy"}:
            md[col] = md[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    lines.append(markdown_table(md))
    lines.extend([
        "",
        "## 判讀重點",
        "",
        "- `avg_effective_holdings` 若遠高於 10，代表低換手後的實際組合已不是嚴格 top-10。",
        "- `avg_tail_count_lt_50bps` / `avg_tail_weight_lt_50bps` 用來檢查小殘倉是否堆積。",
        "- `avg_negative_score_weight` 若偏高，代表有不少權重留在 XGB 當下已不看好的股票。",
        "- `turnover_cap_applied_pct` 越高，代表績效越依賴 turnover cap，而不是單純 top-k ranking。",
        "",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    run_dirs = args.run_dir or DEFAULT_RUN_DIRS
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    for run_dir in run_dirs:
        summaries.extend(analyse_run_dir(run_dir, args.out_dir))

    summary = pd.DataFrame(summaries)
    summary_csv = args.out_dir / "holdings_concentration_summary.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    write_markdown(summary, args.out_dir / "holdings_concentration_summary.md")
    print(f"wrote {summary_csv}")


if __name__ == "__main__":
    main()
