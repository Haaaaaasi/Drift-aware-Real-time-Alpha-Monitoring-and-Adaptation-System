"""分析 admission gate admitted alpha 的後續績效歸因。

注意：這是 period-level attribution，不是逐 alpha causal counterfactual。
同一個 retrain window 可能有多個 quarantine alpha 同時被 admit，也會和 XGB /
portfolio state 交互；因此本診斷用來判斷「是否值得繼續救 alpha expansion」，
而不是宣稱某個 alpha 單獨造成全部 PnL。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path("reports/adaptation_ab")
BEST_RUN_DIR = (
    BASE_DIR
    / "admission_gate_matrix_20260517"
    / "sim_20240701_20260430_top10_sched20_adgate_p4_s0p02_c0p95_nextvwap"
)
INCUMBENT_RUN_DIR = (
    BASE_DIR
    / "rolling_topk_stability_matrix_20260514"
    / "sim_20240701_20260430_top10_sched20_rtop20_w126_pen10_nextvwap"
)
OUT_DIR = BASE_DIR / "admission_gate_attribution_20260517"


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _cumulative_return(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    return float(np.prod(1.0 + returns.astype(float)) - 1.0)


def _read_pnl(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path / "daily_pnl.csv", parse_dates=["date"])
    return df[["date", "net_return"]].rename(columns={"net_return": f"{label}_return"})


def _build_periods() -> pd.DataFrame:
    snapshots = pd.read_csv(BEST_RUN_DIR / "alpha_selection_snapshots.csv", parse_dates=["as_of_date"])
    dates = snapshots["as_of_date"].sort_values().tolist()
    rows = []
    for idx, as_of in enumerate(dates):
        next_as_of = dates[idx + 1] if idx + 1 < len(dates) else pd.Timestamp("2026-05-01")
        rows.append(
            {
                "as_of_date": as_of,
                "period_start": as_of,
                "period_end_exclusive": next_as_of,
                "period_id": idx + 1,
            }
        )
    return pd.DataFrame(rows)


def _period_admission_rows(scores: pd.DataFrame) -> pd.DataFrame:
    scores = scores.copy()
    scores["as_of_date"] = pd.to_datetime(scores["as_of_date"])
    scores["selected"] = _as_bool(scores["selected"])
    admitted = scores[
        scores["alpha_pool"].eq("quarantine")
        & scores["admission_status"].eq("admitted")
    ].copy()
    admitted["selected_admitted"] = admitted["selected"]
    keep_cols = [
        "as_of_date",
        "alpha_id",
        "selected_admitted",
        "weight",
        "score",
        "raw_score",
        "rolling_rank_ic",
        "coverage",
        "admission_score",
        "admission_subwindow_pass_count",
        "max_abs_corr_to_live",
    ]
    return admitted[keep_cols]


def _build_period_attribution(periods: pd.DataFrame, admitted: pd.DataFrame) -> pd.DataFrame:
    admission = (
        admitted.groupby("as_of_date")
        .agg(
            admitted_alpha_ids=("alpha_id", lambda x: ",".join(sorted(map(str, x)))),
            selected_admitted_alpha_ids=(
                "alpha_id",
                lambda x: ",".join(
                    sorted(
                        map(
                            str,
                            admitted.loc[x.index][admitted.loc[x.index, "selected_admitted"]][
                                "alpha_id"
                            ],
                        )
                    )
                ),
            ),
            n_admitted=("alpha_id", "nunique"),
            n_selected_admitted=("selected_admitted", "sum"),
            admitted_weight=("weight", "sum"),
            avg_admission_score=("admission_score", "mean"),
            avg_pre_rolling_rank_ic=("rolling_rank_ic", "mean"),
            avg_max_abs_corr_to_live=("max_abs_corr_to_live", "mean"),
        )
        .reset_index()
    )
    period_df = periods.merge(admission, on="as_of_date", how="left")
    for col in ["admitted_alpha_ids", "selected_admitted_alpha_ids"]:
        period_df[col] = period_df[col].fillna("")
    for col in ["n_admitted", "n_selected_admitted", "admitted_weight"]:
        period_df[col] = period_df[col].fillna(0)

    gate = _read_pnl(BEST_RUN_DIR, "gate")
    incumbent = _read_pnl(INCUMBENT_RUN_DIR, "incumbent")
    pnl = gate.merge(incumbent, on="date", how="inner")

    rows = []
    for _, row in period_df.iterrows():
        mask = (pnl["date"] >= row["period_start"]) & (pnl["date"] < row["period_end_exclusive"])
        seg = pnl.loc[mask].copy()
        excess = seg["gate_return"] - seg["incumbent_return"]
        rows.append(
            {
                **row.to_dict(),
                "n_days": int(len(seg)),
                "gate_cum_return_pct": _cumulative_return(seg["gate_return"]) * 100.0,
                "incumbent_cum_return_pct": _cumulative_return(seg["incumbent_return"]) * 100.0,
                "excess_cum_return_pct": (
                    _cumulative_return(seg["gate_return"]) - _cumulative_return(seg["incumbent_return"])
                )
                * 100.0,
                "mean_daily_excess_bps": float(excess.mean() * 10_000.0) if not excess.empty else np.nan,
                "negative_excess_window": bool(excess.mean() < 0.0) if not excess.empty else None,
            }
        )
    return pd.DataFrame(rows)


def _alpha_attribution(period_attr: pd.DataFrame, admitted: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by_date = admitted.groupby("as_of_date")
    period_by_date = period_attr.set_index("as_of_date")
    for as_of, group in by_date:
        if as_of not in period_by_date.index:
            continue
        period = period_by_date.loc[as_of]
        n_admitted = max(int(period["n_admitted"]), 1)
        for _, alpha_row in group.iterrows():
            rows.append(
                {
                    "alpha_id": str(alpha_row["alpha_id"]),
                    "as_of_date": as_of,
                    "selected_admitted": bool(alpha_row["selected_admitted"]),
                    "weight": float(alpha_row["weight"]),
                    "admission_score": float(alpha_row["admission_score"]),
                    "pre_rolling_rank_ic": float(alpha_row["rolling_rank_ic"]),
                    "coverage": float(alpha_row["coverage"]),
                    "max_abs_corr_to_live": float(alpha_row["max_abs_corr_to_live"]),
                    "period_n_admitted": n_admitted,
                    "period_excess_cum_return_pct": float(period["excess_cum_return_pct"]),
                    "period_mean_daily_excess_bps": float(period["mean_daily_excess_bps"]),
                    "equal_share_excess_bps": float(period["mean_daily_excess_bps"]) / n_admitted,
                    "negative_excess_window": bool(period["negative_excess_window"]),
                }
            )
    exploded = pd.DataFrame(rows)
    if exploded.empty:
        return exploded
    agg = (
        exploded.groupby("alpha_id")
        .agg(
            n_admitted_periods=("as_of_date", "nunique"),
            first_admitted=("as_of_date", "min"),
            last_admitted=("as_of_date", "max"),
            avg_weight=("weight", "mean"),
            avg_admission_score=("admission_score", "mean"),
            avg_pre_rolling_rank_ic=("pre_rolling_rank_ic", "mean"),
            avg_max_abs_corr_to_live=("max_abs_corr_to_live", "mean"),
            avg_period_excess_bps=("period_mean_daily_excess_bps", "mean"),
            median_period_excess_bps=("period_mean_daily_excess_bps", "median"),
            avg_equal_share_excess_bps=("equal_share_excess_bps", "mean"),
            negative_window_rate=("negative_excess_window", "mean"),
        )
        .reset_index()
        .sort_values(["avg_period_excess_bps", "n_admitted_periods"], ascending=[True, False])
    )
    return agg


def _markdown_table(df: pd.DataFrame, columns: list[str] | None = None, limit: int | None = None) -> str:
    out = df.copy()
    if columns is not None:
        out = out[columns]
    if limit is not None:
        out = out.head(limit)
    headers = list(out.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in out.iterrows():
        vals = []
        for col in headers:
            val = row[col]
            if pd.isna(val):
                vals.append("")
            elif isinstance(val, (float, np.floating)):
                vals.append(f"{float(val):.3f}")
            elif isinstance(val, pd.Timestamp):
                vals.append(val.strftime("%Y-%m-%d"))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _write_summary(period_attr: pd.DataFrame, alpha_attr: pd.DataFrame) -> None:
    active_periods = period_attr[period_attr["n_admitted"] > 0].copy()
    manifest = {
        "best_run_dir": str(BEST_RUN_DIR),
        "incumbent_run_dir": str(INCUMBENT_RUN_DIR),
        "n_periods": int(len(period_attr)),
        "n_periods_with_admission": int(len(active_periods)),
        "admitted_period_negative_excess_rate": (
            float(active_periods["negative_excess_window"].mean()) if not active_periods.empty else None
        ),
        "avg_admitted_period_excess_bps": (
            float(active_periods["mean_daily_excess_bps"].mean()) if not active_periods.empty else None
        ),
        "avg_all_period_excess_bps": float(period_attr["mean_daily_excess_bps"].mean()),
    }
    (OUT_DIR / "attribution_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    worst = alpha_attr.sort_values(["avg_period_excess_bps", "n_admitted_periods"], ascending=[True, False])
    best = alpha_attr.sort_values(["avg_period_excess_bps", "n_admitted_periods"], ascending=[False, False])
    lines = [
        "# Admission Gate Admitted Alpha Attribution",
        "",
        "本報告是 period-level attribution，不是逐 alpha causal counterfactual。用途是判斷 alpha expansion 是否值得繼續救。",
        "",
        "## Overall",
        "",
        _markdown_table(pd.DataFrame([manifest])),
        "",
        "## Worst Associated Alpha",
        "",
        _markdown_table(
            worst,
            [
                "alpha_id",
                "n_admitted_periods",
                "avg_period_excess_bps",
                "median_period_excess_bps",
                "negative_window_rate",
                "avg_admission_score",
                "avg_pre_rolling_rank_ic",
            ],
            limit=15,
        ),
        "",
        "## Best Associated Alpha",
        "",
        _markdown_table(
            best,
            [
                "alpha_id",
                "n_admitted_periods",
                "avg_period_excess_bps",
                "median_period_excess_bps",
                "negative_window_rate",
                "avg_admission_score",
                "avg_pre_rolling_rank_ic",
            ],
            limit=15,
        ),
        "",
        "## Worst Periods",
        "",
        _markdown_table(
            period_attr.sort_values("mean_daily_excess_bps"),
            [
                "as_of_date",
                "admitted_alpha_ids",
                "n_admitted",
                "gate_cum_return_pct",
                "incumbent_cum_return_pct",
                "excess_cum_return_pct",
                "mean_daily_excess_bps",
            ],
            limit=10,
        ),
        "",
    ]
    (OUT_DIR / "admitted_alpha_attribution_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(BEST_RUN_DIR / "alpha_scores.csv")
    periods = _build_periods()
    admitted = _period_admission_rows(scores)
    period_attr = _build_period_attribution(periods, admitted)
    alpha_attr = _alpha_attribution(period_attr, admitted)

    admitted.to_csv(OUT_DIR / "admitted_alpha_events.csv", index=False, encoding="utf-8-sig")
    period_attr.to_csv(OUT_DIR / "admitted_period_attribution.csv", index=False, encoding="utf-8-sig")
    alpha_attr.to_csv(OUT_DIR / "admitted_alpha_attribution.csv", index=False, encoding="utf-8-sig")
    _write_summary(period_attr, alpha_attr)
    print(f"[done] wrote attribution to {OUT_DIR}")


if __name__ == "__main__":
    main()
