"""Summarize model-pool selector matrix simulation runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def _sharpe(net_return: pd.Series) -> float:
    std = float(net_return.std())
    if std == 0 or math.isnan(std):
        return float("nan")
    return float(net_return.mean()) / std * math.sqrt(252)


def _run_label(run_dir: Path) -> str:
    name = run_dir.name
    marker = "selector_"
    if marker in name:
        return name.split(marker, 1)[1].replace("_202206_202412", "")
    return name


def summarize_run(run_dir: Path) -> tuple[dict, list[dict]]:
    daily = pd.read_csv(run_dir / "daily_pnl.csv")
    decisions = pd.read_csv(run_dir / "model_pool_decisions.csv")
    retrain = pd.read_csv(run_dir / "retrain_log.csv")
    with (run_dir / "config.json").open("r", encoding="utf-8") as fh:
        config = json.load(fh)

    net = daily["net_return"]
    final_value = float(daily["cumulative_value"].iloc[-1])
    start_value = 10_000_000.0
    cumulative_return = final_value / start_value - 1
    annualized_return = (final_value / start_value) ** (252 / len(daily)) - 1
    drawdown = daily["cumulative_value"] / daily["cumulative_value"].cummax() - 1
    selected = decisions[decisions["selected"].astype(str).str.lower().eq("true")].copy()
    role_counts = selected["selected_role"].value_counts().to_dict()

    cost = daily["commission_cost"] + daily["tax_cost"] + daily["slippage_cost"]
    summary = {
        "label": _run_label(run_dir),
        "run_dir": str(run_dir),
        "selection_metric": config.get("model_pool_selection_metric"),
        "similarity_threshold": config.get("similarity_threshold"),
        "cum_return_pct": cumulative_return * 100,
        "annualized_return_pct": annualized_return * 100,
        "sharpe": _sharpe(net),
        "max_drawdown_pct": float(drawdown.min()) * 100,
        "avg_gross_bps": float(daily["gross_return"].mean()) * 10_000,
        "avg_net_bps": float(daily["net_return"].mean()) * 10_000,
        "avg_total_cost_bps": float(cost.mean()) * 10_000,
        "avg_turnover": float(daily["turnover"].mean()),
        "avg_holdings": float(daily["n_holdings"].mean()),
        "n_days": len(daily),
        "n_retrains": len(retrain),
        "n_pool_reuses": int(retrain["reason"].astype(str).str.contains("reused").sum()),
        "n_pool_misses": int(retrain["reason"].astype(str).str.contains("pool_miss").sum()),
        "selected_current": int(role_counts.get("current", 0)),
        "selected_new": int(role_counts.get("new", 0)),
        "selected_reused": int(role_counts.get("reused", 0)),
        "decision_rows": len(decisions),
        "trigger_events": int(selected["date"].nunique()),
        "avg_selected_shadow_topk_net": float(selected["shadow_topk_net_return"].mean()),
        "avg_selected_proxy_net": float(selected["proxy_net_return"].mean()),
        "selected_proxy_rank_mean": float(selected["proxy_rank_by_net"].mean()),
    }

    role_rows = []
    for role, group in selected.groupby("selected_role"):
        role_rows.append(
            {
                "label": summary["label"],
                "selected_role": role,
                "n": len(group),
                "mean_shadow_topk_net": float(group["shadow_topk_net_return"].mean()),
                "mean_proxy_net": float(group["proxy_net_return"].mean()),
                "mean_proxy_rank": float(group["proxy_rank_by_net"].mean()),
                "mean_similarity": float(group["selected_similarity"].mean()),
            }
        )

    return summary, role_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out-prefix", default="selector_matrix")
    args = parser.parse_args()

    run_dirs = sorted(
        path
        for path in args.root.iterdir()
        if path.is_dir()
        and path.name.startswith("sim_")
        and "selector_" in path.name
        and (path / "model_pool_decisions.csv").exists()
    )
    summaries = []
    roles = []
    for run_dir in run_dirs:
        summary, role_rows = summarize_run(run_dir)
        summaries.append(summary)
        roles.extend(role_rows)

    summary_df = pd.DataFrame(summaries).sort_values(
        ["selection_metric", "similarity_threshold", "label"]
    )
    role_df = pd.DataFrame(roles).sort_values(["label", "selected_role"])

    summary_path = args.root / f"{args.out_prefix}_summary.csv"
    role_path = args.root / f"{args.out_prefix}_selected_role_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    role_df.to_csv(role_path, index=False, encoding="utf-8-sig")

    print(summary_df.to_string(index=False))
    print("\n--- selected roles ---")
    print(role_df.to_string(index=False))
    print(f"\nWrote: {summary_path}")
    print(f"Wrote: {role_path}")


if __name__ == "__main__":
    main()
