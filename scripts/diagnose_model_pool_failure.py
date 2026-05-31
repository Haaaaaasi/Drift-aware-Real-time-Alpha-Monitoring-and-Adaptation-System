"""診斷 model_pool 在正式 WP9 設定下失敗的決策型態。

本腳本刻意分成兩層證據：

1. observed event attribution：實際採用 model_pool 決策後，下一持股週期相對 baseline
   的表現。
2. candidate proxy：同一個 trigger 當下，各候選模型用同一截面形成 top-k 組合後的
   下一持股週期 proxy。

第二層不是完整 counterfactual 回測，只能作為候選品質 proxy。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_OUT_DIR = Path("reports/adaptation_ab/model_pool_failure_diagnosis_20260507")
BASELINES = ["none", "scheduled_20", "triggered"]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(path: str | Path, base: Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if base is not None and (base / p).exists():
        return base / p
    return p


def _load_pnl(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def _post_window_return(pnl: pd.DataFrame, event_date: pd.Timestamp, n_days: int) -> dict:
    idx = pnl.index[pnl["date"] >= event_date]
    if len(idx) == 0:
        return {
            "n_post_days": 0,
            "post_gross_return": np.nan,
            "post_net_return": np.nan,
        }
    start_idx = int(idx[0])
    win = pnl.iloc[start_idx:start_idx + n_days]
    if win.empty:
        return {
            "n_post_days": 0,
            "post_gross_return": np.nan,
            "post_net_return": np.nan,
        }
    gross = np.prod(1.0 + win["gross_return"].astype(float).to_numpy()) - 1.0
    net = np.prod(1.0 + win["net_return"].astype(float).to_numpy()) - 1.0
    return {
        "n_post_days": int(len(win)),
        "post_gross_return": float(gross),
        "post_net_return": float(net),
    }


def _load_ab_daily_pnls(ab_run_dir: Path) -> dict[str, pd.DataFrame]:
    cfg = _read_json(ab_run_dir / "config.json")
    out: dict[str, pd.DataFrame] = {}
    for strat in BASELINES:
        sim_dir = _resolve_path(cfg["run_dirs"][strat])
        out[strat] = _load_pnl(sim_dir / "daily_pnl.csv")
    return out


def _resolve_model_pool_run_dir(ab_run_dir: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    cfg = _read_json(ab_run_dir / "config.json")
    return _resolve_path(cfg["run_dirs"]["model_pool"])


def build_event_attribution(
    *,
    decisions: pd.DataFrame,
    model_pool_pnl: pd.DataFrame,
    baseline_pnls: dict[str, pd.DataFrame],
    n_days: int,
) -> pd.DataFrame:
    selected = decisions[decisions["selected"].astype(bool)].copy()
    selected["date"] = pd.to_datetime(selected["date"])
    rows = []
    for _, row in selected.sort_values(["date", "candidate_model_id"]).iterrows():
        event_date = pd.Timestamp(row["date"])
        actual = _post_window_return(model_pool_pnl, event_date, n_days)
        rec = {
            "date": event_date.strftime("%Y-%m-%d"),
            "day_idx": int(row["day_idx"]),
            "selected_candidate_model_id": row.get("selected_candidate_model_id"),
            "applied_model_id": row.get("applied_model_id"),
            "selected_role": row.get("selected_role"),
            "decision_reason": row.get("decision_reason"),
            "pool_hit": bool(row.get("pool_hit")),
            "raw_best_candidate_model_id": row.get("raw_best_candidate_model_id"),
            "raw_best_role": row.get("raw_best_role"),
            "raw_best_score": row.get("raw_best_score"),
            "best_non_reused_model_id": row.get("best_non_reused_model_id"),
            "best_non_reused_score": row.get("best_non_reused_score"),
            "reuse_score_margin_vs_best_non_reused": row.get("reuse_score_margin_vs_best_non_reused"),
            "reuse_guard_passed": row.get("reuse_guard_passed"),
            "reuse_guard_reason": row.get("reuse_guard_reason"),
            "selected_similarity": row.get("selected_similarity"),
            "best_seen_similarity": row.get("best_seen_similarity"),
            "proxy_rank_by_net": row.get("proxy_rank_by_net"),
            "selected_proxy_net_return": row.get("proxy_net_return"),
            "selected_shadow_ic": row.get("shadow_ic"),
            **actual,
        }
        for strat, pnl in baseline_pnls.items():
            base = _post_window_return(pnl, event_date, n_days)
            rec[f"{strat}_post_net_return"] = base["post_net_return"]
            rec[f"excess_vs_{strat}"] = (
                actual["post_net_return"] - base["post_net_return"]
                if pd.notna(actual["post_net_return"]) and pd.notna(base["post_net_return"])
                else np.nan
            )
        rows.append(rec)
    return pd.DataFrame(rows)


def _plot_diagnostics(event_df: pd.DataFrame, candidate_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if not event_df.empty:
        by_role = event_df.groupby("selected_role")["post_net_return"].mean().sort_index()
        axes[0].bar(by_role.index.astype(str), by_role.values * 100.0, color="#4C78A8")
        axes[0].axhline(0.0, color="gray", linewidth=0.8)
        axes[0].set_title("Observed post-window net return")
        axes[0].set_ylabel("Mean return (%)")
        axes[0].set_xlabel("Selected role")

    if not candidate_df.empty:
        by_candidate = candidate_df.groupby("candidate_role")["proxy_net_return"].mean().sort_index()
        axes[1].bar(by_candidate.index.astype(str), by_candidate.values * 100.0, color="#F58518")
        axes[1].axhline(0.0, color="gray", linewidth=0.8)
        axes[1].set_title("Candidate post-window proxy")
        axes[1].set_ylabel("Mean proxy return (%)")
        axes[1].set_xlabel("Candidate role")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _fmt_pct(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return "NA"
    return f"{float(x) * 100:.2f}%"


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "(無資料)"
    out = df.copy()
    out.insert(0, "index", out.index.astype(str))
    cols = list(out.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _available_aggs(df: pd.DataFrame, specs: dict[str, tuple[str, str]]) -> dict[str, tuple[str, str]]:
    return {name: spec for name, spec in specs.items() if spec[0] in df.columns}


def write_summary(
    *,
    out_path: Path,
    ab_run_dir: Path,
    model_pool_run_dir: Path,
    event_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    n_days: int,
) -> None:
    selected_role_summary = pd.DataFrame()
    if not event_df.empty:
        selected_aggs = _available_aggs(
            event_df,
            {
                "n_events": ("date", "count"),
                "mean_post_net_return": ("post_net_return", "mean"),
                "mean_excess_vs_scheduled20": ("excess_vs_scheduled_20", "mean"),
                "mean_selected_proxy_net": ("selected_proxy_net_return", "mean"),
                "mean_proxy_rank": ("proxy_rank_by_net", "mean"),
                "mean_raw_best_score": ("raw_best_score", "mean"),
                "mean_best_non_reused_score": ("best_non_reused_score", "mean"),
                "mean_reuse_margin_vs_non_reused": (
                    "reuse_score_margin_vs_best_non_reused",
                    "mean",
                ),
            },
        )
        selected_role_summary = (
            event_df.groupby("selected_role")
            .agg(**selected_aggs)
            .sort_index()
        )

    candidate_role_summary = pd.DataFrame()
    if not candidate_df.empty:
        candidate_aggs = _available_aggs(
            candidate_df,
            {
                "n_candidates": ("candidate_model_id", "count"),
                "mean_shadow_ic": ("shadow_ic", "mean"),
                "mean_shadow_sharpe": ("shadow_sharpe", "mean"),
                "mean_shadow_rank_selection": ("shadow_rank_by_selection_metric", "mean"),
                "mean_shadow_rank_topk_net": ("shadow_rank_by_topk_net_return", "mean"),
                "mean_proxy_net_return": ("proxy_net_return", "mean"),
                "mean_proxy_rank": ("proxy_rank_by_net", "mean"),
                "mean_reuse_margin_vs_non_reused": (
                    "reuse_score_margin_vs_best_non_reused",
                    "mean",
                ),
            },
        )
        candidate_role_summary = (
            candidate_df.groupby("candidate_role")
            .agg(**candidate_aggs)
            .sort_index()
        )

    reused_selected = event_df[event_df["selected_role"] == "reused"] if not event_df.empty else pd.DataFrame()
    reused_candidate = candidate_df[candidate_df["candidate_role"] == "reused"] if not candidate_df.empty else pd.DataFrame()
    raw_best_summary = pd.DataFrame()
    if not candidate_df.empty and "raw_best_role" in candidate_df.columns:
        raw_best_events = candidate_df.drop_duplicates(["date", "raw_best_candidate_model_id"]).copy()
        raw_best_events["raw_best_role"] = raw_best_events["raw_best_role"].fillna("none")
        raw_best_summary = (
            raw_best_events.groupby("raw_best_role")
            .agg(
                n_events=("date", "count"),
                mean_raw_best_score=("raw_best_score", "mean"),
                mean_best_non_reused_score=("best_non_reused_score", "mean"),
            )
            .sort_index()
        )
    reuse_guard_summary = pd.DataFrame()
    if not candidate_df.empty and "reuse_guard_reason" in candidate_df.columns:
        guard_events = candidate_df.drop_duplicates(["date", "raw_best_candidate_model_id"]).copy()
        guard_events["reuse_guard_reason"] = guard_events["reuse_guard_reason"].fillna("not_applicable")
        reuse_guard_summary = (
            guard_events.groupby("reuse_guard_reason")
            .agg(
                n_events=("date", "count"),
                mean_raw_best_score=("raw_best_score", "mean"),
                mean_best_non_reused_score=("best_non_reused_score", "mean"),
            )
            .sort_index()
        )

    lines = [
        "# Model Pool 失敗診斷",
        "",
        f"- A/B baseline run：`{ab_run_dir}`",
        f"- Model pool diagnostic run：`{model_pool_run_dir}`",
        f"- Post-window 長度：{n_days} 個交易日",
        "",
        "## 方法說明",
        "",
        "本報告分成兩層證據。Observed event attribution 使用 model_pool 實際採用後的路徑；candidate proxy 使用同一 trigger 當下每個候選模型形成的 top-k equal-weight 組合，估計下一持股週期表現。Candidate proxy 不是完整反事實回測，因此結論以 `evidence suggests` 表述。",
        "",
        "## 實際事件歸因",
        "",
        _df_to_markdown(selected_role_summary.round(4)),
        "",
        "## 候選模型 Shadow / Proxy",
        "",
        _df_to_markdown(candidate_role_summary.round(4)),
        "",
        "## Reuse Guard / Selector Audit",
        "",
        "Raw best role:",
        "",
        _df_to_markdown(raw_best_summary.round(4)),
        "",
        "Guard reason:",
        "",
        _df_to_markdown(reuse_guard_summary.round(4)),
        "",
        "## 初步判讀",
        "",
    ]

    if not reuse_guard_summary.empty:
        rejected = reuse_guard_summary.drop(index="passed", errors="ignore")
        rejected_n = int(rejected["n_events"].sum()) if "n_events" in rejected.columns else 0
        lines.append(
            f"- Reuse guard 共擋下 {rejected_n} 個 raw reused selection；若這個數字高但績效改善有限，問題較可能在 trigger/new model 本身。"
        )

    if reused_selected.empty:
        lines.append("- 本輪沒有 selected_role=reused 的事件，無法判斷 reuse 事件。")
    else:
        mean_excess = reused_selected["excess_vs_scheduled_20"].mean()
        mean_rank = reused_selected["proxy_rank_by_net"].mean()
        lines.append(
            "- Evidence suggests：reused model 被選中後，相對 `scheduled_20` 的 observed post-window excess "
            f"平均為 {_fmt_pct(mean_excess)}，候選 proxy 平均排名為 {mean_rank:.2f}。"
        )
        if mean_excess < 0 and mean_rank > 1.5:
            lines.append(
                "- observed attribution 與 candidate proxy 同向偏弱，reuse decision 可能是 model_pool gross edge 下滑的主要來源之一。"
            )
        elif mean_excess < 0:
            lines.append(
                "- observed attribution 偏弱，但 candidate proxy 未明確顯示 reuse 在 trigger 當下較差；較保守解讀是 shadow window 與實際持股 PnL 的對齊仍有落差。"
            )
        else:
            lines.append(
                "- reused 事件的 observed excess 未明顯為負，model_pool 失敗可能來自 new/kept-current 決策或整體 trigger policy。"
            )

    if not reused_candidate.empty:
        low_sim = reused_candidate[
            pd.to_numeric(reused_candidate["candidate_similarity"], errors="coerce") < 0.6
        ]
        if not low_sim.empty:
            lines.append(
                f"- similarity < 0.6 的 reused candidate 共 {len(low_sim)} 筆，平均 proxy net return "
                f"{_fmt_pct(low_sim['proxy_net_return'].mean())}；若這組明顯弱於高 similarity candidate，下一輪才有理由測 threshold 0.6/0.7。"
            )

    lines.extend([
        "",
        "## 輸出檔案",
        "",
        "- `model_pool_event_attribution.csv`",
        "- `model_pool_candidate_shadow.csv`",
        "- `fig_model_pool_event_pnl.png`",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_diagnosis(
    *,
    ab_run_dir: Path,
    model_pool_run_dir: Path,
    out_dir: Path,
    n_days: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_pnls = _load_ab_daily_pnls(ab_run_dir)
    model_pool_pnl = _load_pnl(model_pool_run_dir / "daily_pnl.csv")
    decisions_path = model_pool_run_dir / "model_pool_decisions.csv"
    if not decisions_path.exists():
        raise FileNotFoundError(
            f"找不到 {decisions_path}；請用 --model-pool-diagnostics 重跑 model_pool"
        )
    decisions = pd.read_csv(decisions_path)

    event_df = build_event_attribution(
        decisions=decisions,
        model_pool_pnl=model_pool_pnl,
        baseline_pnls=baseline_pnls,
        n_days=n_days,
    )
    candidate_df = decisions.copy()

    event_path = out_dir / "model_pool_event_attribution.csv"
    candidate_path = out_dir / "model_pool_candidate_shadow.csv"
    fig_path = out_dir / "fig_model_pool_event_pnl.png"
    summary_path = out_dir / "model_pool_failure_summary.md"

    event_df.to_csv(event_path, index=False)
    candidate_df.to_csv(candidate_path, index=False)
    _plot_diagnostics(event_df, candidate_df, fig_path)
    write_summary(
        out_path=summary_path,
        ab_run_dir=ab_run_dir,
        model_pool_run_dir=model_pool_run_dir,
        event_df=event_df,
        candidate_df=candidate_df,
        n_days=n_days,
    )
    return {
        "event_path": event_path,
        "candidate_path": candidate_path,
        "fig_path": fig_path,
        "summary_path": summary_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ab-run-dir", required=True, help="正式 A/B run 目錄")
    parser.add_argument("--model-pool-run-dir", default=None, help="診斷 model_pool simulation 目錄")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--post-days", type=int, default=10, help="trigger 後觀察幾個交易日")
    args = parser.parse_args()

    ab_run_dir = Path(args.ab_run_dir)
    model_pool_run_dir = _resolve_model_pool_run_dir(ab_run_dir, args.model_pool_run_dir)
    result = run_diagnosis(
        ab_run_dir=ab_run_dir,
        model_pool_run_dir=model_pool_run_dir,
        out_dir=Path(args.out_dir),
        n_days=args.post_days,
    )
    print("=== Model Pool failure diagnosis complete ===")
    for key, path in result.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
