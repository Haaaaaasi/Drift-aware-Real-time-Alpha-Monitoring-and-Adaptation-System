"""WP9 — Adaptation A/B 實驗評估（MVP v3 研究核心）

從 ``pipelines.ab_experiment`` 的輸出出發，對三種 adaptation 策略做 regime-stratified
比較與統計檢定，回答論文 RQ3：

> Adaptation（scheduled / performance-triggered）能否在不同市場 regime 下量化改善績效？

輸入
----
由 ``python -m pipelines.ab_experiment --csv data/tw_stocks_ohlcv.csv \\
    --start 2022-06-01 --end 2024-12-31 --run-tag v3`` 產出的
``reports/adaptation_ab/<run_id>/`` 目錄，內含各策略的 ``reports/simulations/sim_...``
子目錄（daily_pnl.csv / retrain_log.csv / summary.txt）。

分析項目
--------
1. **Regime-stratified 績效**：把交易日切成 2022-H2 / 2023 / 2024 三段，計算各策略
   在每段下的 Sharpe / Max DD / win rate / 重訓次數。
2. **Paired t-test**：triggered 與 scheduled 相對 no-adapt 的每日超額報酬是否顯著 > 0。
3. **重訓時機 vs Drift**：疊加 triggered 策略的重訓日與 rolling IC 時序，檢查觸發
   是否對應 IC 下滑窗口。

輸出
----
reports/adaptation_evaluation/
    regime_stratified.csv          每個 regime × 策略的績效矩陣
    paired_ttest.csv               triggered/scheduled vs none 的 t-stat 與 p-value
    fig1_regime_sharpe.png         Regime × 策略 Sharpe bar chart
    fig2_cumret_overlay.png        累積報酬疊加 regime 底色
    fig3_trigger_timing.png        triggered 重訓時機 vs rolling IC
    evaluation_summary.md          結論摘要（可直接引用於研究報告）

使用
----
    # 自動找最新的 ab_experiment run_id
    python notebooks/03_adaptation_evaluation.py

    # 指定特定 run_id
    python notebooks/03_adaptation_evaluation.py \\
        --run-dir reports/adaptation_ab/ab_20220601_20241231_top10_v3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy import stats

REPORT_DIR = Path("reports/adaptation_evaluation")
AB_DIR = Path("reports/adaptation_ab")

# Taiwan 市場 regime 定義（與 WP4 一致）
REGIMES = {
    "2022_H2_bear": ("2022-07-01", "2022-12-31", "2022-H2 Bear"),
    "2023_recovery": ("2023-01-01", "2023-12-31", "2023 Recovery"),
    "2024_consolidation": ("2024-01-01", "2024-12-31", "2024 Consolidation"),
}


def _find_latest_ab_run(ab_dir: Path) -> Path:
    """找最新（按 mtime）的 ab_experiment run 目錄。"""
    candidates = [p for p in ab_dir.iterdir() if p.is_dir() and p.name.startswith("ab_")]
    if not candidates:
        raise FileNotFoundError(f"在 {ab_dir} 找不到 ab_experiment 輸出")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_ab_run(run_dir: Path) -> dict:
    """載入一次 ab_experiment 的所有策略輸出。"""
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} 不存在；請先執行 pipelines.ab_experiment")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    strategies = list(cfg["run_dirs"].keys())
    data: dict[str, dict] = {}
    for strat in strategies:
        sim_dir = Path(cfg["run_dirs"][strat])
        pnl = pd.read_csv(sim_dir / "daily_pnl.csv", parse_dates=["date"])
        retrains = pd.read_csv(sim_dir / "retrain_log.csv", parse_dates=["date"])
        data[strat] = {"pnl": pnl, "retrains": retrains, "sim_dir": sim_dir}
    comparison = pd.read_csv(run_dir / "comparison.csv", index_col=0)
    return {"config": cfg, "data": data, "comparison": comparison, "run_dir": run_dir}


def compute_regime_metrics(pnl: pd.DataFrame, regime_start: str, regime_end: str) -> dict:
    """計算單一 regime 窗口下的績效指標。"""
    mask = (pnl["date"] >= regime_start) & (pnl["date"] <= regime_end)
    window = pnl[mask].copy()
    if window.empty:
        return {"n_days": 0, "sharpe": np.nan, "max_dd_pct": np.nan,
                "win_rate": np.nan, "cum_return_pct": np.nan}

    ret = window["net_return"].astype(float)
    n_days = len(window)
    cum_val = window["cumulative_value"].astype(float)
    period_return = cum_val.iloc[-1] / cum_val.iloc[0] - 1
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0.0

    peak = cum_val.cummax()
    dd = (cum_val - peak) / peak
    max_dd = dd.min()
    win_rate = float((ret > 0).sum() / n_days) if n_days > 0 else 0.0

    return {
        "n_days": n_days,
        "cum_return_pct": round(float(period_return) * 100, 3),
        "sharpe": round(float(sharpe), 3),
        "max_dd_pct": round(float(max_dd) * 100, 3),
        "win_rate": round(win_rate, 3),
    }


def build_regime_stratified(data: dict) -> pd.DataFrame:
    """產出 regime × 策略的二維 DataFrame。"""
    records = []
    for strat, d in data.items():
        pnl = d["pnl"]
        retrains = d["retrains"]
        for regime_key, (s, e, label) in REGIMES.items():
            metrics = compute_regime_metrics(pnl, s, e)
            # 該 regime 內的重訓次數
            n_retrain_regime = int(
                retrains[(retrains["date"] >= s) & (retrains["date"] <= e)].shape[0]
            )
            records.append({
                "strategy": strat,
                "regime": label,
                "regime_key": regime_key,
                **metrics,
                "n_retrains_in_regime": n_retrain_regime,
            })
    return pd.DataFrame(records)


def paired_ttest_vs_baseline(data: dict, baseline: str = "none") -> pd.DataFrame:
    """對齊日期後，對每個非 baseline 策略做 paired t-test。

    H0: 該策略的每日 net_return 與 baseline 相同
    H1: 該策略的每日 net_return 平均較高（one-sided）
    """
    if baseline not in data:
        return pd.DataFrame()
    base_pnl = data[baseline]["pnl"][["date", "net_return"]].rename(
        columns={"net_return": "base_ret"}
    )

    rows = []
    for strat, d in data.items():
        if strat == baseline:
            continue
        strat_pnl = d["pnl"][["date", "net_return"]].rename(
            columns={"net_return": "strat_ret"}
        )
        merged = base_pnl.merge(strat_pnl, on="date", how="inner")
        if len(merged) < 20:
            continue
        diff = merged["strat_ret"].astype(float) - merged["base_ret"].astype(float)
        t_stat, p_two = stats.ttest_1samp(diff, 0.0)
        p_one = p_two / 2.0 if t_stat > 0 else 1 - p_two / 2.0

        rows.append({
            "strategy": strat,
            "baseline": baseline,
            "n_days": len(merged),
            "mean_daily_excess_ret": round(float(diff.mean()), 6),
            "std_daily_excess_ret": round(float(diff.std()), 6),
            "t_statistic": round(float(t_stat), 3),
            "p_value_two_sided": round(float(p_two), 4),
            "p_value_one_sided": round(float(p_one), 4),
            "significant_5pct_one_sided": bool(p_one < 0.05),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 繪圖
# ---------------------------------------------------------------------------

_COLORS = plt.cm.tab10.colors


def _plot_regime_sharpe(regime_df: pd.DataFrame, out_path: Path) -> None:
    """Fig 1：regime × 策略 Sharpe 分組柱狀圖。"""
    strategies = list(regime_df["strategy"].unique())
    regimes = list(regime_df["regime"].unique())

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(regimes))
    width = 0.8 / max(len(strategies), 1)

    for i, strat in enumerate(strategies):
        vals = [
            regime_df[(regime_df["strategy"] == strat)
                      & (regime_df["regime"] == r)]["sharpe"].iloc[0]
            for r in regimes
        ]
        bars = ax.bar(
            x + i * width - 0.4 + width / 2, vals, width=width,
            label=strat, color=_COLORS[i % len(_COLORS)], alpha=0.85,
        )
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Fig 1. Regime-stratified Sharpe Ratio by Adaptation Strategy")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.7)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _plot_cumret_with_regime_shading(data: dict, out_path: Path) -> None:
    """Fig 2：累積報酬（normalized）疊加 regime 區塊底色。"""
    fig, ax = plt.subplots(figsize=(13, 6))

    # 背景 shading
    regime_colors = {"2022_H2_bear": "#ffeaea", "2023_recovery": "#eaffea",
                     "2024_consolidation": "#eaeaff"}
    for key, (s, e, label) in REGIMES.items():
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                   color=regime_colors.get(key, "lightgray"), alpha=0.5,
                   label=label, zorder=0)

    strategies = list(data.keys())
    for i, strat in enumerate(strategies):
        pnl = data[strat]["pnl"]
        cum = pnl["cumulative_value"] / pnl["cumulative_value"].iloc[0]
        ax.plot(pnl["date"], cum, label=strat,
                color=_COLORS[i % len(_COLORS)], linewidth=1.4, zorder=3)

    ax.set_title("Fig 2. Cumulative Return by Strategy (Regime-shaded)")
    ax.set_ylabel("Normalized cumulative value")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.6)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, zorder=1)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _plot_trigger_timing(data: dict, out_path: Path) -> None:
    """Fig 3：triggered 策略的重訓時機 vs rolling IC 時序。

    若 triggered 不存在則用有重訓的第一個策略。
    """
    focus = "triggered" if "triggered" in data else next(
        (s for s in data if not data[s]["retrains"].empty
         and data[s]["retrains"].iloc[0]["reason"] != "initial_train"),
        list(data.keys())[0],
    )
    pnl = data[focus]["pnl"]
    retrains = data[focus]["retrains"]

    fig, (ax_ic, ax_ret) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                         gridspec_kw={"height_ratios": [1, 1]})

    # 上：rolling IC + 重訓標記
    ax_ic.plot(pnl["date"], pnl["rolling_ic"], color="tab:blue", linewidth=1.2,
               label="Rolling IC (20d)")
    ax_ic.axhline(0.0, color="gray", linestyle="--", linewidth=0.6)
    for _, r in retrains.iterrows():
        ax_ic.axvline(r["date"], color="tab:red", linestyle=":", alpha=0.6)
    ax_ic.set_ylabel("Rolling IC")
    ax_ic.set_title(f"Fig 3. Retrain timing vs Rolling IC — strategy={focus} "
                    f"({len(retrains)} retrains)")
    ax_ic.legend(loc="best")
    ax_ic.grid(True, alpha=0.3)

    # 下：累積報酬 + 重訓標記
    cum = pnl["cumulative_value"] / pnl["cumulative_value"].iloc[0]
    ax_ret.plot(pnl["date"], cum, color="tab:green", linewidth=1.3,
                label="Normalized cumulative value")
    for _, r in retrains.iterrows():
        ax_ret.axvline(r["date"], color="tab:red", linestyle=":", alpha=0.6,
                       label="Retrain" if _ == 0 else None)
    ax_ret.set_ylabel("Normalized cum. value")
    ax_ret.axhline(1.0, color="gray", linestyle="--", linewidth=0.6)
    ax_ret.legend(loc="best")
    ax_ret.grid(True, alpha=0.3)
    ax_ret.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax_ret.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 摘要撰寫
# ---------------------------------------------------------------------------

def write_summary(
    out_path: Path,
    run_dir: Path,
    regime_df: pd.DataFrame,
    ttest_df: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    """寫 evaluation_summary.md。"""
    lines = [
        "# Adaptation A/B 評估摘要",
        "",
        f"資料來源：`{run_dir}`",
        "",
        "## 整體績效（全期間）",
        "",
        _df_to_markdown(comparison.round(3)),
        "",
        "## Regime-stratified Sharpe",
        "",
    ]
    pivot = regime_df.pivot_table(index="strategy", columns="regime",
                                   values="sharpe").round(3)
    lines.append(_df_to_markdown(pivot))

    lines.extend(["", "## Paired t-test（vs no-adapt baseline）", ""])
    if ttest_df.empty:
        lines.append("_未執行（缺 no-adapt baseline 或樣本不足）_")
    else:
        lines.append(_df_to_markdown(ttest_df.round(4)))
        sig = ttest_df[ttest_df["significant_5pct_one_sided"]]
        if not sig.empty:
            lines.append("")
            lines.append(
                f"**顯著性發現**：{', '.join(sig['strategy'].tolist())} "
                f"在 α=0.05 水準下顯著優於 no-adapt baseline（one-sided）。"
            )
        else:
            lines.append("")
            lines.append(
                "**顯著性發現**：在本資料區間上，沒有策略顯著優於 no-adapt baseline "
                "（α=0.05, one-sided）。這可能反映 (a) adaptation 的邊際效益被交易成本"
                "抵消、(b) 樣本期間內 drift 強度不足以讓重訓帶來統計顯著差異、或 "
                "(c) 需要更長的樣本期間累積 power。"
            )

    lines.extend(["", "## 關鍵觀察", ""])
    best_overall = comparison["sharpe"].idxmax()
    lines.append(
        f"- 全期 Sharpe 最佳：`{best_overall}` ({comparison.loc[best_overall, 'sharpe']:.3f})。"
    )
    # 哪個 regime 最難
    worst_regime = (
        regime_df.groupby("regime")["sharpe"].mean().idxmin()
    )
    lines.append(
        f"- 各策略平均 Sharpe 最差的 regime：**{worst_regime}**。"
    )
    # 重訓經濟性：每次重訓帶來多少 Sharpe 增益
    if "none" in comparison.index:
        base_sharpe = comparison.loc["none", "sharpe"]
        econ = (comparison["sharpe"] - base_sharpe) / comparison["n_retrains"].replace(0, np.nan)
        lines.append("")
        lines.append("### 重訓經濟性（每次重訓相對 no-adapt 的 Sharpe 增益）")
        lines.append("")
        lines.append(_df_to_markdown(econ.round(4).to_frame("delta_sharpe_per_retrain")))

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _df_to_markdown(df: pd.DataFrame) -> str:
    """無依賴 tabulate 的 DataFrame → Markdown table。"""
    header = "| " + " | ".join([df.index.name or ""] + [str(c) for c in df.columns]) + " |"
    sep = "| " + " | ".join(["---"] * (len(df.columns) + 1)) + " |"
    rows = [
        "| " + " | ".join([str(idx)] + [str(v) for v in row]) + " |"
        for idx, row in zip(df.index, df.values)
    ]
    return "\n".join([header, sep, *rows])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_evaluation(run_dir: Path, out_dir: Path = REPORT_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] 讀取 ab_experiment 輸出：{run_dir}")
    loaded = load_ab_run(run_dir)
    data = loaded["data"]
    comparison = loaded["comparison"]

    print(f"[eval] 策略數={len(data)}；各策略交易日數：")
    for strat, d in data.items():
        print(f"   {strat}: {len(d['pnl'])} 日, {len(d['retrains'])} 次重訓")

    # Regime-stratified
    regime_df = build_regime_stratified(data)
    regime_path = out_dir / "regime_stratified.csv"
    regime_df.to_csv(regime_path, index=False)
    print(f"[eval] regime_stratified.csv 已寫入")

    # Paired t-test
    ttest_df = paired_ttest_vs_baseline(data, baseline="none")
    ttest_path = out_dir / "paired_ttest.csv"
    ttest_df.to_csv(ttest_path, index=False)
    print(f"[eval] paired_ttest.csv 已寫入")

    # Figures
    _plot_regime_sharpe(regime_df, out_dir / "fig1_regime_sharpe.png")
    _plot_cumret_with_regime_shading(data, out_dir / "fig2_cumret_overlay.png")
    _plot_trigger_timing(data, out_dir / "fig3_trigger_timing.png")
    print(f"[eval] 3 張圖已寫入 {out_dir}")

    # Summary
    summary_path = out_dir / "evaluation_summary.md"
    write_summary(summary_path, run_dir, regime_df, ttest_df, comparison)
    print(f"[eval] evaluation_summary.md 已寫入")

    return {
        "regime_df": regime_df,
        "ttest_df": ttest_df,
        "comparison": comparison,
        "out_dir": out_dir,
    }


def run_cost_sweep_evaluation(sweep_dir: Path, out_dir: Path = REPORT_DIR) -> dict:
    """評估 ab_experiment cost-sweep run 的輸出。

    讀取 ``cost_sensitivity.csv`` 並產出：
        * ``cost_sensitivity_summary.csv`` — 每 cost 場景的策略 Sharpe 排名
        * ``fig4_cost_sensitivity.png`` — 4 panel 圖：Sharpe / cum_ret / max_dd / turnover vs cost
        * ``cost_sensitivity_evaluation.md`` — 排序穩健性解讀
    """
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_csv = sweep_dir / "cost_sensitivity.csv"
    if not sweep_csv.exists():
        raise FileNotFoundError(
            f"找不到 {sweep_csv} — 請確認 ab_experiment 是用 --cost-sweep 模式跑的"
        )

    sweep_df = pd.read_csv(sweep_csv)
    print(f"[cost-sweep] 載入 {len(sweep_df)} 筆 records，"
          f"{sweep_df['strategy'].nunique()} 策略 × {sweep_df['cost_pct'].nunique()} cost 場景")

    # 每 cost 場景的 Sharpe 排名
    rank_rows = []
    for cost in sorted(sweep_df["cost_pct"].unique()):
        sub = sweep_df[sweep_df["cost_pct"] == cost].sort_values("sharpe", ascending=False)
        for rank, (_, row) in enumerate(sub.iterrows(), start=1):
            rank_rows.append({
                "cost_pct": cost,
                "rank": rank,
                "strategy": row["strategy"],
                "sharpe": row["sharpe"],
                "cumulative_return_pct": row["cumulative_return_pct"],
                "max_drawdown_pct": row["max_drawdown_pct"],
                "n_retrains": row["n_retrains"],
            })
    rank_df = pd.DataFrame(rank_rows)
    rank_path = out_dir / "cost_sensitivity_summary.csv"
    rank_df.to_csv(rank_path, index=False)
    print(f"[cost-sweep] cost_sensitivity_summary.csv 已寫入")

    # 圖：每策略一條線（橫軸 cost、縱軸 4 panel）
    strategies = sorted(sweep_df["strategy"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    color_cycle = plt.cm.tab10.colors
    colors = {s: color_cycle[i % len(color_cycle)] for i, s in enumerate(strategies)}

    metrics = [
        ("sharpe", "Sharpe Ratio", axes[0, 0]),
        ("cumulative_return_pct", "Cumulative Return (%)", axes[0, 1]),
        ("max_drawdown_pct", "Max Drawdown (%)", axes[1, 0]),
        ("avg_turnover", "Avg Turnover", axes[1, 1]),
    ]
    for col, ylabel, ax in metrics:
        for strat in strategies:
            sub = sweep_df[sweep_df["strategy"] == strat].sort_values("cost_pct")
            ax.plot(sub["cost_pct"], sub[col],
                    marker="o", label=strat, color=colors[strat], linewidth=1.5)
        ax.set_xlabel("Round-trip cost (%)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("Cost Sensitivity — 5 strategies × N cost scenarios", fontsize=12)
    fig.tight_layout()
    fig_path = out_dir / "fig4_cost_sensitivity.png"
    fig.savefig(fig_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[cost-sweep] fig4_cost_sensitivity.png 已寫入")

    # 排序穩健性指標：每策略在 N 場景中的平均 rank、rank std
    stability_rows = []
    for strat in strategies:
        sub = rank_df[rank_df["strategy"] == strat]
        stability_rows.append({
            "strategy": strat,
            "mean_rank": float(sub["rank"].mean()),
            "rank_std": float(sub["rank"].std()),
            "best_rank": int(sub["rank"].min()),
            "worst_rank": int(sub["rank"].max()),
        })
    stability_df = pd.DataFrame(stability_rows).sort_values("mean_rank")

    # 摘要 MD
    summary_path = out_dir / "cost_sensitivity_evaluation.md"
    lines = [
        f"# Cost Sensitivity Evaluation",
        "",
        f"輸入：`{sweep_dir}`",
        f"場景：{sorted(sweep_df['cost_pct'].unique().tolist())}",
        f"策略：{strategies}",
        "",
        "## 排序穩健性（mean_rank 越低越穩定領先）",
        "",
        _df_to_markdown(stability_df.set_index("strategy").round(3)),
        "",
        "## 解讀",
        "若 `rank_std == 0`，代表該策略在所有成本場景下排名一致；",
        "`rank_std > 1.0` 代表排序對成本敏感，「no-adapt 勝出」結論可能不穩健。",
        "",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[cost-sweep] cost_sensitivity_evaluation.md 已寫入")

    return {
        "rank_df": rank_df,
        "stability_df": stability_df,
        "out_dir": out_dir,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default=None,
                   help="ab_experiment 輸出目錄；省略則自動找最新一次（單跑模式）")
    p.add_argument("--cost-sweep-dir", default=None,
                   help="cost-sweep run 目錄；指定則進入成本敏感度評估模式")
    p.add_argument("--out-dir", default=str(REPORT_DIR))
    args = p.parse_args()

    if args.cost_sweep_dir:
        sweep_dir = Path(args.cost_sweep_dir)
        result = run_cost_sweep_evaluation(sweep_dir, Path(args.out_dir))
        print("\n=== 排序穩健性 ===")
        print(result["stability_df"].set_index("strategy").round(3).to_string())
        print(f"\n結果：{result['out_dir']}/cost_sensitivity_evaluation.md")
        return

    run_dir = Path(args.run_dir) if args.run_dir else _find_latest_ab_run(AB_DIR)
    result = run_evaluation(run_dir, Path(args.out_dir))

    print("\n=== Regime × 策略 Sharpe ===")
    print(result["regime_df"].pivot_table(
        index="strategy", columns="regime", values="sharpe"
    ).round(3).to_string())

    print("\n=== Paired t-test ===")
    if result["ttest_df"].empty:
        print("(無 baseline)")
    else:
        print(result["ttest_df"].to_string(index=False))

    print(f"\n結果：{result['out_dir']}/evaluation_summary.md")


if __name__ == "__main__":
    main()
