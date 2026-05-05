"""WP2 — Per-Alpha IC Analysis（含 In-Sample / Out-of-Sample 切分）

對台股 OHLCV 計算每個 alpha 的 IC / rank-IC / hit-rate / coverage，並選出
將進入 XGBoost meta model（WP3）特徵池的 effective alpha 子集。

預設行為（不指定 --train-end）會以全期資料計算並選擇——這在 P0★#1 修正
之前是 v2 / v3 的作法，但會造成 look-ahead bias：用未來資料挑因子，再用
同一段歷史回測。

新作法（建議）：指定 --train-end YYYY-MM-DD 後，notebook 會
    1. 以 IS 期（tradetime <= train-end）計算 IC 並選 effective alphas
    2. 以 OOS 期（tradetime > train-end）對同一批 alpha 計算 IC 作為驗證
    3. effective_alphas.json 寫入 IS-selected 子集（下游 pipeline 透明讀取）
    4. 額外輸出 effective_alphas_oos_validation.csv 對照 IS/OOS IC 衰減

Outputs
-------
reports/alpha_ic_analysis/
    alpha_ic_summary.csv               Per-alpha IC / rank-IC / hit-rate / coverage（IS 或全期）
    alpha_correlation_matrix.csv       Pairwise Pearson correlation across alphas
    effective_alphas.json              Selected subset（含 split metadata）
    effective_alphas_oos_validation.csv  IS vs OOS IC 對照（僅 --train-end 模式）
    ic_bar_chart.png                   Bar chart of IC / rank-IC per alpha
    rolling_ic.png                     Rolling IC time series (60-day) per alpha
    correlation_heatmap.png            Alpha correlation heatmap

Run
---
    # 舊行為（look-ahead bias，僅作對照用）
    python notebooks/01_alpha_ic_analysis.py --csv data/tw_stocks_ohlcv.csv

    # 建議：IS 選 alpha + OOS 驗證
    python notebooks/01_alpha_ic_analysis.py --csv data/tw_stocks_ohlcv.csv \
        --train-end 2023-06-30

Selection criteria for effective alphas
---------------------------------------
An alpha enters the subset if ALL hold:
    |overall rank-IC| >= 0.01          (weak but non-trivial signal)
    coverage          >= 0.80          (not mostly NaN)

Hit rate is reported but NOT used as a filter — rank-IC already captures
directional edge and hit rate of daily signals tends to hover near 0.50 on
small universes (10 stocks), making it a noisy gate rather than informative.
The goal here is feature selection for the downstream XGBoost meta model, so
we favour recall over precision.
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

from pipelines.daily_batch_pipeline import compute_python_alphas, load_csv_data
from src.common.metrics import (
    information_coefficient,
    rank_information_coefficient,
)
from src.labeling.label_generator import LabelGenerator


def list_alpha_ids_from_dolphindb() -> list[str]:
    """回傳 alpha_features 中所有 distinct alpha_id，已排序。"""
    from src.common.db import get_dolphindb
    c = get_dolphindb()
    r = c.run('select alpha_id from (select distinct alpha_id from loadTable("dfs://darams_alpha","alpha_features")) order by alpha_id')
    return r["alpha_id"].tolist()


def load_single_alpha_from_dolphindb(alpha_id: str) -> pd.DataFrame:
    """載入單一 alpha 的資料（~553k rows，約 17MB），避免一次載入全部 53M rows OOM。

    Returns DataFrame with columns: security_id, tradetime, alpha_id, alpha_value
    """
    from src.common.db import get_dolphindb
    c = get_dolphindb()
    script = f'''
    select security_id, tradetime, alpha_id, alpha_value
    from loadTable("dfs://darams_alpha", "alpha_features")
    where alpha_id = `{alpha_id}
    order by tradetime, security_id
    '''
    df = c.run(script)
    df["tradetime"] = pd.to_datetime(df["tradetime"])
    df["security_id"] = df["security_id"].astype(str)
    return df


def load_bars_for_labels_from_dolphindb() -> pd.DataFrame:
    """從 DolphinDB standardized_bars 讀取 close price 以產生 forward return labels。"""
    from src.common.db import get_dolphindb
    c = get_dolphindb()
    script = '''
    select security_id, tradetime, close
    from loadTable("dfs://darams_market", "standardized_bars")
    where bar_type = "daily" and is_tradable = true
    order by tradetime, security_id
    '''
    df = c.run(script)
    df["tradetime"] = pd.to_datetime(df["tradetime"])
    df["security_id"] = df["security_id"].astype(str)
    return df

REPORT_DIR = Path("reports/alpha_ic_analysis")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------

def compute_alpha_row(
    alpha_id: str,
    alpha_values: pd.Series,
    fwd_returns: pd.Series,
    min_obs: int = 50,
) -> dict | None:
    """單一 alpha 的 IC/rank-IC/hit-rate/coverage 摘要列。樣本不足回 None。

    `alpha_values` 預期 index 為 (security_id, tradetime) 的 Series。
    抽出獨立函式以便對 IS 與 OOS 切片各算一次（dolphindb 模式單 alpha 載入很貴）。
    """
    common = alpha_values.index.intersection(fwd_returns.index)
    if len(common) < min_obs:
        return None
    a = alpha_values.loc[common]
    r = fwd_returns.loc[common]
    ic = information_coefficient(a, r)
    rank_ic = rank_information_coefficient(a, r)
    hit = float((np.sign(a) == np.sign(r)).mean())
    coverage = float(a.notna().mean())
    return {
        "alpha_id": alpha_id,
        "n": int(len(common)),
        "ic": ic,
        "rank_ic": rank_ic,
        "hit_rate": hit,
        "coverage": coverage,
        "abs_ic": abs(ic) if not np.isnan(ic) else 0.0,
    }


def compute_per_alpha_metrics(
    alpha_panel: pd.DataFrame,
    fwd_returns: pd.Series,
) -> pd.DataFrame:
    """Per-alpha summary: IC, rank-IC, hit-rate, coverage, sample size."""
    rows = []
    for aid, g in alpha_panel.groupby("alpha_id"):
        vals = g.set_index(["security_id", "tradetime"])["alpha_value"]
        row = compute_alpha_row(aid, vals, fwd_returns)
        if row is not None:
            rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("abs_ic", ascending=False)
        .reset_index(drop=True)
    )


def split_panel_by_time(
    alpha_panel: pd.DataFrame,
    fwd_returns: pd.Series,
    train_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """以 train_end 將 alpha_panel 與 fwd_returns 切成 (IS, OOS) 兩組。

    切分規則：tradetime <= train_end → IS；tradetime > train_end → OOS。
    fwd_returns 的 index 必須含 'tradetime' level。
    回傳 (alpha_panel_is, fwd_is, alpha_panel_oos, fwd_oos)。
    """
    is_mask_panel = alpha_panel["tradetime"] <= train_end
    panel_is = alpha_panel[is_mask_panel].copy()
    panel_oos = alpha_panel[~is_mask_panel].copy()

    tt = fwd_returns.index.get_level_values("tradetime")
    fwd_is = fwd_returns[tt <= train_end]
    fwd_oos = fwd_returns[tt > train_end]
    return panel_is, fwd_is, panel_oos, fwd_oos


def build_oos_validation(
    summary_is: pd.DataFrame,
    summary_oos: pd.DataFrame,
    selected_ids: list[str],
) -> pd.DataFrame:
    """以 alpha_id 對齊 IS / OOS summary，產出 IC 對照表。

    欄位：alpha_id, selected_in_is, is_*（n/ic/rank_ic/hit_rate/coverage）,
    oos_*, rank_ic_decay (= oos_rank_ic - is_rank_ic), sign_flip。
    """
    is_idx = summary_is.set_index("alpha_id")
    oos_idx = summary_oos.set_index("alpha_id") if not summary_oos.empty else pd.DataFrame()
    rows = []
    for aid in summary_is["alpha_id"]:
        is_row = is_idx.loc[aid]
        has_oos = not oos_idx.empty and aid in oos_idx.index
        oos_row = oos_idx.loc[aid] if has_oos else None
        rec = {
            "alpha_id": aid,
            "selected_in_is": aid in set(selected_ids),
            "is_n": int(is_row["n"]),
            "is_ic": float(is_row["ic"]),
            "is_rank_ic": float(is_row["rank_ic"]),
            "is_hit_rate": float(is_row["hit_rate"]),
            "is_coverage": float(is_row["coverage"]),
            "oos_n": int(oos_row["n"]) if has_oos else 0,
            "oos_ic": float(oos_row["ic"]) if has_oos else float("nan"),
            "oos_rank_ic": float(oos_row["rank_ic"]) if has_oos else float("nan"),
            "oos_hit_rate": float(oos_row["hit_rate"]) if has_oos else float("nan"),
            "oos_coverage": float(oos_row["coverage"]) if has_oos else float("nan"),
        }
        if has_oos:
            rec["rank_ic_decay"] = rec["oos_rank_ic"] - rec["is_rank_ic"]
            rec["sign_flip"] = bool(np.sign(rec["oos_rank_ic"]) != np.sign(rec["is_rank_ic"]))
        else:
            rec["rank_ic_decay"] = float("nan")
            rec["sign_flip"] = False
        rows.append(rec)
    return (
        pd.DataFrame(rows)
        .assign(_abs=lambda d: d["is_rank_ic"].abs())
        .sort_values("_abs", ascending=False)
        .drop(columns=["_abs"])
        .reset_index(drop=True)
    )


def compute_rolling_ic(
    alpha_panel: pd.DataFrame,
    fwd_returns: pd.Series,
    window: int = 60,
) -> pd.DataFrame:
    """Rolling cross-sectional IC over time, one column per alpha.

    For each trade date, computes the cross-sectional Pearson IC over the
    preceding `window` days (pooled across stocks).
    """
    result: dict[str, pd.Series] = {}
    for aid, g in alpha_panel.groupby("alpha_id"):
        df = (
            g.set_index(["security_id", "tradetime"])["alpha_value"]
            .to_frame("alpha")
            .join(fwd_returns.rename("fwd"), how="inner")
            .dropna()
        )
        if len(df) < window:
            continue
        # Per-day IC, then rolling mean
        daily_ic = (
            df.groupby(level="tradetime")
            .apply(lambda x: x["alpha"].corr(x["fwd"]) if len(x) >= 3 else np.nan)
        )
        result[aid] = daily_ic.rolling(window, min_periods=window // 2).mean()

    return pd.DataFrame(result).sort_index()


def compute_alpha_correlation(alpha_panel: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Pearson correlation across alphas (pivoted to wide format)."""
    wide = alpha_panel.pivot_table(
        index=["security_id", "tradetime"],
        columns="alpha_id",
        values="alpha_value",
    )
    return wide.corr()


def select_effective_alphas(
    summary: pd.DataFrame,
    rank_ic_threshold: float = 0.01,
    coverage_threshold: float = 0.80,
) -> list[str]:
    """Apply selection rules and return the effective alpha id list."""
    mask = (
        (summary["rank_ic"].abs() >= rank_ic_threshold)
        & (summary["coverage"] >= coverage_threshold)
    )
    return summary.loc[mask, "alpha_id"].tolist()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ic_bar_chart(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(summary))
    width = 0.38
    ax.bar(x - width / 2, summary["ic"], width, label="IC", color="steelblue")
    ax.bar(x + width / 2, summary["rank_ic"], width, label="Rank IC", color="orange")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["alpha_id"], rotation=45, ha="right")
    ax.set_ylabel("Information Coefficient")
    ax.set_title("Per-alpha IC vs Rank-IC (overall)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_rolling_ic(rolling_ic: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in rolling_ic.columns:
        ax.plot(rolling_ic.index, rolling_ic[col], label=col, linewidth=1)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Rolling IC (60-day)")
    ax.set_title("Rolling IC over time — per alpha")
    ax.legend(ncol=3, fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_correlation_heatmap(corr: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(val) > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson ρ")
    ax.set_title("Alpha correlation matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output orchestration
# ---------------------------------------------------------------------------

def _build_split_meta(
    train_end: pd.Timestamp,
    summary_is: pd.DataFrame,
    summary_oos: pd.DataFrame,
    is_date_range: tuple[pd.Timestamp, pd.Timestamp] | None,
    oos_date_range: tuple[pd.Timestamp, pd.Timestamp] | None,
) -> dict:
    return {
        "train_end": train_end.strftime("%Y-%m-%d"),
        "is_window": (
            f"{is_date_range[0].date()} → {is_date_range[1].date()}"
            if is_date_range else None
        ),
        "oos_window": (
            f"{oos_date_range[0].date()} → {oos_date_range[1].date()}"
            if oos_date_range else None
        ),
        "is_n_alphas": int(len(summary_is)),
        "oos_n_alphas": int(len(summary_oos)),
    }


def emit_selection_outputs(
    primary_summary: pd.DataFrame,
    *,
    horizon: int,
    source: str,
    universe_desc: str,
    train_end: pd.Timestamp | None = None,
    summary_oos: pd.DataFrame | None = None,
    split_meta: dict | None = None,
) -> tuple[list[str], dict]:
    """共用：選 effective alpha、寫 effective_alphas.json，必要時寫 OOS validation。

    primary_summary：被用來選 alpha 的 summary（split 模式為 IS、否則為全期）。
    summary_oos：split 模式下的 OOS summary，用以產生 validation 對照表。
    """
    effective = select_effective_alphas(primary_summary)
    selection_basis = "in_sample" if train_end is not None else "full_sample"

    if train_end is not None and summary_oos is not None:
        validation = build_oos_validation(primary_summary, summary_oos, effective)
        validation.to_csv(REPORT_DIR / "effective_alphas_oos_validation.csv", index=False)
        sign_flip_n = int(validation["sign_flip"].sum())
        sel_oos = validation[validation["selected_in_is"]]
        oos_mean_rank_ic = float(sel_oos["oos_rank_ic"].mean()) if len(sel_oos) else float("nan")
        print(
            f"[split] OOS validation written: rows={len(validation)}  "
            f"sign_flip={sign_flip_n}  selected_oos_mean_rank_ic={oos_mean_rank_ic:.4f}"
        )

    selection = {
        "horizon": horizon,
        "source": source,
        "universe": universe_desc,
        "split": split_meta,
        "selection_basis": selection_basis,
        "criteria": {"abs_rank_ic_min": 0.01, "coverage_min": 0.80},
        "all_alphas": primary_summary["alpha_id"].tolist(),
        "effective_alphas": effective,
        "dropped_alphas": [
            a for a in primary_summary["alpha_id"].tolist() if a not in effective
        ],
    }
    (REPORT_DIR / "effective_alphas.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    return effective, selection


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="WP2 per-alpha IC analysis")
    parser.add_argument("--csv", default="data/tw_stocks_ohlcv.csv",
                        help="Path to OHLCV CSV (from download_tw_stocks.py)")
    parser.add_argument("--source", choices=["python", "dolphindb"], default="python",
                        help="alpha 來源：python=CSV 近似（15 個）/ dolphindb=真實 WQ101（最多 101 個）")
    parser.add_argument("--horizon", type=int, default=5,
                        help="Forward-return horizon in trading days")
    parser.add_argument("--rolling-window", type=int, default=60,
                        help="Rolling IC window size")
    parser.add_argument("--train-end", default=None,
                        help="In-sample 結束日 (YYYY-MM-DD)；指定後將以 IS 期選 alpha、"
                             "OOS 期驗證，避免 look-ahead bias。不指定則沿用全期作法。")
    args = parser.parse_args()

    train_end = pd.Timestamp(args.train_end) if args.train_end else None
    if train_end is not None:
        print(f"[mode] In-sample / Out-of-sample split — train_end={train_end.date()}")
    else:
        print("[mode] Full-sample selection (legacy; subject to look-ahead bias)")

    if args.source == "dolphindb":
        print("[1/6] 讀取 DolphinDB standardized_bars（close price 用於 label 產生）...")
        bars = load_bars_for_labels_from_dolphindb()
        print(f"      rows={len(bars):,}  symbols={bars['security_id'].nunique()}  "
              f"range={bars['tradetime'].min().date()} → {bars['tradetime'].max().date()}")

        print(f"[3/6] Generating forward-return labels (horizon={args.horizon}) ...")
        labels = LabelGenerator(horizons=[args.horizon], bar_type="daily").generate_labels(
            bars[["security_id", "tradetime", "close"]]
        )
        fwd = (
            labels[labels["horizon"] == args.horizon]
            .set_index(["security_id", "signal_time"])["forward_return"]
        )
        fwd.index = fwd.index.set_names(["security_id", "tradetime"])
        print(f"      label rows={len(fwd):,}")

        print("[2/6] 逐 alpha 串流讀取 WQ101 alpha_features（避免 53M rows 一次載入 OOM）...")
        alpha_ids = list_alpha_ids_from_dolphindb()
        print(f"      找到 {len(alpha_ids)} 個 alpha")

        # split mode 下事先切 fwd，迴圈內只切 vals
        if train_end is not None:
            tt_fwd = fwd.index.get_level_values("tradetime")
            fwd_is = fwd[tt_fwd <= train_end]
            fwd_oos = fwd[tt_fwd > train_end]
            print(f"      fwd labels  IS={len(fwd_is):,}  OOS={len(fwd_oos):,}")
        else:
            fwd_is = fwd_oos = None

        ic_rows: list[dict] = []
        ic_rows_oos: list[dict] = []
        rolling_dict: dict[str, pd.Series] = {}
        for i, aid in enumerate(alpha_ids, 1):
            panel = load_single_alpha_from_dolphindb(aid)
            vals = panel.set_index(["security_id", "tradetime"])["alpha_value"]
            if train_end is not None:
                ttimes = vals.index.get_level_values("tradetime")
                vals_is = vals[ttimes <= train_end]
                vals_oos = vals[ttimes > train_end]
                row_is = compute_alpha_row(aid, vals_is, fwd_is)
                if row_is is not None:
                    ic_rows.append(row_is)
                row_oos = compute_alpha_row(aid, vals_oos, fwd_oos)
                if row_oos is not None:
                    ic_rows_oos.append(row_oos)
            else:
                row = compute_alpha_row(aid, vals, fwd)
                if row is not None:
                    ic_rows.append(row)
            # Rolling IC（純視覺化，全期計算以呈現完整時序）
            df_r = (
                panel.set_index(["security_id", "tradetime"])["alpha_value"]
                .to_frame("alpha")
                .join(fwd.rename("fwd"), how="inner")
                .dropna()
            )
            if len(df_r) >= args.rolling_window:
                daily_ic = df_r.groupby(level="tradetime").apply(
                    lambda x: x["alpha"].corr(x["fwd"]) if len(x) >= 3 else np.nan
                )
                rolling_dict[aid] = daily_ic.rolling(args.rolling_window, min_periods=args.rolling_window // 2).mean()
            if i % 10 == 0:
                print(f"      {i}/{len(alpha_ids)} alphas processed...", flush=True)

        summary = (
            pd.DataFrame(ic_rows)
            .sort_values("abs_ic", ascending=False)
            .reset_index(drop=True)
        )
        summary_oos = (
            pd.DataFrame(ic_rows_oos)
            .sort_values("abs_ic", ascending=False)
            .reset_index(drop=True)
            if ic_rows_oos else pd.DataFrame()
        )
        rolling = pd.DataFrame(rolling_dict).sort_index()
        scope = "IS" if train_end is not None else "全期"
        print(f"      {scope} IC summary: {len(summary)} alphas")
        print(summary.to_string(index=False))
        summary.to_csv(REPORT_DIR / "alpha_ic_summary.csv", index=False)

        print("[5/6] 計算相關性（取 IC 前 30 名 alpha）...")
        top30 = summary.head(30)["alpha_id"].tolist()
        corr_frames = []
        for aid in top30:
            panel = load_single_alpha_from_dolphindb(aid)
            wide = panel.pivot_table(index=["security_id", "tradetime"], columns="alpha_id", values="alpha_value")
            corr_frames.append(wide)
        wide_all = pd.concat(corr_frames, axis=1)
        corr = wide_all.corr()
        corr.to_csv(REPORT_DIR / "alpha_correlation_matrix.csv")

        print("[6/6] Plotting ...")
        plot_ic_bar_chart(summary, REPORT_DIR / "ic_bar_chart.png")
        plot_rolling_ic(rolling[top30[:15]], REPORT_DIR / "rolling_ic.png")  # 前 15 條線避免圖太擠
        plot_correlation_heatmap(corr, REPORT_DIR / "correlation_heatmap.png")

        split_meta = None
        if train_end is not None:
            # dolphindb 模式無法輕易 enumerate 整個資料時間範圍，從 fwd index 推
            tt_all = fwd.index.get_level_values("tradetime")
            is_dates = (tt_all[tt_all <= train_end].min(), tt_all[tt_all <= train_end].max())
            oos_tt = tt_all[tt_all > train_end]
            oos_dates = (oos_tt.min(), oos_tt.max()) if len(oos_tt) else None
            split_meta = _build_split_meta(train_end, summary, summary_oos, is_dates, oos_dates)

        effective, selection = emit_selection_outputs(
            summary,
            horizon=args.horizon,
            source="dolphindb",
            universe_desc="556 TW stocks 2022-2026",
            train_end=train_end,
            summary_oos=summary_oos if train_end is not None else None,
            split_meta=split_meta,
        )
        print()
        print("=" * 60)
        print(f"Effective alphas ({len(effective)}/{len(summary)}, basis={selection['selection_basis']}): {effective}")
        print(f"Dropped          : {selection['dropped_alphas']}")
        print(f"Outputs          : {REPORT_DIR}")
        print("=" * 60)
        return  # dolphindb 模式到這裡結束
    else:
        print(f"[1/6] Loading CSV: {args.csv}")
        bars = load_csv_data(args.csv)
        print(f"      rows={len(bars):,}  symbols={bars['security_id'].nunique()}  "
              f"range={bars['tradetime'].min().date()} → {bars['tradetime'].max().date()}")

        print("[2/6] Computing Python-approximated WQ101 alphas ...")
        alpha_panel = compute_python_alphas(bars)
        print(f"      alpha rows={len(alpha_panel):,}  "
              f"distinct alphas={alpha_panel['alpha_id'].nunique()}")

    print(f"[3/6] Generating forward-return labels (horizon={args.horizon}) ...")
    labels = LabelGenerator(horizons=[args.horizon], bar_type="daily").generate_labels(
        bars[["security_id", "tradetime", "close"]]
    )
    fwd = (
        labels[labels["horizon"] == args.horizon]
        .set_index(["security_id", "signal_time"])["forward_return"]
    )
    fwd.index = fwd.index.set_names(["security_id", "tradetime"])
    print(f"      label rows={len(fwd):,}")

    print("[4/6] Computing per-alpha IC / rank-IC / hit-rate ...")
    summary_oos = pd.DataFrame()
    split_meta = None
    if train_end is not None:
        panel_is, fwd_is, panel_oos, fwd_oos = split_panel_by_time(alpha_panel, fwd, train_end)
        print(f"      [split] IS rows={len(panel_is):,}  OOS rows={len(panel_oos):,}")
        summary = compute_per_alpha_metrics(panel_is, fwd_is)
        summary_oos = compute_per_alpha_metrics(panel_oos, fwd_oos)
        is_dates = (panel_is["tradetime"].min(), panel_is["tradetime"].max()) if len(panel_is) else None
        oos_dates = (panel_oos["tradetime"].min(), panel_oos["tradetime"].max()) if len(panel_oos) else None
        split_meta = _build_split_meta(train_end, summary, summary_oos, is_dates, oos_dates)
        print(f"      IS summary  : {len(summary)} alphas")
        print(f"      OOS summary : {len(summary_oos)} alphas")
    else:
        summary = compute_per_alpha_metrics(alpha_panel, fwd)
    print(summary.to_string(index=False))
    summary.to_csv(REPORT_DIR / "alpha_ic_summary.csv", index=False)

    print("[5/6] Computing rolling IC & correlation ...")
    rolling = compute_rolling_ic(alpha_panel, fwd, window=args.rolling_window)
    corr = compute_alpha_correlation(alpha_panel)
    corr.to_csv(REPORT_DIR / "alpha_correlation_matrix.csv")

    print("[6/6] Plotting ...")
    plot_ic_bar_chart(summary, REPORT_DIR / "ic_bar_chart.png")
    plot_rolling_ic(rolling, REPORT_DIR / "rolling_ic.png")
    plot_correlation_heatmap(corr, REPORT_DIR / "correlation_heatmap.png")

    universe_desc = (
        f"{bars['security_id'].nunique()} stocks "
        f"{bars['tradetime'].min().date()}→{bars['tradetime'].max().date()}"
    )
    effective, selection = emit_selection_outputs(
        summary,
        horizon=args.horizon,
        source="python",
        universe_desc=universe_desc,
        train_end=train_end,
        summary_oos=summary_oos if train_end is not None else None,
        split_meta=split_meta,
    )

    print()
    print("=" * 60)
    print(f"Effective alphas ({len(effective)}/{len(summary)}, basis={selection['selection_basis']}): {effective}")
    print(f"Dropped          : {selection['dropped_alphas']}")
    print(f"Outputs          : {REPORT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
