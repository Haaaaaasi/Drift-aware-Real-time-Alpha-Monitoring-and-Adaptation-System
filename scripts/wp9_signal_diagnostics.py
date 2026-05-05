"""WP9 訊號強度診斷實驗。

這個腳本用 TEJ alpha cache 做一組輕量診斷，目標是把「alpha 太弱」、
「XGBoost meta model 過擬合」、「portfolio 週期不對齊」與「交易成本」拆開看。

輸出：
* diagnostic_summary.csv：各訊號/持股方式的 zero-cost gross 表現
* ic_by_horizon.csv：訊號對 1/5/10 日 forward return 的 daily IC
* decile_returns.csv：raw/simple/XGB score 的 decile forward-return test
* daily_returns.csv：各 zero-cost portfolio 的每日 gross return
* summary.md：繁中摘要
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from pipelines.daily_batch_pipeline import load_csv_data
from src.alpha_engine.alpha_cache import cache_path_for_data_path
from src.config.alpha_selection import EFFECTIVE_ALPHAS_PATH, load_effective_alpha_ids
from src.config.data_sources import DATA_SOURCE_DEFAULT_PATHS
from src.labeling.label_generator import LabelGenerator
from src.meta_signal.ml_meta_model import MLMetaModel
from src.portfolio.constructor import PortfolioConstructor
from src.risk.risk_manager import RiskManager


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "reports" / "wp9_signal_diagnostics"


def _parse_date(value: str) -> date:
    return pd.Timestamp(value).date()


def _load_alpha_panel(
    cache_path: Path,
    start: date,
    end: date,
    alpha_ids: list[str],
) -> pd.DataFrame:
    """只從 parquet cache 讀取必要日期與 alpha，避免載入完整 200M rows。"""
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    dataset = ds.dataset(cache_path, format="parquet")
    flt = (
        (pc.field("tradetime") >= pd.Timestamp(start).to_pydatetime())
        & (pc.field("tradetime") <= pd.Timestamp(end).to_pydatetime())
        & pc.field("alpha_id").isin(alpha_ids)
    )
    table = dataset.to_table(
        columns=["security_id", "tradetime", "alpha_id", "alpha_value"],
        filter=flt,
    )
    df = table.to_pandas()
    if df.empty:
        raise RuntimeError(f"alpha cache 在 {start} 到 {end} 沒有資料：{cache_path}")
    df["security_id"] = df["security_id"].astype(str)
    df["tradetime"] = pd.to_datetime(df["tradetime"])
    return df


def _make_forward_returns(
    bars: pd.DataFrame,
    horizons: Iterable[int],
) -> tuple[dict[int, pd.Series], dict[int, pd.Series]]:
    labels = LabelGenerator(horizons=list(horizons), bar_type="daily").generate_labels(
        bars[["security_id", "tradetime", "close"]]
    )
    out: dict[int, pd.Series] = {}
    available_at: dict[int, pd.Series] = {}
    for h in horizons:
        h_labels = (
            labels[labels["horizon"] == h]
            .dropna(subset=["forward_return"])
            .set_index(["security_id", "signal_time"])
        )
        s = h_labels["forward_return"]
        s.index = s.index.set_names(["security_id", "tradetime"])
        out[int(h)] = s
        avail = pd.to_datetime(h_labels["label_available_at"])
        avail.index = avail.index.set_names(["security_id", "tradetime"])
        available_at[int(h)] = avail
    return out, available_at


def _make_next_day_returns(bars: pd.DataFrame) -> pd.Series:
    df = bars[["security_id", "tradetime", "close"]].copy()
    df = df.sort_values(["security_id", "tradetime"])
    df["next_close"] = df.groupby("security_id")["close"].shift(-1)
    df["next_return"] = df["next_close"] / df["close"] - 1.0
    return df.set_index(["security_id", "tradetime"])["next_return"]


def _wide(alpha_panel: pd.DataFrame) -> pd.DataFrame:
    wide = alpha_panel.pivot_table(
        index=["security_id", "tradetime"],
        columns="alpha_id",
        values="alpha_value",
        aggfunc="last",
    )
    return wide.sort_index()


def _load_rank_ic_weights(alpha_ids: list[str]) -> pd.Series:
    summary_path = ROOT / "reports" / "alpha_ic_analysis" / "alpha_ic_summary.csv"
    if not summary_path.exists():
        return pd.Series(1.0 / len(alpha_ids), index=alpha_ids)

    df = pd.read_csv(summary_path)
    weights = df.set_index("alpha_id")["rank_ic"].reindex(alpha_ids).fillna(0.0)
    if weights.abs().sum() <= 0:
        return pd.Series(1.0 / len(alpha_ids), index=alpha_ids)
    return weights / weights.abs().sum()


def _score_frame(score: pd.Series, name: str) -> pd.DataFrame:
    out = score.rename("signal_score").reset_index()
    out["signal_time"] = pd.to_datetime(out["tradetime"])
    out["signal_direction"] = np.where(out["signal_score"] >= 0, 1, -1).astype(int)
    out["confidence"] = out["signal_score"].abs()
    out["signal_name"] = name
    return out[[
        "security_id",
        "tradetime",
        "signal_time",
        "signal_score",
        "signal_direction",
        "confidence",
        "signal_name",
    ]]


def _daily_ic(scores: pd.Series, returns: pd.Series) -> tuple[float, float, int]:
    df = pd.concat([scores.rename("score"), returns.rename("ret")], axis=1).dropna()
    if df.empty:
        return np.nan, np.nan, 0
    rank_ics: list[float] = []
    pearson_ics: list[float] = []
    for _, g in df.groupby(level="tradetime"):
        if len(g) < 20 or g["score"].nunique() <= 1 or g["ret"].nunique() <= 1:
            continue
        rank_ics.append(float(g["score"].corr(g["ret"], method="spearman")))
        pearson_ics.append(float(g["score"].corr(g["ret"], method="pearson")))
    return float(np.nanmean(rank_ics)), float(np.nanmean(pearson_ics)), len(rank_ics)


def _topk_forward_mean(scores: pd.Series, returns: pd.Series, top_k: int) -> float:
    df = pd.concat([scores.rename("score"), returns.rename("ret")], axis=1).dropna()
    vals: list[float] = []
    for _, g in df.groupby(level="tradetime"):
        g = g.sort_values("score", ascending=False).head(top_k)
        if not g.empty:
            vals.append(float(g["ret"].mean()))
    return float(np.nanmean(vals)) if vals else np.nan


def _decile_test(scores: pd.Series, returns: pd.Series, signal_name: str) -> pd.DataFrame:
    df = pd.concat([scores.rename("score"), returns.rename("ret")], axis=1).dropna()
    rows: list[dict] = []
    for t, g in df.groupby(level="tradetime"):
        if len(g) < 50 or g["score"].nunique() < 10:
            continue
        ranked = g.copy()
        ranked["decile"] = pd.qcut(
            ranked["score"].rank(method="first"),
            10,
            labels=False,
            duplicates="drop",
        )
        if ranked["decile"].isna().all():
            continue
        day = ranked.groupby("decile")["ret"].mean()
        for decile, ret in day.items():
            rows.append({
                "date": pd.Timestamp(t).strftime("%Y-%m-%d"),
                "signal": signal_name,
                "decile": int(decile) + 1,
                "forward_return": float(ret),
            })
    return pd.DataFrame(rows)


def _simulate_zero_cost(
    signals: pd.DataFrame,
    next_returns: pd.Series,
    *,
    method: str,
    top_k: int,
    rebalance_every: int,
    entry_rank: int,
    exit_rank: int,
    max_turnover: float,
    min_holding_days: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    constructor = PortfolioConstructor(
        method=method,
        top_k=top_k,
        long_only=True,
        entry_rank=entry_rank,
        exit_rank=exit_rank,
        min_holding_days=min_holding_days,
    )
    risk_mgr = RiskManager(
        max_position_weight=1.0 / top_k,
        max_gross_exposure=1.0,
        max_turnover=max_turnover,
    )
    days = sorted(pd.to_datetime(signals["tradetime"]).unique())
    prev_weights: dict[str, float] = {}
    holding_days: dict[str, int] = {}
    last_rebalance_idx = -10**6
    records: list[dict] = []
    value = 1.0

    for i, t in enumerate(days):
        day_sig = signals[signals["tradetime"] == t].copy()
        if day_sig.empty:
            continue
        day_sig["signal_time"] = t
        rebalance_due = (
            not prev_weights
            or rebalance_every <= 1
            or (i - last_rebalance_idx) >= rebalance_every
        )

        if rebalance_due:
            target = constructor.construct(
                day_sig,
                previous_weights=prev_weights,
                holding_days=holding_days,
            )
            if target.empty:
                current_weights = {}
                cap_applied = False
            else:
                adjusted = risk_mgr.apply_constraints(target, previous_weights=prev_weights)
                current_weights = dict(
                    zip(adjusted["security_id"].astype(str), adjusted["target_weight"])
                )
                cap_applied = bool(adjusted.attrs.get("turnover_cap_applied", False))
                last_rebalance_idx = i
        else:
            tradable = set(day_sig["security_id"].astype(str))
            current_weights = {k: v for k, v in prev_weights.items() if k in tradable}
            cap_applied = False

        all_secs = set(prev_weights) | set(current_weights)
        buys = sum(max(0.0, current_weights.get(s, 0.0) - prev_weights.get(s, 0.0)) for s in all_secs)
        sells = sum(max(0.0, prev_weights.get(s, 0.0) - current_weights.get(s, 0.0)) for s in all_secs)
        gross = 0.0
        for sec, weight in current_weights.items():
            r = next_returns.get((sec, t), np.nan)
            if not np.isnan(r):
                gross += float(weight) * float(r)
        value *= 1.0 + gross
        records.append({
            "date": pd.Timestamp(t).strftime("%Y-%m-%d"),
            "gross_return": gross,
            "turnover": max(buys, sells),
            "n_holdings": len(current_weights),
            "rebalance_flag": bool(rebalance_due),
            "turnover_cap_applied": bool(cap_applied),
            "cumulative_value": value,
        })
        prev_weights = current_weights
        holding_days = {
            sec: (holding_days.get(sec, 0) + 1 if sec in prev_weights else 1)
            for sec in current_weights
            if abs(current_weights.get(sec, 0.0)) > 1e-12
        }

    pnl = pd.DataFrame(records)
    if pnl.empty:
        return pnl, {}
    daily = pnl["gross_return"]
    summary = {
        "n_days": int(len(pnl)),
        "cumulative_return_pct": float((pnl["cumulative_value"].iloc[-1] - 1.0) * 100),
        "avg_gross_return_bps": float(daily.mean() * 10000),
        "annualized_return_pct": float(((1 + daily.mean()) ** 252 - 1) * 100),
        "sharpe": float(daily.mean() / daily.std(ddof=0) * np.sqrt(252)) if daily.std(ddof=0) > 0 else np.nan,
        "max_drawdown_pct": float(((pnl["cumulative_value"] / pnl["cumulative_value"].cummax()) - 1).min() * 100),
        "avg_turnover": float(pnl["turnover"].mean()),
        "avg_holdings": float(pnl["n_holdings"].mean()),
        "rebalance_days": int(pnl["rebalance_flag"].sum()),
    }
    return pnl, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-source", choices=["tej", "csv"], default="tej")
    parser.add_argument("--start", type=_parse_date, default=date(2022, 6, 1))
    parser.add_argument("--end", type=_parse_date, default=date(2022, 8, 31))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--train-window-days", type=int, default=500)
    parser.add_argument("--purge-days", type=int, default=5)
    parser.add_argument("--entry-rank", type=int, default=20)
    parser.add_argument("--exit-rank", type=int, default=40)
    parser.add_argument("--rebalance-every", type=int, default=5)
    parser.add_argument("--max-turnover", type=float, default=0.25)
    parser.add_argument("--min-holding-days", type=int, default=5)
    parser.add_argument("--run-tag", default="short_smoke")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--allow-yfinance", action="store_true")
    args = parser.parse_args()

    data_path = ROOT / DATA_SOURCE_DEFAULT_PATHS[args.data_source]
    out_dir = Path(args.out_dir)
    run_dir = out_dir / f"diag_{args.start:%Y%m%d}_{args.end:%Y%m%d}_{args.run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    alpha_ids = load_effective_alpha_ids(ROOT / EFFECTIVE_ALPHAS_PATH, required=True)
    assert alpha_ids is not None
    train_start = args.start - pd.Timedelta(days=args.train_window_days + args.purge_days + 20)

    bars = load_csv_data(
        data_path,
        start=train_start,
        end=args.end,
        allow_yfinance=args.allow_yfinance,
    )
    bars["security_id"] = bars["security_id"].astype(str)

    cache_path = ROOT / cache_path_for_data_path(data_path)
    alpha_panel = _load_alpha_panel(cache_path, train_start, args.end, alpha_ids)
    bar_keys = bars[["security_id", "tradetime"]].drop_duplicates()
    alpha_panel = alpha_panel.merge(bar_keys, on=["security_id", "tradetime"], how="inner")

    wide = _wide(alpha_panel).reindex(columns=alpha_ids).fillna(0.0)
    fwd, label_available_at = _make_forward_returns(bars, [1, 5, 10])
    next_returns = _make_next_day_returns(bars)

    eval_mask = (
        (wide.index.get_level_values("tradetime") >= pd.Timestamp(args.start))
        & (wide.index.get_level_values("tradetime") <= pd.Timestamp(args.end))
    )
    eval_wide = wide.loc[eval_mask]

    weights = _load_rank_ic_weights(alpha_ids)
    simple_score = eval_wide.mul(weights.reindex(eval_wide.columns).fillna(0.0), axis=1).sum(axis=1)
    simple_signal = _score_frame(simple_score, "simple_signed_rank_ic")

    train_end = pd.Timestamp(args.start) - pd.Timedelta(days=args.purge_days)
    train_mask = (
        (wide.index.get_level_values("tradetime") >= pd.Timestamp(train_start))
        & (wide.index.get_level_values("tradetime") <= train_end)
    )
    train_panel = wide.loc[train_mask]
    labels_5 = fwd[5]
    label_avail_5 = label_available_at[5]
    train_idx = labels_5.index.intersection(train_panel.index)
    train_labels = labels_5.loc[train_idx]
    train_labels = train_labels[label_avail_5.loc[train_idx] <= pd.Timestamp(args.start)]

    model = MLMetaModel(feature_columns=alpha_ids, objective="forward_return", proxy_top_k=args.top_k)
    train_info = model.train(train_panel, train_labels)
    xgb_signal = model.predict(eval_wide)
    xgb_signal["signal_time"] = pd.to_datetime(xgb_signal["tradetime"])
    xgb_signal["signal_name"] = "xgb_forward_return"

    score_series = {
        "simple_signed_rank_ic": simple_score,
        "xgb_forward_return": xgb_signal.set_index(["security_id", "tradetime"])["signal_score"],
    }

    ic_rows: list[dict] = []
    for signal_name, scores in score_series.items():
        for horizon, returns in fwd.items():
            eval_returns = returns.loc[returns.index.intersection(scores.index)]
            rank_ic, ic, n_days = _daily_ic(scores, eval_returns)
            ic_rows.append({
                "signal": signal_name,
                "horizon": horizon,
                "mean_rank_ic": rank_ic,
                "mean_ic": ic,
                "n_days": n_days,
                "topk_forward_mean_bps": _topk_forward_mean(scores, eval_returns, args.top_k) * 10000,
            })

    deciles = []
    for signal_name, scores in score_series.items():
        eval_returns = fwd[5].loc[fwd[5].index.intersection(scores.index)]
        deciles.append(_decile_test(scores, eval_returns, signal_name))
    decile_df = pd.concat(deciles, ignore_index=True) if deciles else pd.DataFrame()

    portfolio_specs = [
        ("simple_daily_topk_zero_cost", simple_signal, "equal_weight_topk", 1, 1.0),
        ("simple_turnover_aware_zero_cost", simple_signal, "turnover_aware_topk", args.rebalance_every, args.max_turnover),
        ("xgb_daily_topk_zero_cost", xgb_signal, "equal_weight_topk", 1, 1.0),
        ("xgb_turnover_aware_zero_cost", xgb_signal, "turnover_aware_topk", args.rebalance_every, args.max_turnover),
        ("simple_rebalance5_hold5_zero_cost", simple_signal, "equal_weight_topk", 5, 1.0),
        ("xgb_rebalance5_hold5_zero_cost", xgb_signal, "equal_weight_topk", 5, 1.0),
    ]

    summary_rows: list[dict] = []
    daily_parts: list[pd.DataFrame] = []
    for name, sig, method, rebalance_every, max_turnover in portfolio_specs:
        pnl, summary = _simulate_zero_cost(
            sig,
            next_returns,
            method=method,
            top_k=args.top_k,
            rebalance_every=rebalance_every,
            entry_rank=args.entry_rank,
            exit_rank=args.exit_rank,
            max_turnover=max_turnover,
            min_holding_days=args.min_holding_days if method == "turnover_aware_topk" else 0,
        )
        summary["experiment"] = name
        summary["signal"] = sig["signal_name"].iloc[0]
        summary["portfolio_method"] = method
        summary["rebalance_every"] = rebalance_every
        summary["max_turnover"] = max_turnover
        summary_rows.append(summary)
        if not pnl.empty:
            pnl = pnl.copy()
            pnl["experiment"] = name
            daily_parts.append(pnl)

    summary_df = pd.DataFrame(summary_rows)
    ic_df = pd.DataFrame(ic_rows)
    daily_df = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()

    decile_agg = pd.DataFrame()
    if not decile_df.empty:
        decile_agg = (
            decile_df.groupby(["signal", "decile"], as_index=False)["forward_return"]
            .mean()
            .assign(forward_return_bps=lambda d: d["forward_return"] * 10000)
        )

    summary_df.to_csv(run_dir / "diagnostic_summary.csv", index=False)
    ic_df.to_csv(run_dir / "ic_by_horizon.csv", index=False)
    decile_df.to_csv(run_dir / "decile_returns_daily.csv", index=False)
    decile_agg.to_csv(run_dir / "decile_returns.csv", index=False)
    daily_df.to_csv(run_dir / "daily_returns.csv", index=False)

    best = summary_df.sort_values("avg_gross_return_bps", ascending=False).iloc[0]
    xgb_ic5 = ic_df[(ic_df["signal"] == "xgb_forward_return") & (ic_df["horizon"] == 5)]
    simple_ic5 = ic_df[(ic_df["signal"] == "simple_signed_rank_ic") & (ic_df["horizon"] == 5)]
    md = [
        "# WP9 訊號診斷摘要",
        "",
        f"- 期間：{args.start} 到 {args.end}",
        f"- 資料源：{args.data_source}",
        f"- effective alphas：{len(alpha_ids)}",
        f"- XGBoost 訓練樣本：{train_info['n_train']}，features：{train_info['n_features']}",
        f"- 最佳 zero-cost portfolio：`{best['experiment']}`，avg gross {best['avg_gross_return_bps']:.3f} bps/day，cum {best['cumulative_return_pct']:.2f}%",
        "",
        "## 5 日訊號 IC",
        "",
        f"- simple signed IC ensemble：{float(simple_ic5['mean_rank_ic'].iloc[0]):.4f}" if not simple_ic5.empty else "- simple signed IC ensemble：NA",
        f"- XGBoost：{float(xgb_ic5['mean_rank_ic'].iloc[0]):.4f}" if not xgb_ic5.empty else "- XGBoost：NA",
        "",
        "## 需要解讀的紅旗",
        "",
        "- 若 5 日 IC 為正但 daily top-k gross 為負，代表 horizon/portfolio 對齊有問題。",
        "- 若 simple ensemble 勝過 XGBoost，代表 meta model 可能把薄訊號過擬合壞。",
        "- 若 simple decile top-bottom spread 也不明顯，才比較能說 alpha 本身在 TEJ OOS 太弱。",
    ]
    (run_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")

    print(f"run_dir={run_dir}")
    print(summary_df[[
        "experiment",
        "cumulative_return_pct",
        "avg_gross_return_bps",
        "sharpe",
        "avg_turnover",
        "avg_holdings",
    ]].to_string(index=False))
    print()
    print(ic_df.to_string(index=False))


if __name__ == "__main__":
    main()
