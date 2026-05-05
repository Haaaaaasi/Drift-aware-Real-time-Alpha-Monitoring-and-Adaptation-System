"""P0★ #1 — In-sample / Out-of-sample alpha selection validation runner.

直接讀 ``data/alpha_cache/wq101_alphas.parquet``（已含 1105 stocks × 101 alphas
× ~1851 days，含 51 檔 TEJ 期間下市股），透過 pyarrow predicate pushdown 逐 alpha
串流（每 alpha < 0.1s），再呼叫 notebook 中的 IS/OOS helper 寫出報告與
``effective_alphas.json``。

universe 篩選：以「IS 期間（tradetime ≤ train_end）內交易天數 ≥ min_is_days」
為條件挑出合格股票池——這個池**含期間下市股**，但排除 IS 末段才上市的新股。
之後若加 ``--max-stocks``，會以固定 seed 隨機抽樣保證 reproducibility。

舊的「top-N by row count」做法被淘汰，原因：對 TEJ 全 1105 檔（含下市）這種寬
universe，row-count 排序會系統性排除任何 2024-06 之前下市的股票，等同重新引入
survivorship bias。

用法
----
    # 預設 TEJ + train_end 2024-06-30 + 200 隨機抽樣
    python scripts/run_is_oos_validation.py

    # 全 universe（不抽樣）
    python scripts/run_is_oos_validation.py --max-stocks 0

    # CSV (yfinance) 對照——保留向後相容
    python scripts/run_is_oos_validation.py --data-source csv --train-end 2024-06-30
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipelines.daily_batch_pipeline import load_csv_data
from src.labeling.label_generator import LabelGenerator


def _load_notebook():
    spec = importlib.util.spec_from_file_location(
        "nb_alpha_ic", ROOT / "notebooks" / "01_alpha_ic_analysis.py"
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["nb_alpha_ic"] = m
    spec.loader.exec_module(m)
    return m


DATA_SOURCE_DEFAULT_PATHS = {
    "csv": ROOT / "data" / "tw_stocks_ohlcv.csv",
    "tej": ROOT / "data" / "tw_stocks_tej.parquet",
}


def resolve_data_path(args: argparse.Namespace) -> Path:
    """``--csv`` 提供時優先；否則依 ``--data-source`` 對應預設路徑。"""
    if args.csv:
        return Path(args.csv)
    return DATA_SOURCE_DEFAULT_PATHS[args.data_source]


def select_survivorship_correct_universe(
    bars: pd.DataFrame,
    train_end: pd.Timestamp,
    min_is_days: int,
    max_stocks: int,
    seed: int,
) -> tuple[list[str], dict]:
    """以 IS 期間交易天數為條件挑選股票，**含期間下市股**。

    步驟
    ----
    1. 取 ``tradetime ≤ train_end`` 的 bars
    2. 計算每檔在 IS 期間的交易天數
    3. ``count >= min_is_days`` 為合格池
    4. ``max_stocks > 0`` 時以 ``seed`` 隨機抽樣，否則用全部合格股票

    回傳 ``(selected_ids, stats)``。``stats`` 含 universe / qualified /
    selected 三段計數，方便寫入報告 metadata。
    """
    is_bars = bars[bars["tradetime"] <= train_end]
    counts = is_bars.groupby("security_id").size()
    universe_n = int(counts.size)
    qualified = counts[counts >= min_is_days].index.tolist()
    qualified_n = len(qualified)

    if max_stocks and max_stocks > 0 and qualified_n > max_stocks:
        rng = np.random.default_rng(seed)
        selected = sorted(rng.choice(qualified, size=max_stocks, replace=False).tolist())
        sample_mode = f"random sample seed={seed}"
    else:
        selected = sorted(qualified)
        sample_mode = "all qualified"

    stats = {
        "universe": universe_n,
        "qualified": qualified_n,
        "selected": len(selected),
        "min_is_days": min_is_days,
        "sample_mode": sample_mode,
    }
    return selected, stats


def list_alpha_ids_from_cache(cache_path: Path) -> list[str]:
    """回傳 cache 中所有 distinct alpha_id（升冪）。"""
    dset = ds.dataset(cache_path, format="parquet")
    table = dset.to_table(columns=["alpha_id"])
    return sorted(table.column("alpha_id").unique().to_pylist())


def load_single_alpha_from_cache(
    cache_path: Path,
    alpha_id: str,
    security_ids: list[str],
) -> pd.DataFrame:
    """以 pyarrow filter 一次只讀單一 alpha 的指定股票切片。"""
    dset = ds.dataset(cache_path, format="parquet")
    sid_strs = [str(s) for s in security_ids]
    flt = (pc.field("alpha_id") == alpha_id) & (pc.field("security_id").isin(sid_strs))
    table = dset.to_table(filter=flt)
    df = table.to_pandas()
    df["tradetime"] = pd.to_datetime(df["tradetime"])
    df["security_id"] = df["security_id"].astype(str)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-source",
        choices=["csv", "tej"],
        default="tej",
        help="OHLCV 來源；省略 --csv 時依此取預設路徑。預設 tej（survivorship-correct）",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="顯式指定 OHLCV 路徑（CSV 或 parquet 皆可）；提供時覆寫 --data-source",
    )
    parser.add_argument("--cache", default=str(ROOT / "data" / "alpha_cache" / "wq101_alphas.parquet"))
    parser.add_argument("--train-end", default="2024-06-30")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument(
        "--max-stocks",
        type=int,
        default=200,
        help="股票抽樣大小；0 表示用全部合格股票（預設 200，與舊 yfinance 跑法樣本數對齊）",
    )
    parser.add_argument(
        "--min-is-days",
        type=int,
        default=252,
        help="IS 期間最少交易天數（預設 252，~1 年），低於此數視為新股或樣本不足",
    )
    parser.add_argument("--seed", type=int, default=42, help="隨機抽樣 seed，固定以保 reproducibility")
    args = parser.parse_args()

    nb = _load_notebook()
    train_end = pd.Timestamp(args.train_end)
    data_path = resolve_data_path(args)

    print(f"[1/5] Loading bars from {data_path}")
    bars = load_csv_data(data_path, allow_yfinance=args.data_source == "csv")
    bars["security_id"] = bars["security_id"].astype(str)
    print(
        f"      total rows={len(bars):,}  symbols={bars['security_id'].nunique()}  "
        f"range={bars['tradetime'].min().date()} → {bars['tradetime'].max().date()}"
    )

    print(
        f"[2/5] Survivorship-correct universe selection "
        f"(min_is_days={args.min_is_days}, max_stocks={args.max_stocks}, seed={args.seed})"
    )
    selected_ids, stats = select_survivorship_correct_universe(
        bars,
        train_end=train_end,
        min_is_days=args.min_is_days,
        max_stocks=args.max_stocks,
        seed=args.seed,
    )

    delisted_in_selected = 0
    universe_path = ROOT / "data" / "tw_stocks_tej_universe.parquet"
    if args.data_source == "tej" and universe_path.exists():
        try:
            uni = pd.read_parquet(universe_path)
            uni["security_id"] = uni["security_id"].astype(str)
            inactive_set = set(uni.loc[~uni["is_active_at_end"], "security_id"])
            delisted_in_selected = sum(1 for s in selected_ids if s in inactive_set)
        except Exception as e:
            print(f"      [warn] could not read TEJ universe parquet: {e}")
    stats["delisted_in_selected"] = delisted_in_selected
    print(
        f"      universe={stats['universe']:,} qualified={stats['qualified']:,} "
        f"selected={stats['selected']:,} (delisted_in_selected={delisted_in_selected})  "
        f"sample_mode={stats['sample_mode']}"
    )

    bars_sub = bars[bars["security_id"].isin(selected_ids)].copy()

    print(f"[3/5] Generating fwd labels (horizon={args.horizon})")
    labels = LabelGenerator(horizons=[args.horizon], bar_type="daily").generate_labels(
        bars_sub[["security_id", "tradetime", "close"]]
    )
    fwd = (
        labels[labels["horizon"] == args.horizon]
        .set_index(["security_id", "signal_time"])["forward_return"]
    )
    fwd.index = fwd.index.set_names(["security_id", "tradetime"])
    tt_fwd = fwd.index.get_level_values("tradetime")
    fwd_is = fwd[tt_fwd <= train_end]
    fwd_oos = fwd[tt_fwd > train_end]
    print(f"      label rows={len(fwd):,}  IS={len(fwd_is):,}  OOS={len(fwd_oos):,}")

    print(f"[4/5] Streaming IC by alpha (cache={args.cache})")
    cache_path = Path(args.cache)
    alpha_ids = list_alpha_ids_from_cache(cache_path)
    print(f"      alphas to process: {len(alpha_ids)}")

    ic_rows_is: list[dict] = []
    ic_rows_oos: list[dict] = []
    is_panel_min: pd.Timestamp | None = None
    is_panel_max: pd.Timestamp | None = None
    oos_panel_min: pd.Timestamp | None = None
    oos_panel_max: pd.Timestamp | None = None
    t_start = time.time()
    for i, aid in enumerate(alpha_ids, 1):
        panel = load_single_alpha_from_cache(cache_path, aid, selected_ids)
        if panel.empty:
            continue
        vals = panel.set_index(["security_id", "tradetime"])["alpha_value"]
        ttimes = vals.index.get_level_values("tradetime")
        vals_is = vals[ttimes <= train_end]
        vals_oos = vals[ttimes > train_end]

        if not vals_is.empty:
            cur_min, cur_max = vals_is.index.get_level_values("tradetime").min(), vals_is.index.get_level_values("tradetime").max()
            is_panel_min = cur_min if is_panel_min is None or cur_min < is_panel_min else is_panel_min
            is_panel_max = cur_max if is_panel_max is None or cur_max > is_panel_max else is_panel_max
        if not vals_oos.empty:
            cur_min, cur_max = vals_oos.index.get_level_values("tradetime").min(), vals_oos.index.get_level_values("tradetime").max()
            oos_panel_min = cur_min if oos_panel_min is None or cur_min < oos_panel_min else oos_panel_min
            oos_panel_max = cur_max if oos_panel_max is None or cur_max > oos_panel_max else oos_panel_max

        row_is = nb.compute_alpha_row(aid, vals_is, fwd_is)
        if row_is is not None:
            ic_rows_is.append(row_is)
        row_oos = nb.compute_alpha_row(aid, vals_oos, fwd_oos)
        if row_oos is not None:
            ic_rows_oos.append(row_oos)
        if i % 20 == 0 or i == len(alpha_ids):
            print(f"      {i}/{len(alpha_ids)} alphas processed  elapsed={time.time()-t_start:.1f}s")

    summary_is = (
        pd.DataFrame(ic_rows_is)
        .sort_values("abs_ic", ascending=False)
        .reset_index(drop=True)
    )
    summary_oos = (
        pd.DataFrame(ic_rows_oos)
        .sort_values("abs_ic", ascending=False)
        .reset_index(drop=True)
        if ic_rows_oos else pd.DataFrame()
    )
    print(f"      IS summary  : {len(summary_is)} alphas")
    print(f"      OOS summary : {len(summary_oos)} alphas")

    nb.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary_is.to_csv(nb.REPORT_DIR / "alpha_ic_summary.csv", index=False)

    is_dates = (is_panel_min, is_panel_max) if is_panel_min is not None else None
    oos_dates = (oos_panel_min, oos_panel_max) if oos_panel_min is not None else None
    split_meta = nb._build_split_meta(train_end, summary_is, summary_oos, is_dates, oos_dates)
    split_meta.update({
        "data_source": args.data_source,
        "data_path": str(data_path),
        "min_is_days": stats["min_is_days"],
        "max_stocks": args.max_stocks,
        "seed": args.seed,
        "universe_total": stats["universe"],
        "qualified_total": stats["qualified"],
        "selected_total": stats["selected"],
        "delisted_in_selected": stats["delisted_in_selected"],
        "sample_mode": stats["sample_mode"],
    })

    universe_desc = (
        f"{stats['selected']} stocks "
        f"({stats['delisted_in_selected']} delisted) "
        f"{bars_sub['tradetime'].min().date()}→{bars_sub['tradetime'].max().date()} "
        f"[{stats['sample_mode']}, {stats['qualified']} qualified, "
        f"min_is_days={stats['min_is_days']}, source={args.data_source}]"
    )

    print(f"[5/5] Writing effective_alphas.json (universe: {universe_desc})")
    effective, selection = nb.emit_selection_outputs(
        summary_is,
        horizon=args.horizon,
        source=f"parquet_cache_{args.data_source}",
        universe_desc=universe_desc,
        train_end=train_end,
        summary_oos=summary_oos,
        split_meta=split_meta,
    )

    print()
    print("=" * 60)
    print(f"IS-selected effective alphas ({len(effective)}/{len(summary_is)}):")
    is_idx = summary_is.set_index("alpha_id")
    oos_idx = summary_oos.set_index("alpha_id") if not summary_oos.empty else pd.DataFrame()
    sign_flips = []
    for aid in effective:
        is_rank_ic = is_idx.loc[aid, "rank_ic"]
        oos_rank_ic = oos_idx.loc[aid, "rank_ic"] if not oos_idx.empty and aid in oos_idx.index else float("nan")
        flip = (
            not np.isnan(oos_rank_ic)
            and np.sign(is_rank_ic) != np.sign(oos_rank_ic)
        )
        if flip:
            sign_flips.append(aid)
        marker = " (sign-flip)" if flip else ""
        print(f"  {aid}: IS rank_ic={is_rank_ic:+.4f}  OOS rank_ic={oos_rank_ic:+.4f}{marker}")
    if sign_flips:
        print(f"\nSign-flips ({len(sign_flips)}): {', '.join(sign_flips)}")
    print(f"\nOutputs : {nb.REPORT_DIR}/effective_alphas.json")
    print(f"          {nb.REPORT_DIR}/effective_alphas_oos_validation.csv")
    print(f"          {nb.REPORT_DIR}/alpha_ic_summary.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
