"""Predict-only pipeline：用截至 T 日的歷史資料，輸出 T+1 應建倉的目標清單。

與 daily_batch_pipeline 的差別
------------------------------
* 不做 backtest 迴圈、不模擬成交、不寫入 fills/orders。
* 訓練窗：嚴格切到 T - purge_days，避免使用尚未實現的 forward return。
* 預測窗：只對 T 日的 alpha 截面 predict 一次。
* 輸出：top-k 目標（security_id / target_weight / signal_score / confidence），
  寫入 reports/predictions/predict_<T>.csv 並印到 stdout。

資料來源（``--data-source``）
-----------------------------
* ``tej``（預設）：``data/tw_stocks_tej.parquet``，TEJ Pro 還原股價匯出，**含 2018+ 期間
  下市股**，已過濾 ETF / 權證。下市股以「下市日報酬計 0、隔日退出 universe」
  規則隱式處理（不需顯式邏輯）。
* ``csv``：``data/tw_stocks_ohlcv.csv``，yfinance 下載的 1083 檔上市股，
  **不含下市股**（survivorship-biased），且已知 stock 8476 還原股價污染，僅保留 demo。

Alpha 來源（``--alpha-source``）
--------------------------------
* ``python_wq101``（預設）：完整 Python WQ101 + parquet 快取，無 Docker 需求。
* ``dolphindb``：從 DolphinDB ``alpha_features`` 讀取預計算 alpha，再套用
  ``reports/alpha_ic_analysis/effective_alphas.json`` 的 TEJ IS-only 清單，
  需要 ``docker-compose up -d``。

使用範例
--------
    # 預設：TEJ parquet（survivorship-correct）+ Python WQ101
    python -m pipelines.predict_next_day

    # DolphinDB alpha_features + TEJ effective_alphas.json（需 Docker）
    python -m pipelines.predict_next_day --alpha-source dolphindb

    # 指定 as-of 日
    python -m pipelines.predict_next_day --data-source tej --as-of 2026-04-17 --top-k 10
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from pipelines.daily_batch_pipeline import compute_python_alphas, load_csv_data
from pipelines.simulate_recent import _load_alphas_from_dolphindb
from src.alpha_engine.alpha_cache import cache_path_for_data_path, compute_with_cache
from src.common.logging import get_logger, setup_logging
from src.config.alpha_selection import EFFECTIVE_ALPHAS_PATH, load_effective_alpha_ids
from src.config.constants import DATA_SOURCE_DEFAULT_PATHS, DEFAULT_DATA_SOURCE
from src.labeling.label_generator import LabelGenerator
from src.meta_signal.ml_meta_model import MLMetaModel
from src.portfolio.constructor import PortfolioConstructor
from src.risk.risk_manager import RiskManager

setup_logging()
logger = get_logger("predict_next_day")


OUTPUT_DIR = Path("reports/predictions")

DATA_SOURCE_DEFAULTS = DATA_SOURCE_DEFAULT_PATHS


def _resolve_as_of(bars: pd.DataFrame, requested: date | None) -> pd.Timestamp:
    """Pick the actual as-of date — prefer requested, else fall back to latest bar."""
    available = pd.to_datetime(bars["tradetime"]).dt.normalize().drop_duplicates().sort_values()
    if requested is None:
        return available.iloc[-1]
    requested_ts = pd.Timestamp(requested)
    if requested_ts in set(available):
        return requested_ts
    fallback = available[available <= requested_ts]
    if fallback.empty:
        raise ValueError(
            f"No bars on or before {requested}; data starts {available.iloc[0].date()}"
        )
    chosen = fallback.iloc[-1]
    logger.warning(
        "as_of_not_in_bars",
        requested=str(requested),
        chosen=str(chosen.date()),
    )
    return chosen


def predict_next_day(
    csv_path: str | Path,
    as_of: date | None = None,
    purge_days: int = 5,
    top_k: int = 10,
    capital: float = 10_000_000.0,
    output_dir: str | Path = OUTPUT_DIR,
    alpha_source: str = "python",
    allow_yfinance: bool = False,
) -> pd.DataFrame:
    """Train on history ≤ T-purge, predict on T, return top-k risk-adjusted targets.

    Args:
        csv_path: Path to OHLCV CSV produced by scripts/download_tw_stocks.py.
        as_of: T — the most recent date whose alphas will drive the prediction.
            Defaults to the latest date in the CSV.
        purge_days: Gap between training tail and prediction date. With horizon=5
            the minimum safe gap is 5 (the forward-return label of date d uses
            close[d+5], which we don't know if d+horizon > T).
        top_k: Number of names to hold.
        capital: Cash size used to convert weights → share counts (informational).
        output_dir: Where to write predict_<as_of>.csv.
        alpha_source: "python_wq101"（完整 Python WQ101，無需 Docker）或
            "dolphindb"（預計算 alpha_features，再套用 effective_alphas.json，需要 Docker）。

    Returns:
        DataFrame with columns:
            security_id, target_weight, target_shares,
            signal_score, confidence, signal_direction, rank
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load bars ---
    bars = load_csv_data(csv_path, allow_yfinance=allow_yfinance)
    as_of_ts = _resolve_as_of(bars, as_of)
    bars = bars[bars["tradetime"] <= as_of_ts].reset_index(drop=True)
    logger.info(
        "bars_loaded",
        rows=len(bars),
        symbols=int(bars["security_id"].nunique()),
        as_of=str(as_of_ts.date()),
        alpha_source=alpha_source,
    )

    # --- 2. Load / compute alphas ---
    eff_alphas = load_effective_alpha_ids(EFFECTIVE_ALPHAS_PATH, required=True)
    if alpha_source == "dolphindb":
        alpha_panel = _load_alphas_from_dolphindb(
            as_of_ts.date(),
            as_of_ts.date(),
            buffer_days=365 * 5,   # 拉足夠的歷史供訓練（~5 年）
            alpha_ids=eff_alphas,
        )
        # 以 bars 的 (security_id, tradetime) 內連接
        bars_key = bars[["security_id", "tradetime"]].copy()
        bars_key["security_id"] = bars_key["security_id"].astype(str)
        alpha_panel = alpha_panel.merge(bars_key, on=["security_id", "tradetime"], how="inner")
        logger.info(
            "dolphindb_alphas_loaded",
            rows=len(alpha_panel),
            alphas=alpha_panel["alpha_id"].nunique(),
            symbols=alpha_panel["security_id"].nunique(),
        )
    elif alpha_source == "python_wq101":
        alpha_panel = compute_with_cache(
            bars,
            alpha_ids=eff_alphas,
            cache_path=cache_path_for_data_path(csv_path),
        )
        logger.info(
            "python_wq101_alphas_loaded",
            rows=len(alpha_panel),
            alphas=alpha_panel["alpha_id"].nunique(),
            symbols=alpha_panel["security_id"].nunique(),
        )
    else:
        # legacy "python" path — also uses WQ101 engine now
        alpha_panel = compute_python_alphas(
            bars,
            cache_path=cache_path_for_data_path(csv_path),
        )
        alpha_panel = alpha_panel[alpha_panel["alpha_id"].isin(eff_alphas)]
        logger.info("effective_alphas_applied", count=len(eff_alphas))

    # --- 3. Build labels & training set with purge gap ---
    label_gen = LabelGenerator(horizons=[5], bar_type="daily")
    labels = label_gen.generate_labels(bars[["security_id", "tradetime", "close"]])
    fwd_5 = labels[labels["horizon"] == 5].set_index(
        ["security_id", "signal_time"]
    )["forward_return"].dropna()
    fwd_5.index = fwd_5.index.set_names(["security_id", "tradetime"])

    purge_cutoff = as_of_ts - pd.Timedelta(days=purge_days)
    train_panel = alpha_panel[alpha_panel["tradetime"] <= purge_cutoff]
    train_labels = fwd_5[
        fwd_5.index.get_level_values("tradetime") <= purge_cutoff
    ]
    logger.info(
        "training_window",
        train_rows=len(train_panel),
        train_labels=len(train_labels),
        purge_cutoff=str(purge_cutoff.date()),
    )

    # --- 4. Train XGBoost on training window ---
    model = MLMetaModel(feature_columns=eff_alphas)
    train_info = model.train(train_panel, train_labels)
    logger.info(
        "model_trained",
        model_id=train_info["model_id"],
        ic=round(train_info["holdout_metrics"].get("ic", 0.0), 4),
        rank_ic=round(train_info["holdout_metrics"].get("rank_ic", 0.0), 4),
        n_features=train_info["n_features"],
    )

    # --- 5. Predict only on the as-of cross-section ---
    todays_panel = alpha_panel[alpha_panel["tradetime"] == as_of_ts]
    if todays_panel.empty:
        raise RuntimeError(
            f"No alpha values for as-of date {as_of_ts.date()}; "
            "check CSV coverage or pick an earlier date."
        )

    signals = model.predict(todays_panel)
    signals = signals.rename(columns={"tradetime": "signal_time"})
    signals["method"] = "ml_meta"
    signals["model_version_id"] = train_info["model_id"]

    # --- 6. Portfolio construction (top-k) ---
    constructor = PortfolioConstructor(
        method="equal_weight_topk", top_k=top_k, long_only=True
    )
    targets = constructor.construct(signals)
    if targets.empty:
        logger.warning("no_long_candidates", as_of=str(as_of_ts.date()))
        return pd.DataFrame()

    # --- 7. Risk constraints ---
    risk_mgr = RiskManager(max_position_weight=0.10, max_gross_exposure=1.0)
    adj = risk_mgr.apply_constraints(targets)

    # --- 8. Attach signal context + share counts ---
    enriched = adj.merge(
        signals[["security_id", "signal_score", "confidence", "signal_direction"]],
        on="security_id",
        how="left",
    )
    last_close = (
        bars[bars["tradetime"] == as_of_ts][["security_id", "close"]]
        .set_index("security_id")["close"]
    )
    enriched["last_close"] = enriched["security_id"].map(last_close)
    enriched["target_shares"] = (
        (enriched["target_weight"] * capital / enriched["last_close"])
        .round()
        .astype("Int64")
    )
    enriched = enriched.sort_values("signal_score", ascending=False).reset_index(drop=True)
    enriched["rank"] = enriched.index + 1

    cols = [
        "rank", "security_id", "target_weight", "target_shares",
        "last_close", "signal_score", "confidence", "signal_direction",
    ]
    out = enriched[cols]

    # --- 9. Persist ---
    out_path = output_dir / f"predict_{as_of_ts.strftime('%Y%m%d')}.csv"
    out.to_csv(out_path, index=False)
    logger.info("predictions_written", path=str(out_path), rows=len(out))

    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--data-source",
        choices=["csv", "tej"],
        default=DEFAULT_DATA_SOURCE,
        help=(
            "tej = TEJ ingest 產出的 data/tw_stocks_tej.parquet（預設，含期間下市股）; "
            "csv = yfinance 下載的 data/tw_stocks_ohlcv.csv（僅 demo，已知 8476 資料污染）"
        ),
    )
    p.add_argument(
        "--csv", default=None,
        help="OHLCV 資料路徑；省略時依 --data-source 取對應預設（csv→tw_stocks_ohlcv.csv / tej→tw_stocks_tej.parquet）",
    )
    p.add_argument(
        "--allow-yfinance",
        action="store_true",
        help="明確允許使用已知污染的 yfinance CSV（僅 demo/反例使用）",
    )
    p.add_argument("--as-of", help="As-of date YYYY-MM-DD (default: latest in data)")
    p.add_argument("--purge-days", type=int, default=5, help="Gap between train tail and predict date")
    p.add_argument("--top-k", type=int, default=10, help="Number of holdings")
    p.add_argument("--capital", type=float, default=10_000_000.0, help="Capital for share-count translation")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Where to write predict_<date>.csv")
    p.add_argument(
        "--alpha-source",
        choices=["python_wq101", "python", "dolphindb"],
        default="python_wq101",
        help=(
            "python_wq101=完整 WQ101 Python 版（預設，無需 Docker，使用 parquet 快取）/ "
            "dolphindb=alpha_features + effective_alphas.json（需 Docker）/ "
            "python=舊版 WQ101 引擎（同 python_wq101）"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    as_of = datetime.strptime(args.as_of, "%Y-%m-%d").date() if args.as_of else None
    csv_path = args.csv or DATA_SOURCE_DEFAULTS[args.data_source]
    logger.info("data_source_resolved", source=args.data_source, path=csv_path)

    targets = predict_next_day(
        csv_path=csv_path,
        as_of=as_of,
        purge_days=args.purge_days,
        top_k=args.top_k,
        capital=args.capital,
        output_dir=args.output_dir,
        alpha_source=args.alpha_source,
        allow_yfinance=args.allow_yfinance,
    )

    if targets.empty:
        print("\n[!] 沒有任何 long 候選股；請檢查資料或放寬 long_only 條件。")
        return

    print("\n=== 明日目標倉位（基於截至 as-of 的歷史） ===")
    print(targets.to_string(index=False))
    print(f"\n  總權重: {targets['target_weight'].sum():.4f}")
    print(f"  名單數: {len(targets)}")


if __name__ == "__main__":
    main()
