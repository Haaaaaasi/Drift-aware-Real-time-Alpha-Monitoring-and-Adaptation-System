"""將 TEJ parquet/CSV 批次寫入 DolphinDB standardized_bars 表。

schema: security_id, tradetime, bar_type, open, high, low, close, vol, vwap,
        cap, indclass, is_tradable, missing_flags

Usage:
    # 預設使用 TEJ survivorship-correct parquet
    python scripts/ingest_csv_to_dolphindb.py

    # 僅流動性 >= 5000 萬 NTD 的股票（加快速度）
    python scripts/ingest_csv_to_dolphindb.py --min-turnover-ntd 50000000

    # 先清空表再重新寫入（避免重複）
    python scripts/ingest_csv_to_dolphindb.py --truncate
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.db import get_dolphindb
from src.common.logging import get_logger, setup_logging

setup_logging()
logger = get_logger("ingest_csv")


DB_MARKET = "dfs://darams_market"
TABLE_NAME = "standardized_bars"


def _load_and_standardize(csv_path: Path) -> pd.DataFrame:
    """讀 OHLCV CSV/parquet 並補齊 DolphinDB 需要的所有欄位。"""
    if csv_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(csv_path)
    else:
        df = pd.read_csv(csv_path)
    df["tradetime"] = pd.to_datetime(df["datetime"])
    df = df.drop(columns=["datetime"])
    df["security_id"] = df["security_id"].astype(str)

    # vwap 近似：用 (H+L+C)/3 × vol 分母補回 vwap（若缺）
    df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0

    # cap proxy：close × 流通股數 proxy（此處無真實股本，用 close × volume 近似 turnover 級距）
    # 實務上應從 financial_master 取；MVP 用 close × 1e6 當簡化 proxy
    df["cap"] = df["close"] * 1_000_000

    # indclass hash proxy（0 為保留，給 1..5）
    df["indclass"] = (df["security_id"].astype(str).apply(lambda s: abs(hash(s)) % 5 + 1)).astype(int)

    df["bar_type"] = "daily"
    df["is_tradable"] = True
    df["missing_flags"] = 0

    # 重新命名 volume -> vol
    if "volume" in df.columns:
        df = df.rename(columns={"volume": "vol"})

    ordered = [
        "security_id", "tradetime", "bar_type", "open", "high", "low", "close",
        "vol", "vwap", "cap", "indclass", "is_tradable", "missing_flags",
    ]
    df = df[ordered].copy()
    df = df.dropna(subset=["close", "open", "high", "low", "vol"])
    logger.info("csv_loaded_and_standardized", rows=len(df),
                symbols=df["security_id"].nunique(),
                start=str(df["tradetime"].min()), end=str(df["tradetime"].max()))
    return df


def _apply_turnover_filter(df: pd.DataFrame, min_turnover_ntd: float) -> pd.DataFrame:
    if min_turnover_ntd <= 0:
        return df
    turnover = (df["vol"] * df["close"]).groupby(df["security_id"]).mean()
    keep = set(turnover[turnover >= min_turnover_ntd].index)
    before = df["security_id"].nunique()
    out = df[df["security_id"].isin(keep)].reset_index(drop=True)
    logger.info("turnover_filtered", min_ntd=min_turnover_ntd,
                symbols_before=before, symbols_after=out["security_id"].nunique())
    return out


def _truncate(client) -> None:
    """重建 standardized_bars 分區（最簡單：drop + 重跑 setup_database）。
    這裡採 DELETE where 1=1 保留 schema。"""
    logger.warning("truncating_table", table=TABLE_NAME)
    client.run(f'delete from loadTable("{DB_MARKET}","{TABLE_NAME}")')


def _chunked_upload(client, df: pd.DataFrame, chunk_size: int = 100_000) -> int:
    """分批 upload 到 DolphinDB（避免單次 payload 過大）。"""
    total = 0
    n_chunks = (len(df) + chunk_size - 1) // chunk_size
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size].copy()
        client.upload({"bars_upload": chunk})
        client.run(f'loadTable("{DB_MARKET}","{TABLE_NAME}").append!(bars_upload)')
        total += len(chunk)
        logger.info("chunk_uploaded", chunk=f"{i // chunk_size + 1}/{n_chunks}",
                    rows=len(chunk), cumulative=total)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="data/tw_stocks_tej.parquet")
    parser.add_argument("--min-turnover-ntd", type=float, default=0.0,
                        help="最小流動性過濾，0=不過濾；示範：50000000 = 5千萬")
    parser.add_argument("--truncate", action="store_true", help="寫入前先清空表")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"OHLCV 檔案不存在：{csv_path}")

    df = _load_and_standardize(csv_path)
    df = _apply_turnover_filter(df, args.min_turnover_ntd)

    client = get_dolphindb()
    try:
        if args.truncate:
            _truncate(client)
        total = _chunked_upload(client, df, chunk_size=args.chunk_size)
        # 驗證
        cnt = client.run(f'exec count(*) from loadTable("{DB_MARKET}","{TABLE_NAME}")')
        n_sec = client.run(
            f'exec count(*) from (select distinct security_id from loadTable("{DB_MARKET}","{TABLE_NAME}"))'
        )
        print(f"\n=== Ingest 完成 ===")
        print(f"本次上傳 : {total:,} 行")
        print(f"表中總行數: {int(cnt):,}")
        print(f"獨立標的數: {int(n_sec):,}")
    finally:
        pass


if __name__ == "__main__":
    main()
