"""TEJ OHLCV CSV → 標準化 parquet（含 survivorship-correct universe）。

合併 TEJ Pro 匯出的多份「調整股價」CSV（UTF-16 LE / tab，含期間下市股），
過濾為 4 碼純數字普通股，輸出與 yfinance ``data/tw_stocks_ohlcv.csv``
完全相同的長表格 schema 供下游 pipeline 直接讀入。

輸出
----
1. ``data/tw_stocks_tej.parquet``           — 主要 OHLCV 資料
2. ``data/tw_stocks_tej_universe.parquet``  — 每檔 (first_date, last_date) bounds，
   供 survivorship 分析與 universe-by-day 過濾使用

下市股處理規則（保守一致版）
----------------------------
保留每檔下市股的所有 OHLCV rows（含停牌期間 vol=0 的延展）。下游 simulation
不需要做任何特別處理就會自然套用以下規則：

* **下市日當天**：``next_close`` 為 NaN（沒有下一筆 bar）→ ``next_return`` NaN
  → simulate 迴圈將該股的當日報酬計為 0（已建好的 NaN-skip 邏輯）。
* **下市日之後**：該股不在 alpha_panel / next_ret 中 → 自動退出 universe。

此規則符合「每檔下市股從下市日往前回推到最後一筆 OHLCV，當天報酬計為 0
（即不交易），下市日後該股退出 universe」。不需要區分下市原因（合併/終止/
轉櫃），對 reviewer 是保守且方法學上 defensible 的選擇。

使用範例
--------
    # 預設讀取根目錄兩份 TEJ CSV，輸出 data/tw_stocks_tej.parquet
    python scripts/ingest_tej_csv.py

    # 自訂輸入清單
    python scripts/ingest_tej_csv.py --input OHLSV20182022.csv OHLSV202320260502.csv

    # 自訂輸出路徑
    python scripts/ingest_tej_csv.py --output data/custom_tej.parquet
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = ["OHLSV20182022.csv", "OHLSV202320260502.csv"]
DEFAULT_OUTPUT = Path("data/tw_stocks_tej.parquet")
DEFAULT_UNIVERSE_OUTPUT = Path("data/tw_stocks_tej_universe.parquet")

COMMON_STOCK_RE = re.compile(r"^\d{4}$")


def load_tej_csv(path: str | Path) -> pd.DataFrame:
    """讀單一 TEJ OHLCV CSV（UTF-16 LE / tab）並轉為標準 schema。

    TEJ 欄位順序固定為：證券代號（含中文名）/ 年月日 / 開盤 / 最高 / 最低 / 收盤 /
    成交量(千股)。我們只用順序而非欄位名稱 match，避免被中文 garble 影響。

    Returns
    -------
    DataFrame：columns = ``security_id`` (str) / ``datetime`` (datetime64[ns]) /
    ``open`` / ``high`` / ``low`` / ``close`` (float64) / ``volume`` (int64, 已從千股換算為股)
    """
    df = pd.read_csv(path, encoding="utf-16-le", sep="\t")
    if df.shape[1] != 7:
        raise ValueError(
            f"{path} 欄位數 = {df.shape[1]}，預期 7 欄"
            f"（證券代號 / 日期 / O / H / L / C / 成交量(千股)）"
        )
    df.columns = ["name_combined", "date", "open", "high", "low", "close", "volume_kshare"]

    df["security_id"] = df["name_combined"].astype(str).str.extract(r"^(\S+)")[0]
    df["datetime"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    # TEJ 成交量單位是千股，轉為股以對齊 yfinance 既有 schema
    df["volume"] = (df["volume_kshare"].astype("int64") * 1000).astype("int64")

    return df[["security_id", "datetime", "open", "high", "low", "close", "volume"]]


def filter_common_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """只保留 4 碼純數字普通股，過濾 ETF（00xx）/ 槓桿反向（5碼+L/R/U）/
    TDR（910xxx）/ 其他衍生商品（DR / IX / LB 等）。

    這是因為 WQ101 alpha 是 cross-sectional rank-based，混入 ETF / 槓桿 ETF
    會嚴重污染 ranking（這些商品的價格動態與普通股本質不同）。
    """
    mask = df["security_id"].str.fullmatch(r"\d{4}", na=False)
    return df[mask].reset_index(drop=True)


def build_universe_bounds(df: pd.DataFrame, active_threshold_days: int = 30) -> pd.DataFrame:
    """為每檔股票建構 (first_date, last_date, is_active_at_end, n_trading_days)。

    ``is_active_at_end`` 定義：``last_date >= data_end - active_threshold_days``。
    用於識別「於資料末段仍正常交易」的標的；其餘視為已下市（implicit delisting）。

    這個表格不是強制給 simulate_recent 使用——下游程式仍透過 alpha_panel 的存在
    與否做 per-day universe filter——但對於 survivorship 分析（例如「sim 期間有
    多少檔下市」、「下市股 last_date 與 TEJ 名冊是否吻合」）非常方便。
    """
    bounds = (
        df.groupby("security_id")["datetime"]
        .agg(first_date="min", last_date="max")
        .reset_index()
    )
    n_days = df.groupby("security_id").size().rename("n_trading_days").reset_index()
    bounds = bounds.merge(n_days, on="security_id", how="left")

    data_end = df["datetime"].max()
    threshold = data_end - pd.Timedelta(days=active_threshold_days)
    bounds["is_active_at_end"] = bounds["last_date"] >= threshold
    return bounds.sort_values("security_id").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", nargs="+", default=DEFAULT_INPUT,
        help=f"TEJ OHLCV CSV 路徑，可指定多份（預設：{' '.join(DEFAULT_INPUT)}）",
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT),
        help=f"OHLCV parquet 輸出路徑（預設：{DEFAULT_OUTPUT}）",
    )
    parser.add_argument(
        "--universe-output", default=str(DEFAULT_UNIVERSE_OUTPUT),
        help=f"Universe bounds parquet 輸出路徑（預設：{DEFAULT_UNIVERSE_OUTPUT}）",
    )
    parser.add_argument(
        "--active-threshold-days", type=int, default=30,
        help="判斷 is_active_at_end 的門檻（last_date 距資料末日的天數）",
    )
    args = parser.parse_args()

    print(f"讀入 {len(args.input)} 份 TEJ CSV ...")
    frames: list[pd.DataFrame] = []
    for p in args.input:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"找不到輸入檔：{path}")
        df = load_tej_csv(path)
        print(
            f"  {path.name}: rows={len(df):,}, ids={df['security_id'].nunique()}, "
            f"{df['datetime'].min().date()} → {df['datetime'].max().date()}"
        )
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    # 兩份檔案理論上不重疊；保險起見對 (id, date) 去重，保留最後一筆
    before = len(combined)
    combined = combined.drop_duplicates(subset=["security_id", "datetime"], keep="last")
    if len(combined) < before:
        print(f"  [WARN] 移除重複 {before - len(combined):,} 筆 (id, date)")
    print(
        f"\n合併後：{len(combined):,} rows, {combined['security_id'].nunique()} ids, "
        f"{combined['datetime'].min().date()} → {combined['datetime'].max().date()}"
    )

    common = filter_common_stocks(combined)
    print(
        f"過濾為 4 碼普通股：{len(common):,} rows, {common['security_id'].nunique()} ids "
        f"（移除 {combined['security_id'].nunique() - common['security_id'].nunique()} 檔非普通股：ETF / 權證 / TDR）"
    )

    common = common.sort_values(["security_id", "datetime"]).reset_index(drop=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    common.to_parquet(out_path, compression="snappy", index=False)
    print(f"\nOHLCV parquet 已寫入：{out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    bounds = build_universe_bounds(common, active_threshold_days=args.active_threshold_days)
    bounds_path = Path(args.universe_output)
    bounds.to_parquet(bounds_path, compression="snappy", index=False)
    n_active = int(bounds["is_active_at_end"].sum())
    n_delisted = int((~bounds["is_active_at_end"]).sum())
    print(f"Universe bounds 已寫入：{bounds_path}")
    print(f"  active at end: {n_active}")
    print(f"  delisted (last_date < end - {args.active_threshold_days}d): {n_delisted}")

    # Sim window 摘要
    sim_start = pd.Timestamp("2022-06-01")
    sim_end = pd.Timestamp("2024-12-31")
    sim_delisted = bounds[
        (~bounds["is_active_at_end"])
        & (bounds["last_date"] >= sim_start)
        & (bounds["last_date"] <= sim_end)
    ]
    print(f"  sim 期間 (2022-06 → 2024-12) implicit 下市檔數：{len(sim_delisted)}")

    print("\n=== 下一步 ===")
    print("  1. simulate：python -m pipelines.simulate_recent --data-source tej --start 2022-06-01 --end 2022-08-31 --strategy scheduled")
    print("  2. predict ：python -m pipelines.predict_next_day --data-source tej")


if __name__ == "__main__":
    main()
