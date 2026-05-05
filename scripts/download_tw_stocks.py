"""Download Taiwan stock daily OHLCV data via yfinance.

產生 data/tw_stocks_ohlcv.csv，欄位：
    security_id, datetime, open, high, low, close, volume

使用方式：
    python scripts/download_tw_stocks.py                       # 所有上市股票，抓到今天
    python scripts/download_tw_stocks.py --include-otc         # 上市 + 上櫃
    python scripts/download_tw_stocks.py --append              # 僅補抓 CSV 缺少的最新資料
    python scripts/download_tw_stocks.py --tickers 2330 2317   # 指定標的
    python scripts/download_tw_stocks.py --start 2020-01-01    # 自訂起始日
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

import httpx
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# TWSE / TPEX ISIN 頁面（提供完整掛牌股票清單）
# ---------------------------------------------------------------------------
ISIN_URL_TWSE = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
ISIN_URL_TPEX = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"

BATCH_SIZE = 80  # 每次 yfinance batch 呼叫的 ticker 數（太大容易 timeout）

# 備用清單（當 ISIN API 無法連線時使用）
FALLBACK_TICKERS: list[str] = [
    "2330", "2303", "2454", "2379", "3711", "2408",
    "2317", "2382", "2357", "2308", "2353", "3008",
    "2395", "2409", "2352", "2881", "2882", "2883",
    "2884", "2885", "2886", "2887", "2891", "2892",
    "2412", "4904", "1301", "1303", "1326", "2002",
    "1402", "2912", "2207", "1216", "2327", "2344",
    "2376", "2501", "2474", "2049", "6505", "1101",
    "2105", "0050", "0056",
]


# ---------------------------------------------------------------------------
# ISIN HTML 解析器
# ---------------------------------------------------------------------------

class _ISINParser(HTMLParser):
    """從 TWSE/TPEX ISIN 頁面 HTML 提取 4 碼數字股票代號。"""

    def __init__(self) -> None:
        super().__init__()
        self._in_td = False
        self._buf = ""
        self.codes: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag == "td":
            self._in_td = True
            self._buf = ""

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self._in_td:
            text = self._buf.strip()
            # 格式："2330　台積電"（全形空白 \u3000 分隔代號與名稱）
            code = text.split("\u3000")[0].strip()
            if len(code) >= 4 and code[:4].isdigit():
                self.codes.append(code[:4])
            self._in_td = False

    def handle_data(self, data: str) -> None:
        if self._in_td:
            self._buf += data


def fetch_tickers(url: str, label: str = "") -> list[str]:
    """從 TWSE 或 TPEX ISIN 頁面取得所有股票代號清單。"""
    try:
        resp = httpx.get(url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        html = resp.content.decode("big5", errors="replace")
        parser = _ISINParser()
        parser.feed(html)
        codes = sorted(set(parser.codes))
        print(f"  {label}: 取得 {len(codes)} 檔")
        return codes
    except Exception as exc:
        print(f"  [WARN] 無法取得 {label} 清單: {exc}")
        return []


# ---------------------------------------------------------------------------
# yfinance 批次下載
# ---------------------------------------------------------------------------

def _download_batch(
    yf_tickers: list[str],
    start: str,
    end: str,
    retry: int = 3,
) -> pd.DataFrame:
    """下載一批 yfinance tickers，返回 long-format DataFrame。"""
    raw: pd.DataFrame = pd.DataFrame()

    for attempt in range(1, retry + 1):
        try:
            raw = yf.download(
                yf_tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
            )
            break
        except Exception as exc:
            if attempt < retry:
                print(f"  [RETRY {attempt}] {exc}")
                time.sleep(3)
            else:
                print(f"  [FAIL] {exc}")
                return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []

    # 單一 ticker → 扁平欄位
    if len(yf_tickers) == 1:
        yt = yf_tickers[0]
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.reindex(columns=["Open", "High", "Low", "Close", "Volume"]).copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index.name = "datetime"
        df = df.reset_index()
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
        df["security_id"] = yt.replace(".TWO", "").replace(".TW", "")
        df = df.dropna(subset=["close", "volume"])
        df = df[df["volume"] > 0]
        if not df.empty:
            frames.append(df[["security_id", "datetime", "open", "high", "low", "close", "volume"]])
    else:
        # 多 ticker → MultiIndex(field, ticker)，level-1 是 ticker
        available = set(raw.columns.get_level_values(1))
        for yt in yf_tickers:
            if yt not in available:
                continue
            try:
                sub = raw.xs(yt, axis=1, level=1)[["Open", "High", "Low", "Close", "Volume"]].copy()
            except KeyError:
                continue
            sub.columns = ["open", "high", "low", "close", "volume"]
            sub.index.name = "datetime"
            sub = sub.reset_index()
            sub["datetime"] = pd.to_datetime(sub["datetime"]).dt.tz_localize(None)
            sub["security_id"] = yt.replace(".TWO", "").replace(".TW", "")
            sub = sub.dropna(subset=["close", "volume"])
            sub = sub[sub["volume"] > 0]
            if not sub.empty:
                frames.append(sub[["security_id", "datetime", "open", "high", "low", "close", "volume"]])

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# 主下載流程
# ---------------------------------------------------------------------------

def download(
    tickers: list[str],
    suffixes: list[str],
    start: str,
    end: str,
    output_path: Path,
    retry: int = 3,
    append: bool = False,
    min_rows: int = 10,
) -> None:
    """下載 OHLCV 並寫出（或追加）至 output_path CSV。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    yf_tickers = [f"{t}{s}" for t, s in zip(tickers, suffixes)]

    # Append 模式：讀現有 CSV，找最新日期，只補抓新資料
    existing_df: pd.DataFrame | None = None
    effective_start = start
    if append and output_path.exists():
        existing_df = pd.read_csv(output_path, parse_dates=["datetime"])
        if not existing_df.empty:
            last_date = existing_df["datetime"].max().date()
            effective_start = (pd.Timestamp(last_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"[APPEND] 現有資料最新日期: {last_date}，補抓 {effective_start} → {end}")

    total = len(yf_tickers)
    n_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\n下載 {total} 檔標的，日期: {effective_start} → {end}，共 {n_batches} 批次\n")

    all_frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for i in range(0, total, BATCH_SIZE):
        batch = yf_tickers[i:i + BATCH_SIZE]
        batch_no = i // BATCH_SIZE + 1
        print(f"  [{batch_no}/{n_batches}] {len(batch)} 檔 …", end="", flush=True)
        result = _download_batch(batch, effective_start, end, retry=retry)

        if result.empty:
            print(" 無資料")
            failed.extend(batch)
        else:
            n_sym = result["security_id"].nunique()
            print(f" {len(result):,} rows，{n_sym} 檔有資料")
            all_frames.append(result)

        time.sleep(1.5)  # 避免對 yfinance 請求過快

    # 對失敗批次逐一重試
    if failed:
        print(f"\n對 {len(failed)} 個失敗標的逐一重試 …")
        for yt in failed:
            result = _download_batch([yt], effective_start, end, retry=retry)
            if not result.empty and len(result) >= min_rows:
                all_frames.append(result)
            time.sleep(0.5)

    if not all_frames:
        print("\nERROR: 沒有下載到任何資料。請確認網路或 ticker 代號。")
        sys.exit(1)

    new_df = pd.concat(all_frames, ignore_index=True)

    # 過濾極少資料的標的（可能是停牌或剛上市）
    counts = new_df.groupby("security_id").size()
    valid_ids = counts[counts >= min_rows].index
    new_df = new_df[new_df["security_id"].isin(valid_ids)]

    # 與現有資料合併（Append 模式）
    if existing_df is not None and not existing_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["security_id", "datetime"])
    else:
        combined = new_df

    combined = combined.sort_values(["security_id", "datetime"]).reset_index(drop=True)
    combined.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"輸出檔  : {output_path}")
    print(f"總筆數  : {len(combined):,}")
    print(f"標的數  : {combined['security_id'].nunique()}")
    print(f"資料期間: {combined['datetime'].min().date()} → {combined['datetime'].max().date()}")
    print(f"{'='*60}")
    print("\n下一步:")
    print(f"  python -m pipelines.daily_batch_pipeline --csv {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    today = date.today().isoformat()

    parser = argparse.ArgumentParser(
        description="下載台灣股票 OHLCV 資料（TWSE 上市 / TPEX 上櫃）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="股票代號（不含 .TW），預設：從 TWSE 自動取得所有上市股票",
    )
    parser.add_argument(
        "--include-otc", action="store_true",
        help="同時納入上櫃（TPEX）股票（加 .TWO 後綴）",
    )
    parser.add_argument(
        "--start", default="2022-01-01",
        help="起始日期 YYYY-MM-DD（預設: 2022-01-01）",
    )
    parser.add_argument(
        "--end", default=today,
        help=f"結束日期 YYYY-MM-DD（預設: 今天 {today}）",
    )
    parser.add_argument(
        "--output", default="data/tw_stocks_ohlcv.csv",
        help="輸出 CSV 路徑（預設: data/tw_stocks_ohlcv.csv）",
    )
    parser.add_argument("--retry", type=int, default=3, help="每批次重試次數（預設: 3）")
    parser.add_argument(
        "--append", action="store_true",
        help="增量更新：只補抓現有 CSV 最新日期之後的資料",
    )
    parser.add_argument(
        "--min-rows", type=int, default=10,
        help="單一標的最低有效 row 數，低於此值將被過濾（預設: 10）",
    )
    args = parser.parse_args()

    tickers: list[str] = []
    suffixes: list[str] = []

    if args.tickers:
        tickers = args.tickers
        suffixes = [".TW"] * len(tickers)
    else:
        # 從 TWSE ISIN 頁面動態取得完整清單
        print("正在從 TWSE 取得所有上市股票清單 …")
        twse = fetch_tickers(ISIN_URL_TWSE, label="上市（TWSE）")
        if twse:
            tickers += twse
            suffixes += [".TW"] * len(twse)
        else:
            print("  使用備用清單（FALLBACK）")
            tickers += FALLBACK_TICKERS
            suffixes += [".TW"] * len(FALLBACK_TICKERS)

        if args.include_otc:
            print("正在從 TPEX 取得所有上櫃股票清單 …")
            tpex = fetch_tickers(ISIN_URL_TPEX, label="上櫃（TPEX）")
            if tpex:
                tickers += tpex
                suffixes += [".TWO"] * len(tpex)

    download(
        tickers=tickers,
        suffixes=suffixes,
        start=args.start,
        end=args.end,
        output_path=Path(args.output),
        retry=args.retry,
        append=args.append,
        min_rows=args.min_rows,
    )


if __name__ == "__main__":
    main()
