"""抓期交所臺指期（TX）夜盤每日漲跌，預設從 2020-01 起，輸出 CSV。

資料源：期交所「期貨每日交易行情下載」``futDataDown``。回傳 Big5 CSV，單次查詢最多一個月；
沒有交易日的區間回傳只有表頭的 CSV，回傳 HTML 代表查詢失敗。

輸出檔已存在時只做增量：從檔內最後一個交易日重抓到今天，重疊的那天以新抓的為準。
每日排程 ``scripts/run_daily_sync.cmd`` 就是這樣呼叫；要整段重抓，刪掉輸出檔再跑。

輸出欄位：

- ``trade_date``：期交所的交易日期。期交所把夜盤算進下一個交易日。
- ``session_start``：這場夜盤開始的日期（當天 15:00 開盤、隔天 05:00 收盤），
  即 ``trade_date`` 的前一個交易日。當特徵用時，以 ``session_start`` 隔天 05:00 為可用時間。
- ``contract_month``：近月合約（到期月份最早的月契約）。
- ``change`` / ``change_pct``：期交所公布的漲跌，基準是 ``session_start`` 當天日盤的結算價。
- ``volume``：近月合約這場夜盤的成交口數（單邊計算，含價差單與鉅額交易成交的口數）。

範例：

    python scripts/fetch_taifex_night.py
    python scripts/fetch_taifex_night.py --start 2024-01-01 --out data/taifex_tx_night.csv
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import time
import urllib.parse
import urllib.request

import pandas as pd

URL = "https://www.taifex.com.tw/cht/3/futDataDown"
COLUMNS = {
    "交易日期": "trade_date",
    "到期月份(週別)": "contract_month",
    "開盤價": "open",
    "最高價": "high",
    "最低價": "low",
    "收盤價": "close",
    "漲跌價": "change",
    "漲跌%": "change_pct",
    "成交量": "volume",
    "交易時段": "session",
}
NUMERIC = ["open", "high", "low", "close", "change", "change_pct", "volume"]
OUTPUT_COLUMNS = ["trade_date", "session_start", "contract_month", *NUMERIC]
LOOKBACK_DAYS = 20  # 多抓一段，讓第一筆夜盤也找得到前一個交易日（春節休市最長約 10 天）


def fetch_range(start: pd.Timestamp, end: pd.Timestamp, retries: int = 3, pause: float = 3.0) -> str:
    """查一段不超過一個月的區間，回傳 CSV 文字。"""
    body = urllib.parse.urlencode(
        {
            "down_type": "1",
            "commodity_id": "TX",
            "queryStartDate": f"{start:%Y/%m/%d}",
            "queryEndDate": f"{end:%Y/%m/%d}",
        }
    ).encode()
    err: Exception | None = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(URL, data=body, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                text = resp.read().decode("cp950")
            if text.startswith("交易日期"):
                return text
            err = RuntimeError("回傳的不是 CSV")
        except Exception as exc:  # noqa: BLE001 - 網路錯誤一律重試
            err = exc
        time.sleep(pause)
    raise RuntimeError(f"期交所 futDataDown {start:%Y-%m-%d}~{end:%Y-%m-%d} 失敗：{err}")


def month_chunks(start: pd.Timestamp, end: pd.Timestamp):
    """把區間切成不跨月的片段（期交所單次最多查一個月）。"""
    while start <= end:
        stop = min(start + pd.offsets.MonthEnd(0), end)
        yield start, stop
        start = stop + pd.Timedelta(days=1)


def parse(text: str) -> pd.DataFrame:
    # 資料列結尾比表頭多一個逗號，要 index_col=False 才不會整列錯位
    df = pd.read_csv(io.StringIO(text), dtype=str, index_col=False)
    df = df[list(COLUMNS)].rename(columns=COLUMNS)
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y/%m/%d")
    df["contract_month"] = df["contract_month"].str.strip()
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col].str.rstrip("%"), errors="coerce")  # "-" 代表無成交
    return df


def night_moves(df: pd.DataFrame) -> pd.DataFrame:
    """每個交易日一列：近月合約的夜盤行情，加上夜盤實際開始的日期。"""
    monthly = df[df["contract_month"].str.fullmatch(r"\d{6}", na=False)]  # 排除價差與週契約
    day_dates = pd.DatetimeIndex(
        monthly.loc[monthly["session"] == "一般", "trade_date"].drop_duplicates().sort_values()
    )
    night = (
        monthly[monthly["session"] == "盤後"]
        .sort_values(["trade_date", "contract_month"])
        .drop_duplicates("trade_date")
    )
    prev = day_dates.searchsorted(night["trade_date"]) - 1
    night = night.assign(session_start=[day_dates[i] if i >= 0 else pd.NaT for i in prev])
    return night[OUTPUT_COLUMNS].reset_index(drop=True)


def fetch_night_moves(start: pd.Timestamp, end: pd.Timestamp, pause: float = 1.0) -> pd.DataFrame:
    frames = []
    for chunk_start, chunk_end in month_chunks(start - pd.Timedelta(days=LOOKBACK_DAYS), end):
        frames.append(parse(fetch_range(chunk_start, chunk_end)))
        time.sleep(pause)
    moves = night_moves(pd.concat(frames, ignore_index=True))
    return moves[moves["trade_date"] >= start].reset_index(drop=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2020-01-01", help="起始交易日期（輸出檔不存在時才用）")
    parser.add_argument("--end", default=None, help="結束日期，預設今天")
    parser.add_argument("--out", default="data/taifex_tx_night.csv", help="輸出 CSV 路徑")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out = Path(args.out)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
    old = None
    if out.exists():
        old = pd.read_csv(out, parse_dates=["trade_date", "session_start"], dtype={"contract_month": str})
    start = pd.Timestamp(args.start) if old is None else old["trade_date"].max()
    # old 為 None 時 concat 會略過；重疊的那天保留新抓的
    moves = pd.concat([old, fetch_night_moves(start, end)], ignore_index=True)
    moves = moves.drop_duplicates("trade_date", keep="last")
    out.parent.mkdir(parents=True, exist_ok=True)
    moves.to_csv(out, index=False)
    added = len(moves) - (0 if old is None else len(old))
    first, last = moves["trade_date"].min(), moves["trade_date"].max()
    print(f"新增 {added} 場，共 {len(moves)} 場夜盤（{first:%Y-%m-%d} 到 {last:%Y-%m-%d}），寫入 {out}")


if __name__ == "__main__":
    main()
