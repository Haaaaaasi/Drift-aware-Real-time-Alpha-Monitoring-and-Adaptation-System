"""證交所官方端點每日同步（取代 TEJ 每日 CSV 追加）。

資料層規則（2026-09-07 起）：

- ``data/tw_stocks_tej.parquet`` 在 ``ANCHOR_DATE``（2026-05-29）以前是 TEJ 還原股價，
  單一還原基準，不再更動。
- ``ANCHOR_DATE`` 之後的列一律由 ``data/twse_raw_bars.parquet``（證交所原始價）乘上
  向前累積還原倍率（``data/corporate_actions.parquet``）重新推導，每次同步整段重寫，
  晚到的除權息 / 減資事件因此會自我修復。若推導結果與既有列不同，會把 alpha cache
  截斷到第一個變動日之前，讓下一次 ``compute_with_cache`` 走增量路徑重算。
- 倍率定義：事件日 ``factor = 參考價 / 前收盤價``；t 日向前倍率
  ``F_t = Π_{ANCHOR < ex_date <= t} 1 / factor``。與 TEJ 還原法一致
  （2026-09-07 以 291 檔驗證相對誤差 < 5e-6；減資以 TWTAUU 參考價驗證）。
- 原始價另存，live 下單數量以原始價計算（``raw_close_on``），因為向前還原價會逐年偏離實際成交價。
- 盤後表沒有的事件（面額變更、換股合併等）放 ``data/corporate_actions_manual.csv``，
  欄位同 ``EVENT_COLUMNS``，``source=manual``；例：6949 沛爾生技 2026-09-07 面額 10→0.5，factor=0.05。
- 品質閘門：單日 |報酬| 超過漲跌幅且前一根 bar 到當天之間沒有任何事件，該列不進 bars
  （raw 仍保留），並列入 warnings。這會讓漏掉事件的股票暫時退出 universe，直到補上事件。

端點（公開、免金鑰；皆為舊式 rwd 介面，非 OpenAPI 合約，parser 以欄位名稱定位表格）：

- 每日全市場行情：``afterTrading/MI_INDEX?date=YYYYMMDD&type=ALLBUT0999``
  （當天晚間可取得；非交易日或尚未發布回「沒有符合條件的資料」）
- 除權息計算結果表：``exRight/TWT49U?startDate&endDate``
- 減資恢復買賣參考價：``reducation/TWTAUU?startDate&endDate``
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
import time
from typing import Callable
import urllib.request
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.alpha_engine.alpha_cache import TEJ_CACHE_PATH, truncate_cache_after
from src.common.logging import get_logger
from src.ingestion.tej_daily_append import (
    DEFAULT_BACKUP_DIR,
    DEFAULT_TEJ_OUTPUT,
    DEFAULT_UNIVERSE_OUTPUT,
    REQUIRED_COLUMNS,
    _backup_existing_files,
    _load_existing_bars,
    _normalize_bars,
    _write_parquet_atomic,
    build_universe_bounds,
)

logger = get_logger("twse_daily")

ANCHOR_DATE = pd.Timestamp("2026-05-29")
RAW_PATH = Path("data/twse_raw_bars.parquet")
EVENTS_PATH = Path("data/corporate_actions.parquet")
MANUAL_EVENTS_PATH = Path("data/corporate_actions_manual.csv")
SYNC_STATE_PATH = Path("data/twse_sync_state.json")
TAIPEI = ZoneInfo("Asia/Taipei")
RAW_COLUMNS = REQUIRED_COLUMNS + ["amount"]
EVENT_COLUMNS = ["security_id", "ex_date", "factor", "kind", "prev_close", "ref_price", "source"]
DAILY_LIMIT = 0.105  # 台股漲跌幅 10%，留 tick 誤差
EVENT_LOOKBACK_DAYS = 45  # 每次同步回看的事件視窗，補晚公告
_BASE = "https://www.twse.com.tw/rwd/zh"
_UA = {"User-Agent": "Mozilla/5.0"}

FetchDaily = Callable[[pd.Timestamp], "pd.DataFrame | None"]
FetchEvents = Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame]


# ---------------------------------------------------------------------------
# parsing / fetching
# ---------------------------------------------------------------------------
def _get_json(url: str, retries: int = 3, pause: float = 3.0) -> dict:
    err: Exception | None = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=40) as resp:
                return json.loads(resp.read().decode("utf-8-sig"))
        except Exception as exc:  # noqa: BLE001 - 網路錯誤一律重試
            err = exc
            time.sleep(pause)
    raise RuntimeError(f"TWSE request failed after {retries} tries: {url}: {err}")


def roc_to_timestamp(value: object) -> pd.Timestamp:
    """民國日期（``115/09/04``、``1150904``、``115年09月04日``）→ Timestamp。"""
    digits = re.sub(r"\D", "", str(value))
    return pd.Timestamp(int(digits[:-4]) + 1911, int(digits[-4:-2]), int(digits[-2:]))


def _num(value: object) -> float:
    text = str(value).replace(",", "").strip()
    if text in {"", "--", "-", "X", "nan", "None"}:
        return float("nan")
    return float(text)


def is_common_stock(code: object) -> bool:
    """與 TEJ 歷史一致：4 碼純數字，排除 00xx ETF；91xx TDR 保留。"""
    code = str(code).strip()
    return bool(re.fullmatch(r"\d{4}", code)) and not code.startswith("00")


def fetch_daily_all(day: pd.Timestamp) -> pd.DataFrame | None:
    """MI_INDEX 一日全市場行情；非交易日或尚未發布回 None。"""
    day = pd.Timestamp(day).normalize()
    payload = _get_json(
        f"{_BASE}/afterTrading/MI_INDEX?date={day:%Y%m%d}&type=ALLBUT0999&response=json"
    )
    table = next(
        (
            t
            for t in payload.get("tables") or []
            if "證券代號" in (t.get("fields") or []) and "收盤價" in t["fields"]
        ),
        None,
    )
    if table is None:
        return None
    df = pd.DataFrame(table["data"], columns=table["fields"])
    out = pd.DataFrame(
        {
            "security_id": df["證券代號"].astype(str).str.strip(),
            "datetime": day,
            "open": df["開盤價"].map(_num),
            "high": df["最高價"].map(_num),
            "low": df["最低價"].map(_num),
            "close": df["收盤價"].map(_num),
            "volume": df["成交股數"].map(_num),
            "amount": df["成交金額"].map(_num),
        }
    )
    out = out[out["security_id"].map(is_common_stock) & out["close"].notna()].copy()
    out["volume"] = out["volume"].fillna(0).round().astype("int64")
    return out[RAW_COLUMNS].reset_index(drop=True)


def fetch_ex_rights(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """除權息計算結果表 → ``factor = 除權息參考價 / 除權息前收盤價``。"""
    payload = _get_json(
        f"{_BASE}/exRight/TWT49U?startDate={pd.Timestamp(start):%Y%m%d}"
        f"&endDate={pd.Timestamp(end):%Y%m%d}&response=json"
    )
    if payload.get("stat") != "OK" or not payload.get("data"):
        return pd.DataFrame(columns=EVENT_COLUMNS)
    df = pd.DataFrame(payload["data"], columns=payload["fields"])
    out = pd.DataFrame(
        {
            "security_id": df["股票代號"].astype(str).str.strip(),
            "ex_date": df["資料日期"].map(roc_to_timestamp),
            "prev_close": df["除權息前收盤價"].map(_num),
            "ref_price": df["除權息參考價"].map(_num),
            "kind": df["權/息"].astype(str).str.strip(),
            "source": "TWT49U",
        }
    )
    out["factor"] = out["ref_price"] / out["prev_close"]
    return _clean_events(out)


def fetch_capital_reduction(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """減資恢復買賣參考價 → ``factor = 恢復買賣參考價 / 停止買賣前收盤價格``。"""
    payload = _get_json(
        f"{_BASE}/reducation/TWTAUU?startDate={pd.Timestamp(start):%Y%m%d}"
        f"&endDate={pd.Timestamp(end):%Y%m%d}&response=json"
    )
    if payload.get("stat") != "OK" or not payload.get("data"):
        return pd.DataFrame(columns=EVENT_COLUMNS)
    df = pd.DataFrame(payload["data"], columns=payload["fields"])
    out = pd.DataFrame(
        {
            "security_id": df["股票代號"].astype(str).str.strip(),
            "ex_date": df["恢復買賣日期"].map(roc_to_timestamp),
            "prev_close": df["停止買賣前收盤價格"].map(_num),
            "ref_price": df["恢復買賣參考價"].map(_num),
            "kind": "減資",
            "source": "TWTAUU",
        }
    )
    out["factor"] = out["ref_price"] / out["prev_close"]
    return _clean_events(out)


def fetch_all_events(start: pd.Timestamp, end: pd.Timestamp, pause: float = 3.0) -> pd.DataFrame:
    ex_rights = fetch_ex_rights(start, end)
    time.sleep(pause)
    reductions = fetch_capital_reduction(start, end)
    return pd.concat([ex_rights, reductions], ignore_index=True)


def _clean_events(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    df = df[df["security_id"].map(is_common_stock)].copy()
    df = df[np.isfinite(df["factor"]) & (df["factor"] > 0)]
    df["ex_date"] = pd.to_datetime(df["ex_date"]).dt.normalize()
    return df[EVENT_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# adjustment
# ---------------------------------------------------------------------------
def forward_factor(
    events: pd.DataFrame,
    keys: pd.DataFrame,
    anchor: pd.Timestamp = ANCHOR_DATE,
) -> pd.Series:
    """對每個 ``(security_id, datetime)`` 回傳 ``F_t = Π_{anchor < ex_date <= t} 1/factor``。"""
    keys = keys[["security_id", "datetime"]].copy()
    keys["datetime"] = pd.to_datetime(keys["datetime"])
    if events.empty:
        return pd.Series(1.0, index=keys.index)
    ev = events.copy()
    ev["ex_date"] = pd.to_datetime(ev["ex_date"])
    ev = ev[ev["ex_date"] > pd.Timestamp(anchor)]
    if ev.empty:
        return pd.Series(1.0, index=keys.index)
    # 同一檔同一天多筆事件先合併，避免 merge_asof 在 tie 上取到不完整的累積值
    ev = ev.groupby(["security_id", "ex_date"], as_index=False)["factor"].prod()
    ev = ev.sort_values(["security_id", "ex_date"])
    ev["cum"] = ev.groupby("security_id")["factor"].transform(lambda f: (1.0 / f).cumprod())
    left = keys.sort_values("datetime", kind="mergesort")
    merged = pd.merge_asof(
        left,
        ev[["security_id", "ex_date", "cum"]].sort_values("ex_date", kind="mergesort"),
        left_on="datetime",
        right_on="ex_date",
        by="security_id",
        direction="backward",
    )
    return merged["cum"].fillna(1.0).set_axis(left.index).reindex(keys.index)


def daily_limit_gate(
    raw: pd.DataFrame,
    events: pd.DataFrame,
    anchor_close: pd.Series | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """剔除「單日 |報酬| 超過漲跌幅、且前一根 bar 到當天之間沒有事件」的原始列。

    前收盤用最後一根被接受的 bar（第一根用 anchor 當天的 TEJ 列，anchor 為原始價），
    所以漏掉事件的股票會一直被擋到事件補上為止，而不是帶著假跳空進 alpha。
    """
    raw = raw.sort_values(["security_id", "datetime"]).reset_index(drop=True)
    ev_dates: dict[str, np.ndarray] = {}
    if not events.empty:
        e = events.copy()
        e["ex_date"] = pd.to_datetime(e["ex_date"])
        ev_dates = {sid: g["ex_date"].sort_values().to_numpy() for sid, g in e.groupby("security_id")}
    prev_close = anchor_close.astype(float).to_dict() if anchor_close is not None else {}
    keep = np.ones(len(raw), dtype=bool)
    warnings: list[str] = []
    sids = raw["security_id"].to_numpy()
    days = raw["datetime"].to_numpy()
    closes = raw["close"].to_numpy(dtype=float)
    prev_day: dict[str, np.datetime64] = {}
    for i in range(len(raw)):
        sid, day, close = sids[i], days[i], closes[i]
        p = prev_close.get(sid)
        if p is not None and p > 0 and abs(close / p - 1.0) > DAILY_LIMIT:
            ex = ev_dates.get(sid)
            lo = prev_day.get(sid, np.datetime64("1900-01-01"))
            has_event = ex is not None and bool(((ex > lo) & (ex <= day)).any())
            if not has_event:
                keep[i] = False
                warnings.append(
                    f"{sid} {pd.Timestamp(day).date()} close {close} vs prev {p}: "
                    f"|ret| > {DAILY_LIMIT:.1%} without event -> dropped"
                )
                continue
        prev_close[sid] = close
        prev_day[sid] = day
    return raw[keep].reset_index(drop=True), warnings


def rebuild_adjusted(
    raw: pd.DataFrame,
    events: pd.DataFrame,
    anchor: pd.Timestamp = ANCHOR_DATE,
    anchor_close: pd.Series | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """原始價 → 品質閘門 → × 向前倍率 → 與 TEJ 歷史同基準的 bars（只含 anchor 之後）。"""
    raw = raw.copy()
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw = raw[raw["datetime"] > pd.Timestamp(anchor)].reset_index(drop=True)
    if raw.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS), []
    raw, warnings = daily_limit_gate(raw, events, anchor_close)
    factor = forward_factor(events, raw, anchor)
    for col in ["open", "high", "low", "close"]:
        raw[col] = raw[col] * factor
    return _normalize_bars(raw[REQUIRED_COLUMNS]), warnings


def raw_close_on(day: pd.Timestamp, raw_path: str | Path = RAW_PATH) -> pd.Series | None:
    """回傳某日各檔原始收盤價（下單數量用）；沒有資料回 None。"""
    path = Path(raw_path)
    if not path.exists():
        return None
    day = pd.Timestamp(day).normalize()
    df = pd.read_parquet(path, columns=["security_id", "datetime", "close"], filters=[("datetime", "==", day)])
    if df.empty:
        return None
    return df.set_index("security_id")["close"]


# ---------------------------------------------------------------------------
# sync
# ---------------------------------------------------------------------------
def _read_parquet_or_empty(path: Path, columns: list[str]) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=columns)
    return pd.read_parquet(path)[columns]


def _load_manual_events(path: Path) -> pd.DataFrame:
    """盤後表沒有的事件（面額變更、換股合併…）由人工維護的 CSV 補上。"""
    if not Path(path).exists():
        return pd.DataFrame(columns=EVENT_COLUMNS)
    df = pd.read_csv(path, dtype={"security_id": str})
    df["ex_date"] = pd.to_datetime(df["ex_date"])
    df["source"] = "manual"
    for col in ("kind", "prev_close", "ref_price"):
        if col not in df.columns:
            df[col] = np.nan
    return _clean_events(df)


def _first_changed_date(old: pd.DataFrame, new: pd.DataFrame) -> pd.Timestamp | None:
    """既有 post-anchor 段與新推導段第一個不同的日期（純追加的新日期不算變動）。"""
    if old.empty:
        return None
    key = ["security_id", "datetime"]
    m = old.merge(new, on=key, how="outer", suffixes=("_old", "_new"), indicator=True)
    diff = (m["_merge"] != "both").to_numpy()
    for col in ["open", "high", "low", "close", "volume"]:
        a = m[f"{col}_old"].to_numpy(dtype=float)
        b = m[f"{col}_new"].to_numpy(dtype=float)
        diff |= ~np.isclose(a, b, rtol=1e-9, atol=1e-9)
    changed = m.loc[diff & (m["datetime"] <= old["datetime"].max()), "datetime"]
    return pd.Timestamp(changed.min()).normalize() if not changed.empty else None


def _prune_backups(backup_dir: Path, keep: int = 5) -> None:
    for stem in ("tw_stocks_tej", "tw_stocks_tej_universe"):
        files = sorted(backup_dir.glob(f"{stem}_*.parquet"), key=lambda p: p.stat().st_mtime)
        for old in files[:-keep]:
            old.unlink()


def sync_twse(
    *,
    as_of: str | pd.Timestamp | None = None,
    raw_path: str | Path = RAW_PATH,
    events_path: str | Path = EVENTS_PATH,
    manual_events_path: str | Path = MANUAL_EVENTS_PATH,
    bars_path: str | Path = DEFAULT_TEJ_OUTPUT,
    universe_path: str | Path = DEFAULT_UNIVERSE_OUTPUT,
    backup_dir: str | Path | None = DEFAULT_BACKUP_DIR,
    cache_path: str | Path = TEJ_CACHE_PATH,
    state_path: str | Path = SYNC_STATE_PATH,
    anchor: pd.Timestamp = ANCHOR_DATE,
    pause: float = 3.0,
    fetch_daily: FetchDaily = fetch_daily_all,
    fetch_events: FetchEvents | None = None,
    dry_run: bool = False,
) -> dict:
    """抓缺的交易日 + 事件，重寫 anchor 之後的 bars，必要時截斷 alpha cache。"""
    as_of_ts = (
        pd.Timestamp(as_of).normalize()
        if as_of is not None
        else pd.Timestamp(datetime.now(TAIPEI).date())
    )
    anchor = pd.Timestamp(anchor).normalize()
    raw_path, events_path, bars_path, universe_path = map(Path, (raw_path, events_path, bars_path, universe_path))
    fetch_events = fetch_events or (lambda s, e: fetch_all_events(s, e, pause=pause))

    raw = _read_parquet_or_empty(raw_path, RAW_COLUMNS)
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    events = _read_parquet_or_empty(events_path, EVENT_COLUMNS)

    # 1. 事件（回看視窗補晚公告；同 key 以新抓的為準）
    ev_start = anchor + pd.Timedelta(days=1)
    if not events.empty:
        ev_start = max(ev_start, pd.to_datetime(events["ex_date"]).max() - pd.Timedelta(days=EVENT_LOOKBACK_DAYS))
    new_events = fetch_events(ev_start, as_of_ts)
    manual = _load_manual_events(Path(manual_events_path))
    events = (
        pd.concat([f for f in (events, new_events, manual) if not f.empty] or [events], ignore_index=True)
        .drop_duplicates(subset=["security_id", "ex_date", "source"], keep="last")
        .sort_values(["ex_date", "security_id"])
        .reset_index(drop=True)
    )

    # 2. 缺的交易日
    start = (raw["datetime"].max() if not raw.empty else anchor) + pd.Timedelta(days=1)
    fetched: list[pd.DataFrame] = []
    skipped: list[str] = []
    for day in pd.bdate_range(start, as_of_ts):
        df = fetch_daily(day)
        time.sleep(pause)
        if df is None or df.empty:
            skipped.append(day.date().isoformat())
            continue
        fetched.append(df)
        logger.info("twse_daily_fetched", date=day.date().isoformat(), rows=len(df))

    # 3. 合併原始價
    existing = _load_existing_bars(bars_path)
    raw_all = pd.concat([f for f in (raw, *fetched) if not f.empty] or [raw], ignore_index=True)
    raw_all = (
        raw_all.drop_duplicates(subset=["security_id", "datetime"], keep="last")
        .sort_values(["security_id", "datetime"])
        .reset_index(drop=True)
    )

    # 4. anchor 之後整段重新推導（含品質閘門），偵測與既有列的差異
    anchor_close = existing[existing["datetime"] == anchor].set_index("security_id")["close"]
    adjusted, warnings = rebuild_adjusted(raw_all, events, anchor, anchor_close)
    old_post = existing[existing["datetime"] > anchor]
    first_changed = _first_changed_date(old_post, adjusted)
    merged = pd.concat([existing[existing["datetime"] <= anchor], adjusted], ignore_index=True)
    merged = _normalize_bars(merged).sort_values(["security_id", "datetime"]).reset_index(drop=True)

    result = {
        "as_of": as_of_ts.date().isoformat(),
        "anchor": anchor.date().isoformat(),
        "fetched_dates": [d["datetime"].iloc[0].date().isoformat() for d in fetched],
        "skipped_dates": skipped,
        "raw_rows": int(len(raw_all)),
        "events_rows": int(len(events)),
        "new_events": int(len(new_events)),
        "bars_rows": int(len(merged)),
        "output_max_date": merged["datetime"].max().date().isoformat() if not merged.empty else None,
        "first_changed_date": first_changed.date().isoformat() if first_changed is not None else None,
        "cache_truncation": None,
        "warnings": warnings,
        "dry_run": dry_run,
        "synced_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
    }
    if dry_run:
        return result

    if fetched:
        _write_parquet_atomic(raw_all, raw_path)
    _write_parquet_atomic(events, events_path)
    if fetched or first_changed is not None:
        if backup_dir is not None:
            _backup_existing_files([bars_path, universe_path], backup_dir=Path(backup_dir))
            _prune_backups(Path(backup_dir))
        _write_parquet_atomic(merged, bars_path)
        _write_parquet_atomic(build_universe_bounds(merged), universe_path)
    if first_changed is not None:
        # 既有列變了 → cache 從變動日起失效，下次 compute_with_cache 增量重算
        result["cache_truncation"] = truncate_cache_after(cache_path, first_changed - pd.Timedelta(days=1))
    Path(state_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("twse_sync_done", **{k: v for k, v in result.items() if k not in {"warnings", "cache_truncation"}})
    return result
