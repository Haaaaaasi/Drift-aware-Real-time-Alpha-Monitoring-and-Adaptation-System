"""
Alpha feature parquet cache — read/write/incremental update.

Layout:
  * TEJ 正式研究：data/alpha_cache/wq101_alphas.parquet
  * yfinance demo：data/alpha_cache/wq101_alphas_csv.parquet
Schema: (security_id: str, tradetime: datetime64, alpha_id: str, alpha_value: float64)

Incremental update strategy
----------------------------
1. Read existing cache → find max(tradetime) T_last.
2. Filter bars to dates > T_last, but extend the window left by `lookback_days`
   so time-series rolling operators have enough warm-up data.
3. Compute alphas for the extended window.
4. Keep only rows where tradetime > T_last (discard the warm-up overlap).
5. Append to existing cache and write back.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.alpha_engine.wq101_python import compute_wq101_alphas
from src.common.logging import get_logger
from src.config.data_sources import infer_data_source_from_path

logger = get_logger("alpha_cache")

# 2026-05 起正式研究預設為 TEJ；舊檔名沿用給 TEJ cache，避免重算 1GB+ parquet。
TEJ_CACHE_PATH = Path("data/alpha_cache/wq101_alphas.parquet")
CSV_CACHE_PATH = Path("data/alpha_cache/wq101_alphas_csv.parquet")
CACHE_PATH = TEJ_CACHE_PATH


def cache_path_for_data_source(data_source: str) -> Path:
    """Return the source-specific WQ101 cache path.

    yfinance 與 TEJ 有大量 security_id 重疊；若共用同一份 cache，pipeline 會把
    來源 A 的 alpha 誤用到來源 B。正式研究預設 TEJ，csv/yfinance 只保留 demo
    專用 cache。
    """
    if data_source == "csv":
        return CSV_CACHE_PATH
    return TEJ_CACHE_PATH


def cache_path_for_data_path(path: str | Path) -> Path:
    """Infer cache path from a user-supplied OHLCV path."""
    if infer_data_source_from_path(path) == "tej":
        return TEJ_CACHE_PATH
    return CSV_CACHE_PATH


def _manifest_path(path: str | Path) -> Path:
    return Path(f"{Path(path)}.manifest.json")


def _infer_data_source_from_cache_path(path: str | Path) -> str:
    path = Path(path)
    if path.name == CSV_CACHE_PATH.name:
        return "csv"
    if path.name == TEJ_CACHE_PATH.name:
        return "tej"
    return "custom"


def read_cache_manifest(path: str | Path) -> dict | None:
    """Read cache sidecar manifest, if present."""
    manifest_path = _manifest_path(path)
    if not manifest_path.exists():
        return None
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def write_cache_manifest(
    path: str | Path,
    *,
    data_source: str,
    rows: int,
    n_securities: int,
    n_alphas: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    """Write the source manifest next to the parquet cache."""
    manifest = {
        "schema_version": 1,
        "data_source": data_source,
        "alpha_engine": "python_wq101",
        "rows": int(rows),
        "n_securities": int(n_securities),
        "n_alphas": int(n_alphas),
        "start": str(pd.Timestamp(start).date()),
        "end": str(pd.Timestamp(end).date()),
    }
    manifest_path = _manifest_path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _validate_cache_manifest(path: Path, expected_data_source: str | None) -> None:
    if expected_data_source is None:
        return
    manifest = read_cache_manifest(path)
    if manifest is None:
        raise RuntimeError(
            f"alpha cache 缺少來源 manifest：{_manifest_path(path)}。"
            "為避免 yfinance/TEJ cache 混用，請刪除 cache 後重算，或先用已驗證的 "
            "TEJ cache 產生 manifest。"
        )
    actual = manifest.get("data_source")
    if actual != expected_data_source:
        raise RuntimeError(
            f"alpha cache 來源不符：{path} manifest={actual!r}, "
            f"expected={expected_data_source!r}。請勿混用 yfinance 與 TEJ cache。"
        )


def read_cache(
    path: Path = CACHE_PATH,
    *,
    expected_data_source: str | None = None,
) -> pd.DataFrame | None:
    """Return cached alpha panel, or None if the file does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    _validate_cache_manifest(path, expected_data_source)
    df = pd.read_parquet(path)
    df["tradetime"] = pd.to_datetime(df["tradetime"])
    df["security_id"] = df["security_id"].astype(str)
    logger.info("cache_read", path=str(path), rows=len(df))
    return df


def write_cache(
    df: pd.DataFrame,
    path: Path = CACHE_PATH,
    *,
    data_source: str | None = None,
) -> None:
    """Persist alpha panel to parquet (snappy compression)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["tradetime"] = pd.to_datetime(df["tradetime"])
    df["security_id"] = df["security_id"].astype(str)
    df.to_parquet(path, index=False, compression="snappy")
    source = data_source or _infer_data_source_from_cache_path(path)
    write_cache_manifest(
        path,
        data_source=source,
        rows=len(df),
        n_securities=df["security_id"].nunique(),
        n_alphas=df["alpha_id"].nunique(),
        start=df["tradetime"].min(),
        end=df["tradetime"].max(),
    )
    logger.info("cache_written", path=str(path), rows=len(df))


def compute_with_cache(
    bars: pd.DataFrame,
    alpha_ids: list[str] | None = None,
    cache_path: Path = CACHE_PATH,
    lookback_days: int = 252,
    force_recompute: bool = False,
    data_source: str | None = None,
) -> pd.DataFrame:
    """Return alpha panel, using and updating the parquet cache.

    If no cache exists (or force_recompute=True), computes the full universe
    and writes the cache.  Otherwise computes only the new dates (plus a
    lookback buffer for ts windows), appends to the cache, and writes back.

    The returned DataFrame is filtered to `alpha_ids` if provided.
    """
    cache_path = Path(cache_path)
    expected_data_source = data_source or _infer_data_source_from_cache_path(cache_path)
    existing = None if force_recompute else read_cache(
        cache_path,
        expected_data_source=expected_data_source,
    )

    bars = bars.copy()
    bars["tradetime"] = pd.to_datetime(bars["tradetime"])

    # Universe consistency check：若 cache 與 bars 的 security_id 完全不交集（例如
    # production cache vs synthetic tests），cache 對本次呼叫不適用，直接重算且不
    # 寫回（避免污染 production cache）。
    if existing is not None:
        bar_sids = set(bars["security_id"].astype(str).unique())
        cache_sids = set(existing["security_id"].astype(str).unique())
        if bar_sids and not (bar_sids & cache_sids):
            logger.info(
                "cache_universe_mismatch_recomputing",
                n_bar_sids=len(bar_sids),
                n_cache_sids=len(cache_sids),
            )
            fresh = compute_wq101_alphas(bars, alpha_ids=None)
            if alpha_ids is not None:
                fresh = fresh[fresh["alpha_id"].isin(alpha_ids)].reset_index(drop=True)
            return fresh

    if existing is None:
        logger.info("cache_cold_start", force=force_recompute)
        fresh = compute_wq101_alphas(bars, alpha_ids=None)
        write_cache(fresh, cache_path, data_source=expected_data_source)
        result = fresh
    else:
        last_cached = existing["tradetime"].max()
        bar_dates = bars["tradetime"].drop_duplicates().sort_values()
        new_dates = bar_dates[bar_dates > last_cached]

        if new_dates.empty:
            logger.info("cache_up_to_date", last_cached=str(last_cached.date()))
            result = existing
        else:
            # Include lookback buffer so rolling operators warm up correctly
            lookback_start = new_dates.iloc[0] - pd.Timedelta(days=lookback_days)
            bars_slice = bars[bars["tradetime"] >= lookback_start]

            logger.info(
                "cache_incremental",
                new_dates=len(new_dates),
                lookback_start=str(lookback_start.date()),
            )
            incremental = compute_wq101_alphas(bars_slice, alpha_ids=None)
            # Keep only truly new rows (discard the warm-up overlap)
            incremental = incremental[incremental["tradetime"] > last_cached]

            updated = pd.concat([existing, incremental], ignore_index=True)
            updated = updated.drop_duplicates(
                subset=["security_id", "tradetime", "alpha_id"], keep="last"
            )
            write_cache(updated, cache_path, data_source=expected_data_source)
            result = updated

    # 只在 result 實際超出 bars 日期範圍時才 filter，避免 104M-row cache 觸發
    # 不必要的 pandas block consolidate / object-dtype copy 而 OOM
    bar_max_date = bars["tradetime"].max()
    bar_min_date = bars["tradetime"].min()
    result_min = result["tradetime"].min()
    result_max = result["tradetime"].max()
    if result_min < bar_min_date or result_max > bar_max_date:
        result = result[
            (result["tradetime"] >= bar_min_date) & (result["tradetime"] <= bar_max_date)
        ].reset_index(drop=True)

    if alpha_ids is not None:
        result = result[result["alpha_id"].isin(alpha_ids)].reset_index(drop=True)

    return result
