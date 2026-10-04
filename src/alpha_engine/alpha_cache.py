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
from uuid import uuid4

import numpy as np
import pandas as pd

from src.alpha_engine.wq101_python import compute_wq101_alphas
from src.common.logging import get_logger
from src.config.data_sources import infer_data_source_from_path

logger = get_logger("alpha_cache")

# 2026-05 起正式研究預設為 TEJ；舊檔名沿用給 TEJ cache，避免重算 1GB+ parquet。
TEJ_CACHE_PATH = Path("data/alpha_cache/wq101_alphas.parquet")
CSV_CACHE_PATH = Path("data/alpha_cache/wq101_alphas_csv.parquet")
CACHE_PATH = TEJ_CACHE_PATH
CACHE_COLUMNS = ["security_id", "tradetime", "alpha_id", "alpha_value"]


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


def _normalise_cache_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tradetime"] = pd.to_datetime(df["tradetime"])
    df["security_id"] = df["security_id"].astype(str)
    return df


def _align_to_bar_keys(alpha_panel: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Return only alpha rows whose (security_id, tradetime) exists in bars."""
    if alpha_panel.empty:
        return alpha_panel
    bars_key = bars[["security_id", "tradetime"]].drop_duplicates().copy()
    bars_key["security_id"] = bars_key["security_id"].astype(str)
    bars_key["tradetime"] = pd.to_datetime(bars_key["tradetime"])
    if bars_key.empty:
        return alpha_panel.iloc[0:0].reset_index(drop=True)
    before = len(alpha_panel)

    # 正式 TEJ cache 已超過 1 億列；一次性 merge 會額外建立大型 join indexer。
    # 依 alpha_id 分批 semi-join，只保留一個全域 boolean mask，降低補跑五月資料時的峰值記憶體。
    keep_mask = np.zeros(before, dtype=bool)
    group_positions = alpha_panel.groupby("alpha_id", sort=False).indices
    for positions in group_positions.values():
        chunk_keys = alpha_panel.iloc[positions][["security_id", "tradetime"]].copy()
        chunk_keys["__row_pos"] = positions
        matched = chunk_keys.merge(
            bars_key,
            on=["security_id", "tradetime"],
            how="inner",
            copy=False,
        )
        if not matched.empty:
            keep_mask[matched["__row_pos"].to_numpy(dtype=np.int64, copy=False)] = True

    rows_after = int(keep_mask.sum())
    if rows_after == before:
        return alpha_panel.reset_index(drop=True)

    aligned = alpha_panel.loc[keep_mask].reset_index(drop=True)
    if rows_after != before:
        logger.info(
            "alpha_panel_aligned_to_bars",
            rows_before=before,
            rows_after=rows_after,
        )
    return aligned


def _read_cache_slice(
    path: Path,
    *,
    expected_data_source: str | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    alpha_ids: list[str] | None = None,
) -> pd.DataFrame | None:
    """Read only the requested cache window and alpha ids."""
    path = Path(path)
    if not path.exists():
        return None
    _validate_cache_manifest(path, expected_data_source)

    filters: list[tuple[str, str, object]] = []
    if start is not None:
        filters.append(("tradetime", ">=", pd.Timestamp(start)))
    if end is not None:
        filters.append(("tradetime", "<=", pd.Timestamp(end)))
    if alpha_ids is not None:
        unique_alpha_ids = sorted({str(a) for a in alpha_ids})
        if not unique_alpha_ids:
            return pd.DataFrame(columns=CACHE_COLUMNS)
        filters.append(("alpha_id", "in", unique_alpha_ids))

    df = pd.read_parquet(
        path,
        columns=CACHE_COLUMNS,
        filters=filters or None,
    )
    df = _normalise_cache_frame(df)
    logger.info(
        "cache_slice_read",
        path=str(path),
        rows=len(df),
        start=str(pd.Timestamp(start).date()) if start is not None else None,
        end=str(pd.Timestamp(end).date()) if end is not None else None,
        n_alpha_ids=len(alpha_ids) if alpha_ids is not None else None,
    )
    return df


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
    df = _normalise_cache_frame(pd.read_parquet(path, columns=CACHE_COLUMNS))
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


def _convert_file_cache_to_dataset(path: Path) -> None:
    """將舊單檔 parquet cache 原地轉成 parquet dataset 目錄。"""
    if not path.exists() or path.is_dir():
        return
    if not path.is_file():
        raise RuntimeError(f"Unsupported alpha cache path type: {path}")

    legacy_path = path.with_name(f"{path.stem}.single_file{path.suffix}")
    counter = 1
    while legacy_path.exists():
        legacy_path = path.with_name(
            f"{path.stem}.single_file_{counter}{path.suffix}"
        )
        counter += 1

    path.replace(legacy_path)
    try:
        path.mkdir(parents=True, exist_ok=False)
        legacy_path.replace(path / "part-00000.parquet")
    except Exception:
        if path.exists() and path.is_dir():
            part = path / "part-00000.parquet"
            if part.exists():
                part.replace(legacy_path)
            path.rmdir()
        legacy_path.replace(path)
        raise


def _append_cache_part(
    incremental: pd.DataFrame,
    path: Path,
    *,
    data_source: str,
    manifest: dict,
) -> None:
    """以新增 part 檔追加 cache，避免為了寫回而載入完整 2 億列 cache。"""
    if incremental.empty:
        return

    path = Path(path)
    _convert_file_cache_to_dataset(path)
    path.mkdir(parents=True, exist_ok=True)

    incremental = incremental.copy()
    incremental["tradetime"] = pd.to_datetime(incremental["tradetime"])
    incremental["security_id"] = incremental["security_id"].astype(str)
    incremental = incremental.drop_duplicates(
        subset=["security_id", "tradetime", "alpha_id"], keep="last"
    )

    start = pd.Timestamp(incremental["tradetime"].min()).date().isoformat()
    end = pd.Timestamp(incremental["tradetime"].max()).date().isoformat()
    part_name = f"part-{start}-{end}-{uuid4().hex[:8]}.parquet"
    tmp = path / f".{part_name}.tmp"
    final = path / part_name
    incremental.to_parquet(tmp, index=False, compression="snappy")
    tmp.replace(final)

    write_cache_manifest(
        path,
        data_source=data_source,
        rows=int(manifest.get("rows", 0)) + len(incremental),
        n_securities=max(
            int(manifest.get("n_securities", 0)),
            int(incremental["security_id"].nunique()),
        ),
        n_alphas=max(
            int(manifest.get("n_alphas", 0)),
            int(incremental["alpha_id"].nunique()),
        ),
        start=pd.Timestamp(manifest["start"]),
        end=incremental["tradetime"].max(),
    )
    logger.info(
        "cache_incremental_part_written",
        path=str(path),
        part=str(final.name),
        rows=len(incremental),
    )


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

    bars = bars.copy()
    bars["tradetime"] = pd.to_datetime(bars["tradetime"])
    bar_max_date = bars["tradetime"].max()
    bar_min_date = bars["tradetime"].min()

    existing = None
    if not force_recompute and cache_path.exists():
        manifest = read_cache_manifest(cache_path)
        _validate_cache_manifest(cache_path, expected_data_source)
        if manifest is not None:
            cache_start = pd.Timestamp(manifest["start"])
            cache_end = pd.Timestamp(manifest["end"])
            if cache_start <= bar_min_date and bar_max_date <= cache_end:
                result = _read_cache_slice(
                    cache_path,
                    expected_data_source=expected_data_source,
                    start=bar_min_date,
                    end=bar_max_date,
                    alpha_ids=alpha_ids,
                )
                if result is not None:
                    bar_sids = set(bars["security_id"].astype(str).unique())
                    cache_sids = set(result["security_id"].astype(str).unique())
                    if not bar_sids or (bar_sids & cache_sids):
                        return _align_to_bar_keys(result, bars)
                    logger.info(
                        "cache_universe_mismatch_recomputing",
                        n_bar_sids=len(bar_sids),
                        n_cache_sids=len(cache_sids),
                    )
                    fresh = compute_wq101_alphas(bars, alpha_ids=None)
                    if alpha_ids is not None:
                        fresh = fresh[fresh["alpha_id"].isin(alpha_ids)].reset_index(drop=True)
                    return _align_to_bar_keys(fresh, bars)

            if cache_start <= bar_min_date and cache_end < bar_max_date:
                bar_dates = bars["tradetime"].drop_duplicates().sort_values()
                new_dates = bar_dates[bar_dates > cache_end]
                if not new_dates.empty:
                    lookback_start = new_dates.iloc[0] - pd.Timedelta(days=lookback_days)
                    bars_slice = bars[bars["tradetime"] >= lookback_start]

                    logger.info(
                        "cache_incremental_manifest_path",
                        new_dates=len(new_dates),
                        lookback_start=str(lookback_start.date()),
                        last_cached=str(cache_end.date()),
                    )
                    incremental = compute_wq101_alphas(bars_slice, alpha_ids=None)
                    incremental = incremental[incremental["tradetime"] > cache_end]
                    _append_cache_part(
                        incremental,
                        cache_path,
                        data_source=expected_data_source,
                        manifest=manifest,
                    )

                result = _read_cache_slice(
                    cache_path,
                    expected_data_source=expected_data_source,
                    start=bar_min_date,
                    end=bar_max_date,
                    alpha_ids=alpha_ids,
                )
                if result is not None:
                    return _align_to_bar_keys(result, bars)

        existing = read_cache(
            cache_path,
            expected_data_source=expected_data_source,
        )

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
            return _align_to_bar_keys(fresh, bars)

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
    result_min = result["tradetime"].min()
    result_max = result["tradetime"].max()
    if result_min < bar_min_date or result_max > bar_max_date:
        result = result[
            (result["tradetime"] >= bar_min_date) & (result["tradetime"] <= bar_max_date)
        ].reset_index(drop=True)

    if alpha_ids is not None:
        result = result[result["alpha_id"].isin(alpha_ids)].reset_index(drop=True)

    return _align_to_bar_keys(result, bars)


# ---------------------------------------------------------------------------
# truncation（bars 修正後讓增量路徑重算）
# ---------------------------------------------------------------------------
def _part_time_range(pf) -> tuple[pd.Timestamp, pd.Timestamp]:
    """用 row-group 統計取 part 的 tradetime 範圍；沒有統計就串流掃描，不進 pandas。"""
    import pyarrow.compute as pc

    idx = pf.schema_arrow.get_field_index("tradetime")
    lo = hi = None
    for i in range(pf.metadata.num_row_groups):
        st = pf.metadata.row_group(i).column(idx).statistics
        if st is None or not st.has_min_max:
            lo = hi = None
            break
        lo = st.min if lo is None else min(lo, st.min)
        hi = st.max if hi is None else max(hi, st.max)
    if lo is None:
        for batch in pf.iter_batches(columns=["tradetime"], batch_size=2_000_000):
            mm = pc.min_max(batch.column("tradetime")).as_py()
            lo = mm["min"] if lo is None else min(lo, mm["min"])
            hi = mm["max"] if hi is None else max(hi, mm["max"])
    return pd.Timestamp(lo), pd.Timestamp(hi)


def truncate_cache_after(path: str | Path, keep_through: pd.Timestamp) -> dict:
    """刪除 cache 中 ``tradetime > keep_through`` 的列。

    整個落在 keep_through 之後的 part 直接刪檔；跨界的 part 以 pyarrow 串流過濾重寫，
    不把 2 億列讀進 pandas。單檔 cache 先原地轉成 dataset 目錄。manifest 的 end / rows
    會同步更新，之後 ``compute_with_cache`` 會走 manifest 增量路徑補算被截掉的日期。
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    path = Path(path)
    keep_through = pd.Timestamp(keep_through).normalize()
    manifest = read_cache_manifest(path)
    summary = {
        "keep_through": str(keep_through.date()),
        "removed_parts": [],
        "rewritten_parts": [],
        "noop": True,
    }
    if manifest is None or not path.exists() or pd.Timestamp(manifest["end"]) <= keep_through:
        return summary
    summary["noop"] = False
    _convert_file_cache_to_dataset(path)

    rows = 0
    end: pd.Timestamp | None = None
    for part in sorted(path.glob("*.parquet")):
        # Windows 下檔案還開著就不能 unlink / replace，所以先在 with 裡讀完再動檔案
        with pq.ParquetFile(part) as pf:
            lo, hi = _part_time_range(pf)
            num_rows = pf.metadata.num_rows
            if lo > keep_through or hi <= keep_through:
                straddle = None
            else:
                straddle = part.with_name(f".{part.name}.tmp")
                writer = None
                kept = 0
                for batch in pf.iter_batches(batch_size=1_000_000):
                    col = batch.column("tradetime")
                    cutoff = pa.scalar(keep_through.to_datetime64()).cast(col.type)
                    filtered = batch.filter(pc.less_equal(col, cutoff))
                    if filtered.num_rows == 0:
                        continue
                    if writer is None:
                        writer = pq.ParquetWriter(straddle, filtered.schema, compression="snappy")
                    writer.write_batch(filtered)
                    kept += filtered.num_rows
                    part_max = pd.Timestamp(pc.max(filtered.column("tradetime")).as_py())
                    end = part_max if end is None else max(end, part_max)
                if writer is not None:
                    writer.close()
        if lo > keep_through:
            part.unlink()
            summary["removed_parts"].append(part.name)
        elif straddle is None:
            rows += num_rows
            end = hi if end is None else max(end, hi)
        elif kept == 0:
            part.unlink()
            summary["removed_parts"].append(part.name)
        else:
            straddle.replace(part)
            rows += kept
            summary["rewritten_parts"].append(part.name)

    write_cache_manifest(
        path,
        data_source=manifest["data_source"],
        rows=rows,
        n_securities=int(manifest.get("n_securities", 0)),
        n_alphas=int(manifest.get("n_alphas", 0)),
        start=pd.Timestamp(manifest["start"]),
        end=end if end is not None else keep_through,
    )
    logger.info("cache_truncated", path=str(path), **summary)
    return summary
