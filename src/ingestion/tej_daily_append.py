"""TEJ daily OHLCV append utilities.

此模組處理每日 TEJ 檔案併入既有 survivorship-correct parquet 的流程。
它只負責安全更新 bars 與 universe bounds；alpha cache 由 live pipeline 透過
FeatureStore / compute_with_cache 在下一步增量更新。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
import shutil
from typing import Iterable

import pandas as pd


DEFAULT_TEJ_OUTPUT = Path("data/tw_stocks_tej.parquet")
DEFAULT_UNIVERSE_OUTPUT = Path("data/tw_stocks_tej_universe.parquet")
DEFAULT_BACKUP_DIR = Path("data/backups/tej_daily")
COMMON_STOCK_PATTERN = r"\d{4}"
REQUIRED_COLUMNS = ["security_id", "datetime", "open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class TejDailyAppendResult:
    """每日 TEJ append 的可序列化結果。"""

    input_paths: list[str]
    output_path: str
    universe_output_path: str
    dry_run: bool
    existing_rows: int
    incoming_rows_raw: int
    incoming_rows_common: int
    incoming_unique_keys: int
    overlap_keys: int
    added_keys: int
    output_rows: int
    existing_max_date: str | None
    incoming_min_date: str | None
    incoming_max_date: str | None
    output_max_date: str | None
    backup_paths: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def load_tej_ohlcv_csv(
    path: str | Path,
    *,
    encoding: str = "utf-16-le",
    sep: str = "\t",
) -> pd.DataFrame:
    """讀取 TEJ OHLCV CSV，轉成專案標準的 parquet 欄位。"""

    raw = pd.read_csv(path, encoding=encoding, sep=sep)
    if raw.shape[1] != 7:
        raise ValueError(
            f"{path} has {raw.shape[1]} columns; expected 7 TEJ OHLCV columns."
        )
    raw.columns = ["name_combined", "date", "open", "high", "low", "close", "volume_kshare"]
    out = pd.DataFrame()
    out["security_id"] = raw["name_combined"].astype(str).str.extract(r"^(\S+)")[0]
    out["datetime"] = pd.to_datetime(raw["date"].astype(str), format="%Y%m%d")
    for col in ["open", "high", "low", "close"]:
        out[col] = pd.to_numeric(raw[col].astype(str).str.replace(",", "", regex=False))
    volume_kshare = pd.to_numeric(
        raw["volume_kshare"].astype(str).str.replace(",", "", regex=False)
    )
    out["volume"] = (volume_kshare * 1000).round().astype("int64")
    return out[REQUIRED_COLUMNS]


def filter_common_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """保留 4 碼普通股代碼，和既有 TEJ ingestion 的 universe 規則一致。"""

    return df[df["security_id"].astype(str).str.fullmatch(COMMON_STOCK_PATTERN, na=False)].copy()


def build_universe_bounds(
    bars: pd.DataFrame,
    *,
    active_threshold_days: int = 30,
) -> pd.DataFrame:
    """重建每檔股票的 first/last date 與 active-at-end 標記。"""

    bounds = (
        bars.groupby("security_id")["datetime"]
        .agg(first_date="min", last_date="max")
        .reset_index()
    )
    counts = bars.groupby("security_id").size().rename("n_trading_days").reset_index()
    bounds = bounds.merge(counts, on="security_id", how="left")
    data_end = pd.to_datetime(bars["datetime"]).max()
    bounds["is_active_at_end"] = bounds["last_date"] >= (
        data_end - pd.Timedelta(days=active_threshold_days)
    )
    return bounds.sort_values("security_id").reset_index(drop=True)


def append_tej_daily_files(
    input_paths: Iterable[str | Path],
    *,
    output_path: str | Path = DEFAULT_TEJ_OUTPUT,
    universe_output_path: str | Path = DEFAULT_UNIVERSE_OUTPUT,
    backup_dir: str | Path | None = DEFAULT_BACKUP_DIR,
    dry_run: bool = False,
    encoding: str = "utf-16-le",
    sep: str = "\t",
    active_threshold_days: int = 30,
) -> TejDailyAppendResult:
    """把一批每日 TEJ CSV 安全併入既有 parquet。

    相同 `(security_id, datetime)` 會採用新檔案的值，方便處理 TEJ 修正資料。
    寫檔前預設會備份既有 bars / universe parquet。
    """

    paths = [Path(p) for p in input_paths]
    if not paths:
        raise ValueError("input_paths must contain at least one TEJ CSV path")
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"TEJ input file not found: {path}")

    output = Path(output_path)
    universe_output = Path(universe_output_path)
    existing = _load_existing_bars(output)
    existing_rows = len(existing)
    existing_max = _date_str(existing["datetime"].max()) if not existing.empty else None

    frames = [load_tej_ohlcv_csv(p, encoding=encoding, sep=sep) for p in paths]
    incoming_raw = pd.concat(frames, ignore_index=True)
    incoming_common = filter_common_stocks(incoming_raw)
    incoming_common = _normalize_bars(incoming_common)
    incoming_common = incoming_common.drop_duplicates(
        subset=["security_id", "datetime"],
        keep="last",
    )

    warnings: list[str] = []
    if incoming_common.empty:
        warnings.append("No 4-digit common-stock rows found in incoming TEJ files.")

    existing_keys = _key_index(existing)
    incoming_keys = _key_index(incoming_common)
    overlap_keys = len(existing_keys.intersection(incoming_keys))
    added_keys = len(incoming_keys.difference(existing_keys))

    merged = pd.concat([existing, incoming_common], ignore_index=True)
    merged = _normalize_bars(merged)
    merged = merged.drop_duplicates(subset=["security_id", "datetime"], keep="last")
    merged = merged.sort_values(["security_id", "datetime"]).reset_index(drop=True)

    backup_paths: list[str] = []
    if not dry_run:
        backup_paths = _backup_existing_files(
            [output, universe_output],
            backup_dir=Path(backup_dir) if backup_dir is not None else None,
        )
        _write_parquet_atomic(merged, output)
        universe = build_universe_bounds(
            merged,
            active_threshold_days=active_threshold_days,
        )
        _write_parquet_atomic(universe, universe_output)

    return TejDailyAppendResult(
        input_paths=[str(p) for p in paths],
        output_path=str(output),
        universe_output_path=str(universe_output),
        dry_run=dry_run,
        existing_rows=existing_rows,
        incoming_rows_raw=len(incoming_raw),
        incoming_rows_common=len(incoming_common),
        incoming_unique_keys=len(incoming_keys),
        overlap_keys=overlap_keys,
        added_keys=added_keys,
        output_rows=len(merged),
        existing_max_date=existing_max,
        incoming_min_date=_date_str(incoming_common["datetime"].min())
        if not incoming_common.empty
        else None,
        incoming_max_date=_date_str(incoming_common["datetime"].max())
        if not incoming_common.empty
        else None,
        output_max_date=_date_str(merged["datetime"].max()) if not merged.empty else None,
        backup_paths=backup_paths,
        warnings=warnings,
    )


def _load_existing_bars(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    bars = pd.read_parquet(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in bars.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    return _normalize_bars(bars[REQUIRED_COLUMNS].copy())


def _normalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    bars = bars.copy()
    if bars.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    bars["security_id"] = bars["security_id"].astype(str)
    bars["datetime"] = pd.to_datetime(bars["datetime"]).dt.normalize()
    for col in ["open", "high", "low", "close"]:
        bars[col] = pd.to_numeric(bars[col])
    bars["volume"] = pd.to_numeric(bars["volume"]).round().astype("int64")
    return bars[REQUIRED_COLUMNS]


def _key_index(bars: pd.DataFrame) -> pd.MultiIndex:
    if bars.empty:
        return pd.MultiIndex.from_arrays([[], []], names=["security_id", "datetime"])
    return pd.MultiIndex.from_frame(bars[["security_id", "datetime"]])


def _backup_existing_files(paths: list[Path], *, backup_dir: Path | None) -> list[str]:
    if backup_dir is None:
        return []
    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths:
        return []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)
    backups: list[str] = []
    for path in existing_paths:
        backup_path = backup_dir / f"{path.stem}_{stamp}{path.suffix}"
        shutil.copy2(path, backup_path)
        backups.append(str(backup_path))
    return backups


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    df.to_parquet(tmp, compression="snappy", index=False)
    tmp.replace(path)


def _date_str(value: object) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()
