"""Alpha feature store 邏輯介面。

第一版先包住現有 bar-aligned parquet cache，不搬動實體檔案 layout。
後續若改成 partitioned parquet 或 DuckDB，pipeline 仍應透過這層讀取 alpha。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.alpha_engine.alpha_cache import (
    cache_path_for_data_path,
    compute_with_cache,
    read_cache_manifest,
)
from src.config.data_sources import infer_data_source_from_path


ALPHA_ENGINE_VERSION = "python_wq101_v1"


def _stable_hash(payload: object) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FeatureStore:
    """提供 alpha panel 的邏輯 feature store。

    目前底層仍是單一 parquet cache；`version` 會把 cache manifest 與檔案指紋
    納入，供 selector snapshot / model log 追蹤當次使用的 feature 來源。
    """

    cache_path: Path
    data_source: str | None = None
    alpha_engine_version: str = ALPHA_ENGINE_VERSION

    @classmethod
    def for_data_path(cls, data_path: str | Path) -> "FeatureStore":
        path = Path(data_path)
        inferred = infer_data_source_from_path(path)
        return cls(
            cache_path=cache_path_for_data_path(path),
            data_source=None if inferred == "custom" else inferred,
        )

    @property
    def version(self) -> str:
        path = Path(self.cache_path)
        stat_payload: dict[str, object] = {"exists": path.exists()}
        if path.exists():
            stat = path.stat()
            stat_payload.update(
                {
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        payload = {
            "kind": "parquet_alpha_cache",
            "cache_path": str(path.as_posix()),
            "data_source": self.data_source,
            "alpha_engine_version": self.alpha_engine_version,
            "manifest": read_cache_manifest(path),
            "file": stat_payload,
        }
        return _stable_hash(payload)

    def load_alpha_panel(
        self,
        bars: pd.DataFrame,
        *,
        alpha_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        return compute_with_cache(
            bars,
            alpha_ids=alpha_ids,
            cache_path=self.cache_path,
            data_source=self.data_source,
        )
