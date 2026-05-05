"""Layer 3 — Batch alpha computation orchestrator (Python side)."""

from __future__ import annotations

from datetime import date, datetime
from uuid import uuid4

import pandas as pd

from src.alpha_engine.dolphindb_client import DolphinDBClient
from src.common.logging import get_logger
from src.config.constants import MVP_V1_ALPHA_IDS

logger = get_logger(__name__)


class BatchAlphaComputer:
    """Orchestrate batch computation of WQAlpha factors via DolphinDB."""

    def __init__(self) -> None:
        self._client = DolphinDBClient()

    def compute(
        self,
        start_date: date,
        end_date: date,
        alpha_ids: list[int] | None = None,
    ) -> pd.DataFrame:
        """Trigger batch alpha computation in DolphinDB.

        Args:
            start_date: Start of computation window.
            end_date: End of computation window.
            alpha_ids: List of alpha numbers (e.g., [1, 2, 3]). Defaults to MVP v1 set.

        Returns:
            DataFrame with columns: security_id, tradetime, alpha_id, alpha_value
        """
        if alpha_ids is None:
            alpha_ids = [int(a.replace("wq", "").lstrip("0")) for a in MVP_V1_ALPHA_IDS]

        version_id = f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"

        # 載入並執行 alpha_batch 模組（prepare101 不存在——alpha_batch.dos 自帶 preparePanels）
        self._client.run('run "/modules/alpha_batch.dos"')

        # computeBatchAlphas 內部 per-alpha 直寫 alpha_features，避免一次 append 31M rows
        # 觸發 DolphinDB 社群版 8GB OOM。回傳單列摘要表：n_rows_written / n_alphas_ok / n_alphas_failed。
        script = (
            f'startDate = {start_date.strftime("%Y.%m.%d")};\n'
            f'endDate = {end_date.strftime("%Y.%m.%d")};\n'
            f'alphaIds = {alpha_ids};\n'
            f'versionId = "{version_id}";\n'
            'summary = computeBatchAlphas(startDate, endDate, alphaIds, versionId);\n'
            'summary'
        )
        summary_df = self._client.run(script)
        n_rows = int(summary_df.iloc[0]["n_rows_written"]) if summary_df is not None and len(summary_df) else 0
        n_ok = int(summary_df.iloc[0]["n_alphas_ok"]) if summary_df is not None and len(summary_df) else 0
        n_failed = int(summary_df.iloc[0]["n_alphas_failed"]) if summary_df is not None and len(summary_df) else 0

        if n_rows == 0:
            logger.warning("batch_compute_empty", start=str(start_date), end=str(end_date))
            return pd.DataFrame(columns=["security_id", "tradetime", "alpha_id", "alpha_value"])

        logger.info(
            "batch_compute_complete",
            rows=n_rows,
            alphas=len(alpha_ids),
            alphas_ok=n_ok,
            alphas_failed=n_failed,
            version=version_id,
        )

        # 回傳精簡摘要 DataFrame（想要完整資料請用 get_alpha_features 或直接查 alpha_features 表）
        return pd.DataFrame({
            "version_id": [version_id],
            "n_rows": [n_rows],
            "n_alphas": [len(alpha_ids)],
            "n_alphas_ok": [n_ok],
            "n_alphas_failed": [n_failed],
            "start_date": [start_date],
            "end_date": [end_date],
        })

    def get_alpha_features(
        self,
        security_ids: list[str],
        start_date: date,
        end_date: date,
        alpha_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        """Retrieve previously computed alpha features from DolphinDB."""
        if alpha_ids is None:
            alpha_ids = MVP_V1_ALPHA_IDS

        symbols_str = "`".join(security_ids)
        alphas_str = "`".join(alpha_ids)
        where = (
            f'security_id in [`{symbols_str}], '
            f'tradetime between timestamp({start_date.strftime("%Y.%m.%d")})'
            f' : timestamp({end_date.strftime("%Y.%m.%d")}), '
            f'alpha_id in [`{alphas_str}]'
        )

        return self._client.query_table("dfs://darams_alpha", "alpha_features", where)

    def close(self) -> None:
        self._client.close()
