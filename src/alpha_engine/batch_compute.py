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

        self._client.load_module("wq101alpha")
        self._client.load_module("prepare101")

        # Upload parameters
        self._client.run(f'startDate = {start_date.strftime("%Y.%m.%d")}')
        self._client.run(f'endDate = {end_date.strftime("%Y.%m.%d")}')
        self._client.run(f"alphaIds = {alpha_ids}")

        # Load and run batch module
        self._client.run('run "/modules/alpha_batch.dos"')
        result = self._client.run("computeBatchAlphas(startDate, endDate, alphaIds)")

        if result is None or (isinstance(result, pd.DataFrame) and result.empty):
            logger.warning("batch_compute_empty", start=str(start_date), end=str(end_date))
            return pd.DataFrame(columns=["security_id", "tradetime", "alpha_id", "alpha_value"])

        df = result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)
        logger.info(
            "batch_compute_complete",
            rows=len(df),
            alphas=len(alpha_ids),
            version=version_id,
        )

        # Persist to DolphinDB
        self._client.run(f'saveAlphaFeatures(result, "{version_id}")')

        return df

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
