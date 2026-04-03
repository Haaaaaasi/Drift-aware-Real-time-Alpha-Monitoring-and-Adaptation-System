"""Layer 2 — Data quality checks for standardized bars."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logging import get_logger

logger = get_logger(__name__)


class QualityChecker:
    """Validate and flag data quality issues in standardized bars."""

    def __init__(
        self,
        max_missing_ratio: float = 0.15,
        price_sigma_threshold: float = 10.0,
    ) -> None:
        self._max_missing_ratio = max_missing_ratio
        self._price_sigma_threshold = price_sigma_threshold

    def check_and_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run quality checks and update is_tradable / missing_flags."""
        df = df.copy()

        # OHLC consistency: high >= max(open, close) and low <= min(open, close)
        invalid_ohlc = (df["high"] < df[["open", "close"]].max(axis=1)) | (
            df["low"] > df[["open", "close"]].min(axis=1)
        )
        if invalid_ohlc.any():
            logger.warning("invalid_ohlc_detected", count=int(invalid_ohlc.sum()))
            df.loc[invalid_ohlc, "is_tradable"] = False

        # Extreme price detection
        for col in ["open", "high", "low", "close"]:
            if df[col].std() > 0:
                z = (df[col] - df[col].mean()) / df[col].std()
                extreme = z.abs() > self._price_sigma_threshold
                if extreme.any():
                    logger.warning(
                        "extreme_price_detected",
                        column=col,
                        count=int(extreme.sum()),
                    )

        # Zero volume
        zero_vol = df["vol"] <= 0
        df.loc[zero_vol, "is_tradable"] = False

        # Stale price detection (per security)
        df["_prev_close"] = df.groupby("security_id")["close"].shift(1)
        stale = (df["close"] == df["_prev_close"]) & df["_prev_close"].notna()
        df["_stale_count"] = stale.groupby(df["security_id"]).cumsum()
        # Mark as untradable if stale for 10+ consecutive bars
        df.loc[df["_stale_count"] >= 10, "is_tradable"] = False
        df.drop(columns=["_prev_close", "_stale_count"], inplace=True)

        tradable_ratio = df["is_tradable"].mean()
        logger.info(
            "quality_check_complete",
            total_rows=len(df),
            tradable_ratio=round(tradable_ratio, 4),
        )
        return df
