"""Layer 4 — Plan B: ML meta model for alpha aggregation.

TODO (MVP v2): Full implementation with XGBoost/LightGBM and purged CV.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.common.logging import get_logger

logger = get_logger(__name__)


class MLMetaModel:
    """Machine-learning meta model that combines alpha features to predict returns.

    MVP v2 models: LogisticRegression (baseline) → XGBoost → LightGBM
    Uses purged expanding-window cross-validation to prevent leakage.
    """

    def __init__(self, model_type: str = "logistic") -> None:
        self._model_type = model_type
        self._model: Any = None
        self._feature_columns: list[str] = []
        self._model_id = f"ml_{model_type}_{uuid4().hex[:8]}"

    def train(
        self,
        alpha_features: pd.DataFrame,
        labels: pd.Series,
        purge_days: int = 5,
    ) -> dict[str, float]:
        """Train the meta model on alpha features with forward-return labels.

        Args:
            alpha_features: Wide-format DataFrame (index=security_id×tradetime, columns=alpha_ids).
            labels: Binary direction labels (1 / -1) aligned with features.
            purge_days: Gap days between train and validation for leakage prevention.

        Returns:
            Dictionary of holdout metrics.
        """
        common = alpha_features.index.intersection(labels.index)
        X = alpha_features.loc[common].fillna(0)
        y = (labels.loc[common] > 0).astype(int)

        self._feature_columns = list(X.columns)

        # Simple expanding-window split (last 20% as holdout)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx + purge_days :]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx + purge_days :]

        if self._model_type == "logistic":
            self._model = LogisticRegression(max_iter=1000, C=0.1)
        else:
            logger.warning("model_type_not_implemented", model_type=self._model_type)
            self._model = LogisticRegression(max_iter=1000, C=0.1)

        self._model.fit(X_train, y_train)

        val_pred = self._model.predict(X_val)
        accuracy = float((val_pred == y_val).mean()) if len(y_val) > 0 else 0.0

        metrics = {"accuracy": accuracy, "val_size": len(y_val)}
        logger.info("ml_meta_model_trained", model_id=self._model_id, metrics=metrics)
        return metrics

    def predict(self, alpha_features: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions from the trained model."""
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X = alpha_features[self._feature_columns].fillna(0)
        proba = self._model.predict_proba(X)[:, 1]

        result = pd.DataFrame(index=alpha_features.index)
        result["signal_score"] = proba - 0.5  # center around 0
        result["signal_direction"] = np.where(proba > 0.5, 1, -1)
        result["confidence"] = np.abs(proba - 0.5) * 2

        return result.reset_index()

    @property
    def model_id(self) -> str:
        return self._model_id
