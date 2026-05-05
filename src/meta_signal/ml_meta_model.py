"""Layer 4 — Plan B: XGBoost meta model for alpha aggregation (MVP v2).

Replaces the MVP-v1 LogisticRegression stub with an XGBoost regressor that
predicts forward returns directly, preserving magnitude for downstream ranking.

Key design
----------
* **Regression, not classification**: XGBRegressor outputs continuous scores.
  Direction and confidence are derived from the score magnitude. This keeps
  the signal schema compatible with RuleBasedSignalGenerator so downstream
  portfolio construction doesn't care which meta model produced it.

* **Purged expanding-window CV**: For time series, we sort the index by
  tradetime, split into `n_splits` expanding folds, and insert a `purge_days`
  gap between each train tail and validation head. This mimics the
  de-facto standard (López de Prado) without needing full PurgedKFold.

* **Feature pool filtering**: Accepts an optional `feature_columns` list so
  callers can restrict training to the effective alpha subset produced by
  WP2 (`configs/alpha_config.yaml: v2_effective_alphas`).

* **Registry registration is optional**: `register_to_registry()` is a thin
  wrapper that catches connection errors so unit tests / offline runs work
  without PostgreSQL.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import xgboost as xgb

from src.common.logging import get_logger
from src.common.metrics import (
    information_coefficient,
    rank_information_coefficient,
)

logger = get_logger(__name__)


DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "tree_method": "hist",
    "verbosity": 0,
}


class MLMetaModel:
    """XGBoost regressor that predicts forward returns from alpha features."""

    def __init__(
        self,
        hyperparams: dict[str, Any] | None = None,
        feature_columns: list[str] | None = None,
        objective: str = "forward_return",
        proxy_top_k: int = 10,
        proxy_round_trip_cost: float = 0.005,
    ) -> None:
        self._hyperparams = {**DEFAULT_XGB_PARAMS, **(hyperparams or {})}
        self._feature_columns: list[str] = list(feature_columns) if feature_columns else []
        self._objective = objective
        self._proxy_top_k = proxy_top_k
        self._proxy_round_trip_cost = proxy_round_trip_cost
        self._model: xgb.XGBRegressor | None = None
        self._model_id = f"ml_xgb_{uuid4().hex[:8]}"
        self._trained_at: datetime | None = None
        self._training_window: tuple[datetime, datetime] | None = None
        self._holdout_metrics: dict[str, float] = {}
        self._feature_importance: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        alpha_features: pd.DataFrame,
        labels: pd.Series,
        purge_days: int = 5,
        n_splits: int = 3,
    ) -> dict[str, Any]:
        """Fit the XGBoost model with purged expanding-window CV.

        Args:
            alpha_features: Wide-format DataFrame indexed by (security_id,
                tradetime) with one column per alpha. NaNs are filled with 0.
            labels: Forward-return labels aligned with `alpha_features` by
                the (security_id, tradetime) multi-index.
            purge_days: Gap (in days) between train-tail and val-head in
                each CV fold to prevent label leakage.
            n_splits: Number of expanding-window CV folds.

        Returns:
            {model_id, holdout_metrics, feature_importance, n_train}
        """
        X, y = self._align_features_labels(alpha_features, labels)
        if len(X) < 50:
            raise ValueError(
                f"Not enough samples to train ML meta model: {len(X)}"
            )

        # Respect explicit feature_columns, else infer from columns.
        if not self._feature_columns:
            self._feature_columns = list(X.columns)
        else:
            missing = [c for c in self._feature_columns if c not in X.columns]
            if missing:
                logger.warning("ml_meta_missing_features", missing=missing)
                self._feature_columns = [c for c in self._feature_columns if c in X.columns]
            X = X[self._feature_columns]

        # Purged expanding-window CV for OOS metrics.
        cv_metrics = self._purged_cv(X, y, purge_days=purge_days, n_splits=n_splits)

        # Refit on the full set for downstream prediction.
        self._model = xgb.XGBRegressor(**self._hyperparams)
        self._model.fit(X.to_numpy(), y.to_numpy())

        # Feature importance (gain-based, normalized).
        booster = self._model.get_booster()
        importance_raw = booster.get_score(importance_type="gain")
        # XGBoost keys features as f0, f1, ... — map back to names.
        self._feature_importance = {
            self._feature_columns[int(k[1:])]: float(v)
            for k, v in importance_raw.items()
            if int(k[1:]) < len(self._feature_columns)
        }
        # Fill missing features with 0 for transparent reporting.
        for col in self._feature_columns:
            self._feature_importance.setdefault(col, 0.0)

        self._holdout_metrics = cv_metrics
        self._trained_at = datetime.utcnow()
        dates = X.index.get_level_values("tradetime")
        self._training_window = (
            dates.min().to_pydatetime(),
            dates.max().to_pydatetime(),
        )

        result = {
            "model_id": self._model_id,
            "holdout_metrics": cv_metrics,
            "feature_importance": self._feature_importance,
            "n_train": int(len(X)),
            "n_features": len(self._feature_columns),
        }
        logger.info("ml_meta_model_trained", **{k: v for k, v in result.items() if k != "feature_importance"})
        return result

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, alpha_features: pd.DataFrame) -> pd.DataFrame:
        """Predict signals from alpha features.

        Returns a long-format DataFrame with:
            security_id, tradetime, signal_score, signal_direction, confidence
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X = self._ensure_wide(alpha_features)
        X = X.reindex(columns=self._feature_columns).fillna(0.0)

        raw = self._model.predict(X.to_numpy())
        out = pd.DataFrame(
            {
                "signal_score": raw,
                "signal_direction": np.where(raw >= 0, 1, -1).astype(int),
                "confidence": np.abs(raw),
            },
            index=X.index,
        ).reset_index()
        return out

    # ------------------------------------------------------------------
    # Optional registry hook
    # ------------------------------------------------------------------

    def register_to_registry(self) -> bool:
        """Attempt to persist this model to the PostgreSQL model_registry.

        Swallows connection errors so offline runs and unit tests without a
        database still succeed. Returns True on success, False otherwise.
        """
        if self._model is None or self._trained_at is None:
            logger.warning("ml_meta_register_skipped_untrained")
            return False
        try:
            from src.adaptation.model_registry import ModelRegistryManager

            ModelRegistryManager().register_model(
                model_id=self._model_id,
                model_type="xgboost_regressor",
                trained_at=self._trained_at,
                training_window=self._training_window or (self._trained_at, self._trained_at),
                features_used=self._feature_columns,
                hyperparams=self._hyperparams,
                holdout_metrics=self._holdout_metrics,
            )
            return True
        except Exception as exc:  # pragma: no cover — network/DB dependent
            logger.warning("ml_meta_register_failed", error=str(exc))
            return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_wide(alpha_features: pd.DataFrame) -> pd.DataFrame:
        """Coerce long-format alpha panel into wide MultiIndex DataFrame."""
        if {"alpha_id", "alpha_value"}.issubset(alpha_features.columns):
            wide = alpha_features.pivot_table(
                index=["security_id", "tradetime"],
                columns="alpha_id",
                values="alpha_value",
            )
            return wide
        return alpha_features

    def _align_features_labels(
        self,
        alpha_features: pd.DataFrame,
        labels: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series]:
        wide = self._ensure_wide(alpha_features)
        common = wide.index.intersection(labels.index)
        X = wide.loc[common].fillna(0.0)
        y = labels.loc[common]
        # Sort by tradetime for correct time-series CV.
        order = X.index.get_level_values("tradetime").argsort(kind="stable")
        return X.iloc[order], y.iloc[order]

    def _purged_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        purge_days: int,
        n_splits: int,
    ) -> dict[str, float]:
        """Expanding-window CV with a time gap between train tail and val head.

        Sorted by tradetime. For split i, train on [0, t_i - purge_days), val
        on [t_i, t_{i+1}). Aggregates IC/rank-IC/RMSE/direction-accuracy across
        folds (sample-weighted).
        """
        dates = X.index.get_level_values("tradetime").unique().sort_values()
        if len(dates) < (n_splits + 1):
            logger.warning("purged_cv_too_few_dates", n_dates=len(dates))
            n_splits = max(2, len(dates) // 20)

        fold_edges = np.linspace(
            int(len(dates) * 0.4), len(dates), n_splits + 1, dtype=int
        )
        ic_list, rank_list, rmse_list, dir_list, weights = [], [], [], [], []
        net_proxy_list, topk_ret_list, turnover_list = [], [], []

        for i in range(n_splits):
            val_start_date = dates[fold_edges[i]] if fold_edges[i] < len(dates) else None
            val_end_date = (
                dates[fold_edges[i + 1] - 1] if fold_edges[i + 1] - 1 < len(dates) else dates[-1]
            )
            if val_start_date is None or val_start_date >= val_end_date:
                continue

            purge_cutoff = val_start_date - pd.Timedelta(days=purge_days)
            train_mask = X.index.get_level_values("tradetime") <= purge_cutoff
            val_mask = (
                (X.index.get_level_values("tradetime") >= val_start_date)
                & (X.index.get_level_values("tradetime") <= val_end_date)
            )
            X_tr, y_tr = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]
            if len(X_tr) < 30 or len(X_val) < 10:
                continue

            model = xgb.XGBRegressor(**self._hyperparams)
            model.fit(X_tr.to_numpy(), y_tr.to_numpy())
            pred = pd.Series(model.predict(X_val.to_numpy()), index=X_val.index)

            ic = information_coefficient(pred, y_val)
            rank_ic = rank_information_coefficient(pred, y_val)
            rmse = float(np.sqrt(np.mean((pred - y_val) ** 2)))
            dir_acc = float((np.sign(pred) == np.sign(y_val)).mean())
            proxy = self._long_only_topk_proxy(pred, y_val)

            if not np.isnan(ic):
                ic_list.append(ic)
                rank_list.append(rank_ic)
                rmse_list.append(rmse)
                dir_list.append(dir_acc)
                net_proxy_list.append(proxy["net_return_proxy"])
                topk_ret_list.append(proxy["long_only_topk_net_return"])
                turnover_list.append(proxy["turnover_proxy"])
                weights.append(len(X_val))

        if not ic_list:
            return {
                "ic": 0.0,
                "rank_ic": 0.0,
                "gross_ic": 0.0,
                "net_ic_proxy": 0.0,
                "long_only_topk_net_return": 0.0,
                "turnover_proxy": 0.0,
                "rmse": 0.0,
                "dir_accuracy": 0.0,
                "n_folds": 0,
            }

        w = np.array(weights, dtype=float)
        w /= w.sum()
        return {
            "ic": float(np.dot(ic_list, w)),
            "rank_ic": float(np.dot(rank_list, w)),
            "gross_ic": float(np.dot(ic_list, w)),
            "net_ic_proxy": float(np.dot(net_proxy_list, w)),
            "long_only_topk_net_return": float(np.dot(topk_ret_list, w)),
            "turnover_proxy": float(np.dot(turnover_list, w)),
            "rmse": float(np.dot(rmse_list, w)),
            "dir_accuracy": float(np.dot(dir_list, w)),
            "n_folds": len(ic_list),
        }

    def _long_only_topk_proxy(
        self,
        pred: pd.Series,
        y_true: pd.Series,
    ) -> dict[str, float]:
        if pred.empty:
            return {
                "net_return_proxy": 0.0,
                "long_only_topk_net_return": 0.0,
                "turnover_proxy": 0.0,
            }

        rows = pd.DataFrame({"pred": pred, "y": y_true}).dropna()
        if rows.empty:
            return {
                "net_return_proxy": 0.0,
                "long_only_topk_net_return": 0.0,
                "turnover_proxy": 0.0,
            }

        daily_net: list[float] = []
        daily_turnover: list[float] = []
        previous: set[str] = set()
        for _, day in rows.groupby(level="tradetime", sort=True):
            day = day[day["pred"] >= 0].sort_values("pred", ascending=False)
            selected = day.head(self._proxy_top_k)
            current = set(selected.index.get_level_values("security_id").astype(str))
            if selected.empty:
                gross = 0.0
            else:
                gross = float(selected["y"].mean())
            if not previous:
                turnover = 1.0 if current else 0.0
            else:
                buys = len(current - previous) / max(self._proxy_top_k, 1)
                sells = len(previous - current) / max(self._proxy_top_k, 1)
                turnover = max(buys, sells)
            net = gross - turnover * self._proxy_round_trip_cost
            daily_net.append(net)
            daily_turnover.append(turnover)
            previous = current

        if not daily_net:
            return {
                "net_return_proxy": 0.0,
                "long_only_topk_net_return": 0.0,
                "turnover_proxy": 0.0,
            }
        return {
            "net_return_proxy": float(np.mean(daily_net)),
            "long_only_topk_net_return": float(np.mean(daily_net)),
            "turnover_proxy": float(np.mean(daily_turnover)),
        }

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def feature_importance(self) -> dict[str, float]:
        return self._feature_importance

    @property
    def holdout_metrics(self) -> dict[str, float]:
        return self._holdout_metrics
