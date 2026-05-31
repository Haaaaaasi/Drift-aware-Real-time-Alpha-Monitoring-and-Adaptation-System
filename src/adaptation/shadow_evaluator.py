"""Layer 10 — Shadow / canary evaluation for candidate models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logging import get_logger
from src.common.metrics import information_coefficient, hit_rate, sharpe_ratio

logger = get_logger(__name__)


class ShadowEvaluator:
    """Evaluate candidate models in shadow mode before promoting to production.

    Compares up to 3 candidates:
    - Current production model
    - Freshly retrained model
    - Reused historical model (from recurring concept pool)
    """

    def __init__(
        self,
        min_improvement_ic: float = 0.005,
        min_evaluation_days: int = 5,
    ) -> None:
        self._min_improvement = min_improvement_ic
        self._min_days = min_evaluation_days

    def evaluate_candidates(
        self,
        candidates: dict[str, pd.DataFrame],
        forward_returns: pd.Series,
        *,
        proxy_top_k: int | None = None,
        commission_rate: float = 0.0,
        tax_rate: float = 0.0,
        slippage_bps: float = 0.0,
        round_trip_cost_pct: float | None = None,
    ) -> dict[str, dict[str, float]]:
        """Evaluate multiple candidate signal sets.

        Args:
            candidates: Mapping model_id -> signals DataFrame [security_id, tradetime, signal_score].
            forward_returns: Indexed by (security_id, tradetime).
            proxy_top_k: When provided, also compute a shadow-window top-k portfolio proxy.

        Returns:
            Mapping model_id -> evaluation metrics.
        """
        results = {}

        for model_id, signals in candidates.items():
            if signals.empty:
                results[model_id] = self._empty_metrics()
                continue

            sig = signals.set_index(["security_id", "tradetime"])["signal_score"]
            common = sig.index.intersection(forward_returns.index)

            if len(common) < 10:
                results[model_id] = self._empty_metrics()
                continue

            ic = information_coefficient(sig.loc[common], forward_returns.loc[common])
            hr = hit_rate(sig.loc[common], forward_returns.loc[common])

            # Approximate Sharpe from signal-weighted returns
            weighted_ret = sig.loc[common] * forward_returns.loc[common]
            sr = sharpe_ratio(weighted_ret) if len(weighted_ret) > 5 else 0.0

            metrics = {
                "ic": float(ic) if not np.isnan(ic) else 0.0,
                "hit_rate": float(hr) if not np.isnan(hr) else 0.0,
                "sharpe": float(sr) if not np.isnan(sr) else 0.0,
                "n_samples": len(common),
            }
            if proxy_top_k is not None and proxy_top_k > 0:
                metrics.update(
                    self._evaluate_topk_proxy(
                        signals=signals,
                        forward_returns=forward_returns,
                        top_k=proxy_top_k,
                        commission_rate=commission_rate,
                        tax_rate=tax_rate,
                        slippage_bps=slippage_bps,
                        round_trip_cost_pct=round_trip_cost_pct,
                    )
                )

            results[model_id] = metrics

        logger.info("shadow_evaluation_complete", candidates=list(results.keys()))
        return results

    def select_best(
        self,
        evaluation_results: dict[str, dict[str, float]],
        current_model_id: str | None = None,
        metric: str = "ic",
    ) -> str | None:
        """Select the best candidate model.

        If a current model exists, the best candidate must improve the selected
        metric by min_improvement.
        """
        if not evaluation_results:
            return None

        if metric not in {
            "ic",
            "hit_rate",
            "sharpe",
            "topk_gross_return",
            "topk_net_return",
        }:
            raise ValueError(f"unsupported shadow selection metric: {metric}")

        ranked = sorted(
            evaluation_results.items(),
            key=lambda x: x[1].get(metric, 0.0),
            reverse=True,
        )

        best_id, best_metrics = ranked[0]

        if current_model_id and current_model_id in evaluation_results:
            current_score = evaluation_results[current_model_id].get(metric, 0.0)
            best_score = best_metrics.get(metric, 0.0)
            if best_score - current_score < self._min_improvement:
                logger.info(
                    "shadow_no_improvement",
                    best_id=best_id,
                    metric=metric,
                    best_score=best_score,
                    current_score=current_score,
                )
                return None

        logger.info("shadow_best_selected", model_id=best_id, metric=metric, metrics=best_metrics)
        return best_id

    @staticmethod
    def _empty_metrics() -> dict[str, float]:
        return {
            "ic": 0.0,
            "hit_rate": 0.0,
            "sharpe": 0.0,
            "n_samples": 0,
            "topk_gross_return": 0.0,
            "topk_net_return": 0.0,
            "topk_turnover": 0.0,
            "topk_n_days": 0,
        }

    def _evaluate_topk_proxy(
        self,
        *,
        signals: pd.DataFrame,
        forward_returns: pd.Series,
        top_k: int,
        commission_rate: float,
        tax_rate: float,
        slippage_bps: float,
        round_trip_cost_pct: float | None,
    ) -> dict[str, float]:
        """Shadow-window top-k equal-weight proxy, with simple turnover costs."""
        required = {"security_id", "tradetime", "signal_score"}
        if signals.empty or not required.issubset(signals.columns):
            return self._empty_metrics() | {"ic": 0.0, "hit_rate": 0.0, "sharpe": 0.0}

        sig_df = signals[list(required)].dropna(subset=["signal_score"]).copy()
        if sig_df.empty:
            return {
                "topk_gross_return": 0.0,
                "topk_net_return": 0.0,
                "topk_turnover": 0.0,
                "topk_n_days": 0,
            }
        sig_df["security_id"] = sig_df["security_id"].astype(str)
        sig_df["tradetime"] = pd.to_datetime(sig_df["tradetime"])

        fwd_df = forward_returns.rename("forward_return").reset_index()
        if fwd_df.empty:
            return {
                "topk_gross_return": 0.0,
                "topk_net_return": 0.0,
                "topk_turnover": 0.0,
                "topk_n_days": 0,
            }
        fwd_df["security_id"] = fwd_df["security_id"].astype(str)
        fwd_df["tradetime"] = pd.to_datetime(fwd_df["tradetime"])
        fwd_lookup = fwd_df.set_index(["security_id", "tradetime"])["forward_return"]

        prev_weights: dict[str, float] = {}
        gross_daily: list[float] = []
        net_daily: list[float] = []
        turnovers: list[float] = []

        for dt, day_signals in sig_df.groupby("tradetime", sort=True):
            ranked = (
                day_signals.sort_values("signal_score", ascending=False)
                .head(max(1, int(top_k)))
            )
            if ranked.empty:
                continue
            weight = 1.0 / len(ranked)
            weights = {str(sec): weight for sec in ranked["security_id"]}

            gross = 0.0
            n_valid = 0
            for sec, w in weights.items():
                ret = fwd_lookup.get((sec, dt), np.nan)
                if pd.notna(ret):
                    gross += w * float(ret)
                    n_valid += 1
            if n_valid == 0:
                continue

            all_secs = set(prev_weights) | set(weights)
            buys = sum(max(0.0, weights.get(s, 0.0) - prev_weights.get(s, 0.0)) for s in all_secs)
            sells = sum(max(0.0, prev_weights.get(s, 0.0) - weights.get(s, 0.0)) for s in all_secs)
            cost = self._compute_cost(
                buys=buys,
                sells=sells,
                commission_rate=commission_rate,
                tax_rate=tax_rate,
                slippage_bps=slippage_bps,
                round_trip_cost_pct=round_trip_cost_pct,
            )
            gross_daily.append(gross)
            net_daily.append(gross - cost)
            turnovers.append(max(buys, sells))
            prev_weights = weights

        if not gross_daily:
            return {
                "topk_gross_return": 0.0,
                "topk_net_return": 0.0,
                "topk_turnover": 0.0,
                "topk_n_days": 0,
            }

        gross_arr = np.array(gross_daily, dtype=float)
        net_arr = np.array(net_daily, dtype=float)
        return {
            "topk_gross_return": float(np.prod(1.0 + gross_arr) - 1.0),
            "topk_net_return": float(np.prod(1.0 + net_arr) - 1.0),
            "topk_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
            "topk_n_days": int(len(gross_daily)),
        }

    @staticmethod
    def _compute_cost(
        *,
        buys: float,
        sells: float,
        commission_rate: float,
        tax_rate: float,
        slippage_bps: float,
        round_trip_cost_pct: float | None,
    ) -> float:
        turnover = max(buys, sells)
        if round_trip_cost_pct is not None:
            return turnover * (round_trip_cost_pct / 100.0)
        return (
            (buys + sells) * commission_rate
            + sells * tax_rate
            + (buys + sells) * (slippage_bps / 10000.0)
        )
