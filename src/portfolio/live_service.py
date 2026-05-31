"""Shared portfolio service for live daily recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

from src.portfolio.constructor import PortfolioConstructor
from src.risk.risk_manager import RiskManager


@dataclass(frozen=True)
class LivePortfolioConfig:
    method: str = "turnover_aware_topk"
    top_k: int = 10
    entry_rank: int = 20
    exit_rank: int = 60
    max_turnover: float = 0.25
    min_holding_days: int = 10
    tail_cleanup_weight: float = 0.0025
    max_position_weight: float = 0.10
    max_gross_exposure: float = 1.0
    renormalize_after_exit_cleanup: bool = False


@dataclass(frozen=True)
class LivePortfolioResult:
    targets: pd.DataFrame
    recommendations: pd.DataFrame
    snapshot: pd.DataFrame
    metrics: dict[str, float | int | bool]


def build_live_portfolio(
    *,
    signals: pd.DataFrame,
    as_of_date: date | datetime | pd.Timestamp,
    previous_weights: dict[str, float] | None,
    previous_shares: dict[str, int] | None,
    holding_days: dict[str, int] | None,
    last_prices: pd.Series | dict[str, float],
    capital: float,
    config: LivePortfolioConfig,
) -> LivePortfolioResult:
    """Build target holdings and trade recommendations for one live day."""
    previous_weights = _normalize_float_map(previous_weights)
    previous_shares = _normalize_int_map(previous_shares)
    holding_days = _normalize_int_map(holding_days)
    as_of_ts = pd.Timestamp(as_of_date)
    prices = _normalize_float_map(last_prices)

    constructor = PortfolioConstructor(
        method=config.method,
        top_k=config.top_k,
        long_only=True,
        entry_rank=config.entry_rank,
        exit_rank=config.exit_rank,
        min_holding_days=config.min_holding_days,
    )
    risk = RiskManager(
        max_position_weight=config.max_position_weight,
        max_gross_exposure=config.max_gross_exposure,
        max_turnover=config.max_turnover,
    )

    targets = constructor.construct(
        signals,
        previous_weights=previous_weights,
        holding_days=holding_days,
    )
    held_from_prev_count = int(targets.attrs.get("held_from_prev_count", 0))
    forced_sells_count = int(targets.attrs.get("forced_sells_count", 0))
    turnover_cap_applied = False
    if targets.empty:
        current_weights: dict[str, float] = {}
    else:
        adjusted = risk.apply_constraints(targets, previous_weights=previous_weights)
        turnover_cap_applied = bool(adjusted.attrs.get("turnover_cap_applied", False))
        current_weights = {
            str(row["security_id"]): float(row["target_weight"])
            for _, row in adjusted.iterrows()
            if abs(float(row["target_weight"])) > 1e-12
        }

    signal_scores = signals.set_index("security_id")["signal_score"]
    current_weights, exit_metrics, exit_reasons = apply_exit_discipline(
        current_weights=current_weights,
        signal_scores=signal_scores,
        holding_days=holding_days,
        hard_exit_score_threshold=None,
        hard_exit_min_holding_days=config.min_holding_days,
        tail_cleanup_weight=config.tail_cleanup_weight,
        renormalize=config.renormalize_after_exit_cleanup,
    )

    targets = _weights_to_targets(
        weights=current_weights,
        as_of_ts=as_of_ts,
        prices=prices,
        capital=capital,
        method=config.method,
    )
    recommendations = build_trade_recommendations(
        as_of_ts=as_of_ts,
        target_weights=current_weights,
        previous_weights=previous_weights,
        previous_shares=previous_shares,
        holding_days=holding_days,
        prices=prices,
        capital=capital,
        signals=signals,
        exit_reasons=exit_reasons,
    )
    snapshot = build_portfolio_snapshot(
        as_of_ts=as_of_ts,
        recommendations=recommendations,
        holding_days=holding_days,
        capital=capital,
    )

    all_secs = set(previous_weights) | set(current_weights)
    buys = sum(
        max(0.0, current_weights.get(sec, 0.0) - previous_weights.get(sec, 0.0))
        for sec in all_secs
    )
    sells = sum(
        max(0.0, previous_weights.get(sec, 0.0) - current_weights.get(sec, 0.0))
        for sec in all_secs
    )
    metrics: dict[str, float | int | bool] = {
        "n_targets": int(len(targets)),
        "n_recommendations": int(len(recommendations)),
        "held_from_prev_count": held_from_prev_count,
        "forced_sells_count": forced_sells_count,
        "turnover_cap_applied": turnover_cap_applied,
        "buys_turnover": float(buys),
        "sells_turnover": float(sells),
        "turnover": float(max(buys, sells)),
        **exit_metrics,
    }
    return LivePortfolioResult(
        targets=targets,
        recommendations=recommendations,
        snapshot=snapshot,
        metrics=metrics,
    )


def apply_exit_discipline(
    *,
    current_weights: dict[str, float],
    signal_scores: pd.Series,
    holding_days: dict[str, int],
    hard_exit_score_threshold: float | None,
    hard_exit_min_holding_days: int,
    tail_cleanup_weight: float,
    renormalize: bool,
) -> tuple[dict[str, float], dict[str, float | int | bool], dict[str, str]]:
    scores = {str(sec): float(score) for sec, score in signal_scores.items()}
    weights = {str(sec): float(weight) for sec, weight in current_weights.items()}

    gross_before = float(sum(max(0.0, weight) for weight in weights.values()))
    negative_before = float(
        sum(
            max(0.0, weight)
            for sec, weight in weights.items()
            if pd.notna(scores.get(sec, np.nan)) and scores[sec] <= 0.0
        )
    )

    exit_reasons: dict[str, str] = {}
    for sec, weight in weights.items():
        age = int(holding_days.get(sec, 0))
        if age < hard_exit_min_holding_days:
            continue
        score = scores.get(sec, np.nan)
        if (
            hard_exit_score_threshold is not None
            and pd.notna(score)
            and float(score) <= hard_exit_score_threshold
        ):
            exit_reasons[sec] = "hard_exit_score"
            continue
        if tail_cleanup_weight > 0 and abs(weight) < tail_cleanup_weight:
            exit_reasons[sec] = "tail_cleanup"

    next_weights = {
        sec: weight
        for sec, weight in weights.items()
        if sec not in exit_reasons and abs(weight) > 1e-12
    }

    gross_after = float(sum(max(0.0, weight) for weight in next_weights.values()))
    did_renormalize = False
    if renormalize and gross_after > 0 and gross_before > gross_after:
        target_gross = min(1.0, gross_before)
        scale = target_gross / gross_after
        next_weights = {sec: weight * scale for sec, weight in next_weights.items()}
        gross_after = float(sum(max(0.0, weight) for weight in next_weights.values()))
        did_renormalize = True

    negative_after = float(
        sum(
            max(0.0, weight)
            for sec, weight in next_weights.items()
            if pd.notna(scores.get(sec, np.nan)) and scores[sec] <= 0.0
        )
    )
    hard_exit_count = sum(1 for reason in exit_reasons.values() if reason == "hard_exit_score")
    tail_exit_count = sum(1 for reason in exit_reasons.values() if reason == "tail_cleanup")
    metrics: dict[str, float | int | bool] = {
        "hard_exit_count": int(hard_exit_count),
        "tail_exit_count": int(tail_exit_count),
        "exit_cleanup_count": int(len(exit_reasons)),
        "exit_cleanup_gross_before": gross_before,
        "exit_cleanup_gross_after": gross_after,
        "exit_cleanup_renormalized": did_renormalize,
        "negative_score_weight_before": negative_before,
        "negative_score_weight_after": negative_after,
    }
    return next_weights, metrics, exit_reasons


def build_trade_recommendations(
    *,
    as_of_ts: pd.Timestamp,
    target_weights: dict[str, float],
    previous_weights: dict[str, float],
    previous_shares: dict[str, int],
    holding_days: dict[str, int],
    prices: dict[str, float],
    capital: float,
    signals: pd.DataFrame,
    exit_reasons: dict[str, str],
) -> pd.DataFrame:
    signal_context = _signal_context(signals)
    rows = []
    all_secs = sorted(set(previous_weights) | set(target_weights))
    for sec in all_secs:
        prev_w = float(previous_weights.get(sec, 0.0))
        target_w = float(target_weights.get(sec, 0.0))
        delta_w = target_w - prev_w
        price = prices.get(sec)
        target_shares = _target_shares(target_w, price, capital)
        current_shares = int(previous_shares.get(sec, 0))
        delta_shares = target_shares - current_shares
        ctx = signal_context.get(sec, {})
        rows.append(
            {
                "as_of_date": as_of_ts.date(),
                "security_id": sec,
                "action": _classify_action(prev_w, target_w),
                "current_weight": prev_w,
                "target_weight": target_w,
                "delta_weight": delta_w,
                "current_shares": current_shares,
                "target_shares": target_shares,
                "delta_shares": delta_shares,
                "last_price": price,
                "signal_score": ctx.get("signal_score"),
                "rank": ctx.get("rank"),
                "holding_days": int(holding_days.get(sec, 0)),
                "reason": _recommendation_reason(sec, prev_w, target_w, exit_reasons, ctx),
                "status": "PENDING",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["action", "delta_weight"], ascending=[True, False]
    ).reset_index(drop=True)


def build_portfolio_snapshot(
    *,
    as_of_ts: pd.Timestamp,
    recommendations: pd.DataFrame,
    holding_days: dict[str, int],
    capital: float,
) -> pd.DataFrame:
    if recommendations.empty:
        return pd.DataFrame()
    snap = recommendations.copy()
    snap["snapshot_time"] = as_of_ts
    snap["market_value"] = snap["target_weight"] * capital
    snap["unrealized_pnl"] = np.nan
    snap["holding_days"] = snap.apply(
        lambda row: int(holding_days.get(str(row["security_id"]), 0)) + 1
        if abs(float(row["target_weight"])) > 1e-12
        else 0,
        axis=1,
    )
    return snap[
        [
            "as_of_date",
            "snapshot_time",
            "security_id",
            "current_weight",
            "target_weight",
            "target_shares",
            "last_price",
            "market_value",
            "unrealized_pnl",
            "signal_score",
            "rank",
            "holding_days",
            "reason",
        ]
    ]


def _weights_to_targets(
    *,
    weights: dict[str, float],
    as_of_ts: pd.Timestamp,
    prices: dict[str, float],
    capital: float,
    method: str,
) -> pd.DataFrame:
    rows = []
    for sec, weight in sorted(weights.items()):
        rows.append(
            {
                "rebalance_time": as_of_ts,
                "security_id": sec,
                "target_weight": float(weight),
                "target_shares": _target_shares(weight, prices.get(sec), capital),
                "construction_method": method,
                "pre_risk": False,
            }
        )
    return pd.DataFrame(rows)


def _signal_context(signals: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    ranked = signals.sort_values("signal_score", ascending=False).reset_index(drop=True)
    out: dict[str, dict[str, float | int]] = {}
    for i, row in ranked.iterrows():
        out[str(row["security_id"])] = {
            "signal_score": float(row["signal_score"]),
            "rank": int(i + 1),
        }
    return out


def _classify_action(prev_w: float, target_w: float) -> str:
    eps = 1e-8
    if abs(prev_w) <= eps and target_w > eps:
        return "BUY"
    if prev_w > eps and abs(target_w) <= eps:
        return "SELL"
    if target_w - prev_w > eps:
        return "INCREASE"
    if prev_w - target_w > eps:
        return "REDUCE"
    return "HOLD"


def _recommendation_reason(
    sec: str,
    prev_w: float,
    target_w: float,
    exit_reasons: dict[str, str],
    signal_context: dict[str, float | int],
) -> str:
    if sec in exit_reasons:
        return exit_reasons[sec]
    action = _classify_action(prev_w, target_w)
    if action == "BUY":
        return "new_entry"
    if action == "SELL":
        return "exit_rank_or_untradable"
    if action == "INCREASE":
        return "increase_weight"
    if action == "REDUCE":
        return "reduce_weight"
    if signal_context:
        return "held_from_previous"
    return "unchanged"


def _target_shares(weight: float, price: float | None, capital: float) -> int:
    if price is None or pd.isna(price) or price <= 0:
        return 0
    return int(round(float(weight) * float(capital) / float(price)))


def _normalize_float_map(values: dict[str, float] | pd.Series | None) -> dict[str, float]:
    if values is None:
        return {}
    if isinstance(values, pd.Series):
        iterable = values.items()
    else:
        iterable = values.items()
    out = {}
    for key, value in iterable:
        if pd.isna(value):
            continue
        val = float(value)
        if abs(val) > 1e-12:
            out[str(key)] = val
    return out


def _normalize_int_map(values: dict[str, int] | pd.Series | None) -> dict[str, int]:
    if values is None:
        return {}
    if isinstance(values, pd.Series):
        iterable = values.items()
    else:
        iterable = values.items()
    out = {}
    for key, value in iterable:
        if pd.isna(value):
            continue
        out[str(key)] = int(value)
    return out
