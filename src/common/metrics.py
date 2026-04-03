"""Common quantitative metrics used across monitoring and evaluation layers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def information_coefficient(
    alpha: pd.Series,
    forward_return: pd.Series,
) -> float:
    """Pearson IC between alpha values and forward returns."""
    valid = alpha.notna() & forward_return.notna()
    if valid.sum() < 5:
        return np.nan
    return alpha[valid].corr(forward_return[valid])


def rank_information_coefficient(
    alpha: pd.Series,
    forward_return: pd.Series,
) -> float:
    """Spearman rank IC between alpha values and forward returns."""
    valid = alpha.notna() & forward_return.notna()
    if valid.sum() < 5:
        return np.nan
    corr, _ = stats.spearmanr(alpha[valid], forward_return[valid])
    return float(corr)


def rolling_ic(
    alpha_df: pd.DataFrame,
    return_series: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Compute rolling cross-sectional IC over a time-indexed panel."""
    results = {}
    dates = alpha_df.index.get_level_values("tradetime").unique().sort_values()
    for i in range(window, len(dates)):
        window_dates = dates[i - window : i]
        mask = alpha_df.index.get_level_values("tradetime").isin(window_dates)
        a = alpha_df[mask].values.flatten()
        r = return_series[mask].values.flatten()
        valid = ~(np.isnan(a) | np.isnan(r))
        if valid.sum() >= 5:
            results[dates[i]] = np.corrcoef(a[valid], r[valid])[0, 1]
        else:
            results[dates[i]] = np.nan
    return pd.Series(results)


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, periods: int = 252) -> float:
    """Annualized Sharpe ratio."""
    excess = returns - risk_free / periods
    if excess.std() == 0:
        return 0.0
    return float(np.sqrt(periods) * excess.mean() / excess.std())


def max_drawdown(cumulative_returns: pd.Series) -> float:
    """Maximum drawdown from a cumulative return series."""
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return float(drawdown.min())


def hit_rate(predictions: pd.Series, actuals: pd.Series) -> float:
    """Directional accuracy (fraction of correct sign predictions)."""
    valid = predictions.notna() & actuals.notna()
    if valid.sum() == 0:
        return np.nan
    correct = (np.sign(predictions[valid]) == np.sign(actuals[valid]))
    return float(correct.mean())


def profit_factor(pnl_series: pd.Series) -> float:
    """Ratio of gross profits to gross losses."""
    gains = pnl_series[pnl_series > 0].sum()
    losses = abs(pnl_series[pnl_series < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def turnover(weights_prev: pd.Series, weights_curr: pd.Series) -> float:
    """Portfolio turnover between two weight snapshots."""
    aligned = pd.DataFrame({"prev": weights_prev, "curr": weights_curr}).fillna(0)
    return float(aligned["prev"].sub(aligned["curr"]).abs().sum() / 2)


def ks_test_drift(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Two-sample KS test for distribution shift."""
    ref_clean = reference[~np.isnan(reference)]
    cur_clean = current[~np.isnan(current)]
    if len(ref_clean) < 5 or len(cur_clean) < 5:
        return 0.0, 1.0
    stat, p_value = stats.ks_2samp(ref_clean, cur_clean)
    return float(stat), float(p_value)


def winsorize(series: pd.Series, n_sigma: float = 3.0) -> pd.Series:
    """Winsorize a series to ±n_sigma standard deviations."""
    mu = series.mean()
    sigma = series.std()
    if sigma == 0 or np.isnan(sigma):
        return series
    lower = mu - n_sigma * sigma
    upper = mu + n_sigma * sigma
    return series.clip(lower=lower, upper=upper)


def cross_sectional_zscore(df: pd.DataFrame, value_col: str) -> pd.Series:
    """Z-score normalize a column cross-sectionally (per timestamp)."""
    grouped = df.groupby("tradetime")[value_col]
    mu = grouped.transform("mean")
    sigma = grouped.transform("std")
    sigma = sigma.replace(0, np.nan)
    return (df[value_col] - mu) / sigma
