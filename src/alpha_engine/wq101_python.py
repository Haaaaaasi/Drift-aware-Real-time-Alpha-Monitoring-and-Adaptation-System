"""
Pure-Python WQ101 Alpha Engine.

Port of Kakushadze (2016) "101 Formulaic Alphas" (arXiv:1601.00991) to
vectorised pandas.  Panel layout: index=tradetime, columns=security_id —
mirrors DolphinDB panel matrix orientation so output schema is identical.

Entry point
-----------
    from src.alpha_engine.wq101_python import compute_wq101_alphas
    alpha_panel = compute_wq101_alphas(bars_df, alpha_ids=["wq001", "wq040"])
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.common.logging import get_logger

logger = get_logger("wq101_python")

# ---------------------------------------------------------------------------
# Operator library
# ---------------------------------------------------------------------------

def _rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank (0–1) per row."""
    return df.rank(axis=1, pct=True, na_option="keep")


def _w(d) -> int:
    """Round float window parameter from WQ101 paper to nearest integer, min 1."""
    return max(1, int(round(d)))


def _delay(df: pd.DataFrame, d) -> pd.DataFrame:
    return df.shift(_w(d))


def _delta(df: pd.DataFrame, d) -> pd.DataFrame:
    return df - df.shift(_w(d))


def _sign(df: pd.DataFrame) -> pd.DataFrame:
    return np.sign(df)


def _log(df: pd.DataFrame) -> pd.DataFrame:
    return np.log(df.clip(lower=1e-12))


def _abs(df: pd.DataFrame) -> pd.DataFrame:
    return df.abs()


def _signedpower(df: pd.DataFrame, e: float) -> pd.DataFrame:
    return np.sign(df) * (df.abs() ** e)


def _scale(df: pd.DataFrame, a: float = 1.0) -> pd.DataFrame:
    s = df.abs().sum(axis=1).replace(0.0, np.nan)
    return df.div(s, axis=0) * a


def _ts_sum(df: pd.DataFrame, d) -> pd.DataFrame:
    return df.rolling(_w(d), min_periods=1).sum()


def _ts_mean(df: pd.DataFrame, d) -> pd.DataFrame:
    return df.rolling(_w(d), min_periods=1).mean()


def _ts_min(df: pd.DataFrame, d) -> pd.DataFrame:
    return df.rolling(_w(d), min_periods=1).min()


def _ts_max(df: pd.DataFrame, d) -> pd.DataFrame:
    return df.rolling(_w(d), min_periods=1).max()


def _ts_stddev(df: pd.DataFrame, d) -> pd.DataFrame:
    return df.rolling(_w(d), min_periods=2).std()


def _ts_rank(df: pd.DataFrame, d) -> pd.DataFrame:
    return df.rolling(_w(d), min_periods=1).apply(
        lambda arr: (arr.argsort().argsort()[-1] + 1) / len(arr), raw=True
    )


def _ts_argmax(df: pd.DataFrame, d) -> pd.DataFrame:
    return df.rolling(_w(d), min_periods=1).apply(
        lambda arr: len(arr) - 1 - int(np.argmax(arr)), raw=True
    )


def _ts_argmin(df: pd.DataFrame, d) -> pd.DataFrame:
    return df.rolling(_w(d), min_periods=1).apply(
        lambda arr: len(arr) - 1 - int(np.argmin(arr)), raw=True
    )


def _ts_product(df: pd.DataFrame, d) -> pd.DataFrame:
    return df.rolling(_w(d), min_periods=1).apply(np.prod, raw=True)


def _correlation(x: pd.DataFrame, y: pd.DataFrame, d) -> pd.DataFrame:
    d = _w(d)
    result = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    for col in x.columns:
        result[col] = x[col].rolling(d, min_periods=2).corr(y[col])
    return result


def _covariance(x: pd.DataFrame, y: pd.DataFrame, d) -> pd.DataFrame:
    d = _w(d)
    result = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    for col in x.columns:
        result[col] = x[col].rolling(d, min_periods=2).cov(y[col])
    return result


def _decay_linear(df: pd.DataFrame, d) -> pd.DataFrame:
    d = _w(d)
    w = np.arange(1, d + 1, dtype=float)
    w /= w.sum()
    return df.rolling(d, min_periods=1).apply(
        lambda arr: float(np.dot(arr, w[-len(arr):]) / w[-len(arr):].sum()),
        raw=True,
    )


def _indneutralize(x: pd.DataFrame, ind: pd.DataFrame) -> pd.DataFrame:
    xs = x.stack()
    gs = ind.stack().astype(int)
    tmp = pd.DataFrame({"x": xs, "g": gs}).dropna()
    date_lv = tmp.index.get_level_values(0)
    tmp["mean"] = tmp.groupby([date_lv, tmp["g"]])["x"].transform("mean")
    tmp["neu"] = tmp["x"] - tmp["mean"]
    return tmp["neu"].unstack().reindex(index=x.index, columns=x.columns)


def _adv(vol: pd.DataFrame, d: int) -> pd.DataFrame:
    return _ts_mean(vol, d)


def _where(cond, a, b):
    """Element-wise conditional — works with DataFrames or scalars."""
    if isinstance(cond, pd.DataFrame):
        return b if not isinstance(a, pd.DataFrame) and not isinstance(b, pd.DataFrame) \
            else pd.DataFrame(np.where(cond.values, _val(a, cond), _val(b, cond)),
                              index=cond.index, columns=cond.columns)
    return pd.DataFrame(np.where(cond.values, _val(a, cond), _val(b, cond)),
                        index=cond.index, columns=cond.columns)


def _val(x, ref: pd.DataFrame):
    if isinstance(x, pd.DataFrame):
        return x.reindex_like(ref).values
    return x


# ---------------------------------------------------------------------------
# Alpha formulas  (signatures mirror alpha_batch.dos dispatch)
# ---------------------------------------------------------------------------

def _wq001(close):
    ret = close.pct_change(1)
    inner = _ts_stddev(ret, 20).where(ret < 0, close)
    return _rank(_ts_argmax(_signedpower(inner, 2), 5)) - 0.5


def _wq002(vol, close, open_):
    x = _rank(_delta(_log(vol), 2))
    y = _rank((close - open_) / open_.replace(0, np.nan))
    return -1 * _correlation(x, y, 6)


def _wq003(vol, open_):
    return -1 * _correlation(_rank(open_), _rank(vol), 10)


def _wq004(low):
    return -1 * _ts_rank(_rank(low), 9)


def _wq005(vwap, open_, close):
    return _rank(open_ - _ts_sum(vwap, 10) / 10) * (-1 * _abs(_rank(close - vwap)))


def _wq006(vol, open_):
    return -1 * _correlation(open_, vol, 10)


def _wq007(vol, close):
    adv20 = _adv(vol, 20)
    cond = adv20 < vol
    a = (-1 * _ts_rank(_abs(_delta(close, 7)), 60)) * _sign(_delta(close, 7))
    return pd.DataFrame(
        np.where(cond.values, a.values, -1.0), index=close.index, columns=close.columns
    )


def _wq008(open_, close):
    ret = close.pct_change(1)
    x = _ts_sum(open_, 5) * _ts_sum(ret, 5)
    return -1 * _rank(x - _delay(x, 10))


def _wq009(close):
    d1 = _delta(close, 1)
    inner = pd.DataFrame(
        np.where(_ts_max(d1, 5).values < 0, d1.values, -d1.values),
        index=close.index, columns=close.columns,
    )
    return pd.DataFrame(
        np.where(_ts_min(d1, 5).values > 0, d1.values, inner.values),
        index=close.index, columns=close.columns,
    )


def _wq010(close):
    d1 = _delta(close, 1)
    inner = pd.DataFrame(
        np.where(_ts_max(d1, 4).values < 0, d1.values, -d1.values),
        index=close.index, columns=close.columns,
    )
    tmp = pd.DataFrame(
        np.where(_ts_min(d1, 4).values > 0, d1.values, inner.values),
        index=close.index, columns=close.columns,
    )
    return _rank(tmp)


def _wq011(vwap, vol, close):
    vc = vwap - close
    return (_rank(_ts_max(vc, 3)) + _rank(_ts_min(vc, 3))) * _rank(_delta(vol, 3))


def _wq012(vol, close):
    return _sign(_delta(vol, 1)) * (-1 * _delta(close, 1))


def _wq013(vol, close):
    return -1 * _rank(_covariance(_rank(close), _rank(vol), 5))


def _wq014(vol, open_, close):
    ret = close.pct_change(1)
    return (-1 * _rank(_delta(ret, 3))) * _correlation(open_, vol, 10)


def _wq015(vol, high):
    return -1 * _ts_sum(_rank(_correlation(_rank(high), _rank(vol), 3)), 3)


def _wq016(vol, high):
    return -1 * _rank(_covariance(_rank(high), _rank(vol), 5))


def _wq017(vol, close):
    adv20 = _adv(vol, 20)
    a = -1 * _rank(_ts_rank(close, 10))
    b = _rank(_delta(_delta(close, 1), 1))
    c = _rank(_ts_rank(vol / adv20.replace(0, np.nan), 5))
    return a * b * c


def _wq018(close, open_):
    return -1 * _rank(
        _ts_stddev(_abs(close - open_), 5) + (close - open_) + _correlation(close, open_, 10)
    )


def _wq019(close):
    ret = close.pct_change(1)
    s = _sign(close - _delay(close, 7)) + _sign(_delta(close, 7))
    return (-1 * s) * (1 + _rank(1 + _ts_sum(ret, 250))) ** 2


def _wq020(open_, close, high, low):
    return ((-1 * _rank(open_ - _delay(high, 1)))
            * _rank(open_ - _delay(close, 1))
            * _rank(open_ - _delay(low, 1)))


def _wq021(close, vol):
    sma8 = _ts_mean(close, 8)
    std8 = _ts_stddev(close, 8)
    sma2 = _ts_mean(close, 2)
    adv20 = _adv(vol, 20)
    cond1 = (sma8 + std8) < sma2
    cond2 = sma2 < (sma8 - std8)
    cond3 = (vol / adv20.replace(0, np.nan)) >= 1
    base = pd.DataFrame(np.where(cond3.values, 1.0, -1.0), index=close.index, columns=close.columns)
    base = pd.DataFrame(np.where(cond2.values, 1.0, base.values), index=close.index, columns=close.columns)
    return pd.DataFrame(np.where(cond1.values, -1.0, base.values), index=close.index, columns=close.columns)


def _wq022(close, vol, high):
    return -1 * _delta(_correlation(high, vol, 5), 5) * _rank(_ts_stddev(close, 20))


def _wq023(high):
    cond = _ts_sum(high, 20) / 20 < high
    res = (-1 * _delta(high, 2))
    return pd.DataFrame(np.where(cond.values, res.values, 0.0), index=high.index, columns=high.columns)


def _wq024(close):
    sma100 = _ts_sum(close, 100) / 100
    cond = (_delta(sma100, 100) / _delay(close, 100).replace(0, np.nan)) <= 0.05
    a = -(close - _ts_min(close, 100))
    b = -1 * _delta(close, 3)
    return pd.DataFrame(np.where(cond.values, a.values, b.values), index=close.index, columns=close.columns)


def _wq025(close, vol, high, vwap):
    ret = close.pct_change(1)
    adv20 = _adv(vol, 20)
    return _rank(-1 * ret * adv20 * vwap * (high - close))


def _wq026(vol, high):
    return -1 * _ts_max(_correlation(_ts_rank(vol, 5), _ts_rank(high, 5), 5), 3)


def _wq027(vol, vwap):
    x = _ts_mean(_correlation(_rank(vol), _rank(vwap), 6), 2)
    return pd.DataFrame(
        np.where(_rank(x).values > 0.5, -1.0, 1.0), index=vol.index, columns=vol.columns
    )


def _wq028(vol, high, low, close):
    adv20 = _adv(vol, 20)
    return _scale(_correlation(adv20, low, 5) + (high + low) / 2 - close)


def _wq029(close):
    ret = close.pct_change(1)
    inner = _ts_sum(_rank(_rank(-1 * _rank(_delta(close - 1, 5)))), 2)
    a = _ts_min(_rank(_rank(_scale(_log(inner.clip(lower=1e-12))))), 5)
    b = _ts_rank(_delay(-1 * ret, 6), 5)
    return a + b


def _wq030(vol, close):
    ret = close.pct_change(1)
    s = _sign(close - _delay(close, 1)) + _sign(ret - _delay(ret, 1))
    denom = _ts_sum(vol, 20).replace(0, np.nan)
    return (1 - _rank(s)) * _ts_sum(vol, 5) / denom


def _wq031(vol, close, low):
    adv12 = _adv(vol, 12)
    a = _rank(_rank(_rank(_decay_linear(-1 * _rank(_rank(_delta(close, 10))), 10))))
    b = _rank(-1 * _delta(close, 3))
    c = _sign(_scale(_correlation(adv12, low, 12)))
    return a + b + c


def _wq032(close, vwap):
    return _scale(_ts_mean(close, 7) - close) + 20 * _scale(_correlation(vwap, _delay(close, 5), 230))


def _wq033(open_, close):
    return _rank(-(1 - (open_ / close.replace(0, np.nan))))


def _wq034(close):
    std25 = _ts_stddev(close, 5).replace(0, np.nan)
    a = 1 - _rank(_ts_stddev(close, 2) / std25)
    b = 1 - _rank(_delta(close, 1))
    return _rank(a + b)


def _wq035(vol, close, high, low):
    ret = close.pct_change(1)
    a = _ts_rank(vol, 32) * (1 - _ts_rank(close + high - low, 16))
    b = 1 - _ts_rank(ret, 32)
    return a * b


def _wq036(vol, open_, close, vwap):
    adv20 = _adv(vol, 20)
    ret = close.pct_change(1)
    a = 2.21 * _rank(_correlation(close - open_, _delay(vol, 1), 15))
    b = 0.7 * _rank(open_ - close)
    c = 0.73 * _rank(_ts_rank(_delay(-1 * ret, 6), 5))
    d = _rank(_abs(_correlation(vwap, adv20, 6)))
    e = 0.6 * _rank(_ts_mean(close, 200) - open_) * _rank(close - open_)
    return a + b + c + d + e


def _wq037(open_, close):
    return _rank(_correlation(_delay(open_ - close, 1), close, 200)) + _rank(open_ - close)


def _wq038(open_, close):
    return -1 * _rank(_ts_rank(close, 10)) * _rank(close / open_.replace(0, np.nan))


def _wq039(vol, close):
    adv20 = _adv(vol, 20)
    ret = close.pct_change(1)
    a = -1 * _rank(_delta(close, 7)) * (1 - _rank(_decay_linear(vol / adv20.replace(0, np.nan), 9)))
    b = 1 + _rank(_ts_sum(ret, 250))
    return a * b


def _wq040(vol, high):
    return -1 * _rank(_ts_stddev(high, 10)) * _correlation(high, vol, 10)


def _wq041(high, low, vwap):
    return _signedpower((high * low) ** 0.5 - vwap, 1)


def _wq042(vwap, close):
    denom = _rank(vwap + close).replace(0, np.nan)
    return _rank(vwap - close) / denom


def _wq043(vol, close):
    adv20 = _adv(vol, 20)
    return _ts_rank(vol / adv20.replace(0, np.nan), 20) * _ts_rank(-1 * _delta(close, 7), 8)


def _wq044(vol, high):
    return -1 * _correlation(high, _rank(vol), 5)


def _wq045(vol, close):
    x = _ts_mean(_delay(close, 5), 20)
    return -1 * _rank(
        _ts_mean(close, 5) * _rank(_correlation(close, vol, 2)) * _rank(_ts_mean(x, 2))
    )


def _wq046(close):
    a = (_delay(close, 20) - _delay(close, 10)) / 10
    b = (_delay(close, 10) - close) / 10
    diff = a - b
    d1 = _delta(close, 1)
    result = -1 * d1
    result = pd.DataFrame(
        np.where(_sign(diff).values == 1, 1.0, result.values),
        index=close.index, columns=close.columns,
    )
    return pd.DataFrame(
        np.where(diff.values > 0.25, -1.0, result.values),
        index=close.index, columns=close.columns,
    )


def _wq047(vol, close, high, vwap):
    adv20 = _adv(vol, 20)
    return (
        _rank(1 / close) * vol / adv20
        * high * _rank(high - close)
        / (_ts_mean(high, 5) / _ts_mean(high, 5).replace(0, np.nan))
        - _rank(vwap - _delay(vwap, 5))
    )


def _wq048(close, ind):
    neu = _indneutralize(_rank(close.pct_change(1)), ind)
    return (-1 * _rank(neu)) * _rank(_delta(close, 1))


def _wq049(close):
    sma12 = _ts_mean(close, 12)
    cond = ((_delay(close, 20) - _delay(close, 10)) / 10) > ((_delay(close, 10) - close) / 10)
    return pd.DataFrame(
        np.where(cond.values, 1.0, -1.0), index=close.index, columns=close.columns
    )


def _wq050(vol, vwap):
    return -1 * _ts_max(
        _rank(_correlation(_rank(vol), _rank(vwap), 5)), 5
    ) * _ts_min(_rank(_correlation(_rank(vol), _rank(vwap), 5)), 5)


def _wq051(close):
    a = (_delay(close, 20) - _delay(close, 10)) / 10
    b = (_delay(close, 10) - close) / 10
    diff = a - b
    return pd.DataFrame(
        np.where(diff.values > 0.05, -1.0, 1.0), index=close.index, columns=close.columns
    )


def _wq052(vol, close, low):
    ret = close.pct_change(1)
    return (
        (_ts_min(low, 5) - _delay(_ts_min(low, 5), 5))
        * _rank((_ts_sum(ret, 240) - _ts_sum(ret, 20)) / 220)
        * _ts_rank(vol, 5)
    )


def _wq053(close, high, low):
    h_l = (high - low).replace(0, np.nan)
    return -1 * _delta(
        (close - low) / h_l - (high - close) / h_l,
        9
    )


def _wq054(open_, close, high, low):
    denom = (low - high).replace(0, np.nan)
    return -1 * (low - close) * (open_ ** 5) / (denom * (close ** 5)).replace(0, np.nan)


def _wq055(vol, close, high, low):
    h_range = (high - low).replace(0, np.nan)
    return -1 * _correlation(
        _rank((close - _ts_min(low, 12)) / ((_ts_max(high, 12) - _ts_min(low, 12)).replace(0, np.nan))),
        _rank(vol),
        6,
    )


def _wq056(close, cap):
    ret = close.pct_change(1)
    return 0 - (1 * (
        _rank(_ts_mean(ret, 10))
        * _rank(_rank(cap))
        * _rank(_rank(_rank(close)))
    ))


def _wq057(close, vwap):
    return 0 - (1 * (
        (_ts_argmax(close, 30) < _ts_argmax(_indneutralize_simple(vwap, close), 30)).astype(float) - 0.5
    ))


def _indneutralize_simple(x, ref):
    return x - x.mean(axis=1).values.reshape(-1, 1) * pd.DataFrame(
        np.ones_like(ref.values), index=ref.index, columns=ref.columns
    )


def _wq058(vol, vwap, ind):
    return -1 * _ts_rank(_decay_linear(_correlation(_indneutralize(_rank(vwap), ind), _rank(vol), 3), 7), 5)


def _wq059(vol, vwap, ind):
    return -1 * _ts_rank(_decay_linear(_correlation(
        _indneutralize(((vwap * 0.728317) + (vwap * (1 - 0.728317))), ind),
        vol,
        4,
    ), 16), 8)


def _wq060(vol, close, high, low):
    h_l = (high - low).replace(0, np.nan)
    x = _rank((2 * close - low - high) / h_l * vol)
    return -1 * (2 * _scale(x, 1) - _scale(x, 1))


def _wq061(vol, vwap):
    adv180 = _adv(vol, 180)
    return (
        _rank(_ts_min(vol, 16)) < _rank(_correlation(vwap, adv180, 18))
    ).astype(float)


def _wq062(vol, vwap, open_, high, low):
    adv20 = _adv(vol, 20)
    return (
        (_rank(_correlation(vwap, _ts_sum(adv20, 22), 9)) < _rank((low * 0.1 + vwap * 0.9 - high)))
    ).astype(float) * -1


def _wq063(vol, open_, close, vwap, ind, low):
    adv180 = _adv(vol, 180)
    return (
        _rank(_decay_linear(
            _delta(_indneutralize(_rank(vwap), ind) * _rank(vol), 2),
            8,
        ))
        - _rank(_decay_linear(
            _correlation(((low * 0.298 + vwap * (1 - 0.298)) - _ts_mean(adv180, 37)),
                         (low * 0.298 + vwap * (1 - 0.298)),
                         19),
            16,
        ))
    )


def _wq064(vol, vwap, open_, high, low):
    adv120 = _adv(vol, 120)
    return (
        _rank(_correlation(_ts_sum((open_ * 0.178 + low * (1 - 0.178)), 12),
                           _ts_sum(adv120, 12), 16))
        < _rank(_delta(((high + low) / 2 * 0.178 + vwap * (1 - 0.178)), 3.65))
    ).astype(float) * -1


def _wq065(vol, vwap, open_):
    adv60 = _adv(vol, 60)
    return (
        _rank(_correlation((open_ * 0.00817205 + vwap * (1 - 0.00817205)), _ts_sum(adv60, 9), 6))
        < _rank(open_ - _ts_min(open_, 14))
    ).astype(float) * -1


def _wq066(vwap, high, low, open_):
    return (
        _rank(_decay_linear(_delta(vwap, 3.51013), 7))
        + _ts_rank(_decay_linear(((low * 0.96633 + low * (1 - 0.96633) - vwap) / (open_ - (high + low) / 2 + 1e-9).replace(0, np.nan)), 11), 7)
    ) * -1


def _wq067(vol, high, vwap, ind):
    adv20 = _adv(vol, 20)
    return (
        _rank(_ts_argmax(_indneutralize(high, ind), 14))
        < _rank(_rank(_correlation(_indneutralize(vwap * -1, ind), _ts_mean(adv20, 30), 15)))
    ).astype(float) * -1


def _wq068(vol, close, high, low):
    adv15 = _adv(vol, 15)
    return (
        _ts_rank(_correlation(_rank(high), _rank(adv15), 9), 14)
        < _rank(_delta(close * 0.518371 + low * (1 - 0.518371), 1.06119))
    ).astype(float) * -1


def _wq069(vol, close, vwap, ind):
    adv20 = _adv(vol, 20)
    return (
        _rank(_ts_max(
            _delta(_indneutralize(vwap, ind), 2.72412),
            4.79521,
        ))
        < _rank(_rank(close / _ts_mean(adv20, 4)))
    ).astype(float) * -1


def _wq070(vol, close, vwap, ind):
    adv50 = _adv(vol, 50)
    return (
        _rank(_delta(_indneutralize(vwap, ind), 1.29456))
        < _rank((_correlation(adv50, close, 17.8717) ** 2))
    ).astype(float) * -1


def _wq071(vol, vwap, close, open_, low):
    adv180 = _adv(vol, 180)
    a = _ts_rank(_decay_linear(_correlation(
        _ts_rank(close, 3), _ts_rank(adv180, 12), 18,
    ), 4), 16)
    b = _ts_rank(_decay_linear(_rank(low + open_ - (vwap + vwap)), 16), 4)
    return pd.DataFrame(
        np.where(a.values < b.values, a.values, b.values),
        index=close.index, columns=close.columns,
    )


def _wq072(vol, vwap, high, low):
    adv40 = _adv(vol, 40)
    return _rank(_decay_linear(
        _correlation(((high + low) / 2), adv40, 9),
        10,
    )) / _rank(_decay_linear(
        _correlation(_ts_rank(vwap, 4), _ts_rank(vol, 9), 7),
        3,
    )).replace(0, np.nan)


def _wq073(vwap, open_, low):
    a = _rank(_decay_linear(_delta(vwap, 4.72775), 2.91864))
    b = _ts_rank(_decay_linear(
        (_delta(open_ * 0.147155 + low * (1 - 0.147155), 2.03608)
         / (open_ * 0.147155 + low * (1 - 0.147155) + 1e-9)),
        3.33829,
    ), 16.7411)
    return pd.DataFrame(
        np.where(a.values < b.values, a.values, b.values),
        index=vwap.index, columns=vwap.columns,
    ) * -1


def _wq074(vol, vwap, close, high):
    adv30 = _adv(vol, 30)
    return (
        _rank(_correlation(close, _ts_sum(adv30, 37), 15))
        < _rank(_correlation(_rank(high * 0.0261661 + vwap * (1 - 0.0261661)), _rank(vol), 11))
    ).astype(float) * -1


def _wq075(vol, vwap, low):
    adv50 = _adv(vol, 50)
    return (
        _rank(_correlation(vwap, vol, 4)) < _rank(_correlation(_rank(low), _rank(adv50), 12))
    ).astype(float)


def _wq076(vol, low, vwap, ind):
    a = _ts_max(_rank(_decay_linear(_delta(vwap, 1), 12)), 14)
    b = _ts_min(_rank(_ts_rank(_indneutralize(
        _rank(_rank(_delta(_log(vol), 3))), ind,
    ), 9)), 14)
    return (a - b) * _sign(_delta(vwap, 5))


# kept for dispatch alias compatibility
_wq076_fixed = _wq076


def _wq077(vol, vwap, high, low):
    adv40 = _adv(vol, 40)
    a = _rank(_decay_linear(((high + low) / 2 + high - (vwap + high)), 20))
    b = _rank(_decay_linear(_correlation(((high + low) / 2), adv40, 3), 6))
    return pd.DataFrame(
        np.where(a.values < b.values, a.values, b.values),
        index=vol.index, columns=vol.columns,
    )


def _wq078(vol, vwap, low):
    adv40 = _adv(vol, 40)
    return (
        _rank(_ts_sum(
            _correlation(_rank(low * 0.352233 + vwap * (1 - 0.352233)), _rank(adv40), 12),
            14,
        ))
        * _rank(_rank(_rank(_correlation(
            _ts_rank(vwap, 20), _ts_rank(vol, 4), 7,
        ))))
    )


def _wq079(vol, open_, close, vwap, ind):
    adv150 = _adv(vol, 150)
    return (
        _rank(_delta(_indneutralize(
            ((close * 0.60733 + open_ * (1 - 0.60733))),
            ind,
        ), 1))
        < _rank(_correlation(
            _ts_rank(vwap, 4), _ts_rank(adv150, 9), 15,
        ))
    ).astype(float)


def _wq080(vol, open_, high, ind):
    adv10 = _adv(vol, 10)
    return (
        _rank(_sign(_delta(_indneutralize(open_ * 0.868128 + high * (1 - 0.868128), ind), 4)))
        * (-1 * _ts_rank(_abs(_correlation(adv10, open_, 5)), 7))
    )


def _wq081(vol, vwap):
    adv10 = _adv(vol, 10)
    return (
        _rank(_log(
            _ts_product(_rank((_rank(_correlation(vwap, _ts_sum(adv10, 49.6054), 8.47743)) ** 4)), 14.9655)
        ))
        - _rank(_correlation(_rank(vwap), _rank(vol), 5.07914))
    )


def _wq082(vol, open_, ind):
    adv10 = _adv(vol, 10)
    return pd.DataFrame(
        np.where(
            (_ts_min(_rank(_decay_linear(_delta(open_, 1), 15)), 7).values
             < _ts_rank(_decay_linear(
                 _correlation(_indneutralize(_rank(open_), ind), _rank(adv10), 14), 13,
             ), 9).values),
            -1.0,
            1.0,
        ),
        index=vol.index, columns=vol.columns,
    )


def _wq083(vol, vwap, close, high, low):
    return (
        _rank(_delay(
            ((high - low) / _ts_mean(close, 5)).replace(0, np.nan),
            2,
        ))
        * _rank(_rank(vol))
        / (high - low)
        / ((_ts_mean(close, 5) - close).replace(0, np.nan))
    )


def _wq084(vwap, close):
    return _signedpower(
        _ts_rank(vwap - _ts_max(vwap, 14.0261), 13.9605),
        _delta(close, 5),
    )


def _wq085(vol, close, high, low):
    adv30 = _adv(vol, 30)
    n = (high - _ts_min(high, 22.4101)) / (0.0001 + _ts_max(close, 84.8193) - _ts_min(close, 84.8193)).replace(0, np.nan)
    return (_rank(n) ** _rank(_correlation(vol, adv30, 17.4842)))


def _wq086(vol, vwap, open_, close):
    adv20 = _adv(vol, 20)
    return (
        _ts_rank(_correlation(close, 20 * _rank(adv20) * _rank(vwap), 15), 20) < 0.5
    ).astype(float) * -1


def _wq087(vol, close, vwap, ind):
    adv81 = _adv(vol, 81)
    a = _rank(_decay_linear(
        _delta(_indneutralize(
            (close * 0.369701 + vwap * (1 - 0.369701)), ind,
        ), 1.91233),
        2.65461,
    ))
    b = _ts_rank(_decay_linear(
        _abs(_correlation(
            _indneutralize(adv81, ind), close, 13.4132,
        )),
        4.89768,
    ), 14.4535)
    return pd.DataFrame(
        np.where(a.values < b.values, a.values, b.values),
        index=close.index, columns=close.columns,
    ) * -1


def _wq088(vol, open_, close, high, low):
    adv60 = _adv(vol, 60)
    a = _rank(_decay_linear(
        (low - close) / (open_ - (high + low) / 2 + 1e-9).replace(0, np.nan),
        7,
    ))
    b = _ts_rank(_decay_linear(_correlation(
        _ts_rank(close, 4), _ts_rank(adv60, 9), 6,
    ), 13), 8)
    return pd.DataFrame(
        np.where(a.values < b.values, a.values, b.values),
        index=close.index, columns=close.columns,
    )


def _wq089(vol, low, vwap, ind):
    adv10 = _adv(vol, 10)
    return (
        2 * (_ts_rank(_decay_linear(_correlation(
            _indneutralize((low * 0.967285 + vwap * (1 - 0.967285)), ind),
            adv10,
            7,
        ), 6), 4))
        - _ts_rank(_decay_linear(_delta(
            _indneutralize(vwap, ind), 3,
        ), 10), 15)
    )


def _wq090(vol, low, close, ind):
    adv40 = _adv(vol, 40)
    return (
        _rank(_correlation(
            _rank(low * 0.967285 + close * (1 - 0.967285)),
            _rank(adv40),
            6,
        ))
        < _rank(_rank(_indneutralize(
            _rank(close), ind,
        )))
    ).astype(float) * -1


def _wq091(vol, close, vwap, ind):
    adv30 = _adv(vol, 30)
    return (
        _ts_rank(_decay_linear(
            _decay_linear(
                _correlation(_indneutralize(close, ind), vol, 9.37637),
                6.21347,
            ),
            18.1713,
        ), 16.2956)
        - _rank(_decay_linear(_correlation(
            vwap, adv30, 3.92131,
        ), 7.58555))
    )


def _wq092(vol, open_, close, high, low):
    adv30 = _adv(vol, 30)
    a = _ts_rank(_decay_linear(
        (high + low + close) / 3 / (open_ + 1e-9).replace(0, np.nan) - 1,
        15,
    ), 19)
    b = _ts_rank(_decay_linear(
        _rank(_correlation(close, adv30, 9)),
        10,
    ), 20)
    return pd.DataFrame(
        np.where(a.values < b.values, a.values, b.values),
        index=close.index, columns=close.columns,
    )


def _wq093(vol, close, vwap, ind):
    adv81 = _adv(vol, 81)
    return (
        _ts_rank(_decay_linear(
            _correlation(_indneutralize(vwap, ind), adv81, 17.4193),
            19.848,
        ), 7.54455)
        / _rank(_decay_linear(
            _delta(((close * 0.524434) + (vwap * (1 - 0.524434))), 2.77377),
            16.2664,
        )).replace(0, np.nan)
    )


def _wq094(vol, vwap):
    adv60 = _adv(vol, 60)
    return (
        0 - (1 * (
            _rank((_scale(_correlation(_rank(vwap), _rank(adv60), 10)))
                  < _rank(_correlation(
                      _ts_rank(vwap, 4), _ts_rank(vol, 5), 9,
                  )))
            .astype(float))
        )
    )


def _wq095(vol, open_, high, low):
    adv40 = _adv(vol, 40)
    return _rank(
        (open_ - _ts_min(open_, 12))
        < _ts_rank(_rank(_correlation(
            _ts_sum(((high + low) / 2), 19.1351),
            _ts_sum(adv40, 19.1351),
            12.8742,
        )), 11.7584)
    )


def _wq096(vol, vwap, close):
    adv60 = _adv(vol, 60)
    a = _ts_rank(_decay_linear(_correlation(
        _rank(vwap), _rank(vol), 3.83878,
    ), 4.16783), 8.38151)
    b = _ts_rank(_decay_linear(
        _ts_argmax(
            _correlation(_ts_rank(close, 7.45404), _ts_rank(adv60, 4.13242), 3.65459),
            12.6556,
        ),
        14.0365,
    ), 13.4143)
    return pd.DataFrame(
        np.where(a.values < b.values, a.values, b.values),
        index=close.index, columns=close.columns,
    ) * -1


def _wq097(vol, low, vwap, ind):
    adv60 = _adv(vol, 60)
    return (
        _rank(_correlation(
            _indneutralize(
                _rank(_rank(_correlation(vwap, _ts_mean(adv60, 14.7444), 6.00049))),
                ind,
            ),
            _indneutralize(_rank(low), ind),
            4.6626,
        ))
        - _rank(_ts_argmin(vwap, 14.0555))
    )


def _wq098(vwap, open_, vol):
    adv5 = _adv(vol, 5)
    adv15 = _adv(vol, 15)
    return (
        _rank(_decay_linear(
            _correlation(vwap, _ts_mean(adv5, 26.4719), 4.58418),
            7.18088,
        ))
        - _rank(_decay_linear(
            _ts_rank(
                _ts_argmin(_correlation(_rank(open_), _rank(adv15), 20.8187), 8.62571),
                6.95668,
            ),
            8.07206,
        ))
    )


def _wq099(vol, high, low):
    adv60 = _adv(vol, 60)
    return (
        -1 * _rank(_correlation(
            _ts_sum(((high + low) / 2), 19.8975),
            _ts_sum(adv60, 19.8975),
            8.8136,
        ))
    )


def _wq100(vol, high, low, close, ind):
    adv20 = _adv(vol, 20)
    return 0 - (1 * (
        (1.5 * _scale(_indneutralize(_indneutralize(
            _rank(((close - low) - (high - close)) / (high - low + 1e-9).replace(0, np.nan)),
            ind,
        ), ind)))
        - _scale(_indneutralize(
            _rank(_correlation(close, _rank(adv20), 5)) - _rank(_ts_argmin(close, 30)),
            ind,
        ))
    ))


def _wq101(close, open_, high, low):
    return (close - open_) / ((high - low) + 0.001)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

def _make_panels(bars: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Convert long-format bars to wide panel dict (index=date, columns=security)."""
    df = bars.copy()
    df["tradetime"] = pd.to_datetime(df["tradetime"])
    panels: dict[str, pd.DataFrame] = {}
    for col in ("open", "high", "low", "close", "vol", "vwap", "cap", "indclass"):
        if col in df.columns:
            panels[col] = df.pivot_table(
                index="tradetime", columns="security_id", values=col, aggfunc="last"
            )
    return panels


_DISPATCH: dict[str, tuple[Callable, list[str]]] = {
    "wq001": (_wq001,  ["close"]),
    "wq002": (_wq002,  ["vol", "close", "open"]),
    "wq003": (_wq003,  ["vol", "open"]),
    "wq004": (_wq004,  ["low"]),
    "wq005": (_wq005,  ["vwap", "open", "close"]),
    "wq006": (_wq006,  ["vol", "open"]),
    "wq007": (_wq007,  ["vol", "close"]),
    "wq008": (_wq008,  ["open", "close"]),
    "wq009": (_wq009,  ["close"]),
    "wq010": (_wq010,  ["close"]),
    "wq011": (_wq011,  ["vwap", "vol", "close"]),
    "wq012": (_wq012,  ["vol", "close"]),
    "wq013": (_wq013,  ["vol", "close"]),
    "wq014": (_wq014,  ["vol", "open", "close"]),
    "wq015": (_wq015,  ["vol", "high"]),
    "wq016": (_wq016,  ["vol", "high"]),
    "wq017": (_wq017,  ["vol", "close"]),
    "wq018": (_wq018,  ["close", "open"]),
    "wq019": (_wq019,  ["close"]),
    "wq020": (_wq020,  ["open", "close", "high", "low"]),
    "wq021": (_wq021,  ["close", "vol"]),
    "wq022": (_wq022,  ["close", "vol", "high"]),
    "wq023": (_wq023,  ["high"]),
    "wq024": (_wq024,  ["close"]),
    "wq025": (_wq025,  ["close", "vol", "high", "vwap"]),
    "wq026": (_wq026,  ["vol", "high"]),
    "wq027": (_wq027,  ["vol", "vwap"]),
    "wq028": (_wq028,  ["vol", "high", "low", "close"]),
    "wq029": (_wq029,  ["close"]),
    "wq030": (_wq030,  ["vol", "close"]),
    "wq031": (_wq031,  ["vol", "close", "low"]),
    "wq032": (_wq032,  ["close", "vwap"]),
    "wq033": (_wq033,  ["open", "close"]),
    "wq034": (_wq034,  ["close"]),
    "wq035": (_wq035,  ["vol", "close", "high", "low"]),
    "wq036": (_wq036,  ["vol", "open", "close", "vwap"]),
    "wq037": (_wq037,  ["open", "close"]),
    "wq038": (_wq038,  ["open", "close"]),
    "wq039": (_wq039,  ["vol", "close"]),
    "wq040": (_wq040,  ["vol", "high"]),
    "wq041": (_wq041,  ["high", "low", "vwap"]),
    "wq042": (_wq042,  ["vwap", "close"]),
    "wq043": (_wq043,  ["vol", "close"]),
    "wq044": (_wq044,  ["vol", "high"]),
    "wq045": (_wq045,  ["vol", "close"]),
    "wq046": (_wq046,  ["close"]),
    "wq047": (_wq047,  ["vol", "close", "high", "vwap"]),
    "wq048": (_wq048,  ["close", "indclass"]),
    "wq049": (_wq049,  ["close"]),
    "wq050": (_wq050,  ["vol", "vwap"]),
    "wq051": (_wq051,  ["close"]),
    "wq052": (_wq052,  ["vol", "close", "low"]),
    "wq053": (_wq053,  ["close", "high", "low"]),
    "wq054": (_wq054,  ["open", "close", "high", "low"]),
    "wq055": (_wq055,  ["vol", "close", "high", "low"]),
    "wq056": (_wq056,  ["close", "cap"]),
    "wq057": (_wq057,  ["close", "vwap"]),
    "wq058": (_wq058,  ["vol", "vwap", "indclass"]),
    "wq059": (_wq059,  ["vol", "vwap", "indclass"]),
    "wq060": (_wq060,  ["vol", "close", "high", "low"]),
    "wq061": (_wq061,  ["vol", "vwap"]),
    "wq062": (_wq062,  ["vol", "vwap", "open", "high", "low"]),
    "wq063": (_wq063,  ["vol", "open", "close", "vwap", "indclass", "low"]),
    "wq064": (_wq064,  ["vol", "vwap", "open", "high", "low"]),
    "wq065": (_wq065,  ["vol", "vwap", "open"]),
    "wq066": (_wq066,  ["vwap", "high", "low", "open"]),
    "wq067": (_wq067,  ["vol", "high", "vwap", "indclass"]),
    "wq068": (_wq068,  ["vol", "close", "high", "low"]),
    "wq069": (_wq069,  ["vol", "close", "vwap", "indclass"]),
    "wq070": (_wq070,  ["vol", "close", "vwap", "indclass"]),
    "wq071": (_wq071,  ["vol", "vwap", "close", "open", "low"]),
    "wq072": (_wq072,  ["vol", "vwap", "high", "low"]),
    "wq073": (_wq073,  ["vwap", "open", "low"]),
    "wq074": (_wq074,  ["vol", "vwap", "close", "high"]),
    "wq075": (_wq075,  ["vol", "vwap", "low"]),
    "wq076": (_wq076_fixed, ["vol", "low", "vwap", "indclass"]),
    "wq077": (_wq077,  ["vol", "vwap", "high", "low"]),
    "wq078": (_wq078,  ["vol", "vwap", "low"]),
    "wq079": (_wq079,  ["vol", "open", "close", "vwap", "indclass"]),
    "wq080": (_wq080,  ["vol", "open", "high", "indclass"]),
    "wq081": (_wq081,  ["vol", "vwap"]),
    "wq082": (_wq082,  ["vol", "open", "indclass"]),
    "wq083": (_wq083,  ["vol", "vwap", "close", "high", "low"]),
    "wq084": (_wq084,  ["vwap", "close"]),
    "wq085": (_wq085,  ["vol", "close", "high", "low"]),
    "wq086": (_wq086,  ["vol", "vwap", "open", "close"]),
    "wq087": (_wq087,  ["vol", "close", "vwap", "indclass"]),
    "wq088": (_wq088,  ["vol", "open", "close", "high", "low"]),
    "wq089": (_wq089,  ["vol", "low", "vwap", "indclass"]),
    "wq090": (_wq090,  ["vol", "low", "close", "indclass"]),
    "wq091": (_wq091,  ["vol", "close", "vwap", "indclass"]),
    "wq092": (_wq092,  ["vol", "open", "close", "high", "low"]),
    "wq093": (_wq093,  ["vol", "close", "vwap", "indclass"]),
    "wq094": (_wq094,  ["vol", "vwap"]),
    "wq095": (_wq095,  ["vol", "open", "high", "low"]),
    "wq096": (_wq096,  ["vol", "vwap", "close"]),
    "wq097": (_wq097,  ["vol", "low", "vwap", "indclass"]),
    "wq098": (_wq098,  ["vwap", "open", "vol"]),
    "wq099": (_wq099,  ["vol", "high", "low"]),
    "wq100": (_wq100,  ["vol", "high", "low", "close", "indclass"]),
    "wq101": (_wq101,  ["close", "open", "high", "low"]),
}

_PANEL_ALIAS = {"open": "open_"}  # function param name differs from panel key


def _call_alpha(alpha_id: str, panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    func, inputs = _DISPATCH[alpha_id]
    args = []
    for inp in inputs:
        p = panels.get(inp)
        if p is None:
            raise KeyError(f"{alpha_id} requires panel '{inp}' which is missing from bars")
        args.append(p)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return func(*args)


# ---------------------------------------------------------------------------
# Cross-sectional z-score normalisation
# ---------------------------------------------------------------------------

def _cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sigma = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sigma, axis=0)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_wq101_alphas(
    bars: pd.DataFrame,
    alpha_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Compute WQ101 alpha factors from standardised OHLCV bars.

    Args:
        bars: Long-format DataFrame with columns
              (security_id, tradetime, open, high, low, close, vol, vwap, cap, indclass).
              Produced by load_csv_data() in daily_batch_pipeline.
        alpha_ids: Subset of alpha IDs to compute (e.g. ["wq001", "wq040"]).
                   None = compute all 101.

    Returns:
        Long-format DataFrame with columns
        (security_id, tradetime, alpha_id, alpha_value).
        Values are cross-sectionally z-scored per date, NaN rows dropped.
    """
    if alpha_ids is None:
        alpha_ids = list(_DISPATCH.keys())
    else:
        unknown = set(alpha_ids) - set(_DISPATCH)
        if unknown:
            raise ValueError(f"Unknown alpha IDs: {unknown}")

    panels = _make_panels(bars)
    parts: list[pd.DataFrame] = []

    for aid in alpha_ids:
        try:
            mat = _call_alpha(aid, panels)
            mat = _cs_zscore(mat)
            long = mat.stack().rename("alpha_value").reset_index()
            long.columns = ["tradetime", "security_id", "alpha_value"]
            long["alpha_id"] = aid
            long = long.dropna(subset=["alpha_value"])
            parts.append(long[["security_id", "tradetime", "alpha_id", "alpha_value"]])
            logger.debug("alpha_computed", alpha_id=aid, rows=len(long))
        except Exception as exc:  # noqa: BLE001
            logger.warning("alpha_failed", alpha_id=aid, error=str(exc))

    if not parts:
        return pd.DataFrame(columns=["security_id", "tradetime", "alpha_id", "alpha_value"])

    result = pd.concat(parts, ignore_index=True)
    result["tradetime"] = pd.to_datetime(result["tradetime"])
    result["security_id"] = result["security_id"].astype(str)
    logger.info(
        "wq101_computed",
        alpha_ids=len(alpha_ids),
        rows=len(result),
        dates=result["tradetime"].nunique(),
        symbols=result["security_id"].nunique(),
    )
    return result
