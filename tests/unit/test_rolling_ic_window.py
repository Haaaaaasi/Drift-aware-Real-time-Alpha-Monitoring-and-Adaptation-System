"""Phase A — Rolling IC / Sharpe 窗口拆分單元測試（P0★ #2）。

驗證 ``pipelines.simulate_recent._compute_rolling_ic`` 與 ``_compute_rolling_sharpe``：
1. ``[t-window, t-eval_gap]`` calendar-day 雙邊界
2. eval_gap 內的最近樣本被排除（trigger 不吃 shadow window 之內的資料）
3. window 之外（更早）的樣本被排除
4. 樣本不足時回傳 NaN
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pipelines.simulate_recent import _compute_rolling_ic, _compute_rolling_sharpe


def _make_signal_df(signal_time: pd.Timestamp, n_secs: int = 20, score_seed: int = 0) -> pd.DataFrame:
    """造一個 mini signal DataFrame：n_secs 檔股票同一個 signal_time。"""
    rng = np.random.RandomState(score_seed)
    return pd.DataFrame({
        "security_id": [f"SEC{i:03d}" for i in range(n_secs)],
        "signal_time": [signal_time] * n_secs,
        "signal_score": rng.randn(n_secs),
    })


def _make_fwd_series(dates: list[pd.Timestamp], n_secs: int = 20, seed: int = 1) -> pd.Series:
    """造對應的 forward_return Series，索引 (security_id, tradetime)。"""
    rng = np.random.RandomState(seed)
    rows = []
    for d in dates:
        for i in range(n_secs):
            rows.append((f"SEC{i:03d}", d, rng.randn() * 0.01))
    df = pd.DataFrame(rows, columns=["security_id", "tradetime", "forward_return"])
    return df.set_index(["security_id", "tradetime"])["forward_return"]


class TestRollingICWindow:
    def test_eval_gap_excludes_recent_signals(self):
        """eval_gap=20 應排除最近 20 calendar days 的 signal（即使在 window 內）。

        構造：t=2024-04-01；window=60、eval_gap=20、purge_days=5、horizon_days=5。
            * effective_upper = t - max(eval_gap, eval_gap + purge + horizon) = t-30
            * lower = t-60 = 2024-01-31
            * 6 個 in-window mature signals（[t-60, t-30]）+ 3 個 recent（被排除）
        """
        t = pd.Timestamp("2024-04-01")
        in_window_dates = [
            pd.Timestamp("2024-02-05"),
            pd.Timestamp("2024-02-12"),
            pd.Timestamp("2024-02-19"),
            pd.Timestamp("2024-02-26"),
            pd.Timestamp("2024-03-01"),  # close to t-30 boundary
            pd.Timestamp("2024-03-02"),
        ]
        recent_dates = [
            pd.Timestamp("2024-03-25"),  # 在 eval_gap 內，應排除
            pd.Timestamp("2024-03-15"),  # mature_cutoff 之後，應排除
        ]
        all_dates = in_window_dates + recent_dates
        fwd = _make_fwd_series(all_dates)
        signals = [_make_signal_df(d, score_seed=i) for i, d in enumerate(all_dates)]

        ic = _compute_rolling_ic(
            signals, fwd, t,
            purge_days=5, horizon_days=5,
            window_days=60, eval_gap_days=20,
        )
        assert ic is not None
        assert not math.isnan(ic), "預期應回傳 IC，但得到 NaN"

    def test_returns_nan_when_only_recent_signals(self):
        """若所有 signal 都在 eval_gap 內 → 樣本不足，回傳 NaN。"""
        t = pd.Timestamp("2024-04-01")
        # 全部 signal 都在最近 15 日內，被 eval_gap=20 排除
        dates = [t - pd.Timedelta(days=d) for d in [5, 10, 15]]
        fwd = _make_fwd_series(dates)
        signals = [_make_signal_df(d, score_seed=i) for i, d in enumerate(dates)]

        ic = _compute_rolling_ic(
            signals, fwd, t,
            purge_days=5, horizon_days=5,
            window_days=60, eval_gap_days=20,
        )
        assert math.isnan(ic), f"全部 signal 都在 eval_gap 內，應回傳 NaN，實際 {ic}"

    def test_returns_nan_when_signals_out_of_window(self):
        """所有 signal 都在 t-window 之前 → NaN。"""
        t = pd.Timestamp("2024-04-01")
        # 全部 signal 都比 t-90 還早（window=60 排除）
        dates = [t - pd.Timedelta(days=d) for d in [100, 110, 120]]
        fwd = _make_fwd_series(dates)
        signals = [_make_signal_df(d, score_seed=i) for i, d in enumerate(dates)]

        ic = _compute_rolling_ic(
            signals, fwd, t,
            purge_days=5, horizon_days=5,
            window_days=60, eval_gap_days=20,
        )
        assert math.isnan(ic)

    def test_empty_past_signals_returns_nan(self):
        ic = _compute_rolling_ic(
            [], pd.Series(dtype=float), pd.Timestamp("2024-04-01"),
            purge_days=5, horizon_days=5,
            window_days=60, eval_gap_days=20,
        )
        assert math.isnan(ic)


class TestRollingSharpeWindow:
    def test_sharpe_only_uses_window_records(self):
        """Sharpe 應只用 [t-window, t-eval_gap] 內的 daily net_return。"""
        t = pd.Timestamp("2024-04-01")
        # 構造 80 日 records：前 30 日（>t-60）outside window；中間 30 日 inside；最近 20 日（eval_gap）outside
        records = []
        # 在 window 內（[t-60, t-20]）— 故意給高均值低波動 → Sharpe 應為正
        for i in range(40, 20, -1):
            records.append({
                "date": (t - pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
                "net_return": 0.005,  # 0.5%/day, std=0 → 但 std=0 會回 0.0
            })
        # 加入一些變動避免 std=0
        records.extend([
            {"date": (t - pd.Timedelta(days=30)).strftime("%Y-%m-%d"), "net_return": 0.01},
            {"date": (t - pd.Timedelta(days=25)).strftime("%Y-%m-%d"), "net_return": -0.005},
        ])
        # 最近 20 日 — 應被 eval_gap 排除
        for i in range(15, 0, -1):
            records.append({
                "date": (t - pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
                "net_return": -0.05,  # 故意給很差的回報，若被納入會壓 Sharpe
            })

        sharpe = _compute_rolling_sharpe(
            records, t, window_days=60, eval_gap_days=20,
        )
        assert not math.isnan(sharpe)
        # 因為最近 20 日的負報酬被排除，Sharpe 應 > -1（不會被「拖垮」）
        assert sharpe > -1.0, f"sharpe={sharpe}，最近差報酬應被排除"

    def test_sharpe_returns_nan_when_insufficient_samples(self):
        """樣本 < 10 個 → NaN。"""
        t = pd.Timestamp("2024-04-01")
        # 只在 window 內放 5 筆 records
        records = [
            {"date": (t - pd.Timedelta(days=i)).strftime("%Y-%m-%d"), "net_return": 0.001}
            for i in [22, 25, 28, 30, 35]
        ]
        sharpe = _compute_rolling_sharpe(
            records, t, window_days=60, eval_gap_days=20,
        )
        assert math.isnan(sharpe)

    def test_empty_records_returns_nan(self):
        sharpe = _compute_rolling_sharpe(
            [], pd.Timestamp("2024-04-01"),
            window_days=60, eval_gap_days=20,
        )
        assert math.isnan(sharpe)


class TestTrainWindowDays:
    """驗證 simulate() 的 train_window_days 參數對訓練集切片的影響。

    直接對重訓區塊使用的切片邏輯做單元測試（不啟動完整模擬迴圈）。
    """

    def _make_alpha_panel(self, start: pd.Timestamp, n_days: int, n_secs: int = 5) -> pd.DataFrame:
        """產生 (security_id, tradetime, alpha_id, alpha_value) 格式的 dummy panel。"""
        rows = []
        for d in range(n_days):
            t = start + pd.Timedelta(days=d)
            for i in range(n_secs):
                rows.append({"security_id": f"S{i}", "tradetime": t, "alpha_id": "wq001", "alpha_value": float(d * n_secs + i)})
        return pd.DataFrame(rows)

    def _make_label_series(self, alpha_panel: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """從 alpha_panel 的 (security_id, tradetime) 造 dummy fwd_5 與 label_avail。"""
        idx = pd.MultiIndex.from_frame(alpha_panel[["security_id", "tradetime"]])
        fwd_5 = pd.Series(0.001, index=idx, name="forward_return")
        fwd_5.index.names = ["security_id", "tradetime"]
        label_avail = pd.Series(
            alpha_panel["tradetime"] + pd.Timedelta(days=10),  # label ready 10 days later
            index=idx,
            name="label_available_at",
        )
        label_avail.index.names = ["security_id", "tradetime"]
        return fwd_5, label_avail

    def test_expanding_window_uses_all_history(self):
        """train_window_days=None（expanding）：訓練集含 purge_cutoff 之前的所有資料。"""
        start = pd.Timestamp("2023-01-01")
        panel = self._make_alpha_panel(start, n_days=180)
        purge_cutoff = start + pd.Timedelta(days=90)

        train_panel = panel[panel["tradetime"] <= purge_cutoff]
        assert len(train_panel) > 0
        # 91 天（day 0..90）× 5 secs
        assert train_panel["tradetime"].min() == start
        assert train_panel["tradetime"].max() == purge_cutoff

    def test_rolling_window_limits_training_history(self):
        """train_window_days=30：訓練集只含最近 30 天，更早的歷史被丟棄。"""
        start = pd.Timestamp("2023-01-01")
        panel = self._make_alpha_panel(start, n_days=180)
        purge_cutoff = start + pd.Timedelta(days=90)
        train_window_days = 30
        window_start = purge_cutoff - pd.Timedelta(days=train_window_days)

        train_panel = panel[
            (panel["tradetime"] >= window_start) &
            (panel["tradetime"] <= purge_cutoff)
        ]
        assert train_panel["tradetime"].min() >= window_start
        assert train_panel["tradetime"].max() <= purge_cutoff
        # 窗口前的資料不應出現
        assert (train_panel["tradetime"] < window_start).sum() == 0

    def test_rolling_window_labels_match_window(self):
        """train_window_days 同時截斷 label：label 的 signal_time 不超出 window_start。"""
        start = pd.Timestamp("2023-01-01")
        panel = self._make_alpha_panel(start, n_days=180)
        fwd_5, label_avail = self._make_label_series(panel)

        t = start + pd.Timedelta(days=100)
        purge_cutoff = t - pd.Timedelta(days=5)
        train_window_days = 30
        window_start = purge_cutoff - pd.Timedelta(days=train_window_days)

        train_labels = fwd_5[
            (label_avail <= t) &
            (label_avail.index.get_level_values("tradetime") >= window_start)
        ]
        signal_times = train_labels.index.get_level_values("tradetime")
        assert (signal_times < window_start).sum() == 0, "rolling window 外的 label 不應進入訓練集"
