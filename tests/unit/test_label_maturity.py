"""Label maturity 時間語義正確性測試。

驗證 ``compute_label_available_at`` 與 ``LabelGenerator.generate_labels``
在以下四種情境下的行為：

1. 一般交易日：label_available_at = 第 h+buffer 個後續 trading day
2. 連假跨越：跳過假日，不用 calendar days 近似
3. 資料尾端：future bars 不足 → 回傳 None，label 不生成（不 fallback 成 timedelta）
4. simulate retrain：新版 train_labels 筆數 ≤ 舊版（不會更多）
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.common.time_utils import compute_label_available_at
from src.labeling.label_generator import LabelGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trading_days(*dates: str) -> pd.DatetimeIndex:
    """以字串清單建立已排序、已 normalize 的 DatetimeIndex。"""
    return pd.DatetimeIndex([pd.Timestamp(d).normalize() for d in sorted(dates)])


def _make_bars(trading_days: pd.DatetimeIndex, n_symbols: int = 5) -> pd.DataFrame:
    """為指定交易日清單產生最小 OHLCV DataFrame。"""
    rng = np.random.RandomState(0)
    rows = []
    for sym in [f"SYM{i}" for i in range(n_symbols)]:
        price = 100.0
        for d in trading_days:
            price *= 1 + rng.randn() * 0.01
            rows.append({"security_id": sym, "tradetime": d, "close": price,
                         "vol": 1_000_000})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Case 1: 一般交易日
# ---------------------------------------------------------------------------

class TestNormalTradingDay:
    """signal_time 在資料中段，label_available_at 應等於第 h+buffer 個後續 trading day。"""

    def test_label_available_at_is_nth_trading_bar(self):
        # 10 個連續「非週末」交易日（2024-01-02 ~ 2024-01-12，跳過週末 01-06、01-07）
        days = _make_trading_days(
            "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
            "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11",
            "2024-01-12",
        )
        signal = pd.Timestamp("2024-01-02").normalize()
        avail = compute_label_available_at(
            signal, horizon_bars=5, buffer_bars=1,
            bar_type="daily", trading_days=days,
        )
        # pos=0; target=0+5+1=6; days[6]="2024-01-10"
        assert avail is not None
        assert pd.Timestamp(avail).normalize() == pd.Timestamp("2024-01-10")

    def test_label_available_at_skips_weekend(self):
        """第 h+buffer 個後續 bar 正好跳過週末。"""
        days = _make_trading_days(
            "2024-01-04", "2024-01-05",          # Thu, Fri
            "2024-01-08", "2024-01-09",          # Mon, Tue (weekend skipped)
        )
        signal = pd.Timestamp("2024-01-04").normalize()
        avail = compute_label_available_at(
            signal, horizon_bars=2, buffer_bars=1,
            bar_type="daily", trading_days=days,
        )
        # pos=0; target=0+2+1=3; days[3]="2024-01-09"
        assert avail is not None
        assert pd.Timestamp(avail).normalize() == pd.Timestamp("2024-01-09")


# ---------------------------------------------------------------------------
# Case 2: 連假跨越
# ---------------------------------------------------------------------------

class TestHolidayCrossing:
    """連假中間應直接跳到下一個有效 trading bar，而非加 calendar days。"""

    def test_chinese_new_year_gap(self):
        """春節假期（2024-02-09 ~ 2024-02-18）：第 6 個後續 bar 應落在 2024-02-19，
        而非 calendar 近似的 2024-02-07（2024-02-01 + 6 calendar days）。
        """
        days = _make_trading_days(
            "2024-02-01", "2024-02-02", "2024-02-05",
            "2024-02-06", "2024-02-07", "2024-02-08",
            # 假期：2024-02-09 ~ 2024-02-18
            "2024-02-19", "2024-02-20", "2024-02-21",
        )
        signal = pd.Timestamp("2024-02-01").normalize()
        avail = compute_label_available_at(
            signal, horizon_bars=5, buffer_bars=1,
            bar_type="daily", trading_days=days,
        )
        # pos=0; target=6; days[6]="2024-02-19"
        assert avail is not None
        expected = pd.Timestamp("2024-02-19")
        assert pd.Timestamp(avail).normalize() == expected

    def test_not_calendar_days(self):
        """確認回傳值不等於 calendar days 的近似結果。"""
        days = _make_trading_days(
            "2024-02-01", "2024-02-02", "2024-02-05",
            "2024-02-06", "2024-02-07", "2024-02-08",
            "2024-02-19", "2024-02-20",
        )
        signal = pd.Timestamp("2024-02-01").normalize()
        avail = compute_label_available_at(
            signal, horizon_bars=5, buffer_bars=1,
            bar_type="daily", trading_days=days,
        )
        calendar_approx = signal + pd.Timedelta(days=6)  # = 2024-02-07
        assert avail is not None
        assert pd.Timestamp(avail).normalize() != calendar_approx, (
            "回傳值不應等於 calendar-day 近似（連假會使兩者不同）"
        )


# ---------------------------------------------------------------------------
# Case 3: 資料尾端 — 應回 None，不 fallback
# ---------------------------------------------------------------------------

class TestDataTail:
    """future bars 不足時，compute_label_available_at 應回傳 None。"""

    def test_returns_none_when_future_bars_insufficient(self):
        days = _make_trading_days(
            "2024-01-02", "2024-01-03", "2024-01-04",
            "2024-01-05", "2024-01-08",  # 只有 4 個後續 bar（index 1-4）
        )
        signal = pd.Timestamp("2024-01-02").normalize()
        avail = compute_label_available_at(
            signal, horizon_bars=5, buffer_bars=1,  # 需要 6 個後續 bar
            bar_type="daily", trading_days=days,
        )
        assert avail is None, f"應回 None，但得到 {avail}"

    def test_generate_labels_skips_tail_rows(self):
        """generate_labels 應跳過資料尾端無法取得 label_available_at 的 rows。"""
        # 只建 10 個交易日；h=5, buffer=1 需要 6 個後續 bar；
        # 所以最後 6 筆 signal rows 應被排除（forward_return 也是 NaN，shift(-5) 的結果）
        n_days = 10
        days = pd.date_range("2024-01-01", periods=n_days, freq="B")
        bars = _make_bars(days, n_symbols=3)
        label_gen = LabelGenerator(horizons=[5], bar_type="daily", buffer_bars=1)
        labels = label_gen.generate_labels(bars[["security_id", "tradetime", "close"]])

        # label_available_at 欄不應有 NaT / None
        assert labels["label_available_at"].notna().all(), (
            "label_available_at 欄不應包含 NaT（None 的 row 應已被 skip）"
        )
        # forward_return 也不應有 NaN（NaN 的 row 同樣會被 skip）
        assert labels["forward_return"].notna().all()

    def test_last_n_rows_absent_from_labels(self):
        """最後 (h + buffer) 個交易日不應出現在 labels 的 signal_time 中。"""
        n_days = 15
        days = pd.date_range("2024-01-01", periods=n_days, freq="B")
        bars = _make_bars(days, n_symbols=2)
        h, buf = 5, 1
        label_gen = LabelGenerator(horizons=[h], bar_type="daily", buffer_bars=buf)
        labels = label_gen.generate_labels(bars[["security_id", "tradetime", "close"]])

        latest_signal = labels["signal_time"].max()
        # 最晚合法 signal_time：days[n_days - 1 - (h + buf)] = days[8]
        expected_latest = days[n_days - 1 - (h + buf)]
        assert pd.Timestamp(latest_signal).normalize() <= pd.Timestamp(expected_latest), (
            f"latest signal_time={latest_signal} 超出允許上界 {expected_latest}"
        )


# ---------------------------------------------------------------------------
# Case 4: simulate retrain — 新版 train_labels 筆數 ≤ 舊版
# ---------------------------------------------------------------------------

class TestSimulateRetrainCount:
    """label_avail <= t 過濾後，訓練筆數應 ≤ 舊版 signal_time <= purge_cutoff 結果。

    舊版 purge_cutoff = t - purge_days(calendar)；
    新版以 label_available_at（trading bars）為門檻。
    因 h + buffer 個 trading bars > purge_days 個 calendar days（在典型參數下），
    新版篩選更嚴格，筆數不應增加。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        n_days = 60
        self.days = pd.date_range("2023-01-01", periods=n_days, freq="B")
        bars = _make_bars(self.days, n_symbols=10)
        h, buf = 5, 1
        label_gen = LabelGenerator(horizons=[h], bar_type="daily", buffer_bars=buf)
        labels_df = label_gen.generate_labels(bars[["security_id", "tradetime", "close"]])
        labels_h = (
            labels_df[labels_df["horizon"] == h]
            .dropna(subset=["forward_return"])
            .set_index(["security_id", "signal_time"])
            .rename_axis(index=["security_id", "tradetime"])
        )
        self.fwd_5 = labels_h["forward_return"]
        self.label_avail = labels_h["label_available_at"]
        self.purge_days = 5
        self.h = h

    def _old_count(self, t: pd.Timestamp) -> int:
        purge_cutoff = t - pd.Timedelta(days=self.purge_days)
        return int((self.fwd_5.index.get_level_values("tradetime") <= purge_cutoff).sum())

    def _new_count(self, t: pd.Timestamp) -> int:
        return int((self.label_avail <= t).sum())

    def test_new_count_le_old_count_mid_period(self):
        t = self.days[30]
        assert self._new_count(t) <= self._old_count(t), (
            f"新版 ({self._new_count(t)}) 應 ≤ 舊版 ({self._old_count(t)}) 在 t={t.date()}"
        )

    def test_new_count_le_old_count_late_period(self):
        t = self.days[50]
        assert self._new_count(t) <= self._old_count(t), (
            f"新版 ({self._new_count(t)}) 應 ≤ 舊版 ({self._old_count(t)}) 在 t={t.date()}"
        )

    def test_new_count_never_exceeds_old_count_all_days(self):
        """對所有模擬日逐一檢查，新版筆數永遠 ≤ 舊版。"""
        violations = []
        for t in self.days[20:]:
            n, o = self._new_count(t), self._old_count(t)
            if n > o:
                violations.append((t.date(), n, o))
        assert not violations, f"新版筆數超過舊版的日期：{violations}"
