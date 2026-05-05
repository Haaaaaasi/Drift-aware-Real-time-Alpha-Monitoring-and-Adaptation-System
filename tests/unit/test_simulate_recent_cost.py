"""Phase A — 交易成本計算單元測試（P0★ #4）。

驗證 ``pipelines.simulate_recent._compute_costs``：
1. 三細項拆分模式：commission per-side（buys+sells）+ tax sell-side only + slippage per-side（buys+sells）
2. ``round_trip_cost_pct`` 覆寫三細項，全部歸入 slippage_cost
3. ``round_trip_cost_pct=0`` 時 net == gross（成本歸零）
4. 首日全買（buys=1.0, sells=0.0）：tax 應為 0，turnover headline 應為 1.0
5. 不對稱再平衡（buys≠sells）：tax 以實際 sells 計算，而非 buys+sells 平均
"""

from __future__ import annotations

import math

import pytest

from pipelines.simulate_recent import _compute_costs


class TestSplitCosts:
    def test_split_costs_baseline_taiwanese_rates(self):
        """台股 baseline：commission 0.0926%/side, tax 0.3%/sell, slippage 5 bps/side。

        穩態再平衡（buys=sells=0.5）：
            commission = (0.5+0.5) × 0.000926  = 0.000926
            tax        = 0.5 × 0.003            = 0.0015
            slippage   = (0.5+0.5) × 5 / 10000  = 0.0005
        """
        c, t, s = _compute_costs(
            buys=0.5,
            sells=0.5,
            commission_rate=0.000926,
            tax_rate=0.003,
            slippage_bps=5.0,
            round_trip_cost_pct=None,
        )
        assert math.isclose(c, 0.000926, rel_tol=1e-9)
        assert math.isclose(t, 0.0015, rel_tol=1e-9)
        assert math.isclose(s, 0.0005, rel_tol=1e-9)

    def test_zero_turnover_zero_costs(self):
        """無換手 → 三細項皆 0。"""
        c, t, s = _compute_costs(
            buys=0.0,
            sells=0.0,
            commission_rate=0.000926,
            tax_rate=0.003,
            slippage_bps=5.0,
            round_trip_cost_pct=None,
        )
        assert c == 0.0 and t == 0.0 and s == 0.0

    def test_commission_scales_with_per_side_doubling(self):
        """commission 應與 per-side rate 成正比（rate 加倍，commission 加倍）。"""
        c1, _, _ = _compute_costs(
            buys=0.5, sells=0.5, commission_rate=0.001, tax_rate=0.0,
            slippage_bps=0.0, round_trip_cost_pct=None,
        )
        c2, _, _ = _compute_costs(
            buys=0.5, sells=0.5, commission_rate=0.002, tax_rate=0.0,
            slippage_bps=0.0, round_trip_cost_pct=None,
        )
        assert math.isclose(c2 / c1, 2.0, rel_tol=1e-9)

    def test_first_day_all_buys_no_tax(self):
        """首日（空倉→滿倉）：buys=1.0, sells=0.0 → tax=0，turnover headline=1.0。

        舊公式 sum(|Δw|)/2=0.5 會錯誤產生 tax=0.5×tax_rate；正確應為 0。
            commission = (1.0+0.0) × 0.000926 = 0.000926
            tax        = 0.0 × 0.003           = 0.0   ← 首日全買，無賣出，不徵稅
            slippage   = (1.0+0.0) × 5 / 10000 = 0.0005
        """
        c, t, s = _compute_costs(
            buys=1.0,
            sells=0.0,
            commission_rate=0.000926,
            tax_rate=0.003,
            slippage_bps=5.0,
            round_trip_cost_pct=None,
        )
        assert math.isclose(c, 0.000926, rel_tol=1e-9)
        assert t == 0.0
        assert math.isclose(s, 0.0005, rel_tol=1e-9)

    def test_asymmetric_rebalance_tax_on_sells_only(self):
        """不對稱再平衡（buys=0.7, sells=0.3）：tax 僅對 sells 計算，turnover=max=0.7。

            commission = (0.7+0.3) × 0.000926 = 0.000926
            tax        = 0.3 × 0.003           = 0.0009   ← 非 0.5×tax_rate = 0.0015
            slippage   = (0.7+0.3) × 5 / 10000 = 0.0005
        """
        c, t, s = _compute_costs(
            buys=0.7,
            sells=0.3,
            commission_rate=0.000926,
            tax_rate=0.003,
            slippage_bps=5.0,
            round_trip_cost_pct=None,
        )
        assert math.isclose(c, 0.000926, rel_tol=1e-9)
        assert math.isclose(t, 0.0009, rel_tol=1e-9)
        assert math.isclose(s, 0.0005, rel_tol=1e-9)


class TestRoundTripPctOverride:
    def test_round_trip_pct_overrides_split(self):
        """``round_trip_cost_pct`` 給定時，三細項全被忽略；總成本歸入 slippage_cost。

        turnover = max(buys, sells) = max(0.5, 0.5) = 0.5
        cost = 0.5 × (0.4 / 100) = 0.002
        """
        c, t, s = _compute_costs(
            buys=0.5,
            sells=0.5,
            commission_rate=0.001,   # 應被忽略
            tax_rate=0.005,           # 應被忽略
            slippage_bps=10.0,        # 應被忽略
            round_trip_cost_pct=0.4,
        )
        assert c == 0.0
        assert t == 0.0
        assert math.isclose(s, 0.002, rel_tol=1e-9)

    def test_zero_round_trip_pct_yields_zero_cost(self):
        """``round_trip_cost_pct=0`` → 三細項皆 0，net == gross。"""
        c, t, s = _compute_costs(
            buys=0.5,
            sells=0.5,
            commission_rate=0.001,
            tax_rate=0.003,
            slippage_bps=5.0,
            round_trip_cost_pct=0.0,
        )
        assert c == 0.0 and t == 0.0 and s == 0.0

    @pytest.mark.parametrize("cost_pct", [0.2, 0.4, 0.6])
    def test_round_trip_pct_proportional(self, cost_pct):
        """sweep 4 個成本場景時，總成本應與 cost_pct 線性相關。

        turnover = max(0.5, 0.5) = 0.5 → cost = 0.5 × (cost_pct / 100)
        """
        _, _, s = _compute_costs(
            buys=0.5,
            sells=0.5,
            commission_rate=0.0,
            tax_rate=0.0,
            slippage_bps=0.0,
            round_trip_cost_pct=cost_pct,
        )
        expected = 0.5 * (cost_pct / 100.0)
        assert math.isclose(s, expected, rel_tol=1e-9)

    def test_round_trip_pct_uses_max_buys_sells(self):
        """round_trip_cost_pct 模式使用 max(buys, sells) 作為 turnover 基數。

        buys=0.8, sells=0.2 → turnover=0.8；cost=0.8 × (0.4/100) = 0.0032
        """
        _, _, s = _compute_costs(
            buys=0.8,
            sells=0.2,
            commission_rate=0.0,
            tax_rate=0.0,
            slippage_bps=0.0,
            round_trip_cost_pct=0.4,
        )
        assert math.isclose(s, 0.8 * 0.004, rel_tol=1e-9)


class TestNetReturnConsistency:
    """驗證主迴圈中 net_return 的構造邏輯（與 _compute_costs 結合）。"""

    def test_net_equals_gross_when_zero_cost(self):
        gross = 0.01  # 1%
        c, t, s = _compute_costs(
            buys=0.5,
            sells=0.5,
            commission_rate=0.0,
            tax_rate=0.0,
            slippage_bps=0.0,
            round_trip_cost_pct=None,
        )
        net = gross - c - t - s
        assert net == gross

    def test_net_subtracts_all_three_components(self):
        """穩態再平衡（buys=sells=0.5）三細項數值與舊公式代數等效。

        commission = 1.0 × 0.000926 = 0.000926
        tax        = 0.5 × 0.003    = 0.0015
        slippage   = 1.0 × 5/10000  = 0.0005
        net = 0.005 - 0.000926 - 0.0015 - 0.0005 = 0.002074
        """
        gross = 0.005
        c, t, s = _compute_costs(
            buys=0.5,
            sells=0.5,
            commission_rate=0.000926,
            tax_rate=0.003,
            slippage_bps=5.0,
            round_trip_cost_pct=None,
        )
        net = gross - c - t - s
        assert math.isclose(net, 0.002074, rel_tol=1e-6)
