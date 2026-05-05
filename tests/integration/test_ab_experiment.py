"""WP9 — Integration test for Adaptation A/B 實驗。

驗證：
1. ``simulate()`` 三種 strategy（none / scheduled / triggered）都能跑完且產出正確 schema
2. Retrain 次數符合預期：
   * none → 1 次（僅 initial_train）
   * scheduled_N → 至少 ceil(days / N) 次，以及每次 reason 開頭為 ``scheduled_``
   * triggered → 至少 1 次（initial_train），其他依冷卻期規則
3. ``run_ab_experiment()`` 產出 comparison.csv / comparison.png / experiment_summary.md
4. comparison.csv 的 schema 正確（包含所有策略欄位）

使用合成 CSV（30 檔標的、~1 年每日棒）作為 fixture，避免依賴真實資料。
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipelines.ab_experiment import DEFAULT_STRATEGIES, run_ab_experiment
from pipelines.simulate_recent import simulate


@pytest.fixture(scope="module")
def synthetic_csv(tmp_path_factory) -> Path:
    """生成合成 OHLCV CSV，符合 ``load_csv_data`` 預期格式：
    columns = datetime, security_id, open, high, low, close, volume
    """
    rng = np.random.RandomState(2026)
    n_symbols = 30
    symbols = [f"SYM{i:04d}" for i in range(1, n_symbols + 1)]
    # 一年多一點，確保有足夠訓練 + 模擬資料
    dates = pd.bdate_range("2023-01-02", "2024-06-30", freq="B")

    rows = []
    for sym in symbols:
        price = 100.0 + rng.randn() * 10
        for d in dates:
            ret = rng.randn() * 0.015
            price = max(1.0, price * (1 + ret))
            o = price * (1 + rng.randn() * 0.003)
            h = max(o, price) * (1 + abs(rng.randn()) * 0.003)
            lo = min(o, price) * (1 - abs(rng.randn()) * 0.003)
            vol = max(1000, int(rng.exponential(300_000)))
            rows.append({
                "datetime": d,
                "security_id": sym,
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(lo, 2),
                "close": round(price, 2),
                "volume": vol,
            })

    df = pd.DataFrame(rows)
    csv_path = tmp_path_factory.mktemp("ab_data") / "synthetic_ohlcv.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(scope="module")
def sim_period():
    # 模擬期間約 100 個交易日，前半部分可訓練
    return date(2024, 1, 2), date(2024, 6, 30)


# ---------------------------------------------------------------------------
# 1. simulate() 三種 strategy 個別驗證
# ---------------------------------------------------------------------------

class TestSimulateStrategies:
    def test_strategy_none_trains_once(self, synthetic_csv, sim_period, tmp_path):
        start, end = sim_period
        result = simulate(
            csv_path=synthetic_csv,
            start=start,
            end=end,
            strategy="none",
            top_k=5,
            out_dir=tmp_path,
            run_tag="test_none",
        )
        retrains = pd.read_csv(result["retrain_log_path"])
        assert len(retrains) == 1, f"strategy=none 應只訓練 1 次，實際 {len(retrains)}"
        assert retrains.iloc[0]["reason"] == "initial_train"

    def test_strategy_scheduled_respects_cadence(self, synthetic_csv, sim_period, tmp_path):
        start, end = sim_period
        retrain_every = 20
        result = simulate(
            csv_path=synthetic_csv,
            start=start,
            end=end,
            strategy="scheduled",
            retrain_every=retrain_every,
            top_k=5,
            out_dir=tmp_path,
            run_tag="test_sched",
        )
        retrains = pd.read_csv(result["retrain_log_path"])
        n_days = result["summary_metrics"].get("n_days", 0)
        # 第一次是 initial_train，之後應約 n_days / retrain_every 次
        expected_min = max(1, math.ceil(n_days / retrain_every) - 1)  # 容許首日扣抵
        assert len(retrains) >= expected_min, (
            f"scheduled_{retrain_every} 模式下期望至少 {expected_min} 次，實際 {len(retrains)}"
        )
        # 後續 retrain reason 應為 scheduled_ 開頭
        later = retrains.iloc[1:]
        if len(later) > 0:
            assert all(later["reason"].str.startswith("scheduled_")), (
                f"scheduled 模式的後續重訓 reason 應以 scheduled_ 開頭：{later['reason'].tolist()}"
            )

    def test_strategy_triggered_has_initial_and_cooldown(self, synthetic_csv, sim_period, tmp_path):
        start, end = sim_period
        result = simulate(
            csv_path=synthetic_csv,
            start=start,
            end=end,
            strategy="triggered",
            top_k=5,
            min_retrain_gap=20,
            out_dir=tmp_path,
            run_tag="test_trig",
        )
        retrains = pd.read_csv(result["retrain_log_path"])
        assert len(retrains) >= 1, "triggered 策略至少要有一次 initial_train"
        assert retrains.iloc[0]["reason"] == "initial_train"
        # 冷卻期驗證：如果有多次重訓，間隔應 ≥ min_retrain_gap
        if len(retrains) >= 2:
            gaps = retrains["day_idx"].diff().dropna()
            assert (gaps >= 20).all(), (
                f"triggered 模式重訓間隔應 ≥ 20 日，實際 gaps={gaps.tolist()}"
            )

    def test_daily_pnl_schema_has_rolling_metrics(self, synthetic_csv, sim_period, tmp_path):
        """daily_pnl.csv 必須包含 rolling_ic 與 rolling_sharpe 欄位。"""
        start, end = sim_period
        result = simulate(
            csv_path=synthetic_csv,
            start=start,
            end=end,
            strategy="scheduled",
            retrain_every=30,
            top_k=5,
            out_dir=tmp_path,
            run_tag="test_schema",
        )
        pnl = pd.read_csv(result["daily_pnl_path"])
        required = {"date", "net_return", "cumulative_value", "rolling_ic", "rolling_sharpe",
                    "n_holdings", "gross_exposure", "turnover"}
        assert required.issubset(set(pnl.columns)), (
            f"daily_pnl 缺欄位：{required - set(pnl.columns)}"
        )

    def test_invalid_strategy_raises(self, synthetic_csv, sim_period, tmp_path):
        start, end = sim_period
        with pytest.raises(ValueError, match="strategy"):
            simulate(
                csv_path=synthetic_csv,
                start=start,
                end=end,
                strategy="bogus",  # type: ignore[arg-type]
                out_dir=tmp_path,
            )


# ---------------------------------------------------------------------------
# 2. run_ab_experiment() 端到端驗證
# ---------------------------------------------------------------------------

class TestABExperiment:
    def test_ab_experiment_produces_all_artifacts(self, synthetic_csv, sim_period, tmp_path):
        start, end = sim_period
        # 跑精簡子集以加快測試
        strategies = ["none", "scheduled_60", "triggered"]
        result = run_ab_experiment(
            csv_path=synthetic_csv,
            start=start,
            end=end,
            strategies=strategies,
            top_k=5,
            out_dir=tmp_path,
            run_tag="test",
        )

        run_dir = Path(result["run_dir"])
        assert (run_dir / "comparison.csv").exists(), "comparison.csv 未產出"
        assert (run_dir / "comparison.png").exists(), "comparison.png 未產出"
        assert (run_dir / "experiment_summary.md").exists(), "experiment_summary.md 未產出"
        assert (run_dir / "config.json").exists(), "config.json 未產出"

    def test_comparison_df_schema(self, synthetic_csv, sim_period, tmp_path):
        start, end = sim_period
        strategies = ["none", "scheduled_60"]
        result = run_ab_experiment(
            csv_path=synthetic_csv,
            start=start,
            end=end,
            strategies=strategies,
            top_k=5,
            out_dir=tmp_path,
            run_tag="schema",
        )
        df = result["comparison_df"]

        # 策略列齊全
        assert set(df.index) == set(strategies), f"策略列缺漏：{set(strategies) - set(df.index)}"
        # 關鍵欄位存在
        required_cols = {
            "n_retrains", "cumulative_return_pct", "annualized_return_pct",
            "sharpe", "max_drawdown_pct", "win_rate_pct", "avg_turnover",
            "avg_gross_return_bps", "avg_total_cost_bps", "avg_net_return_bps",
            "final_value",
        }
        assert required_cols.issubset(df.columns), (
            f"comparison_df 缺欄位：{required_cols - set(df.columns)}"
        )

    def test_none_strategy_has_exactly_one_retrain(self, synthetic_csv, sim_period, tmp_path):
        start, end = sim_period
        result = run_ab_experiment(
            csv_path=synthetic_csv,
            start=start,
            end=end,
            strategies=["none"],
            top_k=5,
            out_dir=tmp_path,
            run_tag="once",
        )
        assert result["comparison_df"].loc["none", "n_retrains"] == 1

    def test_turnover_aware_ab_adds_benchmark(self, synthetic_csv, sim_period, tmp_path):
        start, end = sim_period
        result = run_ab_experiment(
            csv_path=synthetic_csv,
            start=start,
            end=end,
            strategies=["none"],
            top_k=5,
            portfolio_method="turnover_aware_topk",
            rebalance_every=5,
            entry_rank=10,
            exit_rank=15,
            max_turnover=0.25,
            min_holding_days=5,
            objective="net_return_proxy",
            benchmark="ew_buy_hold_universe",
            out_dir=tmp_path,
            run_tag="turnover_aware",
        )

        assert "ew_buy_hold_universe" in result["comparison_df"].index
        config = json.loads((Path(result["run_dir"]) / "config.json").read_text(encoding="utf-8"))
        sub_run = Path(config["run_dirs"]["none"])
        pnl = pd.read_csv(sub_run / "daily_pnl.csv")
        required = {
            "rebalance_flag",
            "held_from_prev_count",
            "forced_sells_count",
            "turnover_cap_applied",
        }
        assert required.issubset(pnl.columns)
        assert Path(config["benchmark_path"]).exists()

    def test_default_strategies_catalog(self):
        """DEFAULT_STRATEGIES 應包含五種預期策略（含 model_pool）。"""
        expected = {"none", "scheduled_20", "scheduled_60", "triggered", "model_pool"}
        assert set(DEFAULT_STRATEGIES) == expected
        # 各配置合法性
        for key, cfg in DEFAULT_STRATEGIES.items():
            assert "strategy" in cfg
            assert cfg["strategy"] in ("none", "scheduled", "triggered", "model_pool")

    def test_unknown_strategy_raises(self, synthetic_csv, sim_period, tmp_path):
        start, end = sim_period
        with pytest.raises(ValueError, match="未知策略"):
            run_ab_experiment(
                csv_path=synthetic_csv,
                start=start,
                end=end,
                strategies=["not_a_strategy"],
                out_dir=tmp_path,
            )


# ---------------------------------------------------------------------------
# 3. Phase A — cost sensitivity sweep（P0★ #4）
# ---------------------------------------------------------------------------

class TestCostSensitivitySweep:
    def test_cost_sweep_produces_sensitivity_csv(self, synthetic_csv, sim_period, tmp_path):
        """``cost_sweep`` 模式應產出 cost_sensitivity.csv 與 PNG。"""
        start, end = sim_period
        result = run_ab_experiment(
            csv_path=synthetic_csv,
            start=start,
            end=end,
            strategies=["none", "scheduled_60"],
            top_k=5,
            out_dir=tmp_path,
            run_tag="sweep",
            cost_sweep=[0.0, 0.4],
        )
        run_dir = Path(result["run_dir"])
        assert (run_dir / "cost_sensitivity.csv").exists()
        assert (run_dir / "cost_sensitivity.png").exists()
        assert (run_dir / "config.json").exists()

        df = pd.read_csv(result["cost_sensitivity_path"])
        assert {"cost_pct", "strategy", "sharpe", "cumulative_return_pct"}.issubset(df.columns)
        # 2 cost × 2 strategies = 4 records
        assert len(df) == 4
        assert set(df["cost_pct"].unique()) == {0.0, 0.4}
        assert set(df["strategy"].unique()) == {"none", "scheduled_60"}

    def test_zero_cost_recovers_gross_returns(self, synthetic_csv, sim_period, tmp_path):
        """``round_trip_cost_pct=0`` 時 daily_pnl 中 net_return == gross_return。"""
        start, end = sim_period
        result = run_ab_experiment(
            csv_path=synthetic_csv,
            start=start,
            end=end,
            strategies=["none"],
            top_k=5,
            out_dir=tmp_path,
            run_tag="zero",
            cost_sweep=[0.0],
        )
        run_dir = Path(result["run_dir"])
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        sub_run = Path(config["sub_runs"]["cost_0"]["none"])
        pnl = pd.read_csv(sub_run / "daily_pnl.csv")
        # net == gross 容許數值精度
        diff = (pnl["net_return"] - pnl["gross_return"]).abs().max()
        assert diff < 1e-12, f"成本=0 時 net 應 == gross，最大差距 {diff}"

    def test_higher_cost_reduces_cumulative_return(self, synthetic_csv, sim_period, tmp_path):
        """提高成本應壓低換手較高策略的 cumulative return。

        用 cumulative_return_pct 而非 Sharpe 作斷言：cost 是 systematic shift，
        必定按 turnover 壓低累積報酬；Sharpe 則因 XGBoost 訓練非確定性可能浮動。
        """
        start, end = sim_period
        result = run_ab_experiment(
            csv_path=synthetic_csv,
            start=start,
            end=end,
            strategies=["scheduled_20"],   # 高換手策略
            top_k=5,
            out_dir=tmp_path,
            run_tag="cost_increasing",
            cost_sweep=[0.0, 0.6],
        )
        df = pd.read_csv(result["cost_sensitivity_path"])
        cr_zero = df[df["cost_pct"] == 0.0]["cumulative_return_pct"].iloc[0]
        cr_high = df[df["cost_pct"] == 0.6]["cumulative_return_pct"].iloc[0]
        assert cr_high < cr_zero, (
            f"成本 0.6% 應壓低累積報酬；zero={cr_zero:.3f}%, high={cr_high:.3f}%"
        )

    def test_daily_pnl_has_split_cost_columns(self, synthetic_csv, sim_period, tmp_path):
        """baseline 模式（無 cost_sweep）daily_pnl.csv 應有四欄成本拆分。"""
        start, end = sim_period
        result = run_ab_experiment(
            csv_path=synthetic_csv,
            start=start,
            end=end,
            strategies=["none"],
            top_k=5,
            out_dir=tmp_path,
            run_tag="split_cols",
        )
        # 找到 sub-run 的 daily_pnl.csv
        config = json.loads((Path(result["run_dir"]) / "config.json").read_text(encoding="utf-8"))
        sub_run = Path(config["run_dirs"]["none"])
        pnl = pd.read_csv(sub_run / "daily_pnl.csv")
        required = {"gross_return", "commission_cost", "tax_cost", "slippage_cost", "net_return"}
        missing = required - set(pnl.columns)
        assert not missing, f"daily_pnl 缺欄位：{missing}"
