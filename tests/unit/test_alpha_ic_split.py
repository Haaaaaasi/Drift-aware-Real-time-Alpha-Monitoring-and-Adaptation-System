"""P0★ #1 — IS/OOS 切分邏輯單元測試。

測試 notebooks/01_alpha_ic_analysis.py 的 IS/OOS 切分相關 helper：
1. split_panel_by_time：train_end 切點正確、IS/OOS 不重疊、邊界 inclusive 規則
2. compute_alpha_row：樣本不足回 None、足夠時回 dict 含預期欄位
3. build_oos_validation：selected_in_is 標記正確、sign_flip 偵測、缺 OOS 時 NaN 處理
4. emit_selection_outputs：split mode 輸出含 split metadata + validation CSV；非 split mode 行為與舊版一致

由於 notebook 檔名以數字開頭，需以 importlib 載入。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = ROOT / "notebooks" / "01_alpha_ic_analysis.py"


def _load_notebook_module():
    """以 importlib 載入 notebook 為 module（檔名數字開頭無法直接 import）。"""
    spec = importlib.util.spec_from_file_location("nb_alpha_ic", NOTEBOOK_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["nb_alpha_ic"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def nb():
    return _load_notebook_module()


# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------

def _make_panel(n_days: int = 100, n_stocks: int = 8, n_alphas: int = 3, seed: int = 0):
    """產生 (alpha_panel, fwd_returns) 合成資料。

    alpha_panel: long format 含 (security_id, tradetime, alpha_id, alpha_value)
    fwd_returns: index=(security_id, tradetime) 的 Series

    透過 alpha0 = fwd + small noise 確保有可預測訊號（IC > 0）。
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    sids = [f"S{i:03d}" for i in range(n_stocks)]
    rows = []
    fwd_records = []
    for d in dates:
        for s in sids:
            r = rng.normal(0, 0.01)
            fwd_records.append({"security_id": s, "tradetime": d, "fwd": r})
            for k in range(n_alphas):
                if k == 0:
                    val = r + rng.normal(0, 0.005)  # 強訊號
                elif k == 1:
                    val = -r + rng.normal(0, 0.01)  # 反向弱訊號
                else:
                    val = rng.normal(0, 1)  # 純噪音
                rows.append({
                    "security_id": s,
                    "tradetime": d,
                    "alpha_id": f"a{k}",
                    "alpha_value": val,
                })
    panel = pd.DataFrame(rows)
    fwd = (
        pd.DataFrame(fwd_records)
        .set_index(["security_id", "tradetime"])["fwd"]
    )
    return panel, fwd


# ---------------------------------------------------------------------------
# split_panel_by_time
# ---------------------------------------------------------------------------

class TestSplitPanelByTime:
    def test_split_at_midpoint_no_overlap(self, nb):
        panel, fwd = _make_panel(n_days=20)
        train_end = pd.Timestamp("2024-01-15")
        is_panel, fwd_is, oos_panel, fwd_oos = nb.split_panel_by_time(panel, fwd, train_end)
        # 不重疊
        assert is_panel["tradetime"].max() <= train_end
        assert oos_panel["tradetime"].min() > train_end
        # 加總等於原長度
        assert len(is_panel) + len(oos_panel) == len(panel)
        assert len(fwd_is) + len(fwd_oos) == len(fwd)

    def test_train_end_inclusive_for_is(self, nb):
        """train_end 當天的列應屬於 IS（<= 規則）。"""
        panel, fwd = _make_panel(n_days=20)
        # 取一個 panel 中存在的日期
        train_end = panel["tradetime"].iloc[len(panel) // 2]
        is_panel, fwd_is, oos_panel, fwd_oos = nb.split_panel_by_time(panel, fwd, train_end)
        assert (is_panel["tradetime"] == train_end).any(), "train_end 當天應在 IS 中"
        assert not (oos_panel["tradetime"] == train_end).any(), "train_end 當天不應出現在 OOS"

    def test_split_before_all_dates_yields_empty_is(self, nb):
        panel, fwd = _make_panel(n_days=10)
        train_end = pd.Timestamp("2020-01-01")  # 早於所有資料
        is_panel, fwd_is, oos_panel, fwd_oos = nb.split_panel_by_time(panel, fwd, train_end)
        assert len(is_panel) == 0
        assert len(fwd_is) == 0
        assert len(oos_panel) == len(panel)


# ---------------------------------------------------------------------------
# compute_alpha_row
# ---------------------------------------------------------------------------

class TestComputeAlphaRow:
    def test_returns_none_when_below_min_obs(self, nb):
        idx = pd.MultiIndex.from_product([["S1"], pd.date_range("2024-01-01", periods=5)])
        vals = pd.Series([0.1] * 5, index=idx, name="alpha_value")
        vals.index = vals.index.set_names(["security_id", "tradetime"])
        fwd = pd.Series([0.01] * 5, index=idx, name="fwd")
        fwd.index = fwd.index.set_names(["security_id", "tradetime"])
        result = nb.compute_alpha_row("a0", vals, fwd, min_obs=50)
        assert result is None

    def test_returns_expected_keys_with_signal(self, nb):
        panel, fwd = _make_panel(n_days=60, n_stocks=10, n_alphas=1)
        vals = panel.set_index(["security_id", "tradetime"])["alpha_value"]
        row = nb.compute_alpha_row("a0", vals, fwd)
        assert row is not None
        assert set(row.keys()) == {"alpha_id", "n", "ic", "rank_ic", "hit_rate", "coverage", "abs_ic"}
        assert row["alpha_id"] == "a0"
        assert row["n"] > 0
        assert row["ic"] > 0  # 合成資料 alpha0 有正訊號


# ---------------------------------------------------------------------------
# build_oos_validation
# ---------------------------------------------------------------------------

class TestBuildOosValidation:
    def test_selected_flag_and_decay_columns(self, nb):
        summary_is = pd.DataFrame([
            {"alpha_id": "a0", "n": 100, "ic": 0.10, "rank_ic": 0.12, "hit_rate": 0.55, "coverage": 1.0, "abs_ic": 0.10},
            {"alpha_id": "a1", "n": 100, "ic": 0.005, "rank_ic": 0.005, "hit_rate": 0.51, "coverage": 1.0, "abs_ic": 0.005},
        ])
        summary_oos = pd.DataFrame([
            {"alpha_id": "a0", "n": 50, "ic": 0.06, "rank_ic": 0.08, "hit_rate": 0.53, "coverage": 1.0, "abs_ic": 0.06},
            {"alpha_id": "a1", "n": 50, "ic": -0.01, "rank_ic": -0.02, "hit_rate": 0.49, "coverage": 1.0, "abs_ic": 0.01},
        ])
        validation = nb.build_oos_validation(summary_is, summary_oos, selected_ids=["a0"])
        # 找到 a0 與 a1 的列
        a0 = validation[validation["alpha_id"] == "a0"].iloc[0]
        a1 = validation[validation["alpha_id"] == "a1"].iloc[0]
        assert a0["selected_in_is"] is True or a0["selected_in_is"] == True  # noqa
        assert a1["selected_in_is"] is False or a1["selected_in_is"] == False  # noqa
        # rank_ic_decay = oos - is
        assert a0["rank_ic_decay"] == pytest.approx(0.08 - 0.12)
        # sign_flip：a1 IS=正、OOS=負
        assert a1["sign_flip"] is True or a1["sign_flip"] == True  # noqa
        assert a0["sign_flip"] is False or a0["sign_flip"] == False  # noqa

    def test_missing_oos_alpha_yields_nan(self, nb):
        summary_is = pd.DataFrame([
            {"alpha_id": "a0", "n": 100, "ic": 0.1, "rank_ic": 0.1, "hit_rate": 0.5, "coverage": 1.0, "abs_ic": 0.1},
        ])
        summary_oos = pd.DataFrame()  # OOS 無任何 alpha 達標
        validation = nb.build_oos_validation(summary_is, summary_oos, selected_ids=["a0"])
        row = validation.iloc[0]
        assert np.isnan(row["oos_ic"])
        assert np.isnan(row["oos_rank_ic"])
        assert np.isnan(row["rank_ic_decay"])
        assert row["sign_flip"] is False or row["sign_flip"] == False  # noqa


# ---------------------------------------------------------------------------
# emit_selection_outputs — split vs full mode
# ---------------------------------------------------------------------------

class TestEmitSelectionOutputs:
    def test_split_mode_writes_validation_and_metadata(self, nb, tmp_path, monkeypatch):
        monkeypatch.setattr(nb, "REPORT_DIR", tmp_path)
        train_end = pd.Timestamp("2024-02-15")

        summary_is = pd.DataFrame([
            {"alpha_id": "a0", "n": 200, "ic": 0.05, "rank_ic": 0.06, "hit_rate": 0.55, "coverage": 1.0, "abs_ic": 0.05},
            {"alpha_id": "a1", "n": 200, "ic": 0.002, "rank_ic": 0.003, "hit_rate": 0.50, "coverage": 1.0, "abs_ic": 0.002},
        ])
        summary_oos = pd.DataFrame([
            {"alpha_id": "a0", "n": 100, "ic": 0.04, "rank_ic": 0.05, "hit_rate": 0.54, "coverage": 1.0, "abs_ic": 0.04},
        ])
        split_meta = nb._build_split_meta(
            train_end, summary_is, summary_oos,
            (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-15")),
            (pd.Timestamp("2024-02-16"), pd.Timestamp("2024-03-31")),
        )

        effective, selection = nb.emit_selection_outputs(
            summary_is,
            horizon=5, source="python", universe_desc="test",
            train_end=train_end, summary_oos=summary_oos,
            split_meta=split_meta,
        )

        # 寫了兩個 artifact
        assert (tmp_path / "effective_alphas.json").exists()
        assert (tmp_path / "effective_alphas_oos_validation.csv").exists()

        # JSON 含 split metadata
        data = json.loads((tmp_path / "effective_alphas.json").read_text(encoding="utf-8"))
        assert data["selection_basis"] == "in_sample"
        assert data["split"] is not None
        assert data["split"]["train_end"] == "2024-02-15"
        assert "is_window" in data["split"]
        assert "oos_window" in data["split"]
        # 下游 pipeline 仍能讀出 effective_alphas
        assert "a0" in data["effective_alphas"]
        assert "a1" not in data["effective_alphas"]  # rank_ic < 0.01

        # validation CSV 結構
        val_df = pd.read_csv(tmp_path / "effective_alphas_oos_validation.csv")
        assert "selected_in_is" in val_df.columns
        assert "rank_ic_decay" in val_df.columns
        assert "sign_flip" in val_df.columns

    def test_full_mode_no_validation_no_split(self, nb, tmp_path, monkeypatch):
        """不指定 train_end 時應與舊行為一致：只有 effective_alphas.json，無 validation CSV。"""
        monkeypatch.setattr(nb, "REPORT_DIR", tmp_path)
        summary = pd.DataFrame([
            {"alpha_id": "a0", "n": 300, "ic": 0.05, "rank_ic": 0.06, "hit_rate": 0.55, "coverage": 1.0, "abs_ic": 0.05},
        ])
        effective, selection = nb.emit_selection_outputs(
            summary,
            horizon=5, source="python", universe_desc="test",
            train_end=None, summary_oos=None, split_meta=None,
        )
        assert (tmp_path / "effective_alphas.json").exists()
        assert not (tmp_path / "effective_alphas_oos_validation.csv").exists()
        data = json.loads((tmp_path / "effective_alphas.json").read_text(encoding="utf-8"))
        assert data["selection_basis"] == "full_sample"
        assert data["split"] is None
        assert data["effective_alphas"] == ["a0"]


# ---------------------------------------------------------------------------
# End-to-end: split actually changes selection on synthetic data
# ---------------------------------------------------------------------------

class TestSplitChangesSelection:
    def test_alpha_with_signal_only_in_is_dropped_in_oos(self, nb):
        """合成資料：alpha 在 IS 期有訊號，OOS 期翻轉——應被 IS 選上但 OOS 出現 sign_flip。"""
        rng = np.random.default_rng(42)
        n_days = 120
        n_stocks = 8
        dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
        sids = [f"S{i:02d}" for i in range(n_stocks)]
        rows, fwd_rows = [], []
        train_end = dates[n_days // 2]
        for d in dates:
            in_is = d <= train_end
            for s in sids:
                r = rng.normal(0, 0.01)
                fwd_rows.append({"security_id": s, "tradetime": d, "fwd": r})
                # IS 期 alpha 與 fwd 同向；OOS 期反向（concept drift 模擬）
                sign = 1.0 if in_is else -1.0
                val = sign * r + rng.normal(0, 0.003)
                rows.append({"security_id": s, "tradetime": d, "alpha_id": "drifty", "alpha_value": val})
        panel = pd.DataFrame(rows)
        fwd = pd.DataFrame(fwd_rows).set_index(["security_id", "tradetime"])["fwd"]

        is_panel, fwd_is, oos_panel, fwd_oos = nb.split_panel_by_time(panel, fwd, train_end)
        s_is = nb.compute_per_alpha_metrics(is_panel, fwd_is)
        s_oos = nb.compute_per_alpha_metrics(oos_panel, fwd_oos)
        eff = nb.select_effective_alphas(s_is)
        validation = nb.build_oos_validation(s_is, s_oos, eff)

        # IS 應有訊號（rank_ic > 0），OOS 反向
        assert s_is["rank_ic"].iloc[0] > 0.05
        assert s_oos["rank_ic"].iloc[0] < -0.05
        # 應出現 sign flip
        assert validation["sign_flip"].iloc[0] == True  # noqa: E712
