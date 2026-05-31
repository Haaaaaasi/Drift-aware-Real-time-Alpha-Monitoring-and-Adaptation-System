from pathlib import Path

import pytest

from pipelines.simulate_recent import _decide_retrain
from src.config.frozen_alpha_selector import load_frozen_alpha_selector


FROZEN_CONFIG = Path("configs/frozen_alpha_selector_20260517.yaml")


def test_frozen_config_maps_to_official_simulation_overrides():
    spec = load_frozen_alpha_selector(FROZEN_CONFIG)

    overrides = spec.simulation_overrides("primary")

    assert spec.frozen_selector_id == "incumbent55_rolling_topk20_w126_pen10_20260517"
    assert overrides["csv_path"] == Path("data/tw_stocks_tej.parquet")
    assert overrides["selector"] == "rolling_topk"
    assert overrides["selector_alpha_top_k"] == 20
    assert overrides["selector_window_days"] == 126
    assert overrides["selector_stability_penalty"] == pytest.approx(0.10)
    assert overrides["selector_admission_gate"] is False
    assert overrides["exclude_indclass_cap_alphas"] is True
    assert overrides["portfolio_method"] == "turnover_aware_topk"
    assert overrides["top_k"] == 10
    assert overrides["retrain_every"] == 20
    assert overrides["train_window_days"] == 500
    assert overrides["execution_price"] == "next_vwap"
    assert overrides["round_trip_cost_pct"] is None


def test_frozen_config_secondary_execution_is_next_open():
    spec = load_frozen_alpha_selector(FROZEN_CONFIG)

    overrides = spec.simulation_overrides("secondary")

    assert overrides["execution_price"] == "next_open"
    assert spec.metadata("secondary")["frozen_execution_price"] == "next_open"


def test_frozen_config_rejects_unknown_execution_mode():
    spec = load_frozen_alpha_selector(FROZEN_CONFIG)

    with pytest.raises(ValueError, match="frozen_execution"):
        spec.simulation_overrides("close")


def test_model_pool_scheduled_trigger_uses_retrain_every_cadence():
    need_retrain, reason = _decide_retrain(
        strategy="model_pool",
        model_pool_trigger_mode="scheduled",
        model=object(),
        day_idx=40,
        last_train_idx=20,
        retrain_every=20,
        min_retrain_gap=999,
        pnl_records=[],
        adapter=object(),
        rolling_window=20,
    )

    assert need_retrain is True
    assert reason == "model_pool_scheduled_every_20d"
