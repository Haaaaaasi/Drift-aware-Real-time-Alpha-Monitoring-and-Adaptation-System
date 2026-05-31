import numpy as np
import pandas as pd

from pipelines.simulate_recent import _apply_placebo_to_signals


def test_shuffle_signal_preserves_score_distribution_and_updates_direction():
    signals = pd.DataFrame({
        "security_id": ["1", "2", "3", "4"],
        "signal_score": [-0.3, -0.1, 0.2, 0.4],
        "signal_direction": [-1, -1, 1, 1],
        "confidence": [0.3, 0.1, 0.2, 0.4],
    })

    shuffled = _apply_placebo_to_signals(
        signals,
        placebo_mode="shuffle_signal",
        rng=np.random.default_rng(7),
    )

    assert sorted(shuffled["signal_score"].tolist()) == sorted(signals["signal_score"].tolist())
    assert not shuffled["signal_score"].equals(signals["signal_score"])
    assert shuffled["signal_direction"].tolist() == [
        1 if score >= 0 else -1 for score in shuffled["signal_score"]
    ]
    assert shuffled["confidence"].tolist() == [abs(score) for score in shuffled["signal_score"]]


def test_placebo_none_returns_original_frame():
    signals = pd.DataFrame({"signal_score": [0.1, 0.2]})
    out = _apply_placebo_to_signals(
        signals,
        placebo_mode="none",
        rng=np.random.default_rng(1),
    )

    assert out is signals
