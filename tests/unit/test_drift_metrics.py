"""Unit tests for WP1 drift metrics: PSI, calibration_error, KS edge cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.metrics import (
    calibration_error,
    ks_test_drift,
    population_stability_index,
)


class TestPopulationStabilityIndex:
    def test_identical_distributions_psi_near_zero(self):
        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, size=5000)
        cur = rng.normal(0, 1, size=5000)
        psi = population_stability_index(ref, cur, n_bins=10)
        assert psi < 0.05, f"identical dists should give small PSI, got {psi}"

    def test_shifted_mean_triggers_moderate_psi(self):
        rng = np.random.default_rng(1)
        ref = rng.normal(0, 1, size=5000)
        cur = rng.normal(0.5, 1, size=5000)  # mean shift
        psi = population_stability_index(ref, cur, n_bins=10)
        assert 0.05 < psi, f"mean shift should raise PSI above noise, got {psi}"

    def test_major_shift_exceeds_critical_threshold(self):
        rng = np.random.default_rng(2)
        ref = rng.normal(0, 1, size=5000)
        cur = rng.normal(3, 2, size=5000)  # large shift + scale
        psi = population_stability_index(ref, cur, n_bins=10)
        assert psi > 0.25, f"major shift should cross critical (0.25), got {psi}"

    def test_small_sample_returns_zero(self):
        ref = np.array([1.0, 2.0, 3.0])
        cur = np.array([1.1, 2.1, 2.9])
        psi = population_stability_index(ref, cur, n_bins=10)
        assert psi == 0.0

    def test_nan_handling(self):
        rng = np.random.default_rng(3)
        ref = rng.normal(0, 1, size=1000)
        cur = rng.normal(0, 1, size=1000)
        ref[::10] = np.nan
        cur[::10] = np.nan
        psi = population_stability_index(ref, cur, n_bins=10)
        assert not np.isnan(psi)
        assert psi >= 0.0

    def test_constant_reference_safe_degenerate(self):
        ref = np.ones(1000)
        cur = np.linspace(-1, 1, 1000)
        psi = population_stability_index(ref, cur, n_bins=10)
        # Degenerate reference: function must not crash and returns finite value
        assert np.isfinite(psi)


class TestCalibrationError:
    def test_perfectly_calibrated_low_ece(self):
        rng = np.random.default_rng(4)
        # Scores drawn such that sign(score) matches sign(actual) ~ 70% — reasonable calibration
        n = 1000
        scores = rng.normal(0, 1, size=n)
        actuals = np.where(rng.random(n) < 0.6, np.abs(scores), -np.abs(scores))
        preds = pd.Series(scores)
        acts = pd.Series(actuals)
        ece = calibration_error(preds, acts, n_bins=10)
        assert 0.0 <= ece <= 0.5

    def test_empty_input_returns_zero(self):
        preds = pd.Series([np.nan, np.nan])
        acts = pd.Series([np.nan, np.nan])
        ece = calibration_error(preds, acts, n_bins=10)
        assert ece == 0.0

    def test_bad_calibration_higher_than_good(self):
        rng = np.random.default_rng(5)
        n = 2000
        # Good: aligned
        scores_good = rng.normal(0, 1, n)
        acts_good = np.sign(scores_good) * np.abs(rng.normal(0, 1, n))
        # Bad: inverted
        scores_bad = rng.normal(0, 1, n)
        acts_bad = -np.sign(scores_bad) * np.abs(rng.normal(0, 1, n))

        ece_good = calibration_error(pd.Series(scores_good), pd.Series(acts_good))
        ece_bad = calibration_error(pd.Series(scores_bad), pd.Series(acts_bad))
        assert ece_bad > ece_good


class TestKSDriftEdgeCases:
    def test_too_few_samples_returns_no_drift(self):
        ref = np.array([1.0, 2.0])
        cur = np.array([3.0, 4.0])
        stat, pval = ks_test_drift(ref, cur)
        assert stat == 0.0 and pval == 1.0

    def test_same_distribution_high_pvalue(self):
        rng = np.random.default_rng(6)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(0, 1, 1000)
        _, pval = ks_test_drift(ref, cur)
        assert pval > 0.05

    def test_large_shift_rejects(self):
        rng = np.random.default_rng(7)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(3, 1, 1000)
        _, pval = ks_test_drift(ref, cur)
        assert pval < 0.01
