"""Unit tests for deepvac/metrics.py: cost functions + GP-BO acquisition math."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from deepvac import metrics


class TestAppendMaeColumn:
    def test_computes_absolute_error(self):
        df = pd.DataFrame({"temp": [1.0, 5.0], "temp_ref": [3.0, 2.0]})
        out = metrics.append_mae_column(df)
        np.testing.assert_array_equal(out["mae"], [2.0, 3.0])

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"temp": [1.0], "temp_ref": [2.0]})
        metrics.append_mae_column(df)
        assert "mae" not in df.columns

    def test_rejects_missing_columns(self):
        with pytest.raises(ValueError):
            metrics.append_mae_column(pd.DataFrame({"temp": [1.0]}))


class TestComputeTailCost:
    def _df(self, temps, target=0.0, dt=1.0):
        n = len(temps)
        return pd.DataFrame({
            "timestamp": np.arange(n) * dt,
            "temp": temps,
            "temp_ref": np.full(n, target),
        })

    def test_zero_cost_when_settled_exactly_on_target(self):
        result = metrics.compute_tail_cost(self._df([0.0, 0.0, 0.0]))
        assert result["cost"] == pytest.approx(0.0)
        assert result["tail_mae"] == pytest.approx(0.0)
        assert result["overshoot"] == pytest.approx(0.0)

    def test_huge_cost_when_never_entering_band(self):
        # Starts at 100, cooling toward 0, never gets within entry_band=2.0.
        result = metrics.compute_tail_cost(self._df([100.0, 90.0, 80.0]))
        assert result["cost"] == 1e9
        assert result["tail_mae"] is None

    def test_direction_cooling_vs_heating(self):
        cooling = metrics.compute_tail_cost(self._df([10.0, 1.0, 0.0], target=0.0))
        heating = metrics.compute_tail_cost(self._df([-10.0, -1.0, 0.0], target=0.0))
        assert cooling["direction"] == 1.0
        assert heating["direction"] == -1.0

    def test_overshoot_penalized_quadratically(self):
        # Cooling toward 0: entering the band then overshooting below 0.
        mild = metrics.compute_tail_cost(self._df([5.0, 1.0, -0.5]), overshoot_weight=10.0)
        severe = metrics.compute_tail_cost(self._df([5.0, 1.0, -2.0]), overshoot_weight=10.0)
        assert severe["overshoot"] > mild["overshoot"]
        assert severe["cost"] > mild["cost"]

    def test_rejects_missing_columns(self):
        with pytest.raises(ValueError):
            metrics.compute_tail_cost(pd.DataFrame({"temp": [1.0]}))


class TestAcquisitionMath:
    def test_expected_improvement_favors_lower_predicted_cost(self):
        mu = np.array([1.0, 5.0])
        sigma = np.array([1.0, 1.0])
        ei = metrics.expected_improvement(mu, sigma, y_best=5.0)
        assert ei[0] > ei[1]

    def test_expected_improvement_zero_when_sigma_negligible(self):
        ei = metrics.expected_improvement(np.array([10.0]), np.array([1e-15]), y_best=1.0)
        assert ei[0] == pytest.approx(0.0)

    def test_expected_improvement_grows_with_uncertainty_at_fixed_mean(self):
        mu = np.array([5.0, 5.0])
        sigma = np.array([0.1, 5.0])
        ei = metrics.expected_improvement(mu, sigma, y_best=5.0)
        assert ei[1] > ei[0]

    def test_lower_confidence_bound_decreases_with_kappa(self):
        mu, sigma = np.array([5.0]), np.array([2.0])
        lcb_low_kappa = metrics.lower_confidence_bound(mu, sigma, kappa=0.0)
        lcb_high_kappa = metrics.lower_confidence_bound(mu, sigma, kappa=3.0)
        assert lcb_high_kappa[0] < lcb_low_kappa[0]

    def test_expected_information_gain_increases_with_sigma(self):
        gains = metrics.expected_information_gain(np.array([0.1, 1.0, 10.0]))
        assert gains[0] < gains[1] < gains[2]

    def test_normal_pdf_peaks_at_zero(self):
        z = np.array([-2.0, 0.0, 2.0])
        pdf = metrics.normal_pdf(z)
        assert pdf[1] > pdf[0]
        assert pdf[1] > pdf[2]

    def test_normal_cdf_monotonic_and_bounded(self):
        cdf = metrics.normal_cdf(np.array([-3.0, 0.0, 3.0]))
        assert cdf[0] < cdf[1] < cdf[2]
        assert 0.0 <= cdf[0] <= 1.0
        assert cdf[1] == pytest.approx(0.5, abs=1e-6)


class TestGpModelFitAndSuggest:
    def _runs_df(self, n=12, seed=0):
        rng = np.random.default_rng(seed)
        kp = rng.uniform(1, 50, size=n)
        ki = rng.uniform(1, 1000, size=n)
        kd = rng.uniform(1, 20, size=n)
        # A simple synthetic response surface with a clear minimum.
        mse = (kp - 6.0) ** 2 + ((ki - 997.0) / 100.0) ** 2 + (kd - 16.0) ** 2
        return pd.DataFrame({"kp": kp, "ki": ki, "kd": kd, "mse": mse})

    def test_fit_gp_model_requires_at_least_three_runs(self):
        with pytest.raises(ValueError):
            metrics.fit_gp_model(self._runs_df(n=2))

    def test_fit_gp_model_rejects_missing_columns(self):
        with pytest.raises(ValueError):
            metrics.fit_gp_model(pd.DataFrame({"kp": [1, 2, 3]}))

    def test_fit_gp_model_reports_best_observed_mse(self):
        df = self._runs_df()
        model = metrics.fit_gp_model(df)
        assert model["best_mse"] == pytest.approx(df["mse"].min())
        assert model["n_samples"] == len(df)

    def test_suggest_next_params_stays_within_bounds(self):
        model = metrics.fit_gp_model(self._runs_df())
        bounds = {"kp": (1.0, 50.0), "ki": (1.0, 1000.0), "kd": (1.0, 20.0)}
        suggestion = metrics.suggest_next_params(model, bounds, n_candidates=200, random_state=1)
        assert 1.0 <= suggestion["kp"] <= 50.0
        assert 1.0 <= suggestion["ki"] <= 1000.0
        assert 1.0 <= suggestion["kd"] <= 20.0

    def test_suggest_next_params_deterministic_given_seed(self):
        model = metrics.fit_gp_model(self._runs_df())
        bounds = {"kp": (1.0, 50.0), "ki": (1.0, 1000.0), "kd": (1.0, 20.0)}
        a = metrics.suggest_next_params(model, bounds, n_candidates=200, random_state=7)
        b = metrics.suggest_next_params(model, bounds, n_candidates=200, random_state=7)
        assert a == b
