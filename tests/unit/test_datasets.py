"""Unit tests for deepvac/datasets.py: run loading, sequence building,
run-level splitting, and scaling -- shared by gru/train_gru.py and
lstm/train_lstm.py."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pytest
from deepvac import datasets
from tests.conftest import write_synthetic_run


class TestInferElapsedS:
    def test_prefers_existing_elapsed_s(self):
        df = pd.DataFrame({"elapsed_s": [5.0, 1.0, 3.0]})
        out = datasets.infer_elapsed_s(df)
        # sorted ascending, values preserved
        assert list(out["elapsed_s"]) == [1.0, 3.0, 5.0]

    def test_derives_from_timestamp(self):
        df = pd.DataFrame({"timestamp": [100.0, 102.0, 101.0]})
        out = datasets.infer_elapsed_s(df)
        assert list(out["elapsed_s"]) == [0.0, 1.0, 2.0]

    def test_falls_back_to_row_index(self):
        df = pd.DataFrame({"temp": [1.0, 2.0, 3.0]})
        out = datasets.infer_elapsed_s(df)
        np.testing.assert_array_equal(out["elapsed_s"], [0.0, 1.0, 2.0])


class TestPrepareRunDataframe:
    def _args(self, **overrides):
        defaults = dict(min_samples=5, min_duration_s=1.0)
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_derives_error_and_fills_missing_control_terms(self, tmp_path):
        run_dir = write_synthetic_run(tmp_path, "run_a", n_samples=50)
        df, meta = datasets.prepare_run_dataframe(run_dir / "run_samples.csv", self._args())
        assert meta["run_id"] == "run_a"
        assert meta["n_samples"] == 50
        np.testing.assert_allclose(df["error"], df["temp_ref"] - df["temp"])

    def test_accepts_target_temp_as_temp_ref_alias(self, tmp_path):
        df = pd.DataFrame({
            "elapsed_s": np.arange(20, dtype=float),
            "temp": np.linspace(20, 10, 20),
            "target_temp": np.full(20, 10.0),
            "kp": np.full(20, 6.0), "ki": np.full(20, 997.0), "kd": np.full(20, 16.0),
        })
        run_dir = tmp_path / "run_b"
        run_dir.mkdir()
        df.to_csv(run_dir / "run_samples.csv", index=False)

        out, _ = datasets.prepare_run_dataframe(run_dir / "run_samples.csv", self._args())
        assert (out["temp_ref"] == 10.0).all()

    def test_rejects_too_few_samples(self, tmp_path):
        run_dir = write_synthetic_run(tmp_path, "short_run", n_samples=3)
        with pytest.raises(ValueError, match="too few samples"):
            datasets.prepare_run_dataframe(run_dir / "run_samples.csv", self._args(min_samples=10))

    def test_rejects_too_short_duration(self, tmp_path):
        run_dir = write_synthetic_run(tmp_path, "brief_run", n_samples=50, dt_s=0.01)
        with pytest.raises(ValueError, match="duration too short"):
            datasets.prepare_run_dataframe(run_dir / "run_samples.csv", self._args(min_duration_s=1000.0))

    def test_rejects_missing_required_column(self, tmp_path):
        # Has temp_ref but no kp/ki/kd -- should fail specifically on kp.
        df = pd.DataFrame({"elapsed_s": [0, 1, 2], "temp": [1, 2, 3], "temp_ref": [0, 0, 0]})
        run_dir = tmp_path / "no_pid"
        run_dir.mkdir()
        df.to_csv(run_dir / "run_samples.csv", index=False)
        with pytest.raises(ValueError, match="kp"):
            datasets.prepare_run_dataframe(run_dir / "run_samples.csv", self._args())


class TestBuildSequences:
    def _args(self, window_steps):
        return argparse.Namespace(window_steps=window_steps)

    def test_shapes_match_window_and_feature_count(self, tmp_path):
        run_dir = write_synthetic_run(tmp_path, "run_a", n_samples=50)
        df, _ = datasets.prepare_run_dataframe(
            run_dir / "run_samples.csv", argparse.Namespace(min_samples=5, min_duration_s=1.0)
        )
        X, y, meta = datasets.build_sequences(df, "run_a", self._args(window_steps=10))
        n_features = len(datasets.FEATURE_NAMES)
        assert X.shape == (len(df) - 10, 10, n_features)
        assert y.shape == (len(df) - 10, 1)
        assert len(meta) == len(X)

    def test_target_is_next_step_temp_delta(self, tmp_path):
        run_dir = write_synthetic_run(tmp_path, "run_a", n_samples=30)
        df, _ = datasets.prepare_run_dataframe(
            run_dir / "run_samples.csv", argparse.Namespace(min_samples=5, min_duration_s=1.0)
        )
        X, y, meta = datasets.build_sequences(df, "run_a", self._args(window_steps=5))
        temp = df["temp"].to_numpy()
        for i in range(len(y)):
            end_idx = int(meta.iloc[i]["end_idx"])
            expected_delta = temp[end_idx + 1] - temp[end_idx]
            assert y[i, 0] == pytest.approx(expected_delta, abs=1e-4)

    def test_too_short_run_yields_empty_arrays(self):
        df = pd.DataFrame({
            "temp": [1.0, 2.0], "temp_ref": [0.0, 0.0], "error": [-1.0, -2.0], "elapsed_s": [0.0, 1.0],
            **{name: [0.0, 0.0] for name in datasets.FEATURE_NAMES if name not in ("temp", "temp_ref", "error")},
        })
        X, y, meta = datasets.build_sequences(df, "short", self._args(window_steps=10))
        assert X.shape == (0, 10, len(datasets.FEATURE_NAMES))
        assert y.shape == (0, 1)
        assert meta.empty


class TestSplitRuns:
    def test_produces_nonoverlapping_partition_of_all_runs(self):
        run_ids = [f"run_{i}" for i in range(10)]
        train, val, test = datasets.split_runs(run_ids, train_fraction=0.6, val_fraction=0.2, seed=0)
        assert sorted(train + val + test) == sorted(run_ids)
        assert not (set(train) & set(val))
        assert not (set(train) & set(test))
        assert not (set(val) & set(test))

    def test_every_split_nonempty(self):
        train, val, test = datasets.split_runs([f"run_{i}" for i in range(5)], 0.8, 0.1, seed=0)
        assert train and val and test

    def test_deterministic_given_seed(self):
        run_ids = [f"run_{i}" for i in range(8)]
        a = datasets.split_runs(run_ids, 0.6, 0.2, seed=42)
        b = datasets.split_runs(run_ids, 0.6, 0.2, seed=42)
        assert a == b

    def test_rejects_fewer_than_three_runs(self):
        with pytest.raises(RuntimeError):
            datasets.split_runs(["only_one", "only_two"], 0.6, 0.2, seed=0)

    def test_deduplicates_run_ids(self):
        train, val, test = datasets.split_runs(["a", "a", "b", "c"], 0.34, 0.33, seed=0)
        assert sorted(train + val + test) == ["a", "b", "c"]


class TestScaleDatasets:
    def test_round_trips_through_inverse_transform(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 5, 3)).astype(np.float32)
        y = rng.normal(size=(20, 1)).astype(np.float32)

        X_s, y_s, _, _, _, _, x_scaler, y_scaler = datasets.scale_datasets(X, y, X, y, X, y)

        recovered_y = y_scaler.inverse_transform(y_s)
        np.testing.assert_allclose(recovered_y, y, atol=1e-4)

    def test_scaled_training_data_is_roughly_standardized(self):
        rng = np.random.default_rng(0)
        X = (rng.normal(size=(200, 4, 3)) * 10 + 5).astype(np.float32)
        y = (rng.normal(size=(200, 1)) * 2 + 1).astype(np.float32)

        X_s, *_ = datasets.scale_datasets(X, y, X, y, X, y)
        assert abs(X_s.mean()) < 0.1
        assert abs(X_s.std() - 1.0) < 0.1
