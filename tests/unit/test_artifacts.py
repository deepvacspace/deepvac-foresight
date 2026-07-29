"""Unit tests for deepvac/artifacts.py: run-id/CSV/JSON persistence, run-dir
discovery, and batch-scenario helpers."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from deepvac import artifacts


class TestMakeRunId:
    def test_has_expected_shape(self):
        run_id = artifacts.make_run_id("mpc")
        prefix, timestamp, suffix = run_id.split("_")
        assert prefix == "mpc"
        assert timestamp.isdigit()
        assert len(suffix) == 8

    def test_default_prefix(self):
        assert artifacts.make_run_id().startswith("run_")

    def test_calls_are_unique(self):
        ids = {artifacts.make_run_id() for _ in range(20)}
        assert len(ids) == 20


class TestSaveJson:
    def test_writes_and_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "summary.json"
        artifacts.save_json(str(path), {"a": 1, "b": [1, 2, 3]})
        assert json.loads(path.read_text()) == {"a": 1, "b": [1, 2, 3]}


class TestAppendRowsCsv:
    def test_noop_on_empty_rows(self, tmp_path):
        path = tmp_path / "out.csv"
        artifacts.append_rows_csv(str(path), [])
        assert not path.exists()

    def test_writes_header_once(self, tmp_path):
        path = tmp_path / "out.csv"
        artifacts.append_rows_csv(str(path), [{"a": 1, "b": 2}])
        artifacts.append_rows_csv(str(path), [{"a": 3, "b": 4}])
        df = pd.read_csv(path)
        assert list(df["a"]) == [1, 3]
        assert list(df["b"]) == [2, 4]


class TestAppendRowCsv:
    def test_creates_file_with_header(self, tmp_path):
        path = tmp_path / "out.csv"
        artifacts.append_row_csv(str(path), {"kp": 6, "ki": 997})
        df = pd.read_csv(path)
        assert list(df.columns) == ["kp", "ki"]
        assert df.iloc[0].to_dict() == {"kp": 6, "ki": 997}

    def test_aligns_new_row_to_existing_header_ignoring_extra_keys(self, tmp_path):
        path = tmp_path / "out.csv"
        artifacts.append_row_csv(str(path), {"kp": 6, "ki": 997})
        # Second row has an extra key not in the header -- should be dropped,
        # not raise or corrupt the file.
        artifacts.append_row_csv(str(path), {"kp": 7, "ki": 998, "extra": "ignored"})
        df = pd.read_csv(path)
        assert list(df.columns) == ["kp", "ki"]
        assert list(df["kp"]) == [6, 7]

    def test_aligns_new_row_missing_a_header_column(self, tmp_path):
        path = tmp_path / "out.csv"
        artifacts.append_row_csv(str(path), {"kp": 6, "ki": 997})
        artifacts.append_row_csv(str(path), {"kp": 7})
        df = pd.read_csv(path)
        assert df["ki"].isna().iloc[1]


class TestHistoryRunFile:
    def test_builds_expected_path(self):
        path = artifacts.history_run_file("run_123", "run_samples.csv")
        assert path.replace("\\", "/") == "history/run_123/run_samples.csv"

    def test_uses_only_the_filename_component(self):
        path = artifacts.history_run_file("run_123", "some/dir/run_samples.csv")
        assert path.replace("\\", "/") == "history/run_123/run_samples.csv"

    def test_respects_custom_folder_name(self):
        path = artifacts.history_run_file("run_123", "f.csv", folder_name="archive")
        assert path.replace("\\", "/") == "archive/run_123/f.csv"


class TestLoadRunsTable:
    def test_loads_direct_csv_when_present(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = tmp_path / "runs.csv"
        pd.DataFrame({"run_id": ["a"], "kp": [1], "ki": [2], "kd": [3], "mse": [0.1]}).to_csv(path, index=False)
        df = artifacts.load_runs_table(str(path))
        assert len(df) == 1

    def test_falls_back_to_per_run_history_scan(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "history" / "run_a").mkdir(parents=True)
        (tmp_path / "history" / "run_b").mkdir(parents=True)
        pd.DataFrame({"run_id": ["a"], "kp": [1], "ki": [2], "kd": [3], "mse": [0.1]}).to_csv(
            tmp_path / "history" / "run_a" / "runs.csv", index=False
        )
        pd.DataFrame({"run_id": ["b"], "kp": [4], "ki": [5], "kd": [6], "mse": [0.2]}).to_csv(
            tmp_path / "history" / "run_b" / "runs.csv", index=False
        )
        df = artifacts.load_runs_table("runs.csv")
        assert len(df) == 2

    def test_returns_empty_frame_with_expected_columns_when_nothing_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        df = artifacts.load_runs_table("does_not_exist.csv")
        assert list(df.columns) == ["run_id", "kp", "ki", "kd", "mse"]
        assert df.empty


class TestIterRunDirs:
    def test_yields_only_directories_sorted(self, tmp_path):
        (tmp_path / "run_b").mkdir()
        (tmp_path / "run_a").mkdir()
        (tmp_path / "not_a_dir.txt").write_text("x")
        dirs = [p.name for p in artifacts.iter_run_dirs(tmp_path)]
        assert dirs == ["run_a", "run_b"]

    def test_rejects_missing_root(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            list(artifacts.iter_run_dirs(tmp_path / "does_not_exist"))


class TestBatchScenarioHelpers:
    def test_cli_flag_name(self):
        assert artifacts.cli_flag_name("mpc_horizon_s") == "--mpc-horizon-s"
        assert artifacts.cli_flag_name("w_abs_error") == "--w-abs-error"

    def test_scenario_to_command_handles_bools_and_values(self):
        cmd = artifacts.scenario_to_command(
            python_exe="python",
            script="mpc_gru.py",
            checkpoint="ckpt.pt",
            output_dir="out",
            scenario={"name": "ignored", "mpc_horizon_s": 40, "print_every_decision": False, "cpu": True},
        )
        assert cmd[:5] == ["python", "mpc_gru.py", "--checkpoint", "ckpt.pt", "--output-dir"]
        assert "--mpc-horizon-s" in cmd and "40" in cmd
        assert "--no-print-every-decision" in cmd
        assert "--cpu" in cmd
        assert "--name" not in cmd

    def test_flatten_summary_prefixes_each_section(self):
        summary = {
            "run_id": "r1",
            "scenario": {"start_temp": 27},
            "mpc": {"optimizer": "cem"},
            "bounds": {"kp": [1.0, 50.0]},
            "cost_weights": {"overshoot_max": 80},
            "metrics": {"tail_mae": 0.5},
        }
        row = artifacts.flatten_summary(summary)
        assert row["scenario_start_temp"] == 27
        assert row["mpc_optimizer"] == "cem"
        assert row["bounds_kp_min"] == 1.0 and row["bounds_kp_max"] == 50.0
        assert row["cost_overshoot_max"] == 80
        assert row["metric_tail_mae"] == 0.5

    def test_add_selection_score_orders_lower_as_better(self):
        df = pd.DataFrame({
            "metric_overshoot_max": [0.0, 5.0],
            "metric_final_abs_error": [0.1, 0.1],
            "metric_tail_mae": [0.1, 0.1],
            "metric_tail_std": [0.1, 0.1],
            "metric_pid_changes": [1, 1],
        })
        out = artifacts.add_selection_score(df)
        assert out["selection_score"].iloc[0] < out["selection_score"].iloc[1]

    def test_add_selection_score_no_op_when_columns_missing(self):
        df = pd.DataFrame({"foo": [1, 2]})
        out = artifacts.add_selection_score(df)
        assert "selection_score" not in out.columns
