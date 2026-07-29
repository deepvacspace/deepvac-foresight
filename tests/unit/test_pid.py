"""Unit tests for deepvac/pid.py: bounds/clipping, banding, scheduling."""

from __future__ import annotations

import argparse
import random

import numpy as np
import pytest
from deepvac import pid as dv_pid


def _mpc_args(**overrides) -> argparse.Namespace:
    defaults = dict(kp_min=1.0, kp_max=50.0, ki_min=1.0, ki_max=1000.0, kd_min=1.0, kd_max=20.0)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestParseBounds:
    def test_valid(self):
        assert dv_pid.parse_bounds("1.5, 3.5") == (1.5, 3.5)

    def test_rejects_reversed_bounds(self):
        with pytest.raises(ValueError):
            dv_pid.parse_bounds("5,1")

    def test_rejects_equal_bounds(self):
        with pytest.raises(ValueError):
            dv_pid.parse_bounds("3,3")

    def test_rejects_wrong_part_count(self):
        with pytest.raises(ValueError):
            dv_pid.parse_bounds("1,2,3")

    def test_rejects_non_numeric(self):
        with pytest.raises(ValueError):
            dv_pid.parse_bounds("a,b")

    def test_rejects_non_string_input(self):
        with pytest.raises(ValueError):
            dv_pid.parse_bounds(5.0)  # type: ignore[arg-type]


class TestPidBoundsAndClip:
    def test_pid_bounds_reads_args(self):
        lo, hi = dv_pid.pid_bounds(_mpc_args())
        np.testing.assert_array_equal(lo, [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(hi, [50.0, 1000.0, 20.0])

    def test_pid_bounds_rejects_inverted_range(self):
        with pytest.raises(ValueError):
            dv_pid.pid_bounds(_mpc_args(kp_min=50.0, kp_max=1.0))

    def test_clip_pid_clamps_to_bounds(self):
        clipped = dv_pid.clip_pid(np.array([-10.0, 5000.0, 15.0]), _mpc_args())
        np.testing.assert_array_equal(clipped, [1.0, 1000.0, 15.0])

    def test_clip_pid_rounds_to_nearest_integer(self):
        clipped = dv_pid.clip_pid(np.array([6.4, 997.5, 16.49]), _mpc_args())
        np.testing.assert_array_equal(clipped, [6.0, 998.0, 16.0])

    def test_clip_pid_stays_within_bounds_after_rounding(self):
        # 49.6 rounds to 50, exactly at kp_max -- must not exceed it.
        clipped = dv_pid.clip_pid(np.array([49.6, 1.0, 1.0]), _mpc_args())
        assert clipped[0] == 50.0


class TestBandScheduling:
    def test_band_range_prefers_band_specific_override(self):
        args = argparse.Namespace(far_kp_min=5, kp_min=1)
        assert dv_pid._band_range(args, "far", "kp", "min") == 5

    def test_band_range_falls_back_to_global(self):
        args = argparse.Namespace(far_kp_min=None, kp_min=1)
        assert dv_pid._band_range(args, "far", "kp", "min") == 1

    def test_random_pid_band_respects_band_specific_bounds(self):
        args = argparse.Namespace(
            far_kp_min=10, far_kp_max=10, far_ki_min=20, far_ki_max=20, far_kd_min=30, far_kd_max=30,
        )
        rng = random.Random(0)
        kp, ki, kd = dv_pid.random_pid_band(args, rng, "far")
        assert (kp, ki, kd) == (10, 20, 30)

    def test_plan_pid_uses_schedule_when_given(self):
        args = argparse.Namespace()
        rng = random.Random(0)
        schedule = [(1, 2, 3, 4, 5, 6)]
        planned, source = dv_pid.plan_pid(1, args, rng, schedule, bands=("far", "near"))
        assert planned == {"far": (1, 2, 3), "near": (4, 5, 6)}
        assert source == "schedule[0]"

    def test_plan_pid_cycles_through_multiple_schedules(self):
        args = argparse.Namespace()
        rng = random.Random(0)
        schedules = [(1, 1, 1), (2, 2, 2)]
        _, source_run1 = dv_pid.plan_pid(1, args, rng, schedules, bands=("far",))
        _, source_run2 = dv_pid.plan_pid(2, args, rng, schedules, bands=("far",))
        _, source_run3 = dv_pid.plan_pid(3, args, rng, schedules, bands=("far",))
        assert (source_run1, source_run2, source_run3) == ("schedule[0]", "schedule[1]", "schedule[0]")

    def test_plan_pid_rejects_wrong_length_schedule(self):
        args = argparse.Namespace()
        rng = random.Random(0)
        with pytest.raises(ValueError):
            dv_pid.plan_pid(1, args, rng, [(1, 2, 3)], bands=("far", "near"))

    def test_plan_pid_falls_back_to_random_bands(self):
        args = argparse.Namespace(far_kp_min=1, far_kp_max=1, far_ki_min=2, far_ki_max=2, far_kd_min=3, far_kd_max=3)
        rng = random.Random(0)
        planned, source = dv_pid.plan_pid(1, args, rng, [], bands=("far",))
        assert planned == {"far": (1, 2, 3)}
        assert source == "random-band-ranges"
