"""Tests that deepvac.mpc_batch reproduces deepvac.mpc's scalar rollout exactly."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from deepvac import mpc as _mpc
from deepvac import mpc_batch

gru_common = pytest.importorskip("gru.gru_common")

CHECKPOINT = gru_common.DEFAULT_CHECKPOINT
pytestmark = pytest.mark.skipif(
    not CHECKPOINT.exists(), reason=f"GRU checkpoint not available at {CHECKPOINT}"
)


@pytest.fixture(scope="module")
def loaded():
    device = torch.device("cpu")
    model, checkpoint = gru_common.load_model(CHECKPOINT, device)
    feature_names = list(checkpoint.get("feature_names", gru_common.DEFAULT_FEATURE_NAMES))
    window_steps = int(checkpoint.get("window_steps", 60))
    return model, checkpoint, feature_names, window_steps, device


def make_args(**overrides):
    import argparse

    ap = argparse.ArgumentParser()
    _mpc.add_common_mpc_args(ap)
    _mpc.add_mpc_cost_args(ap)
    args = ap.parse_args([])
    args.dt_s = 2.0
    args.start_temp = 25.0
    args.target_temp = 0.0
    args.mpc_horizon_s = 20.0
    for key, value in overrides.items():
        setattr(args, key, value)
    assert isinstance(args, argparse.Namespace)
    return args


def make_state(feature_names, window_steps, args, *, temp=8.0, i_part=0.2):
    window = _mpc.initialize_feature_window(
        feature_names, window_steps, temp, args.target_temp, args.dt_s, 10.0, 500.0, 20.0
    )
    pid = gru_common.ChamberPID(u_min=args.u_min, u_max=args.u_max,
                                pid_i_reverse_mul=args.pid_i_reverse_mul)
    pid.i_part = i_part
    diff = gru_common.CodesysDiff()
    diff.prev_value = temp
    diff.filter_out = -0.01
    return _mpc.SimState(
        elapsed_s=120.0, temp=temp, previous_temp=temp + 0.05,
        feature_window=window, pid=pid, diff=diff, kp=10.0, ki=500.0, kd=20.0,
    )


CANDIDATES = np.array(
    [[10.0, 500.0, 20.0], [3.0, 900.0, 5.0], [20.0, 50.0, 18.0], [1.0, 1000.0, 1.0]]
)


def test_batched_rollout_matches_scalar_rollout(loaded):
    model, checkpoint, feature_names, window_steps, device = loaded
    args = make_args()
    state = make_state(feature_names, window_steps, args)
    horizon_steps = 10

    batched = mpc_batch.rollout_population(
        initial_state=state, candidate_pids=CANDIDATES, model=model, checkpoint=checkpoint,
        feature_names=feature_names, device=device, args=args,
        horizon_steps=horizon_steps, cost_fn=_mpc.horizon_cost,
    )

    for idx, candidate in enumerate(CANDIDATES):
        scalar_metrics, scalar_temps = _mpc.rollout_constant_pid(
            initial_state=state, candidate_pid=candidate, model=model, checkpoint=checkpoint,
            feature_names=feature_names, device=device, args=args,
            horizon_steps=horizon_steps, predict_fn=gru_common.predict_delta_t1,
            cost_fn=_mpc.horizon_cost,
        )
        assert batched[idx]["cost"] == pytest.approx(scalar_metrics["cost"], rel=1e-5, abs=1e-6)
        assert batched[idx]["horizon_mae"] == pytest.approx(scalar_metrics["horizon_mae"], rel=1e-5)
        assert batched[idx]["horizon_overshoot_max"] == pytest.approx(
            scalar_metrics["horizon_overshoot_max"], rel=1e-5, abs=1e-6
        )
        assert (batched[idx]["kp"], batched[idx]["ki"], batched[idx]["kd"]) == (
            scalar_metrics["kp"], scalar_metrics["ki"], scalar_metrics["kd"]
        )
        assert len(scalar_temps) == horizon_steps


def test_batched_rollout_does_not_mutate_initial_state(loaded):
    model, checkpoint, feature_names, window_steps, device = loaded
    args = make_args()
    state = make_state(feature_names, window_steps, args)
    before_window = state.feature_window.copy()
    before = (state.temp, state.pid.i_part, state.diff.prev_value, state.diff.filter_out)

    mpc_batch.rollout_population(
        initial_state=state, candidate_pids=CANDIDATES, model=model, checkpoint=checkpoint,
        feature_names=feature_names, device=device, args=args,
        horizon_steps=5, cost_fn=_mpc.horizon_cost,
    )

    assert (state.temp, state.pid.i_part, state.diff.prev_value, state.diff.filter_out) == before
    np.testing.assert_array_equal(state.feature_window, before_window)


def test_invalid_candidate_is_flagged_and_penalized(loaded):
    model, checkpoint, feature_names, window_steps, device = loaded
    # A tiny --max-abs-temp forces every rollout out of bounds immediately.
    args = make_args(max_abs_temp=0.001)
    state = make_state(feature_names, window_steps, args)

    batched = mpc_batch.rollout_population(
        initial_state=state, candidate_pids=CANDIDATES, model=model, checkpoint=checkpoint,
        feature_names=feature_names, device=device, args=args,
        horizon_steps=6, cost_fn=_mpc.horizon_cost,
    )
    for row in batched:
        assert row["valid"] is False
        assert row["cost"] >= args.w_invalid


def test_vectorized_pid_matches_chamber_pid():
    """The PID/diff vectorization is the easiest thing to get subtly wrong."""
    rng = np.random.default_rng(0)
    kp = rng.uniform(1, 20, size=32)
    ki = rng.uniform(1, 1000, size=32)
    kd = rng.uniform(1, 20, size=32)
    temps = rng.uniform(-5, 25, size=32)
    i_part = rng.uniform(-1, 1, size=32)

    diff_prev = temps + rng.uniform(-1, 1, size=32)
    diff_filter = rng.uniform(-0.5, 0.5, size=32)

    vec_out, vec_i, vec_prev, vec_filter = mpc_batch._vec_run_pid_substeps(
        temp=temps, temp_ref=0.0, kp=kp, ki=ki, kd=kd, i_part=i_part.copy(),
        diff_prev=diff_prev.copy(), diff_filter=diff_filter.copy(),
        dt_s=2.0, period_s=0.1, feature_scale=100.0, diff_dc=0.995,
        u_min=-1.0, u_max=1.0, i_reverse_mul=0.333,
    )

    for j in range(32):
        pid = gru_common.ChamberPID(u_min=-1.0, u_max=1.0, pid_i_reverse_mul=0.333)
        pid.i_part = float(i_part[j])
        diff = gru_common.CodesysDiff()
        diff.prev_value = float(diff_prev[j])
        diff.filter_out = float(diff_filter[j])

        terms = _mpc.run_pid_substeps(
            pid=pid, diff=diff, temp_start=float(temps[j]), temp_end=float(temps[j]),
            temp_ref=0.0, kp=float(kp[j]), ki=float(ki[j]), kd=float(kd[j]),
            dt_s=2.0, period_s=0.1, feature_scale=100.0, temp_mode="hold",
        )
        for key in ("u", "u_p", "u_i", "u_d"):
            assert vec_out[key][j] == pytest.approx(terms[key], rel=1e-9, abs=1e-12)
        assert vec_i[j] == pytest.approx(pid.i_part, rel=1e-9, abs=1e-12)
        assert vec_prev[j] == pytest.approx(diff.prev_value, rel=1e-9, abs=1e-12)
        assert vec_filter[j] == pytest.approx(diff.filter_out, rel=1e-9, abs=1e-12)
