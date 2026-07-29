"""Integration tests for deepvac/mpc.py's receding-horizon rollout +
CEM/random-shooting optimizer -- the logic gru/mpc_gru.py and
lstm/mpc_lstm.py share (see deepvac/mpc.py's module docstring).

Uses a small deterministic fake plant model (predict_fn) instead of a real
GRU/LSTM checkpoint: the control-loop logic (candidate rollout, cost
composition, the safety gate that keeps the current PID unless a candidate
is a meaningful improvement) is what's under test here, not any specific
trained model's predictions. This keeps these tests fast and exactly
reproducible.
"""

from __future__ import annotations

import argparse

import numpy as np
from deepvac import mpc
from deepvac.schemas import DEFAULT_FEATURE_NAMES
from gru.gru_common import ChamberPID, CodesysDiff


def fake_predict_fn(model, checkpoint, feature_window, device):
    """First-order decay toward temp_ref: bigger kp -> faster approach.
    Reads the *current* row of the feature window (the one step_state just
    wrote via make_feature_row), mirroring what a real plant model would be
    conditioned on."""
    row = feature_window[-1]
    names = DEFAULT_FEATURE_NAMES
    temp = float(row[names.index("temp")])
    temp_ref = float(row[names.index("temp_ref")])
    kp = float(row[names.index("kp")])
    rate = min(0.5, kp / 100.0)
    return (temp_ref - temp) * rate


def simple_cost_fn(*, temps, candidate_pid, previous_pid, start_temp, target_temp, valid, args):
    temps = np.asarray(temps, dtype=float)
    if temps.size == 0:
        return {"cost": float(args.w_invalid), "valid": False}
    abs_error = np.abs(target_temp - temps)
    overshoot = mpc.overshoot_array(temps, start_temp=start_temp, target_temp=target_temp)
    cost = float(np.mean(abs_error)) + 10.0 * float(np.max(overshoot))
    if not valid:
        cost += float(args.w_invalid)
    return {"cost": cost, "valid": bool(valid), "horizon_mae": float(np.mean(abs_error))}


def _base_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        start_temp=20.0, target_temp=0.0, dt_s=1.0, seed=0,
        mpc_horizon_s=10.0, mpc_hold_s=5.0,
        kp_min=1.0, kp_max=50.0, ki_min=1.0, ki_max=1000.0, kd_min=1.0, kd_max=20.0,
        u_min=-1.0, u_max=1.0, pid_i_reverse_mul=0.333, pid_period_s=0.1, control_feature_scale=100.0,
        max_abs_temp=200.0,
        optimizer="cem", cem_population=16, cem_elite_frac=0.25, cem_iterations=2, cem_min_std_frac=0.05,
        include_current_candidate=True, apply_margin=0.0, max_pid_delta_frac=1.0,
        history_candidates=0, history_neighbor_pool=0, history_score_weight=0.0,
        history_temp_scale=10.0, history_error_scale=10.0, history_time_scale=600.0, history_velocity_scale=0.2,
        w_invalid=1_000_000.0, near_band=2.0,
        print_optimizer_progress=False, optimizer_progress_every=0,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_state(temp: float, kp: float = 6.0, ki: float = 997.0, kd: float = 16.0) -> mpc.SimState:
    feature_window = mpc.initialize_feature_window(
        DEFAULT_FEATURE_NAMES, window_steps=5, start_temp=temp, precondition_ref=temp,
        dt_s=1.0, kp=kp, ki=ki, kd=kd,
    )
    return mpc.SimState(
        elapsed_s=0.0, temp=temp, previous_temp=temp, feature_window=feature_window,
        pid=ChamberPID(-1.0, 1.0, 0.333), diff=CodesysDiff(),
        kp=kp, ki=ki, kd=kd,
    )


class TestRolloutConstantPid:
    def test_higher_kp_reaches_target_faster(self):
        args = _base_args()
        state = _make_state(temp=20.0)
        low_kp, _ = mpc.rollout_constant_pid(
            initial_state=state, candidate_pid=np.array([2.0, 997.0, 16.0]),
            model=None, checkpoint={}, feature_names=DEFAULT_FEATURE_NAMES, device=None,
            args=args, horizon_steps=20, predict_fn=fake_predict_fn, cost_fn=simple_cost_fn,
        )
        high_kp, _ = mpc.rollout_constant_pid(
            initial_state=state, candidate_pid=np.array([40.0, 997.0, 16.0]),
            model=None, checkpoint={}, feature_names=DEFAULT_FEATURE_NAMES, device=None,
            args=args, horizon_steps=20, predict_fn=fake_predict_fn, cost_fn=simple_cost_fn,
        )
        assert high_kp["cost"] < low_kp["cost"]

    def test_does_not_mutate_initial_state(self):
        args = _base_args()
        state = _make_state(temp=20.0)
        original_temp = state.temp
        original_window = state.feature_window.copy()
        mpc.rollout_constant_pid(
            initial_state=state, candidate_pid=np.array([6.0, 997.0, 16.0]),
            model=None, checkpoint={}, feature_names=DEFAULT_FEATURE_NAMES, device=None,
            args=args, horizon_steps=10, predict_fn=fake_predict_fn, cost_fn=simple_cost_fn,
        )
        assert state.temp == original_temp
        np.testing.assert_array_equal(state.feature_window, original_window)


class TestOptimizePidForState:
    def test_keeps_current_pid_when_already_at_kp_max(self):
        # kp is already at its max (fastest possible approach for this fake
        # plant) -- no candidate can beat it, so the safety gate should keep
        # the current PID rather than "changing" to an equivalent one.
        args = _base_args(kp_min=5.0, kp_max=6.0)
        state = _make_state(temp=20.0, kp=6.0)
        rng = np.random.default_rng(0)
        decision = mpc.optimize_pid_for_state(
            state=state, model=None, checkpoint={}, feature_names=DEFAULT_FEATURE_NAMES, device=None,
            args=args, rng=rng, predict_fn=fake_predict_fn, cost_fn=simple_cost_fn,
        )
        assert decision["changed"] is False
        assert decision["kp"] == 6.0

    def test_random_optimizer_also_runs(self):
        args = _base_args(optimizer="random")
        state = _make_state(temp=20.0)
        rng = np.random.default_rng(0)
        decision = mpc.optimize_pid_for_state(
            state=state, model=None, checkpoint={}, feature_names=DEFAULT_FEATURE_NAMES, device=None,
            args=args, rng=rng, predict_fn=fake_predict_fn, cost_fn=simple_cost_fn,
        )
        assert "cost" in decision and "kp" in decision


class TestRunMpcSimulation:
    def test_full_receding_horizon_run_reaches_near_target(self):
        args = _base_args(
            start_temp=20.0, target_temp=0.0, duration_s=60.0, dt_s=1.0,
            mpc_horizon_s=10.0, mpc_hold_s=5.0, kp_init=40.0, ki_init=997.0, kd_init=16.0,
            precondition_ref=None, initial_p=0.0, initial_i=0.0, initial_d=0.0,
            tail_window_s=20.0, settle_band=0.5,
            save_trajectory=True, print_every_decision=False,
        )
        trajectory, decisions, metrics_out = mpc.run_mpc_simulation(
            model=None, checkpoint={}, feature_names=DEFAULT_FEATURE_NAMES, window_steps=5,
            device=None, args=args, predict_fn=fake_predict_fn, cost_fn=simple_cost_fn,
        )
        assert len(trajectory) == 60
        assert len(decisions) > 0
        assert metrics_out["valid"] is True
        assert metrics_out["final_abs_error"] < 5.0
