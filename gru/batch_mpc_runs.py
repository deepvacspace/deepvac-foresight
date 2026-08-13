#!/usr/bin/env python3
"""
Batch runner for mpc_gru.py.

Defines the GRU-MPC scenario matrix and this script's defaults; the batch
orchestration itself is deepvac.artifacts.run_batch_scenarios.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from deepvac.artifacts import run_batch_scenarios


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Run multiple GRU+MPC PID simulations and compare results."
    )

    ap.add_argument(
        "--script",
        default="mpc_gru.py",
        help="Path to the single-run MPC script.",
    )
    ap.add_argument(
        "--checkpoint",
        default="./validation_t1/gru_t1.pt",
        help="Path to the trained GRU checkpoint.",
    )
    ap.add_argument(
        "--output-dir",
        default="mpc_pid_runs_batch",
        help="Directory where all batch runs will be saved.",
    )
    ap.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use. Defaults to the current interpreter.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    ap.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the batch if one scenario fails.",
    )

    return ap


# -----------------------------------------------------------------------------
# Scenario definition
# -----------------------------------------------------------------------------


def make_scenarios() -> list[dict[str, Any]]:
    """
    Define all test cases here.

    The start and target temperatures are fixed because all tests must use
    the same thermal transition. The batch compares only MPC behavior,
    optimizer settings, and cost-function weights.
    """

    base = {
        # Fixed thermal scenario.
        "start_temp": 27,
        "target_temp": 0,

        # Simulation setup.
        "duration_s": 1200,
        "dt_s": 2,

        # Optimizer setup.
        "optimizer": "cem",
        "cem_population": 256,
        "cem_iterations": 3,
        "cem_elite_frac": 0.12,

        # Default MPC setup.
        "mpc_horizon_s": 80,
        "mpc_hold_s": 20,

        # PID bounds.
        "kp_min": 1,
        "kp_max": 50,
        "ki_min": 1,
        "ki_max": 1000,
        "kd_min": 1,
        "kd_max": 20,

        # Initial PID.
        "kp_init": 6,
        "ki_init": 997,
        "kd_init": 16,

        # Cost weights.
        "w_overshoot_max": 80,
        "w_abs_error": 2,
        "w_motion": 20,
        "motion_error_scale": 5,
        "w_near_std": 1,

        # Disable noisy output from the inner script.
        "print_every_decision": False,
        "print_optimizer_progress": False,
    }

    scenarios: list[dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Group 1: different MPC horizon/hold combinations.
    # -------------------------------------------------------------------------
    for horizon_s, hold_s in [
        (40, 10),
        (60, 10),
        (80, 20),
        (120, 20),
        (180, 30),
    ]:
        s = dict(base)
        s.update(
            {
                "name": f"h{horizon_s}_hold{hold_s}",
                "mpc_horizon_s": horizon_s,
                "mpc_hold_s": hold_s,
            }
        )
        scenarios.append(s)

    # -------------------------------------------------------------------------
    # Group 2: different braking/stability cost settings.
    # -------------------------------------------------------------------------
    for w_motion, motion_error_scale, w_near_std in [
        (5, 3, 1),
        (20, 5, 1),
        (50, 5, 1),
        (20, 10, 1),
        (20, 5, 5),
    ]:
        s = dict(base)
        s.update(
            {
                "name": f"motion{w_motion}_scale{motion_error_scale}_std{w_near_std}",
                "w_motion": w_motion,
                "motion_error_scale": motion_error_scale,
                "w_near_std": w_near_std,
            }
        )
        scenarios.append(s)

    # -------------------------------------------------------------------------
    # Group 3: different overshoot penalties.
    # -------------------------------------------------------------------------
    for w_overshoot_max in [
        40,
        80,
        120,
        200,
    ]:
        s = dict(base)
        s.update(
            {
                "name": f"overshoot{w_overshoot_max}",
                "w_overshoot_max": w_overshoot_max,
            }
        )
        scenarios.append(s)

    # -------------------------------------------------------------------------
    # Group 4: different CEM budgets.
    # -------------------------------------------------------------------------
    for population, iterations in [
        (128, 2),
        (256, 3),
        (512, 3),
        (512, 5),
    ]:
        s = dict(base)
        s.update(
            {
                "name": f"cem{population}_iter{iterations}",
                "cem_population": population,
                "cem_iterations": iterations,
            }
        )
        scenarios.append(s)

    return scenarios


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = build_arg_parser().parse_args()
    run_batch_scenarios(args=args, scenarios=make_scenarios(), label="GRU")


if __name__ == "__main__":
    main()
