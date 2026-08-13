#!/usr/bin/env python3
"""Unified `deepvac` command-line entry point.

Each subcommand imports its script module and forwards the remaining argv to
that module's own `main()` / `build_arg_parser()`.

Usage:
    deepvac <command> [--help] [command-specific flags...]
    deepvac --list          # print every command with its one-line description
"""

from __future__ import annotations

import importlib
import sys

# command name -> (module to import, one-line description)
# Grouped to match the module layout; run `deepvac --list` for the live list.
COMMANDS: dict[str, tuple[str, str]] = {
    # --- GRU digital twin --------------------------------------------------
    "train-gru": ("gru.train_gru", "Train/validate/tune/test the one-step GRU plant model."),
    "train-gru-multihorizon": ("gru.gru_multi_time", "Train/validate/test the multi-horizon GRU plant model."),
    "simulate-gru": ("gru.simulate_gru", "Closed-loop reconstruction via trained GRU + ChamberPID."),
    "simulate-runs": ("gru.simulate_runs", "Rank PID triplets via GRU + CODESYS PID simulation."),
    "mpc-gru": ("gru.mpc_gru", "Continuous GRU + MPC PID scheduler (CEM/random shooting)."),
    "batch-mpc-gru": ("gru.batch_mpc_runs", "Batch GRU + MPC runs across scenarios, compare results."),
    "gp-gru": ("gru.gp_gru", "GRU-ranked PID scheduler using historical/GP candidates."),
    "batch-gp-gru": ("gru.batch_gp_gru", "Batch runner for gp-gru."),
    "gp-build": ("gru.gp_build", "Build the historical PID candidate table for GRU ranking."),
    "mpc-build": ("gru.mpc_build", "Build the candidate table for history-seeded GRU MPC."),
    "diagnose-codesys": ("gru.diagnose_codesys", "Validate GRU replay-control + CODESYS PID reconstruction."),

    # --- LSTM digital twin --------------------------------------------------
    "train-lstm": ("lstm.train_lstm", "Train/validate/tune/test the one-step LSTM plant model."),
    "predict-lstm": ("lstm.predict", "Evaluate a trained LSTM checkpoint against run history."),
    "mpc-lstm": ("lstm.mpc_lstm", "Continuous LSTM + MPC PID scheduler (CEM/random shooting)."),
    "batch-mpc-lstm": ("lstm.batch_mpc_runs", "Batch LSTM + MPC runs across scenarios, compare results."),
    "train-lstm-legacy": ("lstm.lstm", "Older standalone LSTM pipeline, kept for comparison only."),
    "train-gru-rollout": ("gru.train_gru_rollout", "Train the GRU on multi-step rollouts, selected on rollout error."),

    # --- Bayesian optimization / AI advisor (optimization/) ----------------
    "optimize-training-loop": ("optimization.training_loop", "Closed-loop band-BO runner against the live chamber."),
    "optimize-band-bo": ("optimization.band_bo_gp", "Fit per-band GP-BO models (--band-mode 3|5) and suggest next PID candidates."),
    "optimize-compute-model": ("optimization.compute_one_model", "Fit one GP-BO model over far/mid/near PID coeffs."),
    "optimize-random-pid": ("optimization.random_pid_tests", "Automated multi-run TCP tests with random PIDs."),
    "optimize-tocero-3band": ("optimization.tocero_3band", "Automated multi-run TCP tests, 3-band PID schedule."),
    "optimize-tocero-5band": ("optimization.tocero_5band", "Automated multi-run TCP tests, 5-band PID schedule."),
    "optimize-tocero-gp-mpc": ("optimization.tocero_gp_mpc", "Live run: GP far-band PID, then GRU+MPC replanning inside the band."),
    "collect-runs": ("optimization.collect_runs", "Repeated, start-gated runs driven by a time-indexed PID profile (logs the reheat)."),
    "settling-metrics": ("optimization.settling_metrics", "Score overshoot, jitter, bias and settling per setpoint episode."),
    "replay-gp": ("optimization.gp_experiment", "Replay one GRU-ranked GP decision schedule over TCP."),
    "replay-gp-batch": ("optimization.batch_gp_experiment", "Replay multiple GRU-ranked GP decision schedules."),
    "replay-mpc": ("optimization.mpc_experiment", "Replay MPC PID decisions over TCP."),
    "compare-advisor-improvement": ("optimization.ai_advisor_improvement", "Compare a baseline vs AI-advisor run."),

    # --- Analysis and reporting (optimization/) -----------------------------
    "list-runs": ("optimization.list_runs", "List/summarize run history, including NaN counts."),
    "rank-runs": ("optimization.rank_runs", "Rank runs by tail MAE / time-to-zero / cost / overshoot."),
    "plot-metrics": ("optimization.plot_metrics", "Plot run-history metrics."),
    "plot-best": ("optimization.plot_best", "Plot trajectories for best/worst/average run groups."),

    # --- Packaging (deepvac/) -----------------------------------------------
    "package-model": ("deepvac.package_model", "Package a checkpoint for the insight/control2-client desktop apps."),

    # --- Chamber TCP protocol (tcp/) ----------------------------------------
    "tcp-get-states": ("tcp.get_states", "Read and print current controller states."),
    "tcp-get-settings": ("tcp.get_settings", "Read and print current controller settings."),
    "tcp-temp-ref": ("tcp.temp_ref", "Publish one temperature setpoint job and verify temp_ref changed."),
    "tcp-write-setting": ("tcp.write_setting", "Write one PID row and verify the readback."),
}


def _print_list() -> None:
    width = max(len(name) for name in COMMANDS)
    print("Available deepvac commands:\n")
    for name in sorted(COMMANDS):
        _, description = COMMANDS[name]
        print(f"  {name.ljust(width)}  {description}")
    print("\nRun `deepvac <command> --help` for that command's full flag list.")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help"):
        _print_list()
        return 0
    if argv[0] == "--list":
        _print_list()
        return 0

    command, rest = argv[0], argv[1:]
    entry = COMMANDS.get(command)
    if entry is None:
        print(f"Unknown command: {command!r}\n", file=sys.stderr)
        _print_list()
        return 2

    module_name, _ = entry
    module = importlib.import_module(module_name)
    # The target module parses sys.argv itself, so present argv as if it had
    # been invoked directly (python -m <module_name> <rest>).
    sys.argv = [module_name, *rest]
    return module.main() or 0


if __name__ == "__main__":
    raise SystemExit(main())
