#!/usr/bin/env python3
"""Unified `deepvac` command-line entry point.

Usage:
    deepvac <command> [--help] [command-specific flags...]
    deepvac --list
"""

from __future__ import annotations

import importlib
import sys

COMMANDS: dict[str, tuple[str, str] | tuple[str, str, tuple[str, ...]]] = {
    # --- GRU/LSTM digital twin (digitaltwin/) -------------------------------
    "train-gru": ("digitaltwin.train", "Pretrain the one-step GRU plant model (stage 1).",
                 ("--model-family", "gru")),
    "train-lstm": ("digitaltwin.train", "Pretrain the one-step LSTM plant model (stage 1).",
                  ("--model-family", "lstm")),
    "train-gru-rollout": ("digitaltwin.train_rollout",
                          "Train the GRU on rollouts, selected on rollout error (stage 2).",
                          ("--model-family", "gru")),
    "train-lstm-rollout": ("digitaltwin.train_rollout",
                           "Train the LSTM on rollouts, selected on rollout error (stage 2).",
                           ("--model-family", "lstm")),
    "twin-acceptance": ("digitaltwin.twin_acceptance",
                        "Score a GRU checkpoint as a whole-run digital twin, live or offline.",
                        ("--model-family", "gru")),
    "twin-acceptance-lstm": ("digitaltwin.twin_acceptance",
                             "Score an LSTM checkpoint as a whole-run digital twin, live or offline.",
                             ("--model-family", "lstm")),
    "predict-run": ("digitaltwin.predict_run",
                    "Predict a whole-run trajectory for a candidate PID, no chamber run needed.",
                    ("--model-family", "gru")),
    "predict-run-lstm": ("digitaltwin.predict_run",
                        "Predict a whole-run trajectory (LSTM twin), no chamber run needed.",
                        ("--model-family", "lstm")),

    # --- AI advisor (advisor/) -----------------------------------------------
    "advise-pid": ("advisor.advise_pid", "Suggest a PID triplet or adaptively control it via CEM search over the twin."),
    "train-cost-surrogate": ("advisor.train_cost_surrogate", "Train a fast MLP to screen PID candidates before CEM."),

    # --- Bayesian optimization / AI advisor (optimization/) ----------------
    "optimize-training-loop": ("optimization.training_loop", "Closed-loop band-BO runner against the live chamber."),
    "optimize-band-bo": ("optimization.band_bo_gp", "Fit per-band GP-BO models (--band-mode 3|5) and suggest next PID candidates."),
    "optimize-compute-model": ("optimization.compute_one_model", "Fit one GP-BO model over far/mid/near PID coeffs."),
    "optimize-random-pid": ("optimization.random_pid_tests", "Automated multi-run TCP tests with random PIDs."),
    "optimize-tocero-3band": ("optimization.tocero_3band", "Automated multi-run TCP tests, 3-band PID schedule."),
    "optimize-tocero-5band": ("optimization.tocero_5band", "Automated multi-run TCP tests, 5-band PID schedule."),
    "collect-runs": ("optimization.collect_runs", "Repeated, start-gated runs driven by a time-indexed PID profile (logs the reheat)."),
    "replay-gp": ("optimization.gp_experiment", "Replay one GRU-ranked GP decision schedule over TCP."),
    "replay-gp-batch": ("optimization.batch_gp_experiment", "Replay multiple GRU-ranked GP decision schedules."),
    "replay-mpc": ("optimization.mpc_experiment", "Replay MPC PID decisions over TCP."),

    # --- Analysis, ranking, and plotting (benchmarks/) -----------------------
    "settling-metrics": ("benchmarks.settling_metrics", "Score overshoot, jitter, bias and settling per setpoint episode."),
    "compare-advisor-improvement": ("benchmarks.ai_advisor_improvement", "Compare a baseline vs AI-advisor run."),
    "list-runs": ("benchmarks.list_runs", "List/summarize run history, including NaN counts."),
    "rank-runs": ("benchmarks.rank_runs", "Rank runs by tail MAE / time-to-zero / cost / overshoot."),
    "plot-metrics": ("benchmarks.plot_metrics", "Plot run-history metrics."),
    "plot-best": ("benchmarks.plot_best", "Plot trajectories for best/worst/average run groups."),

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
        _, description = COMMANDS[name][:2]
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

    module_name, _, *extra = entry
    extra_args = list(extra[0]) if extra else []
    module = importlib.import_module(module_name)
    sys.argv = [module_name, *extra_args, *rest]
    return module.main() or 0


if __name__ == "__main__":
    raise SystemExit(main())
