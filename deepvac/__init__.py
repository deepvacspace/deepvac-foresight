"""Shared library code for the deepvac chamber-control toolkit.

Submodules
----------
protocol   Chamber TCP protocol client (re-exports tcp.tcp_common).
schemas    Canonical feature names and run/telemetry data shapes.
pid        PID coefficient bounds, banding, scheduling, and TCP readback.
metrics    Cost functions and Bayesian-optimization acquisition math.
datasets   Run-history discovery, sequence building, and scaling for the
           GRU/LSTM one-step plant models.
models     Model-agnostic PyTorch dataset helpers.
mpc        Model-predictive-control rollout/optimization loop shared by the
           GRU and LSTM MPC schedulers.
artifacts  Run-id/CSV/JSON persistence helpers for experiment output.

gru/, lstm/, optimization/, and tcp/ are the executable entry points and own
their output directories (validation_t1/, plots_t1/, mlruns/, run_history/,
etc, each resolved relative to the script's own file location).
"""

__version__ = "0.1.0"
