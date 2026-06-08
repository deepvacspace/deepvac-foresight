# GP Runtime Build

Self-contained files for GRU-ranked GP PID control.

## Files

- `live_gp_controller.py` - live closed-loop controller. Waits for finite `temp_u`, then replans and writes PID coefficients every `--hold-s`.
- `gp_experiment.py` - replays the copied ranked decision schedule over TCP.
- `gp_gru.py`, `gru_common.py` - copied GRU ranking/model code.
- `model/gru_t1.pt` - copied GRU checkpoint.
- `model/band_gp_models.pkl` - copied band GP models used to propose extra PID candidates.
- `data/gru_pid_candidate_table.csv` - copied candidate table.
- `data/gru_ranked_decisions.csv` - copied ranked decision schedule.
- `data/band_bo_next_params.json` - copied latest GP suggestion metadata.
- `tcp/`, `utils/` - copied TCP and logging helpers.
- `requirements_runtime.txt` - minimal runtime/package dependencies.
- `package_app.ps1` - builds a PyInstaller onedir app.

The live controller does not let the GP directly control the chamber. The GP
proposes current-band PID candidates, then the GRU rollout ranks those candidates
against history, anchors, current PID, and ranked-decision PIDs.

## Dry Run

```powershell
python .\live_gp_controller.py --dry-run --duration-s 120
```

## Live Control

```powershell
python .\live_gp_controller.py --target-temp 0 --duration-s 1200 --dt-s 2 --hold-s 30
```

By default, TCP read/write/connect failures keep retrying forever. Use
`--max-consecutive-failures N` or `--tcp-write-retries N` only if you want the
controller to abort after a fixed number of failures.

## Replay Ranked Decisions

```powershell
python .\gp_experiment.py
```

## Package As App

```powershell
.\package_app.ps1 -Clean
```

The packaged app is written to `dist/DeepVacGPController/`.
