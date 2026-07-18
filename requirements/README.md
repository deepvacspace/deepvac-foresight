# Dependency groups

This project separates dependencies into four `pip install -e ".[<group>]"`
extras, each with an exact-pin lock file here:

| Group           | Lock file                | What it's for |
|-----------------|---------------------------|----------------|
| `runtime`       | `runtime.lock.txt`        | Chamber protocol, PID/BO helpers, and running an already-trained GRU/LSTM checkpoint (MPC, inference, `optimization/*_experiment.py`). |
| `training`      | `training.lock.txt`       | Everything in `runtime` plus training new models: matplotlib, optuna (hyperparameter search), mlflow (experiment tracking). |
| `visualization` | `visualization.lock.txt`  | Plotting/analysis scripts only (matplotlib), when you already have run-history CSVs and don't need torch/scikit-learn. |
| `dev`           | `dev.lock.txt`             | Test tooling (pytest). |

`pyproject.toml`'s `[project.optional-dependencies]` declares the direct,
loosely-versioned requirements per group (what you'd hand-edit); the
`*.lock.txt` files here pin the exact transitive closure for reproducible
installs, generated against the validated Windows/Python 3.10.19 environment
(see `environment.yml`).

## Install

```powershell
python -m pip install -e ".[runtime]" -r requirements\runtime.lock.txt        # chamber control, inference only
python -m pip install -e ".[training]" -r requirements\training.lock.txt      # + training, optuna, mlflow
python -m pip install -e ".[visualization]" -r requirements\visualization.lock.txt  # plotting only, no torch
python -m pip install -e ".[dev]" -r requirements\dev.lock.txt                # + pytest
python -m pip install -e ".[training,visualization,dev]" \
    -r requirements\training.lock.txt -r requirements\visualization.lock.txt -r requirements\dev.lock.txt  # everything
```

## Regenerating a lock file

Install the extra into a clean virtual environment and freeze it:

```powershell
python -m venv .venv-lock
.venv-lock\Scripts\pip install -e ".[training]"
.venv-lock\Scripts\pip freeze > requirements\training.lock.txt
```
