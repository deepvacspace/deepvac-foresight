# Deepvac
## Overview
Short explanation of what this project does.

## Features
- Deepvac AI Advisor
- Feature 2
- Feature 3

## Main repository for Deepvac AI Advisor

- **gru/** - GRU model for digital twin that predicts chamber temperature
- **lstm/** - LSTM model for digital twin that predicts chamber temperature
- **optimization/** - Current Bayesian Optimization implementation using Gaussian Process with tests for selecting PID coefficient values based on temperature ranges relative to target
- **tcp/** - Utilities and code for TCP connection and data transfer
- **flask/** - Code for data visualization dashboard
- **utils/** - General utility functions
- **utils/bo_common.py** - Utility functions for Bayesian Optimization

## Supported environment

The supported interpreter is Python 3.10 (the validated Conda environment uses
Python 3.10.19). Create a clean environment from the project root:

```powershell
$env:PYTHONNOUSERSITE = "1"
conda env create -f environment.yml
conda activate deepvac
python -m pip check
python -c "import torch, mlflow, optuna; print(torch.__version__)"
```

`requirements.txt` is the minimal direct dependency set.
`requirements.lock.txt` captures the complete validated Windows environment and
is used by `environment.yml` for repeatable Conda recreation.

`PYTHONNOUSERSITE=1` prevents packages in the per-user Python directory from
overriding packages installed in the Conda environment. This matters when a
user-site `pip` installation is incomplete or incompatible.

Run tools as modules from the repository root so imports are deterministic:

```powershell
python -m gru.train_gru --help
python -m lstm.train_lstm --help
python -m optimization.training_loop --help
python -m tcp.get_states  # connects to the configured chamber
```

The default PyPI `torch` package provides portable CPU operation. For CUDA,
install the PyTorch wheel matching the host driver before this project, using
the official PyTorch installation selector.
