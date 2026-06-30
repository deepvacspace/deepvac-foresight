# DeepVac Desktop Visualization

This folder contains a PySide6 desktop version of the Flask dashboard.

## Install

From `.\scripts`:

```powershell
python -m pip install -r visualization\requirements.txt
```

## Run

```powershell
python visualization\app.py
```

The app reads the same data as the Flask dashboard. Optional environment variables are the same:

```powershell
$env:DEEPVAC_WORKSPACE_ROOT=".\scripts"
$env:DEEPVAC_GRU_ROOT=".\scripts\gru"
python visualization\app.py
```

## Local run database

On startup, the app syncs the current run history into:

```text
visualization\deepvac_runs_cache.sqlite3
```

After the first sync, run lists, details, tables, and chart series are read from this SQLite database when the source CSV files have not changed. Set `DEEPVAC_VISUALIZATION_DB` to use a different database path.
