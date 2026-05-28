# DeepVac Chamber Monitor

A small Tkinter GUI that polls the chamber through `tcp/tcp_common.py`.

When opened, it sends `get_states` immediately and then every 30 seconds:

- Any successful response marks the chamber as connected.
- `temp_u` present and not `nan` marks the chamber as running.
- The first transition into running shows a Yes/No prompt asking whether to use the AI suggestion. The buttons are placeholders for now.

## Run Locally

From the `scripts` folder:

```powershell
python app\chamber_monitor.py
```

The app uses the default host, port, and timeout from `tcp\tcp_common.py`, but you can edit them in the GUI before pressing `Refresh`.

## Package As A Windows .exe

Install PyInstaller:

```powershell
python -m pip install pyinstaller
```

Build the executable from the `scripts` folder:

```powershell
python -m PyInstaller --onefile --windowed --name DeepVacChamberMonitor --paths tcp app\chamber_monitor.py
```

The packaged app will be created at:

```text
dist\DeepVacChamberMonitor.exe
```

If you prefer a faster startup and do not mind a folder distribution, omit `--onefile`:

```powershell
python -m PyInstaller --windowed --name DeepVacChamberMonitor --paths tcp app\chamber_monitor.py
```
