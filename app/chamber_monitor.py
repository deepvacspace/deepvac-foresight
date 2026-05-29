#!/usr/bin/env python3
"""Small GUI monitor for the DeepVac chamber TCP state stream."""

from __future__ import annotations

import math
import queue
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import messagebox, ttk

APP_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = APP_DIR.parent
TCP_DIR = SCRIPTS_DIR / "tcp"
if str(TCP_DIR) not in sys.path:
    sys.path.insert(0, str(TCP_DIR))

try:
    from tcp_common import DEFAULT_HOST, DEFAULT_PORT, DEFAULT_TIMEOUT, request_states
except Exception as exc:  
    raise RuntimeError(
        "Could not import tcp_common.py. Keep this app folder next to the tcp folder."
    ) from exc


POLL_INTERVAL_MS = 30_000


@dataclass
class PollResult:
    ok: bool
    states: Optional[Dict[str, float]] = None
    error: str = ""
    checked_at: str = ""


class ChamberMonitorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("DeepVac Chamber Monitor")
        self.geometry("620x420")
        self.minsize(560, 380)

        self.result_queue: queue.Queue[PollResult] = queue.Queue()
        self.poll_in_flight = False
        self.connected = False
        self.running = False
        self.ai_prompt_shown = False
        self.temperature_history: List[Tuple[datetime, float, Optional[float]]] = []

        self.host_var = tk.StringVar(value=DEFAULT_HOST)
        self.port_var = tk.StringVar(value=str(DEFAULT_PORT))
        self.timeout_var = tk.StringVar(value=str(DEFAULT_TIMEOUT))
        self.connection_var = tk.StringVar(value="Disconnected")
        self.running_var = tk.StringVar(value="Not running")
        self.last_checked_var = tk.StringVar(value="Never")
        self.temp_var = tk.StringVar(value="-")
        self.temp_ref_var = tk.StringVar(value="-")
        self.temp_u_var = tk.StringVar(value="-")
        self.error_var = tk.StringVar(value="")
        self.plot_status_var = tk.StringVar(value="No running experiment detected.")

        self._build_ui()
        self.after(100, self.poll_now)
        self.after(200, self.process_results)

    def _build_ui(self) -> None:
        self.configure(bg="#f6f7f9")

        root = ttk.Frame(self, padding=18)
        root.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(root, text="DeepVac", font=("Segoe UI", 18, "bold"))
        title.pack(anchor="w")

        notebook = ttk.Notebook(root)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        monitor_tab = ttk.Frame(notebook, padding=(0, 12, 0, 0))
        plot_tab = ttk.Frame(notebook, padding=(0, 12, 0, 0))
        notebook.add(monitor_tab, text="Monitor")
        notebook.add(plot_tab, text="Plot")

        self._build_monitor_tab(monitor_tab)
        self._build_plot_tab(plot_tab)
        self._apply_badge_styles()

    def _build_monitor_tab(self, root: ttk.Frame) -> None:
        status_frame = ttk.Frame(root)
        status_frame.pack(fill=tk.X)
        status_frame.columnconfigure((0, 1), weight=1)

        self.connection_badge = ttk.Label(
            status_frame,
            textvariable=self.connection_var,
            anchor="center",
            padding=(12, 12),
            font=("Segoe UI", 12, "bold"),
        )
        self.connection_badge.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        self.running_badge = ttk.Label(
            status_frame,
            textvariable=self.running_var,
            anchor="center",
            padding=(12, 12),
            font=("Segoe UI", 12, "bold"),
        )
        self.running_badge.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        readings = ttk.LabelFrame(root, text="Latest states", padding=12)
        readings.pack(fill=tk.X, pady=(18, 12))
        readings.columnconfigure(1, weight=1)

        self._add_reading(readings, 0, "Temperature", self.temp_var)
        self._add_reading(readings, 1, "Target", self.temp_ref_var)
        self._add_reading(readings, 2, "temp_u", self.temp_u_var)
        self._add_reading(readings, 3, "Last checked", self.last_checked_var)

        connection = ttk.LabelFrame(root, text="TCP connection", padding=12)
        connection.pack(fill=tk.X)
        for col in range(6):
            connection.columnconfigure(col, weight=1)

        ttk.Label(connection, text="Host").grid(row=0, column=0, sticky="w")
        ttk.Entry(connection, textvariable=self.host_var).grid(row=1, column=0, columnspan=3, sticky="ew", padx=(0, 8))

        ttk.Label(connection, text="Port").grid(row=0, column=3, sticky="w")
        ttk.Entry(connection, textvariable=self.port_var, width=8).grid(row=1, column=3, sticky="ew", padx=(0, 8))

        ttk.Label(connection, text="Timeout").grid(row=0, column=4, sticky="w")
        ttk.Entry(connection, textvariable=self.timeout_var, width=8).grid(row=1, column=4, sticky="ew", padx=(0, 8))

        ttk.Button(connection, text="Refresh", command=self.poll_now).grid(row=1, column=5, sticky="ew")

        ttk.Label(root, textvariable=self.error_var, foreground="#a33b2b", wraplength=560).pack(
            anchor="w", fill=tk.X, pady=(12, 0)
        )

    def _build_plot_tab(self, root: ttk.Frame) -> None:
        controls = ttk.Frame(root)
        controls.pack(fill=tk.X)

        ttk.Label(controls, textvariable=self.plot_status_var).pack(side=tk.LEFT, anchor="w")
        ttk.Button(controls, text="Clear", command=self.clear_plot).pack(side=tk.RIGHT)

        self.plot_canvas = tk.Canvas(root, height=260, background="#ffffff", highlightthickness=1, highlightbackground="#d7dde5")
        self.plot_canvas.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        self.plot_canvas.bind("<Configure>", lambda _event: self.redraw_plot())

    def _add_reading(self, parent: ttk.Frame, row: int, label: str, value_var: tk.StringVar) -> None:
        ttk.Label(parent, text=label, foreground="#526070").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Label(parent, textvariable=value_var, font=("Segoe UI", 10, "bold")).grid(
            row=row, column=1, sticky="w", pady=4
        )

    def _apply_badge_styles(self) -> None:
        style = ttk.Style(self)
        style.configure("Connected.TLabel", background="#d9f2df", foreground="#145c2f")
        style.configure("Disconnected.TLabel", background="#f6d9d5", foreground="#7a2118")
        style.configure("Running.TLabel", background="#dce9ff", foreground="#173d79")
        style.configure("Idle.TLabel", background="#eceff3", foreground="#46515f")

        self.connection_badge.configure(
            style="Connected.TLabel" if self.connected else "Disconnected.TLabel"
        )
        self.running_badge.configure(style="Running.TLabel" if self.running else "Idle.TLabel")

    def poll_now(self) -> None:
        if self.poll_in_flight:
            return
        self.poll_in_flight = True
        self.error_var.set("")
        thread = threading.Thread(target=self._poll_worker, daemon=True)
        thread.start()

    def _poll_worker(self) -> None:
        checked_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            host = self.host_var.get().strip()
            port = int(self.port_var.get().strip())
            timeout = float(self.timeout_var.get().strip())
            states = request_states(host=host, port=port, timeout=timeout)
            self.result_queue.put(PollResult(ok=True, states=states, checked_at=checked_at))
        except Exception as exc:
            self.result_queue.put(PollResult(ok=False, error=str(exc), checked_at=checked_at))

    def process_results(self) -> None:
        try:
            while True:
                result = self.result_queue.get_nowait()
                self._handle_result(result)
        except queue.Empty:
            pass
        self.after(200, self.process_results)

    def _handle_result(self, result: PollResult) -> None:
        self.poll_in_flight = False
        was_running = self.running

        self.last_checked_var.set(result.checked_at)
        if result.ok and result.states is not None:
            self.connected = True
            self.running = self._is_running(result.states)
            self.connection_var.set("Connected")
            self.running_var.set("Running" if self.running else "Not running")
            self.temp_var.set(self._format_state(result.states, "temp"))
            self.temp_ref_var.set(self._format_state(result.states, "temp_ref"))
            self.temp_u_var.set(self._format_state(result.states, "temp_u"))
            self.error_var.set("")
            self._record_temperature_reading(result)
        else:
            self.connected = False
            self.running = False
            self.connection_var.set("Disconnected")
            self.running_var.set("Not running")
            self.error_var.set(result.error)
            self._update_plot_status()

        self._apply_badge_styles()
        self.redraw_plot()
        if self.running and not was_running and not self.ai_prompt_shown:
            self.ai_prompt_shown = True
            self._show_ai_suggestion_prompt()

        self.after(POLL_INTERVAL_MS, self.poll_now)

    def _show_ai_suggestion_prompt(self) -> None:
        messagebox.askyesno(
            title="Chamber experiment began",
            message=(
                "The chamber experiment appears to have started.\n\n"
                "Do you want to use the AI suggestion for this experiment?"
            ),
            parent=self,
        )

    def _is_running(self, states: Dict[str, Any]) -> bool:
        value = states.get("temp_u")
        if value is None:
            return False
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return False
        return not math.isnan(numeric)

    def _numeric_state(self, states: Dict[str, Any], key: str) -> Optional[float]:
        value = states.get(key)
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(numeric):
            return None
        return numeric

    def _record_temperature_reading(self, result: PollResult) -> None:
        if result.states is None:
            return
        if not self.running:
            self._update_plot_status()
            return

        temp = self._numeric_state(result.states, "temp")
        if temp is None:
            self._update_plot_status()
            return

        temp_ref = self._numeric_state(result.states, "temp_ref")
        try:
            checked_at = datetime.strptime(result.checked_at, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            checked_at = datetime.now()

        self.temperature_history.append((checked_at, temp, temp_ref))
        self.temperature_history = self.temperature_history[-500:]
        self._update_plot_status()

    def _update_plot_status(self) -> None:
        if self.temperature_history:
            self.plot_status_var.set(
                f"Temperature readings: {len(self.temperature_history)}"
                + (" | running" if self.running else " | experiment not running")
            )
        elif self.running:
            self.plot_status_var.set("Experiment running; waiting for temperature reading.")
        else:
            self.plot_status_var.set("No running experiment detected.")

    def clear_plot(self) -> None:
        self.temperature_history.clear()
        self._update_plot_status()
        self.redraw_plot()

    def redraw_plot(self) -> None:
        canvas = getattr(self, "plot_canvas", None)
        if canvas is None:
            return

        canvas.delete("all")
        width = max(int(canvas.winfo_width()), 320)
        height = max(int(canvas.winfo_height()), 220)
        left, right, top, bottom = 54, 18, 18, 40
        plot_w = max(width - left - right, 1)
        plot_h = max(height - top - bottom, 1)

        canvas.create_rectangle(left, top, width - right, height - bottom, outline="#d7dde5")
        canvas.create_text(left, height - 16, text="time", anchor="w", fill="#526070")
        canvas.create_text(12, top, text="temp", anchor="nw", fill="#526070")

        if not self.temperature_history:
            canvas.create_text(
                width / 2,
                height / 2,
                text="No temperature readings yet",
                fill="#526070",
                font=("Segoe UI", 11, "bold"),
            )
            return

        times = [row[0] for row in self.temperature_history]
        temps = [row[1] for row in self.temperature_history]
        refs = [row[2] for row in self.temperature_history if row[2] is not None]
        values = temps + refs
        y_min = min(values)
        y_max = max(values)
        if math.isclose(y_min, y_max):
            y_min -= 1.0
            y_max += 1.0
        y_pad = max((y_max - y_min) * 0.08, 0.5)
        y_min -= y_pad
        y_max += y_pad

        t0 = times[0].timestamp()
        t1 = times[-1].timestamp()
        if math.isclose(t0, t1):
            t1 = t0 + 1.0

        def xy(ts: datetime, value: float) -> Tuple[float, float]:
            x = left + ((ts.timestamp() - t0) / (t1 - t0)) * plot_w
            y = top + (1.0 - ((value - y_min) / (y_max - y_min))) * plot_h
            return x, y

        for frac in (0.0, 0.5, 1.0):
            y_value = y_min + (1.0 - frac) * (y_max - y_min)
            y = top + frac * plot_h
            canvas.create_line(left, y, width - right, y, fill="#eef1f5")
            canvas.create_text(left - 8, y, text=f"{y_value:.1f}", anchor="e", fill="#526070")

        if refs:
            ref_points: List[float] = []
            for ts, _temp, ref in self.temperature_history:
                if ref is not None:
                    ref_points.extend(xy(ts, ref))
            if len(ref_points) >= 4:
                canvas.create_line(*ref_points, fill="#7a8797", width=2, dash=(5, 4))

        temp_points: List[float] = []
        for ts, temp, _ref in self.temperature_history:
            temp_points.extend(xy(ts, temp))

        if len(temp_points) >= 4:
            canvas.create_line(*temp_points, fill="#1f6fd1", width=2)
        else:
            x, y = temp_points
            canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#1f6fd1", outline="")

        latest_ts, latest_temp, latest_ref = self.temperature_history[-1]
        latest_label = f"temp {latest_temp:.2f}"
        if latest_ref is not None:
            latest_label += f" | target {latest_ref:.2f}"
        canvas.create_text(width - right, top + 8, text=latest_label, anchor="ne", fill="#173d79")
        canvas.create_text(left, height - 16, text=times[0].strftime("%H:%M:%S"), anchor="w", fill="#526070")
        canvas.create_text(width - right, height - 16, text=latest_ts.strftime("%H:%M:%S"), anchor="e", fill="#526070")

    def _format_state(self, states: Dict[str, Any], key: str) -> str:
        value = states.get(key)
        if value is None:
            return "-"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        if math.isnan(numeric):
            return "nan"
        return f"{numeric:.4f}"


def main() -> None:
    app = ChamberMonitorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
