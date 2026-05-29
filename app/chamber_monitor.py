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
from typing import Any, Dict, Optional

import tkinter as tk
from tkinter import messagebox, ttk

APP_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = APP_DIR.parent
TCP_DIR = SCRIPTS_DIR / "tcp"
if str(TCP_DIR) not in sys.path:
    sys.path.insert(0, str(TCP_DIR))

try:
    from tcp_common import DEFAULT_HOST, DEFAULT_PORT, DEFAULT_TIMEOUT, request_states
except Exception as exc:  # pragma: no cover - shown in GUI at runtime
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

        self._build_ui()
        self.after(100, self.poll_now)
        self.after(200, self.process_results)

    def _build_ui(self) -> None:
        self.configure(bg="#f6f7f9")

        root = ttk.Frame(self, padding=18)
        root.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(root, text="DeepVac Chamber", font=("Segoe UI", 18, "bold"))
        title.pack(anchor="w")

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

        self._apply_badge_styles()

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
        else:
            self.connected = False
            self.running = False
            self.connection_var.set("Disconnected")
            self.running_var.set("Not running")
            self.error_var.set(result.error)

        self._apply_badge_styles()
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
