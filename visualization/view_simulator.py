"""SimulatorMixin — builds the GRU Simulator page and handles animation."""
import numpy as np
import pyqtgraph as pg

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from common import COLORS, fmt
from chart_widget import ChartWidget
from run_tab import SimWorker


class SimulatorMixin:
    def _sim_view(self):
        page = QScrollArea()
        page.setObjectName("workspaceScroll")
        page.setWidgetResizable(True)
        page.setFrameShape(QFrame.NoFrame)
        body = QWidget()
        body.setObjectName("workspaceBody")
        lay  = QVBoxLayout(body)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)
        page.setWidget(body)

        form_card = QFrame()
        form_card.setObjectName("card")
        form_lay = QVBoxLayout(form_card)
        form_lay.setContentsMargins(12, 12, 12, 12)
        form_lay.setSpacing(10)
        hdr = QLabel("GRU SIMULATOR")
        hdr.setObjectName("sectionLabel")
        form_lay.addWidget(hdr)

        form_grid = QGridLayout()
        form_grid.setHorizontalSpacing(16)
        form_grid.setVerticalSpacing(10)
        self.sim_inputs = {}
        defaults = {
            "kp": "7", "ki": "700", "kd": "10",
            "start_temp": "27", "target_temp": "0",
            "duration_s": "1200", "dt_s": "2",
            "initial_p": "0", "initial_i": "0", "initial_d": "0",
        }
        for i, (key, val) in enumerate(defaults.items()):
            cell = QWidget()
            cl   = QVBoxLayout(cell)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(4)
            lbl = QLabel(key.replace("_", " ").title())
            lbl.setObjectName("sectionLabel")
            ed = QLineEdit(val)
            self.sim_inputs[key] = ed
            cl.addWidget(lbl)
            cl.addWidget(ed)
            form_grid.addWidget(cell, i // 5, i % 5)
            form_grid.setColumnStretch(i % 5, 1)
        run_btn = QPushButton("Run Simulation")
        run_btn.setObjectName("primaryButton")
        run_btn.clicked.connect(self.run_simulation)
        self.sim_button = run_btn
        form_grid.addWidget(run_btn, 2, 0, 1, 5)
        form_lay.addLayout(form_grid)
        lay.addWidget(form_card)

        self.sim_summary_grid = QGridLayout()
        lay.addLayout(self.sim_summary_grid)

        sim_card = QFrame()
        sim_card.setObjectName("card")
        sim_lay  = QVBoxLayout(sim_card)
        sim_lay.setContentsMargins(12, 12, 12, 12)
        sim_lay.setSpacing(8)
        sim_tb = QHBoxLayout()
        self.sim_title   = QLabel("Awaiting simulation")
        self.sim_title.setObjectName("title")
        self.sim_channel = QComboBox()
        self.sim_channel.addItems(["temp", "error", "u", "pred_delta"])
        self.sim_channel.currentTextChanged.connect(self._on_sim_channel_changed)
        sim_tb.addWidget(self.sim_title, 1)
        sim_tb.addWidget(self.sim_channel)
        sim_lay.addLayout(sim_tb)
        self.sim_chart = ChartWidget()
        sim_lay.addWidget(self.sim_chart)
        lay.addWidget(sim_card)
        return page

    def run_simulation(self):
        payload = {}
        try:
            for key, ed in self.sim_inputs.items():
                payload[key] = float(ed.text())
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid input", str(exc))
            return
        self.sim_button.setEnabled(False)
        self.sim_title.setText("Running simulation…")
        self.sim_worker = SimWorker(payload)
        self.sim_worker.finished_ok.connect(lambda r: self._sim_done(payload, r))
        self.sim_worker.failed.connect(self._sim_failed)
        self.sim_worker.start()

    def _sim_done(self, payload, result):
        self.sim_series = result
        self.sim_button.setEnabled(True)
        self.sim_title.setText(
            f"Kp {payload['kp']:g} / Ki {payload['ki']:g} / Kd {payload['kd']:g}")
        self._render_sim_summary()
        self._start_sim_animation()

    def _sim_failed(self, msg):
        self.sim_button.setEnabled(True)
        self.sim_title.setText("Simulation failed")
        QMessageBox.critical(self, "Simulation failed", msg)

    def _render_sim_summary(self):
        self._clear_layout(self.sim_summary_grid)
        metrics = self.sim_series.get("metrics", {}) if self.sim_series else {}
        stats = [
            ("Cost",      metrics.get("cost")),
            ("Tail MAE",  metrics.get("tail_mae")),
            ("End Temp",  metrics.get("end_temp")),
            ("Overshoot", metrics.get("overshoot_max")),
            ("Settle s",  metrics.get("time_to_settle_s")),
        ]
        for i, (lbl, val) in enumerate(stats):
            box = QFrame()
            box.setObjectName("card")
            bl = QVBoxLayout(box)
            bl.setContentsMargins(14, 10, 14, 10)
            bl.setSpacing(4)
            cap = QLabel(lbl)
            cap.setObjectName("sectionLabel")
            bl.addWidget(cap)
            num = QLabel(fmt(val))
            num.setStyleSheet("font-size: 22px; font-weight: 850; background: transparent;")
            bl.addWidget(num)
            self.sim_summary_grid.addWidget(box, 0, i)

    def _on_sim_channel_changed(self):
        if self._sim_anim_timer and self._sim_anim_timer.isActive():
            self._sim_anim_timer.stop()
        self.draw_simulation()

    def _start_sim_animation(self):
        if self._sim_anim_timer and self._sim_anim_timer.isActive():
            self._sim_anim_timer.stop()
        if not self.sim_series:
            return
        ch   = self.sim_channel.currentText()
        cols = ["temp", "temp_ref"] if ch == "temp" else [ch]
        all_points = self.sim_series.get("points", [])
        if not all_points:
            return
        finite_t = [p["t"] for p in all_points if isinstance(p.get("t"), (int, float))]
        first_t  = finite_t[0] if finite_t else 0
        self._sim_anim_data = []
        for i, col in enumerate(cols):
            color = COLORS[i % len(COLORS)]
            xs, ys = [], []
            for p in all_points:
                y = p.get("values", {}).get(col)
                if y is None:
                    continue
                t = p.get("t")
                x = (t - first_t) if isinstance(t, (int, float)) else float(p.get("i", 0))
                xs.append(x)
                ys.append(float(y))
            self._sim_anim_data.append({"col": col, "color": color, "xs": xs, "ys": ys})
        self._sim_anim_total = max(
            (len(d["xs"]) for d in self._sim_anim_data), default=0)
        if self._sim_anim_total == 0:
            self.draw_simulation()
            return
        self._sim_anim_idx = 0
        self.sim_chart.clear_plot_items()
        self.sim_chart.plot.setLabel("bottom", "Time elapsed (s)")
        self.sim_chart.plot.setLabel(
            "left", "Temperature [deg C]" if ch == "temp" else "Value")
        self.sim_chart.auto_range_on_draw = True
        self._sim_anim_curves = []
        for d in self._sim_anim_data:
            curve = self.sim_chart.plot.plot(
                [], [], pen=pg.mkPen(d["color"], width=1.8), name=d["col"])
            self._sim_anim_curves.append(curve)
        self._sim_anim_timer = QTimer(self)
        self._sim_anim_timer.timeout.connect(self._sim_anim_step)
        self._sim_anim_timer.start(16)

    def _sim_anim_step(self):
        chunk = max(1, self._sim_anim_total // 60)
        self._sim_anim_idx = min(self._sim_anim_idx + chunk, self._sim_anim_total)
        for curve, d in zip(self._sim_anim_curves, self._sim_anim_data):
            xs = np.array(d["xs"][:self._sim_anim_idx], dtype=float)
            ys = np.array(d["ys"][:self._sim_anim_idx], dtype=float)
            curve.setData(xs, ys)
        if self.sim_chart.auto_range_on_draw and self._sim_anim_idx > 0:
            self.sim_chart.plot.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
            self.sim_chart.auto_range_on_draw = False
        if self._sim_anim_idx >= self._sim_anim_total:
            self._sim_anim_timer.stop()
            self.sim_chart.curves = list(self._sim_anim_curves)
            self.sim_chart.hover_points = []
            for d in self._sim_anim_data:
                for x, y in zip(d["xs"], d["ys"]):
                    self.sim_chart.hover_points.append(
                        {"x": x, "y": y, "label": d["col"], "color": d["color"]})

    def draw_simulation(self):
        if not self.sim_series:
            return
        if self._sim_anim_timer and self._sim_anim_timer.isActive():
            self._sim_anim_timer.stop()
        ch      = self.sim_channel.currentText()
        payload = {"columns": ["temp", "temp_ref"] if ch == "temp" else [ch],
                   "points":  self.sim_series["points"]}
        self.sim_chart.draw(payload, "line")
