"""Chart infrastructure: CrosshairSyncHub, BoxZoomViewBox, ChartWidget."""
import numpy as np
import pyqtgraph as pg

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox, QDialog, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QWidget,
)

from common import COLORS, RULE_COLOR_OPTIONS, fmt


class CrosshairSyncHub:
    def __init__(self):
        self._charts = []
        self._enabled = False
        self._busy = False

    @property
    def enabled(self):
        return self._enabled

    def register(self, chart):
        if chart not in self._charts:
            self._charts.append(chart)
            chart.sync_hub = self

    def set_enabled(self, enabled):
        self._enabled = enabled
        if not enabled:
            self._broadcast_op(None, lambda c: c.hide_sync_crosshair())

    def broadcast(self, source, x, y):
        if not self._enabled or self._busy:
            return
        self._broadcast_op(source, lambda c: c.set_sync_xy(x, y))

    def broadcast_hide(self, source):
        if not self._enabled or self._busy:
            return
        self._broadcast_op(source, lambda c: c.hide_sync_crosshair())

    def _broadcast_op(self, source, op):
        self._busy = True
        dead = []
        for c in self._charts:
            if c is source:
                continue
            try:
                if c.isVisible():
                    op(c)
            except RuntimeError:
                dead.append(c)
        for c in dead:
            self._charts.remove(c)
        self._busy = False


class BoxZoomViewBox(pg.ViewBox):
    annotate_drag    = Signal(float, float)  # final commit: x0, x1
    annotate_preview = Signal(float, float)  # live during drag: x0, x1

    def __init__(self):
        super().__init__()
        self.setMouseMode(pg.ViewBox.PanMode)
        self.setMenuEnabled(False)
        self.annotate_mode = False
        self._ann_start = None

    def mouseDragEvent(self, event, axis=None):
        if self.annotate_mode and event.button() == Qt.LeftButton and axis is None:
            event.accept()
            if event.isStart():
                self._ann_start = self.mapToView(event.buttonDownPos(Qt.LeftButton))
            elif event.isFinish() and self._ann_start is not None:
                end_pt = self.mapToView(event.pos())
                x0 = min(self._ann_start.x(), end_pt.x())
                x1 = max(self._ann_start.x(), end_pt.x())
                if x1 > x0:
                    self.annotate_drag.emit(x0, x1)
                else:
                    self.annotate_preview.emit(0.0, 0.0)  # zero-width = cancel
                self._ann_start = None
            elif self._ann_start is not None:
                end_pt = self.mapToView(event.pos())
                x0 = min(self._ann_start.x(), end_pt.x())
                x1 = max(self._ann_start.x(), end_pt.x())
                self.annotate_preview.emit(x0, x1)
            return
        if event.button() == Qt.RightButton and axis is None:
            event.accept()
            if event.isFinish():
                self.rbScaleBox.hide()
                rect = QRectF(
                    pg.Point(event.buttonDownPos(Qt.RightButton)),
                    pg.Point(event.pos()),
                )
                mapped = self.childGroup.mapRectFromParent(rect)
                if mapped.width() != 0 and mapped.height() != 0:
                    self.showAxRect(mapped)
            else:
                self.updateScaleBox(event.buttonDownPos(Qt.RightButton), event.pos())
            return
        super().mouseDragEvent(event, axis=axis)


class ChartWidget(QWidget):
    annotation_committed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.payload = None
        self.mode = "line"
        self.annotations = []
        self.setpoint = None
        self.sync_hub = None
        self.datasets = []
        self.curves = []
        self.hover_points = []
        self.marker_items = []
        self.overlay_items = []
        self.dark = True
        self.smoothing_window = 1
        self.overlay_flags = {"min": False, "max": False, "avg": False}
        self.marker_flags = {"events": True, "alarms": True, "controller": True, "state": True}
        self.auto_range_on_draw = True

        self.plot = pg.PlotWidget(viewBox=BoxZoomViewBox())
        self.plot.setObjectName("chartCanvas")
        self.plot.setMinimumHeight(400)
        self.plot.setBackground("#111827")
        self.plot.showGrid(x=True, y=True, alpha=0.28)
        self.plot.setLabel("bottom", "Time elapsed (s)")
        self.plot.setLabel("left", "Value")
        self.legend = self.plot.addLegend(offset=(-12, 12))
        self.plot.setMouseEnabled(x=True, y=True)
        self.plot.getViewBox().setMouseMode(pg.ViewBox.PanMode)

        self.crosshair_v = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("#94a3b8", width=1, style=Qt.DashLine))
        self.crosshair_h = pg.InfiniteLine(
            angle=0,  movable=False, pen=pg.mkPen("#94a3b8", width=1, style=Qt.DashLine))
        self.crosshair_v.setVisible(False)
        self.crosshair_h.setVisible(False)
        self.plot.addItem(self.crosshair_v, ignoreBounds=True)
        self.plot.addItem(self.crosshair_h, ignoreBounds=True)

        self.hover_label = pg.TextItem(
            "", anchor=(0, 1), color="#f8fafc",
            fill=pg.mkBrush(17, 24, 39, 235), border=pg.mkPen("#60a5fa"))
        self.hover_label.setVisible(False)
        self.plot.addItem(self.hover_label, ignoreBounds=True)
        self.plot.scene().sigMouseMoved.connect(self.on_mouse_moved)

        self.region = pg.LinearRegionItem(
            brush=pg.mkBrush(96, 165, 250, 38), pen=pg.mkPen("#60a5fa"))
        self.region.setZValue(-10)
        self.region.setVisible(False)
        self.region.sigRegionChanged.connect(self.apply_time_region)
        self.plot.addItem(self.region, ignoreBounds=True)

        self._ann_preview = pg.LinearRegionItem(
            brush=pg.mkBrush(96, 165, 250, 55),
            pen=pg.mkPen("#60a5fa", width=1.5, style=Qt.DashLine),
            movable=False,
        )
        self._ann_preview.setZValue(-14)
        self._ann_preview.setVisible(False)
        self.plot.addItem(self._ann_preview, ignoreBounds=True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.plot)

    def set_dark(self, dark=True):
        self.dark = dark
        self.plot.setBackground("#111827" if dark else "#ffffff")
        self.redraw()

    def redraw(self):
        self.draw(self.payload, self.mode, self.annotations, self.setpoint)

    def draw(self, payload, mode="line", annotations=None, setpoint=None):
        view_range = self.plot.getViewBox().viewRange()
        self.payload = payload
        self.mode = mode
        self.annotations = annotations or []
        self.setpoint = setpoint
        self.datasets = self._datasets(payload)
        self.clear_plot_items()
        self.hover_points = []
        self.plot.setLabel("left", self.y_axis_label())
        if not self.datasets:
            self.plot.setTitle("No chart data")
            return
        self.plot.setTitle("")
        for index, dataset in enumerate(self.datasets):
            color = COLORS[index % len(COLORS)]
            points = self.apply_smoothing(dataset["points"])
            if not points:
                continue
            xs = np.array([p[0] for p in points], dtype=float)
            ys = np.array([p[1] for p in points], dtype=float)
            pen = pg.mkPen(color, width=1.8)
            symbol = "o" if mode == "scatter" else None
            curve = self.plot.plot(
                xs, ys,
                pen=None if mode == "scatter" else pen,
                symbol=symbol,
                symbolSize=6 if mode == "scatter" else None,
                symbolBrush=color if mode == "scatter" else None,
                name=dataset["label"],
            )
            self.curves.append(curve)
            self.hover_points.extend(
                {"x": float(x), "y": float(y), "label": dataset["label"], "color": color}
                for x, y in zip(xs, ys)
            )
        self.draw_annotations()
        self.draw_overlays()
        if setpoint is not None:
            self.add_horizontal_marker(setpoint, f"Setpoint {fmt(setpoint, 2)}", "#94a3b8")
        if self.auto_range_on_draw:
            self.plot.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
            self.plot.autoRange()
            self.auto_range_on_draw = False
        else:
            self.plot.getViewBox().setRange(
                xRange=view_range[0], yRange=view_range[1], padding=0)

    def clear_plot_items(self):
        for item in self.curves + self.marker_items + self.overlay_items:
            try:
                self.plot.removeItem(item)
            except Exception:
                pass
        self.curves = []
        self.marker_items = []
        self.overlay_items = []
        if self.legend:
            self.legend.clear()

    def reset_view(self):
        self.auto_range_on_draw = True
        self.region.setVisible(False)
        self.redraw()

    def set_smoothing_window(self, window):
        self.smoothing_window = max(1, int(window))
        self.redraw()

    def set_overlay_flags(self, **flags):
        self.overlay_flags.update(flags)
        self.redraw()

    def set_marker_flags(self, **flags):
        self.marker_flags.update(flags)
        self.redraw()

    def set_time_range(self, start, end):
        if start is None or end is None or end <= start:
            self.region.setVisible(False)
            self.auto_range_on_draw = True
            self.redraw()
            return
        self.region.blockSignals(True)
        self.region.setRegion((float(start), float(end)))
        self.region.setVisible(True)
        self.region.blockSignals(False)
        self.plot.setXRange(float(start), float(end), padding=0)

    def apply_time_region(self):
        if self.region.isVisible():
            start, end = self.region.getRegion()
            if end > start:
                self.plot.setXRange(start, end, padding=0)

    def current_x_range(self):
        return self.plot.getViewBox().viewRange()[0]

    def data_x_range(self):
        xs = [p[0] for ds in self.datasets for p in ds["points"]]
        if not xs:
            return 0.0, 0.0
        return float(min(xs)), float(max(xs))

    def export_view(self, path):
        exporter = pg.exporters.ImageExporter(self.plot.plotItem)
        exporter.parameters()["width"] = max(1200, int(self.plot.width()))
        exporter.export(path)

    def y_axis_label(self):
        labels = " ".join(ds["label"].lower() for ds in self.datasets)
        return "Temperature [deg C]" if "temp" in labels else "Value"

    def _datasets(self, payload):
        if not payload:
            return []
        if payload.get("series"):
            return [
                {"label": item["label"],
                 "points": self._normalize_points(item.get("points", []), payload.get("channel"))}
                for item in payload["series"]
            ]
        return [
            {"label": col, "points": self._normalize_points(payload.get("points", []), col)}
            for col in payload.get("columns", [])
        ]

    def _normalize_points(self, points, column):
        if not column:
            return []
        finite_t = [p.get("t") for p in points if isinstance(p.get("t"), (int, float))]
        first_t = finite_t[0] if finite_t else 0
        result = []
        for p in points:
            y = p.get("values", {}).get(column)
            if y is None:
                continue
            x_raw = p.get("t") if isinstance(p.get("t"), (int, float)) else p.get("i", 0)
            x = x_raw - first_t if isinstance(p.get("t"), (int, float)) else x_raw
            result.append((float(x), float(y)))
        return result

    def apply_smoothing(self, points):
        if self.smoothing_window <= 1 or len(points) < self.smoothing_window:
            return points
        xs = np.array([p[0] for p in points], dtype=float)
        ys = np.array([p[1] for p in points], dtype=float)
        kernel = np.ones(self.smoothing_window, dtype=float) / self.smoothing_window
        smoothed = np.convolve(ys, kernel, mode="same")
        return list(zip(xs.tolist(), smoothed.tolist()))

    def draw_overlays(self):
        all_points = [p for ds in self.datasets for p in self.apply_smoothing(ds["points"])]
        if not all_points:
            return
        ys = np.array([p[1] for p in all_points], dtype=float)
        if self.overlay_flags.get("min"):
            self.add_horizontal_marker(float(np.min(ys)), f"Min {fmt(np.min(ys), 3)}", "#51d6c7", overlay=True)
        if self.overlay_flags.get("max"):
            self.add_horizontal_marker(float(np.max(ys)), f"Max {fmt(np.max(ys), 3)}", "#ff6f7d", overlay=True)
        if self.overlay_flags.get("avg"):
            self.add_horizontal_marker(float(np.mean(ys)), f"Avg {fmt(np.mean(ys), 3)}", "#f2bd52", overlay=True)

    def draw_annotations(self):
        for ann in self.annotations:
            cat = self._marker_category(ann)
            if not self.marker_flags.get(cat, True):
                continue
            color = {"events": "#8bd66f", "alarms": "#ff6f7d",
                     "controller": "#f2bd52", "state": "#b792ff"}.get(cat, "#94a3b8")
            atype = ann.get("type")
            if atype == "line-x":
                self.add_vertical_marker(ann.get("x"), ann.get("label", "Marker"), color)
            elif atype == "line-y":
                self.add_horizontal_marker(ann.get("y"), ann.get("label", "Marker"), color)
            elif atype == "point":
                item = pg.ScatterPlotItem(
                    [ann.get("x")], [ann.get("y")], size=9,
                    brush=pg.mkBrush(color), pen=pg.mkPen("#111827", width=1),
                    name=ann.get("label", "Event"))
                self.plot.addItem(item)
                self.marker_items.append(item)
            elif atype == "region-x":
                region = pg.LinearRegionItem(
                    values=(ann.get("x0"), ann.get("x1")), movable=False,
                    brush=pg.mkBrush(color + "33"), pen=pg.mkPen(color, width=1))
                region.setZValue(-20)
                self.plot.addItem(region)
                self.marker_items.append(region)

    def _marker_category(self, ann):
        kind = ann.get("kind")
        if kind == "pid":
            return "controller"
        if kind == "invalid":
            return "alarms"
        if kind == "settling":
            return "state"
        return "events"

    def add_vertical_marker(self, x, label, color):
        if x is None:
            return
        line = pg.InfiniteLine(
            pos=float(x), angle=90, movable=False,
            pen=pg.mkPen(color, width=1.1, style=Qt.DashLine), label=label)
        self.plot.addItem(line)
        self.marker_items.append(line)

    def add_horizontal_marker(self, y, label, color, overlay=False):
        if y is None:
            return
        line = pg.InfiniteLine(
            pos=float(y), angle=0, movable=False,
            pen=pg.mkPen(color, width=1.1, style=Qt.DashLine), label=label)
        self.plot.addItem(line)
        (self.overlay_items if overlay else self.marker_items).append(line)

    def add_range_band(self, lo, hi, color, label=""):
        qc = QColor(color)
        region = pg.LinearRegionItem(
            values=(float(lo), float(hi)),
            orientation=pg.LinearRegionItem.Horizontal,
            movable=False,
            brush=pg.mkBrush(qc.red(), qc.green(), qc.blue(), 45),
            pen=pg.mkPen(color, width=0.8, style=Qt.DashLine),
        )
        region.setZValue(-15)
        self.plot.addItem(region)
        self.overlay_items.append(region)
        if label:
            txt = pg.TextItem(label, color=color, anchor=(0, 1))
            txt.setPos(self.plot.getViewBox().viewRange()[0][0], float(hi))
            self.plot.addItem(txt)
            self.overlay_items.append(txt)

    def add_x_annotation(self, x0, x1, label, color):
        qc = QColor(color)
        region = pg.LinearRegionItem(
            values=(float(x0), float(x1)),
            movable=False,
            brush=pg.mkBrush(qc.red(), qc.green(), qc.blue(), 40),
            pen=pg.mkPen(color, width=0.8, style=Qt.DashLine),
        )
        region.setZValue(-15)
        self.plot.addItem(region)
        self.overlay_items.append(region)
        if label:
            line = pg.InfiniteLine(
                pos=float(x0), angle=90, movable=False,
                pen=pg.mkPen(color, width=1, style=Qt.DashLine),
                label=label,
                labelOpts={"color": color, "position": 0.95, "anchors": [(0, 0), (0, 0)]},
            )
            self.plot.addItem(line)
            self.overlay_items.append(line)

    def set_annotate_mode(self, enabled: bool):
        vb = self.plot.getViewBox()
        vb.annotate_mode = enabled
        if enabled:
            vb.annotate_drag.connect(self._on_annotate_drag)
            vb.annotate_preview.connect(self._on_annotate_preview)
        else:
            self._ann_preview.setVisible(False)
            try:
                vb.annotate_drag.disconnect(self._on_annotate_drag)
                vb.annotate_preview.disconnect(self._on_annotate_preview)
            except RuntimeError:
                pass

    def _on_annotate_preview(self, x0, x1):
        if x1 > x0:
            self._ann_preview.setRegion((x0, x1))
            self._ann_preview.setVisible(True)
        else:
            self._ann_preview.setVisible(False)

    def _on_annotate_drag(self, x0, x1):
        self._ann_preview.setVisible(False)
        dlg = QDialog(self)
        dlg.setWindowTitle("Add Annotation")
        lay = QVBoxLayout(dlg)
        form = QHBoxLayout()
        form.setSpacing(8)
        form.addWidget(QLabel("Label:"))
        label_ed = QLineEdit()
        label_ed.setPlaceholderText("e.g. Steady state")
        form.addWidget(label_ed)
        form.addWidget(QLabel("Color:"))
        color_combo = QComboBox()
        for name, _ in RULE_COLOR_OPTIONS:
            color_combo.addItem(name)
        form.addWidget(color_combo)
        lay.addLayout(form)
        btns = QHBoxLayout()
        btns.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dlg.reject)
        ok_btn = QPushButton("Add")
        ok_btn.setObjectName("primaryButton")
        ok_btn.clicked.connect(dlg.accept)
        ok_btn.setDefault(True)
        btns.addWidget(cancel_btn)
        btns.addWidget(ok_btn)
        lay.addLayout(btns)
        if dlg.exec() == QDialog.Accepted:
            _, color = RULE_COLOR_OPTIONS[color_combo.currentIndex()]
            text = label_ed.text().strip() or f"t={x0:.1f}–{x1:.1f} s"
            self.annotation_committed.emit({"x0": x0, "x1": x1, "label": text, "color": color})

    def on_mouse_moved(self, scene_pos):
        if not self.plot.sceneBoundingRect().contains(scene_pos):
            self.crosshair_v.setVisible(False)
            self.crosshair_h.setVisible(False)
            self.hover_label.setVisible(False)
            if self.sync_hub:
                self.sync_hub.broadcast_hide(self)
            return
        point = self.plot.plotItem.vb.mapSceneToView(scene_pos)
        self.crosshair_v.setPos(point.x())
        self.crosshair_h.setPos(point.y())
        self.crosshair_v.setVisible(True)
        self.crosshair_h.setVisible(True)
        if self.sync_hub:
            self.sync_hub.broadcast(self, point.x(), point.y())
        best = self._nearest_point(scene_pos)
        if best is None:
            self.hover_label.setVisible(False)
            return
        self.hover_label.setText(
            f"{best['label']}\nTime: {fmt(best['x'], 1)} s\nValue: {fmt(best['y'], 3)}")
        self.hover_label.setColor(best["color"])
        self.hover_label.setPos(best["x"], best["y"])
        self.hover_label.setVisible(True)

    def set_sync_xy(self, x, y):
        self.crosshair_v.setPos(float(x))
        self.crosshair_h.setPos(float(y))
        self.crosshair_v.setVisible(True)
        self.crosshair_h.setVisible(True)
        if self.hover_points:
            best = min(self.hover_points, key=lambda p: abs(p["x"] - x))
            vr = self.plot.getViewBox().viewRange()[0]
            if abs(best["x"] - x) <= (vr[1] - vr[0]) * 0.04:
                self.hover_label.setText(
                    f"{best['label']}\nTime: {fmt(best['x'], 1)} s\nValue: {fmt(best['y'], 3)}")
                self.hover_label.setColor(best["color"])
                self.hover_label.setPos(best["x"], best["y"])
                self.hover_label.setVisible(True)
                return
        self.hover_label.setVisible(False)

    def hide_sync_crosshair(self):
        self.crosshair_v.setVisible(False)
        self.crosshair_h.setVisible(False)
        self.hover_label.setVisible(False)

    def _nearest_point(self, scene_pos):
        best = None
        best_dist = float("inf")
        for p in self.hover_points:
            pixel = self.plot.plotItem.vb.mapViewToScene(pg.Point(p["x"], p["y"]))
            dist = ((pixel.x() - scene_pos.x()) ** 2 +
                    (pixel.y() - scene_pos.y()) ** 2) ** 0.5
            if dist < best_dist:
                best = p
                best_dist = dist
        return best if best_dist <= 18 else None
