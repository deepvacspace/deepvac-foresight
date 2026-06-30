import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters

from PySide6.QtCore import (
    QByteArray, QMimeData, QPoint, QRectF, Qt, QThread, QUrl, Signal,
)
from PySide6.QtGui import (
    QAction, QColor, QDesktopServices, QDrag, QFont, QIcon, QPainter, QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMenu, QMessageBox, QPushButton,
    QScrollArea, QSizeGrip, QSpinBox, QSplashScreen, QSplitter,
    QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

import data_service as data

HERE = Path(__file__).resolve().parent
LOGO_PATH = str(HERE / "logo.png")
ICON_PATH = str(HERE / "icon.png")

COLORS = ["#8bd66f", "#4f7cff", "#51d6c7", "#f2bd52",
          "#ff6f7d", "#b792ff", "#f48fb1", "#7ec8ff"]

_TAB_MIME = "application/deepvac-tab"
_gid = 0


def _new_gid():
    global _gid
    _gid += 1
    return _gid


# ── CrosshairSyncHub ─────────────────────────────────────────────────────────

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


def fmt(value, digits=3):
    if value in (None, ""):
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 1000:
        return f"{number:,.1f}"
    return f"{number:,.{digits}f}".rstrip("0").rstrip(".")


def csv_escape(value):
    if value is None:
        return ""
    text = str(value)
    if any(ch in text for ch in [",", '"', "\n", "\r"]):
        return '"' + text.replace('"', '""') + '"'
    return text


# ── SimWorker ────────────────────────────────────────────────────────────────

class SimWorker(QThread):
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, payload):
        super().__init__()
        self.payload = payload

    def run(self):
        try:
            self.finished_ok.emit(data.simulate_gru_run(self.payload))
        except Exception as exc:
            self.failed.emit(str(exc))


# ── BoxZoomViewBox ───────────────────────────────────────────────────────────

class BoxZoomViewBox(pg.ViewBox):
    def __init__(self):
        super().__init__()
        self.setMouseMode(pg.ViewBox.PanMode)
        self.setMenuEnabled(False)

    def mouseDragEvent(self, event, axis=None):
        if event.button() == Qt.RightButton and axis is None:
            event.accept()
            if event.isFinish():
                self.rbScaleBox.hide()
                rect = QRectF(pg.Point(event.buttonDownPos(Qt.RightButton)), pg.Point(event.pos()))
                mapped = self.childGroup.mapRectFromParent(rect)
                if mapped.width() != 0 and mapped.height() != 0:
                    self.showAxRect(mapped)
            else:
                self.updateScaleBox(event.buttonDownPos(Qt.RightButton), event.pos())
            return
        super().mouseDragEvent(event, axis=axis)


# ── ChartWidget ──────────────────────────────────────────────────────────────

class ChartWidget(QWidget):
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
        self.plot.setLabel("bottom", "Time elapsed", units="s")
        self.plot.setLabel("left", "Value")
        self.legend = self.plot.addLegend(offset=(-12, 12))
        self.plot.setMouseEnabled(x=True, y=True)
        self.plot.getViewBox().setMouseMode(pg.ViewBox.PanMode)

        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#94a3b8", width=1, style=Qt.DashLine))
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("#94a3b8", width=1, style=Qt.DashLine))
        self.crosshair_v.setVisible(False)
        self.crosshair_h.setVisible(False)
        self.plot.addItem(self.crosshair_v, ignoreBounds=True)
        self.plot.addItem(self.crosshair_h, ignoreBounds=True)

        self.hover_label = pg.TextItem("", anchor=(0, 1), color="#f8fafc",
                                       fill=pg.mkBrush(17, 24, 39, 235), border=pg.mkPen("#60a5fa"))
        self.hover_label.setVisible(False)
        self.plot.addItem(self.hover_label, ignoreBounds=True)
        self.plot.scene().sigMouseMoved.connect(self.on_mouse_moved)

        self.region = pg.LinearRegionItem(brush=pg.mkBrush(96, 165, 250, 38), pen=pg.mkPen("#60a5fa"))
        self.region.setZValue(-10)
        self.region.setVisible(False)
        self.region.sigRegionChanged.connect(self.apply_time_region)
        self.plot.addItem(self.region, ignoreBounds=True)

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
            curve = self.plot.plot(xs, ys,
                                   pen=None if mode == "scatter" else pen,
                                   symbol=symbol,
                                   symbolSize=6 if mode == "scatter" else None,
                                   symbolBrush=color if mode == "scatter" else None,
                                   name=dataset["label"])
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
            self.plot.getViewBox().setRange(xRange=view_range[0], yRange=view_range[1], padding=0)

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
            return [{"label": item["label"],
                     "points": self._normalize_points(item.get("points", []), payload.get("channel"))}
                    for item in payload["series"]]
        return [{"label": col, "points": self._normalize_points(payload.get("points", []), col)}
                for col in payload.get("columns", [])]

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
            color = {"events": "#8bd66f", "alarms": "#ff6f7d", "controller": "#f2bd52", "state": "#b792ff"}.get(cat, "#94a3b8")
            atype = ann.get("type")
            if atype == "line-x":
                self.add_vertical_marker(ann.get("x"), ann.get("label", "Marker"), color)
            elif atype == "line-y":
                self.add_horizontal_marker(ann.get("y"), ann.get("label", "Marker"), color)
            elif atype == "point":
                item = pg.ScatterPlotItem([ann.get("x")], [ann.get("y")], size=9,
                                          brush=pg.mkBrush(color), pen=pg.mkPen("#111827", width=1),
                                          name=ann.get("label", "Event"))
                self.plot.addItem(item)
                self.marker_items.append(item)
            elif atype == "region-x":
                region = pg.LinearRegionItem(values=(ann.get("x0"), ann.get("x1")), movable=False,
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
        line = pg.InfiniteLine(pos=float(x), angle=90, movable=False,
                               pen=pg.mkPen(color, width=1.1, style=Qt.DashLine), label=label)
        self.plot.addItem(line)
        self.marker_items.append(line)

    def add_horizontal_marker(self, y, label, color, overlay=False):
        if y is None:
            return
        line = pg.InfiniteLine(pos=float(y), angle=0, movable=False,
                               pen=pg.mkPen(color, width=1.1, style=Qt.DashLine), label=label)
        self.plot.addItem(line)
        (self.overlay_items if overlay else self.marker_items).append(line)

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
        self.hover_label.setText(f"{best['label']}\nTime: {fmt(best['x'], 1)} s\nValue: {fmt(best['y'], 3)}")
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
            dist = ((pixel.x() - scene_pos.x()) ** 2 + (pixel.y() - scene_pos.y()) ** 2) ** 0.5
            if dist < best_dist:
                best = p
                best_dist = dist
        return best if best_dist <= 18 else None


# ── TabBar ───────────────────────────────────────────────────────────────────

class TabBar(QWidget):
    tab_activated = Signal(int)
    tab_closed = Signal(int)
    split_right_requested = Signal(int)
    move_to_group_requested = Signal(int, int)  # tab_index, group_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("tabBar")
        self.setFixedHeight(35)
        self._entries = []
        self._active = -1
        self._drag_start = None
        self._drag_idx = -1
        self.get_other_groups = None  # injected by EditorGroup

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setObjectName("tabScroll")
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setFixedHeight(35)

        self._container = QWidget()
        self._container.setObjectName("tabContainer")
        self._tlayout = QHBoxLayout(self._container)
        self._tlayout.setContentsMargins(0, 0, 0, 0)
        self._tlayout.setSpacing(0)
        self._tlayout.addStretch(1)

        self._scroll.setWidget(self._container)
        outer.addWidget(self._scroll, 1)

    def wheelEvent(self, event):
        sb = self._scroll.horizontalScrollBar()
        sb.setValue(sb.value() - event.angleDelta().y() // 3)

    def add_or_focus(self, key, run_id):
        for i, e in enumerate(self._entries):
            if e["key"] == key:
                self._set_active(i)
                return i
        idx = len(self._entries)
        frame = self._make_tab(key, run_id)
        self._tlayout.insertWidget(idx, frame)
        self._entries.append({"key": key, "id": run_id, "frame": frame})
        self._set_active(idx)
        return idx

    def _make_tab(self, key, run_id):
        frame = QFrame()
        frame.setObjectName("tabItem")
        frame.setFixedHeight(35)
        frame.setCursor(Qt.PointingHandCursor)
        frame.setProperty("tabActive", False)

        lay = QHBoxLayout(frame)
        lay.setContentsMargins(10, 0, 4, 0)
        lay.setSpacing(5)

        dot = QLabel("◈")
        dot.setObjectName("tabIcon")

        lbl = QLabel(run_id)
        lbl.setObjectName("tabLabel")
        lbl.setMaximumWidth(200)

        btn = QPushButton("×")
        btn.setObjectName("tabClose")
        btn.setFixedSize(16, 16)
        btn.clicked.connect(lambda: self._close_key(key))

        lay.addWidget(dot)
        lay.addWidget(lbl, 1)
        lay.addWidget(btn)

        frame.mousePressEvent = lambda e, k=key: self._press(e, k)
        frame.mouseMoveEvent = lambda e, k=key: self._move(e, k)
        frame.mouseReleaseEvent = lambda _ev: setattr(self, "_drag_start", None)
        frame.contextMenuEvent = lambda e, k=key: self._ctx(e, k)
        return frame

    def _idx(self, key):
        for i, e in enumerate(self._entries):
            if e["key"] == key:
                return i
        return -1

    def _press(self, event, key):
        i = self._idx(key)
        if i >= 0:
            self._set_active(i)
            self.tab_activated.emit(i)
            self._drag_start = event.globalPosition().toPoint()
            self._drag_idx = i

    def _move(self, event, key):
        if self._drag_start is None:
            return
        delta = event.globalPosition().toPoint() - self._drag_start
        if delta.manhattanLength() < 8:
            return
        i = self._idx(key)
        if i < 0:
            return
        mime = QMimeData()
        payload = json.dumps({"key": key, "id": self._entries[i]["id"]}).encode()
        mime.setData(_TAB_MIME, QByteArray(payload))
        drag = QDrag(self)
        drag.setMimeData(mime)
        pix = self._entries[i]["frame"].grab()
        drag.setPixmap(pix)
        drag.setHotSpot(QPoint(pix.width() // 2, pix.height() // 2))
        self._drag_start = None
        drag.exec(Qt.MoveAction | Qt.CopyAction)

    def _close_key(self, key):
        i = self._idx(key)
        if i >= 0:
            self.tab_closed.emit(i)

    def _ctx(self, event, key):
        i = self._idx(key)
        menu = QMenu(self)
        a_sr = menu.addAction("Split Right")
        moves = []
        if callable(self.get_other_groups):
            others = self.get_other_groups()
            if others:
                menu.addSeparator()
                for g in others:
                    act = menu.addAction(f"Move to Pane {g.group_id}")
                    moves.append((act, g.group_id))
        menu.addSeparator()
        a_cl = menu.addAction("Close")
        chosen = menu.exec(event.globalPos())
        if chosen == a_sr:
            self.split_right_requested.emit(i)
        elif chosen == a_cl:
            self.tab_closed.emit(i)
        else:
            for act, gid in moves:
                if chosen == act:
                    self.move_to_group_requested.emit(i, gid)

    def remove_at(self, index):
        if not (0 <= index < len(self._entries)):
            return
        e = self._entries.pop(index)
        self._tlayout.removeWidget(e["frame"])
        e["frame"].deleteLater()
        if self._active >= len(self._entries):
            self._active = len(self._entries) - 1
        if self._active >= 0:
            self._set_active(self._active)

    def _set_active(self, index):
        self._active = index
        for i, e in enumerate(self._entries):
            e["frame"].setProperty("tabActive", i == index)
            e["frame"].style().unpolish(e["frame"])
            e["frame"].style().polish(e["frame"])

    def active_index(self):
        return self._active

    def active_key(self):
        if 0 <= self._active < len(self._entries):
            return self._entries[self._active]["key"]
        return None

    def count(self):
        return len(self._entries)

    def key_at(self, index):
        if 0 <= index < len(self._entries):
            return self._entries[index]["key"]
        return None

    def all_keys(self):
        return [e["key"] for e in self._entries]



# ── RunTabPage ───────────────────────────────────────────────────────────────

class RunTabPage(QWidget):
    compare_changed = Signal(set)

    def __init__(self, run_key, all_runs, dark=True, parent=None):
        super().__init__(parent)
        self.run_key = run_key
        self.all_runs = all_runs
        self.active_run = run_key
        self.detail = None
        self.series = None
        self.selected_columns = {"temp", "temp_ref"}
        self.compare_runs = {run_key}
        self.dark = dark
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        scroll = QScrollArea()
        scroll.setObjectName("workspaceScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget()
        body.setObjectName("workspaceBody")
        self._body_layout = QVBoxLayout(body)
        self._body_layout.setContentsMargins(14, 14, 14, 14)
        self._body_layout.setSpacing(10)
        scroll.setWidget(body)
        root.addWidget(scroll, 1)

        # ── Chart card ──────────────────────────────────────────────────────
        chart_card = QFrame()
        chart_card.setObjectName("card")
        chart_card_lay = QVBoxLayout(chart_card)
        chart_card_lay.setContentsMargins(10, 10, 10, 10)
        chart_card_lay.setSpacing(8)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(8)

        self.title_label = QLabel("Loading…")
        self.title_label.setObjectName("title")
        toolbar.addWidget(self.title_label)

        self.run_metric_labels = {}
        for lbl, key in [("Samples", "samples"), ("Duration", "duration"), ("Tail MAE", "tail_mae")]:
            box = self._inline_stat(lbl, key)
            toolbar.addWidget(box)

        toolbar.addStretch(1)

        self.chart_mode = QComboBox()
        self.chart_mode.addItems(["line", "scatter"])
        self.chart_mode.currentTextChanged.connect(self.refresh_chart)

        self.show_setpoint = QCheckBox("Setpoint")
        self.show_setpoint.setChecked(True)
        self.show_setpoint.stateChanged.connect(self.refresh_chart)

        self.setpoint_value = QLineEdit("0")
        self.setpoint_value.setFixedWidth(64)
        self.setpoint_value.textChanged.connect(self.refresh_chart)

        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self._reset_views)

        download = QPushButton("Export")
        download.setObjectName("primaryButton")
        dl_menu = QMenu(download)
        for label, slot in [("Chart PNG", self.export_chart_png),
                             ("Run CSV", self.export_run_csv),
                             ("Comparison CSV", self.export_comparison_csv),
                             ("Run Report", self.open_report)]:
            act = QAction(label, dl_menu)
            act.triggered.connect(slot)
            dl_menu.addAction(act)
        download.setMenu(dl_menu)

        for w in [self.show_setpoint, self.setpoint_value, reset_btn, self.chart_mode, download]:
            toolbar.addWidget(w)

        chart_card_lay.addLayout(toolbar)

        self.chart = ChartWidget()
        chart_card_lay.addWidget(self.chart)
        self._body_layout.addWidget(chart_card)

        # ── Controls card ───────────────────────────────────────────────────
        ctrl_card = QFrame()
        ctrl_card.setObjectName("card")
        ctrl_lay = QVBoxLayout(ctrl_card)
        ctrl_lay.setContentsMargins(10, 10, 10, 10)
        ctrl_lay.setSpacing(8)

        lbl = QLabel("PLOT CONTROLS")
        lbl.setObjectName("sectionLabel")
        ctrl_lay.addWidget(lbl)

        time_row = QHBoxLayout()
        time_row.setSpacing(6)
        self.time_start = QDoubleSpinBox()
        self.time_start.setRange(0, 1_000_000)
        self.time_start.setDecimals(1)
        self.time_start.setSuffix(" s")
        self.time_end = QDoubleSpinBox()
        self.time_end.setRange(0, 1_000_000)
        self.time_end.setDecimals(1)
        self.time_end.setSuffix(" s")
        apply_time = QPushButton("Apply Range")
        apply_time.clicked.connect(self._apply_time_range)
        reset_time = QPushButton("Full Range")
        reset_time.clicked.connect(self._reset_time_range)
        self.smoothing = QSpinBox()
        self.smoothing.setRange(1, 501)
        self.smoothing.setSingleStep(2)
        self.smoothing.setValue(1)
        self.smoothing.setSuffix(" pt")
        self.smoothing.valueChanged.connect(self._set_smoothing)
        for w in [QLabel("Start"), self.time_start, QLabel("End"), self.time_end,
                  apply_time, reset_time, QLabel("Smooth"), self.smoothing]:
            time_row.addWidget(w)
        time_row.addStretch(1)
        ctrl_lay.addLayout(time_row)

        ov_row = QHBoxLayout()
        ov_row.setSpacing(12)
        ov_row.addWidget(QLabel("Overlays"))
        self.overlay_min = QCheckBox("Min")
        self.overlay_max = QCheckBox("Max")
        self.overlay_avg = QCheckBox("Average")
        for cb in [self.overlay_min, self.overlay_max, self.overlay_avg]:
            cb.stateChanged.connect(self._update_overlays)
            ov_row.addWidget(cb)
        ov_row.addSpacing(20)
        ov_row.addWidget(QLabel("Markers"))
        self.mk_events = QCheckBox("Events")
        self.mk_alarms = QCheckBox("Alarms")
        self.mk_ctrl = QCheckBox("Controller")
        self.mk_state = QCheckBox("State")
        for cb in [self.mk_events, self.mk_alarms, self.mk_ctrl, self.mk_state]:
            cb.setChecked(True)
            cb.stateChanged.connect(self._update_markers)
            ov_row.addWidget(cb)
        ov_row.addStretch(1)
        ctrl_lay.addLayout(ov_row)
        self._body_layout.addWidget(ctrl_card)

        # ── Lower splitter (channels + details) ─────────────────────────────
        lower = QSplitter(Qt.Horizontal)
        lower.setObjectName("contentSplitter")
        lower.setHandleWidth(3)
        lower.addWidget(self._channels_panel())
        lower.addWidget(self._details_panel())
        lower.setSizes([340, 800])
        self._body_layout.addWidget(lower)

        # ── Raw data ────────────────────────────────────────────────────────
        raw_card = QFrame()
        raw_card.setObjectName("card")
        raw_lay = QVBoxLayout(raw_card)
        raw_lay.setContentsMargins(10, 10, 10, 10)
        raw_lay.setSpacing(6)
        raw_lbl = QLabel("RAW DATA")
        raw_lbl.setObjectName("sectionLabel")
        raw_lay.addWidget(raw_lbl)
        self.sample_table = QTableWidget()
        self.sample_table.setMinimumHeight(350)
        raw_lay.addWidget(self.sample_table)
        self._body_layout.addWidget(raw_card)

        self.chart.set_dark(self.dark)

    def _inline_stat(self, label, key):
        box = QFrame()
        box.setObjectName("inlineStat")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(10, 4, 10, 4)
        lay.setSpacing(0)
        cap = QLabel(label)
        cap.setObjectName("inlineStatLabel")
        val = QLabel("-")
        val.setObjectName("inlineStatValue")
        lay.addWidget(cap)
        lay.addWidget(val)
        self.run_metric_labels[key] = val
        return box

    def _channels_panel(self):
        card = QFrame()
        card.setObjectName("card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(6)
        top = QHBoxLayout()
        top.addWidget(self._sec_lbl("CHANNELS"))
        top.addStretch(1)
        self.toggle_ch_btn = QPushButton("Select All")
        self.toggle_ch_btn.setObjectName("secondaryButton")
        self.toggle_ch_btn.clicked.connect(self._toggle_channels)
        top.addWidget(self.toggle_ch_btn)
        lay.addLayout(top)
        self.channel_list = QListWidget()
        self.channel_list.setObjectName("channelList")
        self.channel_list.setUniformItemSizes(True)
        self.channel_list.itemChanged.connect(self._channel_checked)
        lay.addWidget(self.channel_list)
        return card

    def _details_panel(self):
        card = QFrame()
        card.setObjectName("card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(6)
        lay.addWidget(self._sec_lbl("BAND METRICS"))
        self.band_table = QTableWidget()
        self.band_table.setMinimumHeight(200)
        lay.addWidget(self.band_table)
        lay.addWidget(self._sec_lbl("RUN SUMMARY"))
        self.comparison_table = QTableWidget()
        self.comparison_table.setMinimumHeight(160)
        lay.addWidget(self.comparison_table)
        return card

    def _sec_lbl(self, text):
        lbl = QLabel(text)
        lbl.setObjectName("sectionLabel")
        return lbl

    def _stat_card(self, label, value):
        box = QFrame()
        box.setObjectName("card")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(4)
        cap = QLabel(label)
        cap.setObjectName("sectionLabel")
        lay.addWidget(cap)
        val = QLabel(fmt(value))
        val.setStyleSheet("font-size: 22px; font-weight: 850; background: transparent;")
        lay.addWidget(val)
        return box

    # ── Data loading ─────────────────────────────────────────────────────────

    def load(self):
        try:
            self.detail = data.run_detail(self.run_key)
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            return
        if not any(c in self.detail["numeric_columns"] for c in self.selected_columns):
            preferred = [c for c in ["temp", "temp_ref"] if c in self.detail["numeric_columns"]]
            self.selected_columns = set(preferred or self.detail["numeric_columns"][:3])
        run = self.detail.get("run", {})
        self.title_label.setText(run.get("id", self.run_key))
        self._render_summary()
        self._render_channels()
        self._render_bands()
        self._render_comparison_table()
        self._render_table()
        self._load_series()

    def set_compare_run(self, key, checked):
        if checked:
            self.compare_runs.add(key)
        else:
            self.compare_runs.discard(key)
        if not self.compare_runs:
            self.compare_runs.add(self.run_key)
        self._render_comparison_table()
        self._load_series()
        self.compare_changed.emit(self.compare_runs)

    def update_theme(self, dark):
        self.dark = dark
        self.chart.set_dark(dark)

    def primary_chart(self):
        return self.chart

    def _compare_mode(self):
        return len(self.compare_runs) > 1

    def _first_col(self):
        return next(iter(self.selected_columns), None)

    def _active_charts(self):
        return [self.chart]

    def _render_summary(self):
        run = (self.detail or {}).get("run", {})
        summary = (self.detail or {}).get("summary", {})
        values = {
            "samples": run.get("samples"),
            "duration": f'{fmt(run.get("duration_s"), 1)} s' if run.get("duration_s") is not None else None,
            "tail_mae": summary.get("tail_mae"),
        }
        for key, lbl in self.run_metric_labels.items():
            lbl.setText(fmt(values.get(key)))

    def _render_channels(self):
        if not self.detail:
            return
        self.channel_list.blockSignals(True)
        self.channel_list.clear()
        for col in self.detail.get("numeric_columns", []):
            item = QListWidgetItem(col)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if col in self.selected_columns else Qt.Unchecked)
            self.channel_list.addItem(item)
        self.channel_list.blockSignals(False)
        total = len(self.detail.get("numeric_columns", []))
        self.toggle_ch_btn.setText("Clear All" if len(self.selected_columns) >= total else "Select All")

    def _channel_checked(self, item):
        if item.checkState() == Qt.Checked:
            self.selected_columns.add(item.text())
        else:
            self.selected_columns.discard(item.text())
        self._load_series()

    def _toggle_channels(self):
        cols = set(self.detail.get("numeric_columns", []))
        self.selected_columns = set() if len(self.selected_columns) >= len(cols) else cols
        self._render_channels()
        self._load_series()

    def _render_bands(self):
        if not self.detail:
            return
        cols = ["band", "kp", "ki", "kd", "cost", "n_samples", "far_mae", "mid_mae", "tail_mae", "overshoot"]
        self._fill_table(self.band_table, cols, self.detail.get("bands", []))

    def _render_comparison_table(self):
        rows = [r for r in self.all_runs if r["key"] in (self.compare_runs or {self.run_key})]
        cols = ["Run", "Cost", "MAE", "Tail MAE", "Overshoot"]
        mapped = [{"Run": r["id"], "Cost": r["cost"], "MAE": r["mae"],
                   "Tail MAE": r["tail_mae"], "Overshoot": r["overshoot"]} for r in rows]
        self._fill_table(self.comparison_table, cols, mapped)

    def _render_table(self):
        if self._compare_mode():
            self.sample_table.setRowCount(0)
            self.sample_table.setColumnCount(1)
            self.sample_table.setHorizontalHeaderLabels(["Raw data hidden during comparison"])
            return
        try:
            table = data.run_table(self.run_key)
            self._fill_table(self.sample_table, table["columns"], table["rows"], max_rows=5000)
        except Exception:
            pass

    def _fill_table(self, table, columns, rows, max_rows=None):
        shown = rows[:max_rows] if max_rows else rows
        table.setUpdatesEnabled(False)
        table.setAlternatingRowColors(True)
        table.setShowGrid(False)
        table.setWordWrap(False)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        table.setColumnCount(len(columns))
        table.setRowCount(len(shown))
        table.setHorizontalHeaderLabels(columns)
        for ri, row in enumerate(shown):
            for ci, col in enumerate(columns):
                table.setItem(ri, ci, QTableWidgetItem(fmt(row.get(col))))
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        table.setUpdatesEnabled(True)

    def _load_series(self):
        cols = list(self.selected_columns)
        if not cols:
            self.series = {"columns": [], "points": []}
            self.refresh_chart()
            return
        try:
            if self._compare_mode():
                ch = self._first_col()
                series = []
                for key in self.compare_runs:
                    payload = data.run_series(key, [ch])
                    run = next((r for r in self.all_runs if r["key"] == key), None)
                    series.append({"label": run["id"] if run else key, "points": payload["points"]})
                self.series = {"series": series, "channel": ch}
            else:
                self.series = data.run_series(self.run_key, cols)
        except Exception as exc:
            QMessageBox.critical(self, "Series error", str(exc))
            return
        self.refresh_chart()

    def refresh_chart(self):
        setpoint = None
        if self._compare_mode() and self.show_setpoint.isChecked():
            try:
                setpoint = float(self.setpoint_value.text())
            except ValueError:
                pass
        annotations = [] if self._compare_mode() else (self.detail or {}).get("annotations", [])
        for ch in self._active_charts():
            ch.draw(self.series, self.chart_mode.currentText(), annotations, setpoint)
        self._update_time_controls()

    def _update_time_controls(self):
        start, end = self.chart.data_x_range()
        if end <= start:
            return
        for sb in [self.time_start, self.time_end]:
            sb.blockSignals(True)
            sb.setRange(start, end)
        self.time_start.setValue(start)
        self.time_end.setValue(end)
        for sb in [self.time_start, self.time_end]:
            sb.blockSignals(False)

    def _apply_time_range(self):
        for ch in self._active_charts():
            ch.set_time_range(self.time_start.value(), self.time_end.value())

    def _reset_time_range(self):
        for ch in self._active_charts():
            ch.set_time_range(None, None)
        self._update_time_controls()

    def _set_smoothing(self, value):
        if value > 1 and value % 2 == 0:
            value += 1
            self.smoothing.blockSignals(True)
            self.smoothing.setValue(value)
            self.smoothing.blockSignals(False)
        for ch in self._active_charts():
            ch.set_smoothing_window(value)

    def _update_overlays(self):
        for ch in self._active_charts():
            ch.set_overlay_flags(min=self.overlay_min.isChecked(),
                                 max=self.overlay_max.isChecked(),
                                 avg=self.overlay_avg.isChecked())

    def _update_markers(self):
        for ch in self._active_charts():
            ch.set_marker_flags(events=self.mk_events.isChecked(),
                                alarms=self.mk_alarms.isChecked(),
                                controller=self.mk_ctrl.isChecked(),
                                state=self.mk_state.isChecked())

    def _reset_views(self):
        for ch in self._active_charts():
            ch.reset_view()

    # ── Exports ──────────────────────────────────────────────────────────────

    def export_chart_png(self):
        if not self.detail:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save chart",
                                              f"{self.detail['run']['id']}.png", "PNG (*.png)")
        if path:
            self.chart.export_view(path)

    def export_run_csv(self):
        if not self.detail:
            return
        table = data.run_table(self.run_key)
        path, _ = QFileDialog.getSaveFileName(self, "Save run CSV",
                                              f"{self.detail['run']['id']}-samples.csv", "CSV (*.csv)")
        if path:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=table["columns"])
                writer.writeheader()
                writer.writerows(table["rows"])

    def export_comparison_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save comparison CSV",
                                              "deepvac-comparison.csv", "CSV (*.csv)")
        if not path:
            return
        rows = [r for r in self.all_runs if r["key"] in (self.compare_runs or {self.run_key})]
        lines = ["Run,Cost,MAE,Tail MAE,Overshoot"]
        for r in rows:
            lines.append(",".join(csv_escape(r[k]) for k in ["id", "cost", "mae", "tail_mae", "overshoot"]))
        Path(path).write_text("\n".join(lines), encoding="utf-8")

    def open_report(self):
        if not self.detail:
            return
        reports = HERE / "reports"
        reports.mkdir(exist_ok=True)
        path = reports / f"{self.detail['run']['id']}.html"
        path.write_text(data.make_report_html(self.run_key), encoding="utf-8")
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


# ── EditorGroup ──────────────────────────────────────────────────────────────

class EditorGroup(QWidget):
    active_changed = Signal(object)       # RunTabPage or None
    split_requested = Signal(object, str, str)  # self, direction, key
    group_empty = Signal(object)          # self
    tab_received = Signal(object, str, str)  # self, key, id (from drop)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.group_id = _new_gid()
        self._pages = {}   # key -> RunTabPage
        self.setAcceptDrops(True)
        self.setMinimumSize(300, 300)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Drop indicator strip at top
        self._drop_bar = QFrame()
        self._drop_bar.setObjectName("dropIndicator")
        self._drop_bar.setFixedHeight(3)
        self._drop_bar.setVisible(False)
        lay.addWidget(self._drop_bar)

        self.tab_bar = TabBar()
        self.tab_bar.get_other_groups = None  # set by EditorArea
        self.tab_bar.tab_activated.connect(self._on_activated)
        self.tab_bar.tab_closed.connect(self._on_closed)
        self.tab_bar.split_right_requested.connect(lambda i: self._on_split("right", i))
        self.tab_bar.move_to_group_requested.connect(self._on_move_to_group)
        lay.addWidget(self.tab_bar)

        self.stack = QStackedWidget()
        lay.addWidget(self.stack, 1)

    def add_page(self, key, run_id, page):
        if key not in self._pages:
            self._pages[key] = page
            self.stack.addWidget(page)
        self.tab_bar.add_or_focus(key, run_id)
        self.stack.setCurrentWidget(self._pages[key])
        self.active_changed.emit(self._pages[key])

    def remove_page(self, key):
        if key not in self._pages:
            return
        page = self._pages.pop(key)
        self.stack.removeWidget(page)
        idx = self.tab_bar._idx(key)
        if idx >= 0:
            self.tab_bar.remove_at(idx)
        if self.tab_bar.count() == 0:
            self.active_changed.emit(None)
            self.group_empty.emit(self)
        else:
            active = self.tab_bar.active_key()
            if active and active in self._pages:
                self.stack.setCurrentWidget(self._pages[active])
                self.active_changed.emit(self._pages[active])

    def take_page(self, key):
        """Remove and return (page, run_id) without deleting."""
        if key not in self._pages:
            return None, None
        page = self._pages.pop(key)
        self.stack.removeWidget(page)
        e = next((e for e in self.tab_bar._entries if e["key"] == key), None)
        run_id = e["id"] if e else key
        idx = self.tab_bar._idx(key)
        if idx >= 0:
            self.tab_bar.remove_at(idx)
        if self.tab_bar.count() == 0:
            self.active_changed.emit(None)
            self.group_empty.emit(self)
        else:
            active = self.tab_bar.active_key()
            if active and active in self._pages:
                self.stack.setCurrentWidget(self._pages[active])
                self.active_changed.emit(self._pages[active])
        return page, run_id

    def active_page(self):
        key = self.tab_bar.active_key()
        return self._pages.get(key)

    def has_key(self, key):
        return key in self._pages

    def all_keys(self):
        return list(self._pages.keys())

    def _on_activated(self, index):
        key = self.tab_bar.key_at(index)
        if key and key in self._pages:
            self.stack.setCurrentWidget(self._pages[key])
            self.active_changed.emit(self._pages[key])

    def _on_closed(self, index):
        key = self.tab_bar.key_at(index)
        if key:
            self.remove_page(key)

    def _on_split(self, direction, index):
        key = self.tab_bar.key_at(index)
        if key:
            self.split_requested.emit(self, direction, key)

    def _on_move_to_group(self, tab_index, target_gid):
        key = self.tab_bar.key_at(tab_index)
        if key:
            self.split_requested.emit(self, f"move:{target_gid}", key)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(_TAB_MIME):
            self._drop_bar.setVisible(True)
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self._drop_bar.setVisible(False)

    def dropEvent(self, event):
        self._drop_bar.setVisible(False)
        if not event.mimeData().hasFormat(_TAB_MIME):
            return
        raw = bytes(event.mimeData().data(_TAB_MIME))
        info = json.loads(raw.decode())
        self.tab_received.emit(self, info["key"], info["id"])
        event.acceptProposedAction()


# ── EditorArea ───────────────────────────────────────────────────────────────

class EditorArea(QWidget):
    active_page_changed = Signal(object)  # RunTabPage or None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("editorArea")
        self._groups = []
        self._pages = {}   # key -> (RunTabPage, EditorGroup)
        self._active_group = None
        self._sync_hub = CrosshairSyncHub()

        self._root_splitter = QSplitter(Qt.Horizontal)
        self._root_splitter.setHandleWidth(4)
        self._root_splitter.setObjectName("editorSplitter")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._root_splitter, 1)

        first = self._make_group()
        self._root_splitter.addWidget(first)

        # Floating overlay button — sits on top of the tab-bar row, far right,
        # without claiming its own layout column.
        self._sync_btn = QPushButton("⇔", self)
        self._sync_btn.setObjectName("editorSyncButton")
        self._sync_btn.setCheckable(True)
        self._sync_btn.setFixedSize(28, 28)
        self._sync_btn.setToolTip("Sync crosshair across all plots")
        self._sync_btn.toggled.connect(self._on_sync_toggled)
        self._sync_btn.raise_()
        self._position_sync_btn()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_sync_btn()

    def _position_sync_btn(self):
        if hasattr(self, "_sync_btn"):
            self._sync_btn.move(self.width() - self._sync_btn.width() - 7, 4)
            self._sync_btn.raise_()

    def _make_group(self):
        g = EditorGroup()
        g.active_changed.connect(self._on_group_active_changed)
        g.split_requested.connect(self._on_split_requested)
        g.group_empty.connect(self._on_group_empty)
        g.tab_received.connect(self._on_tab_received)
        g.tab_bar.get_other_groups = lambda: [x for x in self._groups if x is not g]
        self._groups.append(g)
        self._active_group = g
        return g

    def register_chart(self, chart):
        self._sync_hub.register(chart)

    def open_run(self, key, run_id, page):
        # If already open somewhere, focus it
        if key in self._pages:
            _, grp = self._pages[key]
            grp.tab_bar.add_or_focus(key, run_id)
            grp.stack.setCurrentWidget(self._pages[key][0])
            self._active_group = grp
            self.active_page_changed.emit(page)
            return
        grp = self._groups[0]
        self._pages[key] = (page, grp)
        grp.add_page(key, run_id, page)
        self._active_group = grp

    def active_page(self):
        if self._active_group:
            return self._active_group.active_page()
        return None

    def all_groups(self):
        return list(self._groups)

    def update_theme(self, dark):
        for page, _ in self._pages.values():
            page.update_theme(dark)

    def _on_group_active_changed(self, page):
        sender_group = self.sender()
        if sender_group:
            self._active_group = sender_group
        self.active_page_changed.emit(page)

    def _on_split_requested(self, source_group, direction, key):
        if key not in self._pages:
            return
        page, _ = self._pages[key]

        if direction.startswith("move:"):
            target_gid = int(direction.split(":")[1])
            target = next((g for g in self._groups if g.group_id == target_gid), None)
            if target and target is not source_group:
                run_id = next((e["id"] for e in source_group.tab_bar._entries if e["key"] == key), key)
                taken_page, taken_id = source_group.take_page(key)
                if taken_page:
                    self._pages[key] = (taken_page, target)
                    target.add_page(key, taken_id, taken_page)
                    self._active_group = target
            return

        run_id = next((e["id"] for e in source_group.tab_bar._entries if e["key"] == key), key)
        taken_page, taken_id = source_group.take_page(key)
        if not taken_page:
            return

        new_group = self._make_group()

        if direction == "right":
            idx = self._root_splitter.indexOf(source_group)
            if idx >= 0:
                self._root_splitter.insertWidget(idx + 1, new_group)
            else:
                self._root_splitter.addWidget(new_group)
            sizes = self._root_splitter.sizes()
            if sizes:
                total = sum(sizes)
                equal = total // len(sizes)
                self._root_splitter.setSizes([equal] * len(sizes))
        elif direction == "down":
            idx = self._root_splitter.indexOf(source_group)
            vert = QSplitter(Qt.Vertical)
            vert.setHandleWidth(4)
            self._root_splitter.replaceWidget(idx, vert)
            vert.addWidget(source_group)
            vert.addWidget(new_group)
            vert.setSizes([400, 400])
        else:
            self._root_splitter.addWidget(new_group)

        self._pages[key] = (taken_page, new_group)
        new_group.add_page(key, taken_id, taken_page)
        self._active_group = new_group

    def _on_group_empty(self, group):
        if len(self._groups) <= 1:
            return
        self._groups.remove(group)
        parent = group.parent()
        if isinstance(parent, QSplitter):
            if parent.count() <= 1:
                # Lift sibling up
                remaining = [parent.widget(i) for i in range(parent.count()) if parent.widget(i) is not group]
                grandparent = parent.parent()
                if isinstance(grandparent, QSplitter) and remaining:
                    idx = grandparent.indexOf(parent)
                    grandparent.replaceWidget(idx, remaining[0])
            group.setParent(None)
        group.deleteLater()
        if self._active_group is group:
            self._active_group = self._groups[-1] if self._groups else None
        self.active_page_changed.emit(self.active_page())

    def _on_tab_received(self, target_group, key, run_id):
        if key not in self._pages:
            return
        page, source_group = self._pages[key]
        if source_group is target_group:
            return
        taken_page, taken_id = source_group.take_page(key)
        if not taken_page:
            return
        self._pages[key] = (taken_page, target_group)
        target_group.add_page(key, taken_id, taken_page)
        self._active_group = target_group

    def _on_sync_toggled(self, enabled):
        self._sync_hub.set_enabled(enabled)


# ── TitleBar ─────────────────────────────────────────────────────────────────

class TitleBar(QWidget):
    def __init__(self, main_window):
        super().__init__(main_window)
        self._win = main_window
        self.setObjectName("titleBar")
        self.setFixedHeight(40)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Logo — unconstrained width so full logo text is readable
        logo_w = QWidget()
        logo_w.setObjectName("titleBarLogoArea")
        logo_w.setFixedHeight(40)
        ll = QHBoxLayout(logo_w)
        ll.setContentsMargins(12, 4, 16, 4)
        lbl = QLabel()
        pix = QPixmap(LOGO_PATH)
        if not pix.isNull():
            lbl.setPixmap(pix.scaledToHeight(30, Qt.SmoothTransformation))
        else:
            lbl.setText("DEEPVAC")
            lbl.setObjectName("titleBarBrand")
        lbl.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        ll.addWidget(lbl)
        lay.addWidget(logo_w)

        lay.addStretch(1)

        # Center quick-access bar
        self.center_btn = QPushButton("⌕   DeepVac Dashboard")
        self.center_btn.setObjectName("titleCenter")
        self.center_btn.setFixedSize(320, 26)
        lay.addWidget(self.center_btn)

        lay.addStretch(1)

        # ── Status indicators ────────────────────────────────────────────────
        self.status_system = self._status_pill("●", "System Offline", "sysDotErr", "statusText")
        lay.addWidget(self.status_system)
        lay.addSpacing(4)

        self.status_chamber = self._status_pill("⊟", "Chamber Offline", "chamberIconOff", "statusText")
        lay.addWidget(self.status_chamber)
        lay.addSpacing(4)

        self.btn_bell = QPushButton("🔔")
        self.btn_bell.setObjectName("titleIconBtn")
        self.btn_bell.setFixedSize(32, 40)
        self.btn_bell.setFocusPolicy(Qt.NoFocus)
        self.btn_bell.setToolTip("Notifications")
        lay.addWidget(self.btn_bell)

        lay.addSpacing(4)

        # Window control buttons
        self.btn_min = QPushButton("─")
        self.btn_max = QPushButton("□")
        self.btn_close = QPushButton("✕")
        self.btn_min.setObjectName("winBtn")
        self.btn_max.setObjectName("winBtn")
        self.btn_close.setObjectName("winBtnClose")
        for btn in [self.btn_min, self.btn_max, self.btn_close]:
            btn.setFixedSize(46, 40)
            btn.setFocusPolicy(Qt.NoFocus)
            lay.addWidget(btn)

        self.btn_min.clicked.connect(main_window.showMinimized)
        self.btn_max.clicked.connect(self._toggle_max)
        self.btn_close.clicked.connect(main_window.close)

    def _status_pill(self, icon_text, label_text, icon_obj, label_obj):
        pill = QFrame()
        pill.setObjectName("titleStatusPill")
        pill.setFixedHeight(28)
        pl = QHBoxLayout(pill)
        pl.setContentsMargins(8, 0, 8, 0)
        pl.setSpacing(5)
        icon = QLabel(icon_text)
        icon.setObjectName(icon_obj)
        txt = QLabel(label_text)
        txt.setObjectName(label_obj)
        pl.addWidget(icon)
        pl.addWidget(txt)
        pill._icon = icon
        pill._txt = txt
        return pill

    def set_system_status(self, ok, message=None):
        self.status_system._icon.setObjectName("sysDotOk" if ok else "sysDotErr")
        self.status_system._txt.setText(message or ("System OK" if ok else "System Error"))
        self.status_system._icon.style().unpolish(self.status_system._icon)
        self.status_system._icon.style().polish(self.status_system._icon)

    def set_chamber_status(self, online):
        self.status_chamber._icon.setObjectName("chamberIconOn" if online else "chamberIconOff")
        self.status_chamber._txt.setText("Chamber Online" if online else "Chamber Offline")
        self.status_chamber._icon.style().unpolish(self.status_chamber._icon)
        self.status_chamber._icon.style().polish(self.status_chamber._icon)

    def _toggle_max(self):
        if self._win.isMaximized():
            self._win.showNormal()
            self.btn_max.setText("□")
        else:
            self._win.showMaximized()
            self.btn_max.setText("❐")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and not self._win.isMaximized():
            handle = self._win.windowHandle()
            if handle:
                handle.startSystemMove()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._toggle_max()


# ── DeepVacDesktop ───────────────────────────────────────────────────────────

class DeepVacDesktop(QMainWindow):
    def __init__(self, splash=None):
        super().__init__()
        self.splash = splash
        self.setWindowTitle("DeepVac Dashboard")
        icon = QIcon(ICON_PATH)
        self.setWindowIcon(icon)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.resize(1500, 920)
        self.setMinimumSize(1100, 700)

        self.runs = []
        self.dark = True
        self.sim_worker = None
        self.sim_series = None

        self._build_ui()
        self.apply_theme()
        self.load_runs()

    def _build_ui(self):
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.title_bar = TitleBar(self)
        outer.addWidget(self.title_bar)

        body = QWidget()
        body_lay = QHBoxLayout(body)
        body_lay.setContentsMargins(0, 0, 0, 0)
        body_lay.setSpacing(0)
        body_lay.addWidget(self._sidebar())

        self.content_stack = QStackedWidget()
        self.dashboard_widget = self._dashboard_view()   # 0
        self.runs_widget = self._runs_view()              # 1
        self.editor_area = EditorArea()                   # 2
        self.editor_area.active_page_changed.connect(self._on_active_page_changed)
        self.sim_widget = self._sim_view()                # 3
        self.reports_widget = self._reports_view()        # 4
        for w in [self.dashboard_widget, self.runs_widget,
                  self.editor_area, self.sim_widget, self.reports_widget]:
            self.content_stack.addWidget(w)

        body_lay.addWidget(self.content_stack, 1)
        outer.addWidget(body, 1)

        grip = QSizeGrip(central)
        grip.setFixedSize(14, 14)
        grip_lay = QHBoxLayout()
        grip_lay.setContentsMargins(0, 0, 0, 0)
        grip_lay.addStretch(1)
        grip_lay.addWidget(grip)
        outer.addLayout(grip_lay)

        self.setCentralWidget(central)
        self._nav_to(0)

    def _sidebar(self):
        bar = QFrame()
        bar.setObjectName("activityBar")
        bar.setFixedWidth(48)
        lay = QVBoxLayout(bar)
        lay.setContentsMargins(0, 8, 0, 8)
        lay.setSpacing(2)

        nav_defs = [
            ("act_dashboard_btn", "⌂", "Dashboard",  0),
            ("act_runs_btn",      "≡", "Runs",        1),
            ("act_analysis_btn",  "∿", "Analysis",    2),
            ("act_sim_btn",       "⚗", "Simulator",   3),
            ("act_reports_btn",   "⊡", "Reports",     4),
        ]
        self._nav_buttons = []
        for attr, icon, label, idx in nav_defs:
            btn = QPushButton(icon)
            btn.setObjectName("navButton")
            btn.setCheckable(True)
            btn.setFixedSize(48, 46)
            btn.setToolTip(label)
            btn.clicked.connect(lambda _checked, i=idx: self._nav_to(i))
            setattr(self, attr, btn)
            self._nav_buttons.append(btn)
            lay.addWidget(btn)

        lay.addStretch(1)

        self.act_account_btn = QPushButton("◎")
        self.act_account_btn.setObjectName("navButton")
        self.act_account_btn.setFixedSize(48, 46)
        self.act_account_btn.setToolTip("Account")
        lay.addWidget(self.act_account_btn)

        self.act_settings_btn = QPushButton("⚙")
        self.act_settings_btn.setObjectName("navButton")
        self.act_settings_btn.setFixedSize(48, 46)
        self.act_settings_btn.setToolTip("Settings")
        self.act_settings_btn.clicked.connect(self._show_settings)
        lay.addWidget(self.act_settings_btn)

        return bar

    def _nav_to(self, index):
        for i, btn in enumerate(self._nav_buttons):
            btn.setChecked(i == index)
        self.content_stack.setCurrentIndex(index)
        if index == 0:
            self._refresh_dashboard()
        elif index == 4:
            self._refresh_reports()

    def _runs_view(self):
        container = QWidget()
        container.setObjectName("workspaceBody")
        lay = QHBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        left = QFrame()
        left.setObjectName("runsPanel")
        left.setFixedWidth(280)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(0)

        header = QFrame()
        header.setObjectName("runsPanelHeader")
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(12, 10, 8, 10)
        h_lay.setSpacing(0)
        lbl = QLabel("RUNS")
        lbl.setObjectName("sidebarPanelLabel")
        h_lay.addWidget(lbl, 1)
        left_lay.addWidget(header)

        self.search_box = QLineEdit()
        self.search_box.setObjectName("searchBox")
        self.search_box.setPlaceholderText("Search run id…")
        self.search_box.textChanged.connect(self.render_runs)
        left_lay.addWidget(self.search_box)

        self.run_list = QListWidget()
        self.run_list.setObjectName("runList")
        self.run_list.setUniformItemSizes(True)
        self.run_list.itemChanged.connect(self._run_checked)
        self.run_list.itemDoubleClicked.connect(self._open_run_item)
        self.run_list.currentItemChanged.connect(self._on_run_selected)
        self.run_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.run_list.customContextMenuRequested.connect(self._run_list_context_menu)
        left_lay.addWidget(self.run_list, 1)
        lay.addWidget(left)

        right = QFrame()
        right.setObjectName("card")
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(16, 16, 16, 16)
        right_lay.setSpacing(8)
        self._raw_run_label = QLabel("Select a run to view raw data")
        self._raw_run_label.setObjectName("title")
        right_lay.addWidget(self._raw_run_label)
        self._raw_run_table = QTableWidget()
        self._raw_run_table.setMinimumHeight(400)
        right_lay.addWidget(self._raw_run_table)
        lay.addWidget(right, 1)

        return container

    def _dash_chart_card(self, title):
        card = QFrame()
        card.setObjectName("card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 12, 12, 10)
        lay.setSpacing(6)
        lbl = QLabel(title)
        lbl.setObjectName("sectionLabel")
        lay.addWidget(lbl)
        plot = pg.PlotWidget()
        plot.setBackground("#111827" if self.dark else "#f8fafc")
        plot.showGrid(x=True, y=True, alpha=0.22)
        plot.setMinimumHeight(210)
        plot.setMenuEnabled(False)
        lay.addWidget(plot)
        return card, plot

    def _dashboard_view(self):
        scroll = QScrollArea()
        scroll.setObjectName("workspaceScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget()
        body.setObjectName("workspaceBody")
        self._dash_body = QVBoxLayout(body)
        self._dash_body.setContentsMargins(24, 24, 24, 24)
        self._dash_body.setSpacing(16)
        scroll.setWidget(body)

        hdr = QLabel("Dashboard")
        hdr.setObjectName("pageTitle")
        self._dash_body.addWidget(hdr)

        self._dash_stats_row = QHBoxLayout()
        self._dash_stats_row.setSpacing(12)
        self._dash_body.addLayout(self._dash_stats_row)

        charts_row = QHBoxLayout()
        charts_row.setSpacing(12)
        card1, self._dash_cost_plot = self._dash_chart_card("COST OVER RUNS")
        card2, self._dash_mae_plot  = self._dash_chart_card("TAIL MAE DISTRIBUTION")
        card3, self._dash_ovr_plot  = self._dash_chart_card("OVERSHOOT vs COST")
        for c in [card1, card2, card3]:
            charts_row.addWidget(c, 1)
        self._dash_body.addLayout(charts_row)
        self._dash_body.addStretch(1)

        return scroll

    def _refresh_dashboard(self):
        while self._dash_stats_row.count():
            item = self._dash_stats_row.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        if not self.runs:
            return

        costs = [r["cost"] for r in self.runs if r.get("cost") is not None]
        maes  = [r["tail_mae"] for r in self.runs if r.get("tail_mae") is not None]
        best  = min(self.runs, key=lambda r: (r.get("cost") or float("inf")))

        for lbl, val in [
            ("Total Runs",   str(len(self.runs))),
            ("Best Cost",    fmt(min(costs)) if costs else "-"),
            ("Avg Tail MAE", fmt(sum(maes) / len(maes)) if maes else "-"),
            ("Best Run",     best["id"]),
        ]:
            box = QFrame()
            box.setObjectName("card")
            bl = QVBoxLayout(box)
            bl.setContentsMargins(16, 12, 16, 12)
            bl.setSpacing(4)
            cap = QLabel(lbl)
            cap.setObjectName("sectionLabel")
            bl.addWidget(cap)
            num = QLabel(val)
            num.setStyleSheet("font-size: 20px; font-weight: 800; background: transparent;")
            bl.addWidget(num)
            self._dash_stats_row.addWidget(box)
        self._dash_stats_row.addStretch(1)

        self._draw_dashboard_charts()

    def _draw_dashboard_charts(self):
        self._dash_cost_plot.clear()
        self._dash_mae_plot.clear()
        self._dash_ovr_plot.clear()
        if not self.runs:
            return

        ordered = list(reversed(self.runs))
        xs = list(range(len(ordered)))
        costs = [(x, r["cost"]) for x, r in zip(xs, ordered) if r.get("cost") is not None]
        if costs:
            vx, vy = zip(*costs)
            self._dash_cost_plot.plot(list(vx), list(vy),
                                      pen=pg.mkPen("#60a5fa", width=1.8),
                                      symbol="o", symbolSize=5,
                                      symbolBrush="#60a5fa", symbolPen=None)
        self._dash_cost_plot.setLabel("bottom", "Run (oldest → newest)")
        self._dash_cost_plot.setLabel("left", "Cost")

        maes = [r["tail_mae"] for r in self.runs if r.get("tail_mae") is not None]
        if maes:
            bins = min(25, max(5, len(maes) // 4))
            y, x = np.histogram(maes, bins=bins)
            w = (x[1] - x[0]) * 0.85 if len(x) > 1 else 1.0
            bar = pg.BarGraphItem(x=x[:-1], height=y, width=w, brush="#51d6c7", pen=pg.mkPen("#111827", width=0.5))
            self._dash_mae_plot.addItem(bar)
        self._dash_mae_plot.setLabel("bottom", "Tail MAE")
        self._dash_mae_plot.setLabel("left", "Count")

        pairs = [(r["cost"], r["overshoot"]) for r in self.runs
                 if r.get("cost") is not None and r.get("overshoot") is not None]
        if pairs:
            cx, cy = zip(*pairs)
            scatter = pg.ScatterPlotItem(list(cx), list(cy), size=7,
                                         brush=pg.mkBrush("#f2bd52"),
                                         pen=pg.mkPen("#111827", width=0.5))
            self._dash_ovr_plot.addItem(scatter)
        self._dash_ovr_plot.setLabel("bottom", "Cost")
        self._dash_ovr_plot.setLabel("left", "Overshoot")

    def _fill_generic_table(self, table, columns, rows, max_rows=None):
        shown = rows[:max_rows] if max_rows else rows
        table.setUpdatesEnabled(False)
        table.setAlternatingRowColors(True)
        table.setShowGrid(False)
        table.setWordWrap(False)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        table.setColumnCount(len(columns))
        table.setRowCount(len(shown))
        table.setHorizontalHeaderLabels(columns)
        for ri, row in enumerate(shown):
            for ci, col in enumerate(columns):
                table.setItem(ri, ci, QTableWidgetItem(fmt(row.get(col))))
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        table.setUpdatesEnabled(True)

    def _on_run_selected(self, item, _prev):
        if not item:
            return
        key = item.data(Qt.UserRole)
        run = next((r for r in self.runs if r["key"] == key), None)
        if not run:
            return
        self._raw_run_label.setText(run["id"])
        try:
            table = data.run_table(key)
            self._fill_generic_table(self._raw_run_table, table["columns"], table["rows"], max_rows=2000)
        except Exception:
            pass

    def _reports_view(self):
        self._report_rows = []
        self._report_selected_key = None

        container = QWidget()
        container.setObjectName("workspaceBody")
        outer = QVBoxLayout(container)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.setSpacing(14)

        hdr = QLabel("Reports")
        hdr.setObjectName("pageTitle")
        outer.addWidget(hdr)

        sub = QLabel("Generate, preview, and export validation reports from completed thermal runs.")
        sub.setObjectName("sectionLabel")
        outer.addWidget(sub)

        self._report_stats_row = QHBoxLayout()
        self._report_stats_row.setSpacing(12)
        outer.addLayout(self._report_stats_row)

        filter_row = QHBoxLayout()
        filter_row.setSpacing(8)
        self._report_search = QLineEdit()
        self._report_search.setObjectName("searchBox")
        self._report_search.setPlaceholderText("Search run id…")
        self._report_search.textChanged.connect(self._refresh_reports)
        self._report_status_filter = QComboBox()
        self._report_status_filter.addItems(["All", "Ready", "Missing", "Outdated"])
        self._report_status_filter.currentTextChanged.connect(self._refresh_reports)
        filter_row.addWidget(self._report_search, 1)
        filter_lbl = QLabel("Status")
        filter_lbl.setObjectName("sectionLabel")
        filter_row.addWidget(filter_lbl)
        filter_row.addWidget(self._report_status_filter)
        filter_row.addStretch(1)
        outer.addLayout(filter_row)

        body_split = QHBoxLayout()
        body_split.setSpacing(12)

        table_card = QFrame()
        table_card.setObjectName("card")
        tc_lay = QVBoxLayout(table_card)
        tc_lay.setContentsMargins(10, 10, 10, 10)
        tc_lay.setSpacing(6)
        self._report_table = QTableWidget()
        self._report_table.setMinimumHeight(460)
        self._report_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._report_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._report_table.itemSelectionChanged.connect(self._on_report_selected)
        self._report_table.itemDoubleClicked.connect(self._on_report_double_clicked)
        tc_lay.addWidget(self._report_table)
        body_split.addWidget(table_card, 2)

        preview_card = QFrame()
        preview_card.setObjectName("card")
        preview_card.setFixedWidth(280)
        pv_lay = QVBoxLayout(preview_card)
        pv_lay.setContentsMargins(14, 14, 14, 14)
        pv_lay.setSpacing(10)
        pv_title = QLabel("SELECTED REPORT")
        pv_title.setObjectName("sectionLabel")
        pv_lay.addWidget(pv_title)
        self._report_preview_label = QLabel("Select a run to see\nreport details.")
        self._report_preview_label.setWordWrap(True)
        self._report_preview_label.setStyleSheet("color: #94a3b8; font-size: 10pt; background: transparent;")
        pv_lay.addWidget(self._report_preview_label)
        pv_lay.addStretch(1)
        self._report_open_btn = QPushButton("Open")
        self._report_open_btn.setObjectName("primaryButton")
        self._report_open_btn.clicked.connect(self._report_action_open)
        self._report_gen_btn = QPushButton("Generate / Regenerate")
        self._report_gen_btn.clicked.connect(self._report_action_generate)
        self._report_del_btn = QPushButton("Delete Report")
        self._report_del_btn.clicked.connect(self._report_action_delete)
        for b in [self._report_open_btn, self._report_gen_btn, self._report_del_btn]:
            b.setEnabled(False)
            pv_lay.addWidget(b)
        body_split.addWidget(preview_card)

        outer.addLayout(body_split, 1)
        return container

    def _refresh_reports(self):
        reports_dir = HERE / "reports"
        query = self._report_search.text().lower().strip() if hasattr(self, "_report_search") else ""
        status_filter = self._report_status_filter.currentText() if hasattr(self, "_report_status_filter") else "All"

        all_rows = []
        counts = {"Ready": 0, "Missing": 0, "Outdated": 0}
        for run in self.runs:
            path = reports_dir / f"{run['id']}.html"
            if path.exists():
                try:
                    src_mt = data.run_source_mtime(run["key"])
                except Exception:
                    src_mt = None
                if src_mt is not None and path.stat().st_mtime < src_mt:
                    status = "Outdated"
                else:
                    status = "Ready"
                generated = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            else:
                status = "Missing"
                generated = "Never"
            counts[status] = counts.get(status, 0) + 1
            all_rows.append({"run": run, "status": status, "generated": generated, "path": path})

        self._report_rows = all_rows
        self._refresh_report_stats(counts)

        rows = all_rows
        if query:
            rows = [r for r in rows if query in r["run"]["id"].lower()]
        if status_filter != "All":
            rows = [r for r in rows if r["status"] == status_filter]

        cols = ["Run ID", "Status", "Tail MAE", "Overshoot", "Duration", "Last Generated"]
        table = self._report_table
        table.setUpdatesEnabled(False)
        table.setAlternatingRowColors(True)
        table.setShowGrid(False)
        table.setWordWrap(False)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        table.setColumnCount(len(cols))
        table.setRowCount(len(rows))
        table.setHorizontalHeaderLabels(cols)
        for ri, r in enumerate(rows):
            run = r["run"]
            dur = f'{fmt(run.get("duration_s"), 1)} s' if run.get("duration_s") is not None else "-"
            values = [run["id"], r["status"], fmt(run.get("tail_mae")),
                      fmt(run.get("overshoot")), dur, r["generated"]]
            for ci, v in enumerate(values):
                item = QTableWidgetItem(str(v))
                item.setData(Qt.UserRole, run["key"])
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                table.setItem(ri, ci, item)
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        table.setUpdatesEnabled(True)

    def _refresh_report_stats(self, counts):
        while self._report_stats_row.count():
            item = self._report_stats_row.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        for lbl, val in [("Total Runs", str(len(self.runs))),
                          ("Ready", str(counts.get("Ready", 0))),
                          ("Missing", str(counts.get("Missing", 0))),
                          ("Outdated", str(counts.get("Outdated", 0)))]:
            box = QFrame()
            box.setObjectName("card")
            bl = QVBoxLayout(box)
            bl.setContentsMargins(16, 10, 16, 10)
            bl.setSpacing(4)
            cap = QLabel(lbl)
            cap.setObjectName("sectionLabel")
            bl.addWidget(cap)
            num = QLabel(val)
            num.setStyleSheet("font-size: 20px; font-weight: 800; background: transparent;")
            bl.addWidget(num)
            self._report_stats_row.addWidget(box)
        self._report_stats_row.addStretch(1)

    def _on_report_selected(self):
        items = self._report_table.selectedItems()
        if not items:
            self._report_selected_key = None
            for b in [self._report_open_btn, self._report_gen_btn, self._report_del_btn]:
                b.setEnabled(False)
            return
        key = items[0].data(Qt.UserRole)
        self._report_selected_key = key
        row = next((r for r in self._report_rows if r["run"]["key"] == key), None)
        if not row:
            return
        run = row["run"]
        path = row["path"]
        exists = path.exists()
        size_str = f"{path.stat().st_size / 1024:.1f} KB" if exists else "-"
        self._report_preview_label.setText(
            f"Run: {run['id']}\n\n"
            f"Status: {row['status']}\n"
            f"Generated: {row['generated']}\n"
            f"File: {path.name if exists else '(not generated)'}\n"
            f"Size: {size_str}"
        )
        self._report_open_btn.setEnabled(exists)
        self._report_gen_btn.setEnabled(True)
        self._report_del_btn.setEnabled(exists)

    def _on_report_double_clicked(self, item):
        key = item.data(Qt.UserRole)
        row = next((r for r in self._report_rows if r["run"]["key"] == key), None)
        if not row:
            return
        if row["path"].exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(row["path"])))
        else:
            self._report_selected_key = key
            self._report_action_generate()

    def _report_action_open(self):
        if not self._report_selected_key:
            return
        row = next((r for r in self._report_rows if r["run"]["key"] == self._report_selected_key), None)
        if row and row["path"].exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(row["path"])))

    def _report_action_generate(self):
        key = self._report_selected_key
        if not key:
            return
        run = next((r for r in self.runs if r["key"] == key), None)
        if not run:
            return
        reports_dir = HERE / "reports"
        reports_dir.mkdir(exist_ok=True)
        path = reports_dir / f"{run['id']}.html"
        try:
            path.write_text(data.make_report_html(key), encoding="utf-8")
        except Exception as exc:
            QMessageBox.critical(self, "Report error", str(exc))
            return
        self._refresh_reports()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _report_action_delete(self):
        key = self._report_selected_key
        if not key:
            return
        row = next((r for r in self._report_rows if r["run"]["key"] == key), None)
        if row and row["path"].exists():
            row["path"].unlink()
        self._refresh_reports()

    def _show_settings(self):
        menu = QMenu(self)
        themes = menu.addMenu("Themes")
        act_dark = themes.addAction("Dark")
        act_dark.setCheckable(True)
        act_dark.setChecked(self.dark)
        act_light = themes.addAction("Light")
        act_light.setCheckable(True)
        act_light.setChecked(not self.dark)
        btn = self.act_settings_btn
        chosen = menu.exec(btn.mapToGlobal(QPoint(btn.width() + 4, 0)))
        if chosen == act_dark and not self.dark:
            self.dark = True
            self.apply_theme()
        elif chosen == act_light and self.dark:
            self.dark = False
            self.apply_theme()

    def _sim_view(self):
        page = QScrollArea()
        page.setObjectName("workspaceScroll")
        page.setWidgetResizable(True)
        page.setFrameShape(QFrame.NoFrame)
        body = QWidget()
        body.setObjectName("workspaceBody")
        lay = QVBoxLayout(body)
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
        defaults = {"kp": "7", "ki": "700", "kd": "10", "start_temp": "27", "target_temp": "0",
                    "duration_s": "1200", "dt_s": "2", "initial_p": "0", "initial_i": "0", "initial_d": "0"}
        for i, (key, val) in enumerate(defaults.items()):
            cell = QWidget()
            cl = QVBoxLayout(cell)
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
        sim_lay = QVBoxLayout(sim_card)
        sim_lay.setContentsMargins(12, 12, 12, 12)
        sim_lay.setSpacing(8)
        sim_tb = QHBoxLayout()
        self.sim_title = QLabel("Awaiting simulation")
        self.sim_title.setObjectName("title")
        self.sim_channel = QComboBox()
        self.sim_channel.addItems(["temp", "error", "u", "pred_delta"])
        self.sim_channel.currentTextChanged.connect(self.draw_simulation)
        sim_tb.addWidget(self.sim_title, 1)
        sim_tb.addWidget(self.sim_channel)
        sim_lay.addLayout(sim_tb)
        self.sim_chart = ChartWidget()
        sim_lay.addWidget(self.sim_chart)
        lay.addWidget(sim_card)
        return page

    # ── Runs management ───────────────────────────────────────────────────────

    def load_runs(self):
        try:
            self.splash_msg("Loading runs…")

            def progress(i, total, msg):
                self.splash_msg("Loading runs…")

            payload = data.list_runs(progress=progress)
        except Exception as exc:
            QMessageBox.critical(self, "Unable to load runs", str(exc))
            return
        self.runs = payload["runs"]
        self.render_runs()
        self._refresh_dashboard()
        if self.runs:
            self._open_run(self.runs[0]["key"])

    def render_runs(self):
        query = self.search_box.text().lower().strip()
        active_page = self.editor_area.active_page()
        compare_keys = active_page.compare_runs if active_page else set()
        self.run_list.blockSignals(True)
        self.run_list.clear()
        for run in self.runs:
            haystack = " ".join(str(v) for v in run.values()).lower()
            if query and query not in haystack:
                continue
            item = QListWidgetItem(run["id"])
            item.setData(Qt.UserRole, run["key"])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if run["key"] in compare_keys else Qt.Unchecked)
            self.run_list.addItem(item)
        self.run_list.blockSignals(False)

    def _open_run_item(self, item):
        self._open_run(item.data(Qt.UserRole))
        self._nav_to(2)

    def _run_list_context_menu(self, pos):
        item = self.run_list.itemAt(pos)
        if not item:
            return
        key = item.data(Qt.UserRole)
        menu = QMenu(self)
        act_open = menu.addAction("Open in Analysis")
        act_pane = menu.addAction("Open in New Pane")
        chosen = menu.exec(self.run_list.viewport().mapToGlobal(pos))
        if chosen == act_open:
            self._open_run(key)
            self._nav_to(2)
        elif chosen == act_pane:
            self._open_run(key)
            self._nav_to(2)

    def _open_run(self, key):
        run = next((r for r in self.runs if r["key"] == key), None)
        if not run:
            return
        run_id = run["id"]
        # Check if already open
        for grp in self.editor_area.all_groups():
            if grp.has_key(key):
                grp.tab_bar.add_or_focus(key, run_id)
                return
        page = RunTabPage(key, self.runs, dark=self.dark)
        page.compare_changed.connect(lambda keys: self.render_runs())
        self.editor_area.register_chart(page.chart)
        self.editor_area.open_run(key, run_id, page)
        page.load()

    def _run_checked(self, item):
        key = item.data(Qt.UserRole)
        checked = item.checkState() == Qt.Checked
        page = self.editor_area.active_page()
        if page:
            page.set_compare_run(key, checked)

    def _on_active_page_changed(self, page):
        self.render_runs()

    # ── Simulator ─────────────────────────────────────────────────────────────

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
        self.sim_title.setText(f"Kp {payload['kp']:g} / Ki {payload['ki']:g} / Kd {payload['kd']:g}")
        self._render_sim_summary()
        self.draw_simulation()

    def _sim_failed(self, msg):
        self.sim_button.setEnabled(True)
        self.sim_title.setText("Simulation failed")
        QMessageBox.critical(self, "Simulation failed", msg)

    def _render_sim_summary(self):
        self._clear_layout(self.sim_summary_grid)
        metrics = self.sim_series.get("metrics", {}) if self.sim_series else {}
        stats = [("Cost", metrics.get("cost")), ("Tail MAE", metrics.get("tail_mae")),
                 ("End Temp", metrics.get("end_temp")), ("Overshoot", metrics.get("overshoot_max")),
                 ("Settle s", metrics.get("time_to_settle_s"))]
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

    def draw_simulation(self):
        if not self.sim_series:
            return
        ch = self.sim_channel.currentText()
        payload = {"columns": ["temp", "temp_ref"] if ch == "temp" else [ch],
                   "points": self.sim_series["points"]}
        self.sim_chart.draw(payload, "line")

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    # ── Theme ─────────────────────────────────────────────────────────────────

    def toggle_theme(self):
        self.dark = not self.dark
        self.apply_theme()

    def splash_msg(self, msg):
        if self.splash:
            self.splash.showMessage(msg, Qt.AlignCenter | Qt.AlignBottom, Qt.white)
            QApplication.processEvents()

    def apply_theme(self):
        if self.dark:
            c = {
                "bg": "#0b1020", "panel": "#111827", "panel2": "#0f172a",
                "panel3": "#1e293b", "text": "#f8fafc", "muted": "#94a3b8",
                "border": "#253247", "border2": "#1f2937", "accent": "#60a5fa",
                "accent2": "#2563eb", "atext": "#ffffff", "hover": "#172033",
                "sel": "#1d4ed8", "talt": "#0d1526", "sbg": "#0b1020", "sh": "#334155",
                "tab_bg": "#1e2433", "tab_active": "#111827", "tab_border": "#60a5fa",
                "tab_text": "#94a3b8", "tab_active_text": "#f8fafc",
            }
        else:
            c = {
                "bg": "#eef2f7", "panel": "#ffffff", "panel2": "#f8fafc",
                "panel3": "#eef2ff", "text": "#0f172a", "muted": "#64748b",
                "border": "#d8e0ea", "border2": "#e5eaf2", "accent": "#2563eb",
                "accent2": "#1d4ed8", "atext": "#ffffff", "hover": "#f1f5f9",
                "sel": "#dbeafe", "talt": "#f8fafc", "sbg": "#eef2f7", "sh": "#cbd5e1",
                "tab_bg": "#ececec", "tab_active": "#ffffff", "tab_border": "#2563eb",
                "tab_text": "#64748b", "tab_active_text": "#0f172a",
            }

        css = f"""
            QWidget {{
                background: {c['bg']};
                color: {c['text']};
                font-family: "Segoe UI", "Inter", "Arial";
                font-size: 10.5pt;
            }}
            QWidget#titleBar {{
                background: {c['panel2']};
                border-bottom: 1px solid {c['border']};
                min-height: 40px; max-height: 40px;
            }}
            QWidget#titleBarLogoArea {{
                background: {c['panel2']};
            }}
            QLabel#titleBarBrand {{
                font-size: 11px; font-weight: 900;
                color: {c['accent']};
                background: transparent;
            }}
            QPushButton#titleCenter {{
                background: {c['panel3']};
                border: 1px solid {c['border']};
                border-radius: 4px;
                color: {c['muted']};
                font-size: 10pt;
                padding: 0 10px;
                text-align: center;
            }}
            QPushButton#titleCenter:hover {{
                border-color: {c['accent']};
                color: {c['text']};
                background: {c['hover']};
            }}
            QPushButton#winBtn {{
                background: transparent;
                border: none;
                border-radius: 0;
                color: {c['muted']};
                font-size: 11px;
                padding: 0;
            }}
            QPushButton#winBtn:hover {{
                background: {c['hover']};
                color: {c['text']};
            }}
            QPushButton#winBtnClose {{
                background: transparent;
                border: none;
                border-radius: 0;
                color: {c['muted']};
                font-size: 11px;
                padding: 0;
            }}
            QPushButton#winBtnClose:hover {{
                background: #c42b1c;
                color: #ffffff;
            }}
            QFrame#titleStatusPill {{
                background: {c['panel3']};
                border: 1px solid {c['border']};
                border-radius: 6px;
                min-height: 28px; max-height: 28px;
            }}
            QLabel#sysDotOk {{
                color: #22c55e;
                font-size: 9px;
                background: transparent;
            }}
            QLabel#sysDotErr {{
                color: #ef4444;
                font-size: 9px;
                background: transparent;
            }}
            QLabel#chamberIconOn {{
                color: #22c55e;
                font-size: 11px;
                background: transparent;
            }}
            QLabel#chamberIconOff {{
                color: #ef4444;
                font-size: 11px;
                background: transparent;
            }}
            QLabel#statusText {{
                font-size: 9.5pt;
                color: {c['muted']};
                background: transparent;
            }}
            QPushButton#titleIconBtn {{
                background: transparent;
                border: none;
                border-radius: 4px;
                font-size: 14px;
                padding: 0;
            }}
            QPushButton#titleIconBtn:hover {{
                background: {c['hover']};
            }}
            QFrame#activityBar {{
                background: {c['panel2']};
                border-right: 1px solid {c['border']};
                min-width: 48px; max-width: 48px;
            }}
            QPushButton#navButton {{
                background: transparent;
                border: none;
                border-left: 3px solid transparent;
                color: {c['muted']};
                font-size: 20px;
                padding: 0;
                border-radius: 0;
                text-align: center;
            }}
            QPushButton#navButton:hover {{
                color: {c['text']};
                background: {c['hover']};
            }}
            QPushButton#navButton:checked {{
                color: {c['accent']};
                border-left-color: {c['accent']};
                background: {c['hover']};
            }}
            QLabel#pageTitle {{
                font-size: 22px;
                font-weight: 800;
                color: {c['text']};
                background: transparent;
            }}
            QFrame#runsPanel {{
                background: {c['panel']};
                border-right: 1px solid {c['border']};
            }}
            QFrame#runsPanelHeader {{
                background: {c['panel']};
                border-bottom: 1px solid {c['border2']};
                min-height: 36px; max-height: 36px;
            }}
            QLabel#sidebarPanelLabel {{
                font-size: 10px;
                font-weight: 800;
                color: {c['muted']};
                letter-spacing: 1.2px;
                background: transparent;
            }}
            QWidget#workspaceBody {{ background: {c['bg']}; }}
            QFrame#card {{
                background: {c['panel']};
                border: 1px solid {c['border']};
                border-radius: 10px;
            }}
            QLabel#title {{
                font-size: 18px;
                font-weight: 800;
                color: {c['text']};
            }}
            QLabel#sectionLabel {{
                font-size: 10px;
                font-weight: 800;
                color: {c['muted']};
                letter-spacing: 0.9px;
                background: transparent;
            }}
            QFrame#inlineStat {{
                background: {c['panel2']};
                border: 1px solid {c['border2']};
                border-radius: 7px;
                min-width: 76px;
            }}
            QLabel#inlineStatLabel {{
                font-size: 9px;
                font-weight: 800;
                color: {c['muted']};
                letter-spacing: 0.7px;
                background: transparent;
            }}
            QLabel#inlineStatValue {{
                font-size: 13px;
                font-weight: 800;
                color: {c['text']};
                background: transparent;
            }}
            QPushButton {{
                background: {c['panel2']};
                color: {c['text']};
                border: 1px solid {c['border']};
                border-radius: 8px;
                padding: 6px 12px;
                font-weight: 650;
            }}
            QPushButton:hover {{ background: {c['hover']}; border-color: {c['accent']}; }}
            QPushButton:pressed {{ background: {c['panel3']}; padding-top: 7px; padding-bottom: 5px; }}
            QPushButton:disabled {{ color: {c['muted']}; background: {c['panel2']}; border-color: {c['border2']}; }}
            QPushButton#primaryButton {{
                background: {c['accent2']};
                color: {c['atext']};
                border-color: {c['accent2']};
            }}
            QPushButton#primaryButton:hover {{ background: {c['accent']}; border-color: {c['accent']}; }}
            QPushButton#secondaryButton {{ color: {c['muted']}; }}
            QLineEdit, QComboBox {{
                background: {c['panel2']}; color: {c['text']};
                border: 1px solid {c['border']}; border-radius: 8px;
                padding: 6px 9px;
                selection-background-color: {c['accent2']};
            }}
            QLineEdit:focus, QComboBox:hover, QComboBox:focus {{
                border-color: {c['accent']}; background: {c['panel']};
            }}
            QLineEdit#searchBox {{ padding: 8px 10px; }}
            QComboBox::drop-down {{ width: 22px; border: 0; }}
            QComboBox QAbstractItemView {{
                background: {c['panel']}; color: {c['text']};
                border: 1px solid {c['border']}; border-radius: 8px;
                selection-background-color: {c['sel']}; padding: 4px; outline: 0;
            }}
            QMenu {{
                background: {c['panel']}; color: {c['text']};
                border: 1px solid {c['border']}; border-radius: 8px; padding: 4px;
            }}
            QMenu::item {{ padding: 7px 22px 7px 10px; border-radius: 6px; }}
            QMenu::item:selected {{ background: {c['sel']}; color: {c['atext']}; }}
            QCheckBox {{ spacing: 7px; color: {c['muted']}; background: transparent; font-weight: 650; }}
            QCheckBox::indicator {{
                width: 15px; height: 15px; border-radius: 4px;
                border: 1px solid {c['border']}; background: {c['panel2']};
            }}
            QCheckBox::indicator:hover {{ border-color: {c['accent']}; }}
            QCheckBox::indicator:checked {{ background: {c['accent2']}; border-color: {c['accent2']}; }}
            QListWidget, QTableWidget {{
                background: {c['panel2']}; color: {c['text']};
                border: 1px solid {c['border2']}; border-radius: 8px;
                padding: 4px; outline: 0; gridline-color: transparent;
                alternate-background-color: {c['talt']};
                selection-background-color: {c['sel']};
                selection-color: {c['atext']};
            }}
            QListWidget::item {{
                min-height: 28px; padding: 6px 8px; border-radius: 6px; margin: 1px;
            }}
            QListWidget::item:hover {{ background: {c['hover']}; }}
            QListWidget::item:selected {{ background: {c['sel']}; color: {c['atext']}; }}
            QTableWidget::item {{ padding: 5px; border: 0; }}
            QHeaderView::section {{
                background: {c['panel']}; color: {c['muted']}; padding: 8px 6px;
                border: 0; border-bottom: 1px solid {c['border']}; font-weight: 800;
            }}
            QTableCornerButton::section {{
                background: {c['panel']}; border: 0; border-bottom: 1px solid {c['border']};
            }}
            QScrollArea, QScrollArea > QWidget > QWidget {{ background: {c['bg']}; border: 0; }}
            QSplitter::handle {{ background: {c['border']}; }}
            QScrollBar:vertical {{
                background: {c['sbg']}; width: 10px; margin: 2px; border-radius: 5px;
            }}
            QScrollBar::handle:vertical {{
                background: {c['sh']}; min-height: 28px; border-radius: 5px;
            }}
            QScrollBar::handle:vertical:hover {{ background: {c['accent']}; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                height: 0; background: transparent; border: 0;
            }}
            QScrollBar:horizontal {{
                background: {c['sbg']}; height: 10px; margin: 2px; border-radius: 5px;
            }}
            QScrollBar::handle:horizontal {{
                background: {c['sh']}; min-width: 28px; border-radius: 5px;
            }}
            QScrollBar::handle:horizontal:hover {{ background: {c['accent']}; }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
                width: 0; background: transparent; border: 0;
            }}
            /* ── Tab bar ── */
            QWidget#tabBar {{
                background: {c['tab_bg']};
                border-bottom: 1px solid {c['border']};
            }}
            QWidget#tabContainer {{
                background: {c['tab_bg']};
            }}
            QFrame#tabItem {{
                background: {c['tab_bg']};
                border-right: 1px solid {c['border']};
                border-bottom: 2px solid transparent;
                min-height: 35px; max-height: 35px;
            }}
            QFrame#tabItem:hover {{
                background: {c['hover']};
            }}
            QFrame#tabItem[tabActive="true"] {{
                background: {c['tab_active']};
                border-bottom: 2px solid {c['tab_border']};
            }}
            QLabel#tabIcon {{
                font-size: 9px;
                color: {c['accent']};
                background: transparent;
            }}
            QLabel#tabLabel {{
                font-size: 10pt;
                color: {c['tab_text']};
                background: transparent;
            }}
            QFrame#tabItem[tabActive="true"] QLabel#tabLabel {{
                color: {c['tab_active_text']};
            }}
            QPushButton#tabClose {{
                background: transparent;
                border: none;
                color: {c['muted']};
                font-size: 13px;
                font-weight: 900;
                padding: 0;
                border-radius: 3px;
                min-width: 16px; max-width: 16px;
                min-height: 16px; max-height: 16px;
            }}
            QPushButton#tabClose:hover {{
                background: #e74c3c;
                color: #ffffff;
            }}
            QPushButton#editorSyncButton {{
                background: {c['panel3']};
                border: 1px solid {c['border']};
                border-radius: 6px;
                color: {c['muted']};
                font-size: 13px;
                padding: 0;
                min-width: 28px; max-width: 28px;
                min-height: 28px; max-height: 28px;
            }}
            QPushButton#editorSyncButton:hover {{
                color: {c['text']};
                border-color: {c['accent']};
                background: {c['hover']};
            }}
            QPushButton#editorSyncButton:checked {{
                color: {c['accent']};
                border-color: {c['accent']};
                background: {c['panel3']};
            }}
            QFrame#dropIndicator {{
                background: {c['accent']};
            }}
            /* ── Editor area ── */
            QWidget#editorArea {{
                background: {c['bg']};
            }}
            QSplitter#editorSplitter::handle {{
                background: {c['border']};
                width: 4px;
            }}
            QSplitter#chartSplitter::handle {{
                background: {c['border']};
                width: 4px;
            }}
            QSplitter#contentSplitter::handle {{
                background: {c['border']};
                width: 4px;
            }}
        """
        self.setStyleSheet(css)
        self.sim_chart.set_dark(self.dark)
        self.editor_area.update_theme(self.dark)
        dash_bg = "#111827" if self.dark else "#f8fafc"
        for p in [self._dash_cost_plot, self._dash_mae_plot, self._dash_ovr_plot]:
            p.setBackground(dash_bg)


# ── Splash ───────────────────────────────────────────────────────────────────

class LoadingSplash(QSplashScreen):
    def drawContents(self, painter):
        painter.setPen(QColor("#f8fafc"))
        painter.setFont(QFont("Segoe UI", 11, 600))
        painter.drawText(QRectF(0, 224, 520, 54), Qt.AlignCenter | Qt.TextWordWrap, self.message())


def make_splash():
    pixmap = QPixmap(520, 300)
    pixmap.fill(QColor("#0b1020"))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    painter.setPen(QColor("#60a5fa"))
    painter.setBrush(QColor("#111827"))
    painter.drawRoundedRect(18, 18, 484, 264, 14, 14)

    logo = QPixmap(LOGO_PATH)
    if not logo.isNull():
        scaled = logo.scaledToHeight(90, Qt.SmoothTransformation)
        x = (520 - scaled.width()) // 2
        painter.drawPixmap(x, 62, scaled)
    else:
        painter.setPen(QColor("#f8fafc"))
        painter.setFont(QFont("Segoe UI", 26, QFont.Black))
        painter.drawText(QRectF(0, 60, 520, 80), Qt.AlignCenter, "DeepVac")

    painter.setPen(QColor("#60a5fa"))
    painter.drawLine(60, 192, 460, 192)

    painter.end()
    return LoadingSplash(pixmap)


def main():
    app = QApplication(sys.argv)
    icon = QIcon(ICON_PATH)
    app.setWindowIcon(icon)

    splash = make_splash()
    splash.show()
    splash.showMessage("Starting DeepVac…", Qt.AlignCenter | Qt.AlignBottom, Qt.white)
    app.processEvents()

    window = DeepVacDesktop(splash=splash)
    window.showMaximized()
    splash.finish(window)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
