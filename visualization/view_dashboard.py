"""DashboardMixin — builds and refreshes the Dashboard page."""
import pyqtgraph as pg

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget,
)

from common import fmt


class DashboardMixin:
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

        costs = [r["cost"]     for r in self.runs if r.get("cost")     is not None]
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
        import numpy as np
        self._dash_cost_plot.clear()
        self._dash_mae_plot.clear()
        self._dash_ovr_plot.clear()
        if not self.runs:
            return

        ordered = list(reversed(self.runs))
        xs    = list(range(len(ordered)))
        costs = [(x, r["cost"]) for x, r in zip(xs, ordered) if r.get("cost") is not None]
        if costs:
            vx, vy = zip(*costs)
            self._dash_cost_plot.plot(list(vx), list(vy),
                                      pen=pg.mkPen("#60a5fa", width=1.8),
                                      symbol="o", symbolSize=5,
                                      symbolBrush="#60a5fa", symbolPen=None)
        self._dash_cost_plot.setLabel("bottom", "Run (oldest → newest)")
        self._dash_cost_plot.setLabel("left",   "Cost")

        maes = [r["tail_mae"] for r in self.runs if r.get("tail_mae") is not None]
        if maes:
            bins = min(25, max(5, len(maes) // 4))
            y, x = np.histogram(maes, bins=bins)
            w    = (x[1] - x[0]) * 0.85 if len(x) > 1 else 1.0
            bar  = pg.BarGraphItem(x=x[:-1], height=y, width=w,
                                   brush="#51d6c7", pen=pg.mkPen("#111827", width=0.5))
            self._dash_mae_plot.addItem(bar)
        self._dash_mae_plot.setLabel("bottom", "Tail MAE")
        self._dash_mae_plot.setLabel("left",   "Count")

        pairs = [(r["cost"], r["overshoot"]) for r in self.runs
                 if r.get("cost") is not None and r.get("overshoot") is not None]
        if pairs:
            cx, cy  = zip(*pairs)
            scatter = pg.ScatterPlotItem(list(cx), list(cy), size=7,
                                         brush=pg.mkBrush("#f2bd52"),
                                         pen=pg.mkPen("#111827", width=0.5))
            self._dash_ovr_plot.addItem(scatter)
        self._dash_ovr_plot.setLabel("bottom", "Cost")
        self._dash_ovr_plot.setLabel("left",   "Overshoot")
