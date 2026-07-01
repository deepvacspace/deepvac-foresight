"""ReportsMixin — builds the Reports page and handles report generation."""
from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from PySide6.QtCore import QUrl

from common import HERE, fmt, _html_to_pdf, _svg_icon
import data_service as data


class ReportsMixin:
    def _reports_view(self):
        self._report_rows         = []
        self._report_selected_key = None

        container = QWidget()
        container.setObjectName("workspaceBody")
        outer = QVBoxLayout(container)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.setSpacing(14)

        hdr = QLabel("Reports")
        hdr.setObjectName("pageTitle")
        outer.addWidget(hdr)

        sub = QLabel(
            "Generate, preview, and export validation reports from completed thermal runs.")
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
        self._report_search.addAction(
            _svg_icon("search", "#64748b", 13), QLineEdit.LeadingPosition)
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
        self._report_preview_label.setStyleSheet(
            "color: #94a3b8; font-size: 10pt; background: transparent;")
        pv_lay.addWidget(self._report_preview_label)
        pv_lay.addStretch(1)
        self._report_open_btn = QPushButton("Open")
        self._report_open_btn.setObjectName("primaryButton")
        self._report_open_btn.clicked.connect(self._report_action_open)
        self._report_gen_btn  = QPushButton("Generate / Regenerate")
        self._report_gen_btn.clicked.connect(self._report_action_generate)
        self._report_del_btn  = QPushButton("Delete Report")
        self._report_del_btn.clicked.connect(self._report_action_delete)
        for b in [self._report_open_btn, self._report_gen_btn, self._report_del_btn]:
            b.setEnabled(False)
            pv_lay.addWidget(b)
        body_split.addWidget(preview_card)

        outer.addLayout(body_split, 1)
        return container

    def _refresh_reports(self):
        reports_dir   = HERE / "reports"
        query         = (self._report_search.text().lower().strip()
                         if hasattr(self, "_report_search") else "")
        status_filter = (self._report_status_filter.currentText()
                         if hasattr(self, "_report_status_filter") else "All")

        all_rows = []
        counts   = {"Ready": 0, "Missing": 0, "Outdated": 0}
        for run in self.runs:
            path = reports_dir / f"{run['id']}.pdf"
            if path.exists():
                try:
                    src_mt = data.run_source_mtime(run["key"])
                except Exception:
                    src_mt = None
                if src_mt is not None and path.stat().st_mtime < src_mt:
                    status = "Outdated"
                else:
                    status = "Ready"
                generated = datetime.fromtimestamp(
                    path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            else:
                status    = "Missing"
                generated = "Never"
            counts[status] = counts.get(status, 0) + 1
            all_rows.append({"run": run, "status": status,
                             "generated": generated, "path": path})

        self._report_rows = all_rows
        self._refresh_report_stats(counts)

        rows = all_rows
        if query:
            rows = [r for r in rows if query in r["run"]["id"].lower()]
        if status_filter != "All":
            rows = [r for r in rows if r["status"] == status_filter]

        cols  = ["Run ID", "Status", "Tail MAE", "Overshoot", "Duration", "Last Generated"]
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
            dur = (f'{fmt(run.get("duration_s"), 1)} s'
                   if run.get("duration_s") is not None else "-")
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
            w    = item.widget()
            if w:
                w.deleteLater()
        for lbl, val in [
            ("Total Runs", str(len(self.runs))),
            ("Ready",      str(counts.get("Ready",   0))),
            ("Missing",    str(counts.get("Missing", 0))),
            ("Outdated",   str(counts.get("Outdated",0))),
        ]:
            box = QFrame()
            box.setObjectName("card")
            bl  = QVBoxLayout(box)
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
        key  = items[0].data(Qt.UserRole)
        self._report_selected_key = key
        row  = next((r for r in self._report_rows if r["run"]["key"] == key), None)
        if not row:
            return
        run    = row["run"]
        path   = row["path"]
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
        self._report_selected_key = key
        if row["path"].exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(row["path"])))
        else:
            self._report_action_generate()

    def _report_action_open(self):
        if not self._report_selected_key:
            return
        row = next((r for r in self._report_rows
                    if r["run"]["key"] == self._report_selected_key), None)
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
        path = reports_dir / f"{run['id']}.pdf"
        try:
            _html_to_pdf(data.make_report_html(key), str(path))
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
