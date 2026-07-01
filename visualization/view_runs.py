"""RunsMixin — builds the Runs browser page and manages run opening/comparison."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMenu, QTableWidget, QVBoxLayout, QWidget,
)

from common import fmt, _svg_icon
from run_tab import RunTabPage
import data_service as data


class RunsMixin:
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
        self.search_box.addAction(
            _svg_icon("search", "#64748b", 13), QLineEdit.LeadingPosition)
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

    def load_runs(self):
        from PySide6.QtWidgets import QMessageBox

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
        query       = self.search_box.text().lower().strip()
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
            item.setCheckState(
                Qt.Checked if run["key"] in compare_keys else Qt.Unchecked)
            self.run_list.addItem(item)
        self.run_list.blockSignals(False)

    def _open_run_item(self, item):
        self._open_run(item.data(Qt.UserRole))
        self._nav_to(2)

    def _run_list_context_menu(self, pos):
        item = self.run_list.itemAt(pos)
        if not item:
            return
        key  = item.data(Qt.UserRole)
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
        key     = item.data(Qt.UserRole)
        checked = item.checkState() == Qt.Checked
        page    = self.editor_area.active_page()
        if page:
            page.set_compare_run(key, checked)

    def _on_active_page_changed(self, page):
        self.render_runs()

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
            self._fill_generic_table(
                self._raw_run_table, table["columns"], table["rows"], max_rows=2000)
        except Exception:
            pass

    def _fill_generic_table(self, table, columns, rows, max_rows=None):
        from PySide6.QtWidgets import QTableWidgetItem
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
