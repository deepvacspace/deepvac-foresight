"""MonitoringMixin — builds the Live Monitoring page and alarm management."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)


class MonitoringMixin:
    def _monitoring_view(self):
        container = QWidget()
        container.setObjectName("workspaceBody")
        outer = QVBoxLayout(container)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.setSpacing(14)

        hdr = QLabel("Live Monitoring")
        hdr.setObjectName("pageTitle")
        outer.addWidget(hdr)
        sub = QLabel("Real-time chamber data streaming and alarm management.")
        sub.setObjectName("sectionLabel")
        sub.setWordWrap(True)
        outer.addWidget(sub)

        # ── Top row: connection + live data ──────────────────────────────────
        top_row = QHBoxLayout()
        top_row.setSpacing(12)

        conn_card = QFrame()
        conn_card.setObjectName("card")
        conn_card.setFixedWidth(248)
        cl = QVBoxLayout(conn_card)
        cl.setContentsMargins(14, 14, 14, 14)
        cl.setSpacing(10)

        lbl = QLabel("CONNECTION")
        lbl.setObjectName("sectionLabel")
        cl.addWidget(lbl)

        conn_grid = QGridLayout()
        conn_grid.setSpacing(6)
        conn_grid.setColumnStretch(1, 1)

        self._mon_protocol = QComboBox()
        self._mon_protocol.addItems(["TCP / IP", "Serial / RS-232", "Modbus TCP"])

        host_row = QWidget()
        host_lay = QHBoxLayout(host_row)
        host_lay.setContentsMargins(0, 0, 0, 0)
        host_lay.setSpacing(4)
        self._mon_host = QLineEdit("127.0.0.1")
        self._mon_port = QSpinBox()
        self._mon_port.setRange(1, 65535)
        self._mon_port.setValue(5555)
        self._mon_port.setFixedWidth(68)
        host_lay.addWidget(self._mon_host)
        host_lay.addWidget(self._mon_port)

        self._mon_interval = QComboBox()
        self._mon_interval.addItems(["250 ms", "500 ms", "1 s", "2 s", "5 s"])
        self._mon_interval.setCurrentIndex(2)

        for row_idx, (cap, w) in enumerate([
            ("Protocol",    self._mon_protocol),
            ("Host / Port", host_row),
            ("Poll interval", self._mon_interval),
        ]):
            l = QLabel(cap)
            l.setObjectName("sectionLabel")
            conn_grid.addWidget(l, row_idx, 0)
            conn_grid.addWidget(w, row_idx, 1)
        cl.addLayout(conn_grid)

        self._mon_connect_btn = QPushButton("Connect")
        self._mon_connect_btn.setObjectName("primaryButton")
        self._mon_connect_btn.clicked.connect(self._on_mon_connect)
        cl.addWidget(self._mon_connect_btn)

        status_row = QHBoxLayout()
        self._mon_dot       = QLabel("●")
        self._mon_dot.setObjectName("chamberIconOff")
        self._mon_status_lbl = QLabel("Offline — not connected")
        self._mon_status_lbl.setObjectName("statusText")
        status_row.addWidget(self._mon_dot)
        status_row.addWidget(self._mon_status_lbl, 1)
        cl.addLayout(status_row)
        cl.addStretch(1)
        top_row.addWidget(conn_card)

        live_card = QFrame()
        live_card.setObjectName("card")
        ll = QVBoxLayout(live_card)
        ll.setContentsMargins(14, 14, 14, 14)
        ll.setSpacing(8)
        live_hdr = QHBoxLayout()
        live_lbl = QLabel("LIVE DATA")
        live_lbl.setObjectName("sectionLabel")
        live_hdr.addWidget(live_lbl)
        live_hdr.addStretch(1)
        self._mon_update_lbl = QLabel("—")
        self._mon_update_lbl.setObjectName("sectionLabel")
        live_hdr.addWidget(self._mon_update_lbl)
        ll.addLayout(live_hdr)
        placeholder = QLabel(
            "Chamber not connected\n\n"
            "Configure the connection settings and click Connect\n"
            "to begin streaming live data."
        )
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setObjectName("monitorPlaceholder")
        placeholder.setMinimumHeight(240)
        ll.addWidget(placeholder, 1)
        top_row.addWidget(live_card, 1)
        outer.addLayout(top_row)

        # ── Alarms ───────────────────────────────────────────────────────────
        alarms_card = QFrame()
        alarms_card.setObjectName("card")
        al = QVBoxLayout(alarms_card)
        al.setContentsMargins(14, 14, 14, 14)
        al.setSpacing(10)

        alarms_hdr = QHBoxLayout()
        albl = QLabel("ALARMS")
        albl.setObjectName("sectionLabel")
        alarms_hdr.addWidget(albl)
        adesc = QLabel("Define thresholds that notify you when values go out of range.")
        adesc.setObjectName("sectionLabel")
        adesc.setWordWrap(True)
        alarms_hdr.addWidget(adesc, 1)
        add_alarm_btn = QPushButton("+ Add Alarm")
        add_alarm_btn.setObjectName("primaryButton")
        add_alarm_btn.clicked.connect(self._toggle_alarm_form)
        alarms_hdr.addWidget(add_alarm_btn)
        al.addLayout(alarms_hdr)

        self._alarm_form = QFrame()
        self._alarm_form.setObjectName("ruleRow")
        afl = QHBoxLayout(self._alarm_form)
        afl.setContentsMargins(8, 8, 8, 8)
        afl.setSpacing(8)

        self._alarm_name_ed   = QLineEdit()
        self._alarm_name_ed.setPlaceholderText("Alarm name")
        self._alarm_name_ed.setFixedWidth(130)
        self._alarm_var_combo = QComboBox()
        self._alarm_var_combo.addItems(["temp", "temp_ref", "error", "u", "kp", "ki", "kd"])
        self._alarm_var_combo.setEditable(True)
        self._alarm_var_combo.setFixedWidth(100)
        self._alarm_cond_combo = QComboBox()
        self._alarm_cond_combo.addItems(["above", "below", "outside range"])
        self._alarm_cond_combo.setFixedWidth(110)
        self._alarm_val_ed  = QLineEdit()
        self._alarm_val_ed.setPlaceholderText("threshold")
        self._alarm_val_ed.setFixedWidth(80)
        self._alarm_val2_ed = QLineEdit()
        self._alarm_val2_ed.setPlaceholderText("upper (range)")
        self._alarm_val2_ed.setFixedWidth(100)
        self._alarm_sev_combo = QComboBox()
        self._alarm_sev_combo.addItems(["Info", "Warning", "Critical"])
        self._alarm_sev_combo.setFixedWidth(90)

        save_alarm_btn   = QPushButton("Add")
        save_alarm_btn.setObjectName("primaryButton")
        save_alarm_btn.clicked.connect(self._save_alarm)
        cancel_alarm_btn = QPushButton("Cancel")
        cancel_alarm_btn.clicked.connect(self._toggle_alarm_form)

        for cap, w in [
            ("Name",      self._alarm_name_ed),
            ("Variable",  self._alarm_var_combo),
            ("Condition", self._alarm_cond_combo),
            ("Value",     self._alarm_val_ed),
            ("",          self._alarm_val2_ed),
            ("Severity",  self._alarm_sev_combo),
            ("",          save_alarm_btn),
            ("",          cancel_alarm_btn),
        ]:
            if cap:
                l = QLabel(cap)
                l.setObjectName("sectionLabel")
                afl.addWidget(l)
            afl.addWidget(w)
        afl.addStretch(1)
        self._alarm_form.setVisible(False)
        al.addWidget(self._alarm_form)

        self._alarms_table = QTableWidget()
        self._alarms_table.setMinimumHeight(180)
        self._alarms_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._alarms_table.setSelectionMode(QAbstractItemView.SingleSelection)
        al.addWidget(self._alarms_table)
        self._refresh_alarms_table()
        outer.addWidget(alarms_card, 1)
        return container

    def _on_mon_connect(self):
        QMessageBox.information(
            self, "Live Monitoring",
            "Chamber connection is not yet implemented.\n\n"
            "This panel will be enabled when real-time communication\n"
            "with the thermal chamber is developed.",
        )

    def _toggle_alarm_form(self):
        self._alarm_form.setVisible(not self._alarm_form.isVisible())

    def _save_alarm(self):
        name      = self._alarm_name_ed.text().strip()
        var       = self._alarm_var_combo.currentText().strip()
        cond      = self._alarm_cond_combo.currentText()
        val_text  = self._alarm_val_ed.text().strip()
        val2_text = self._alarm_val2_ed.text().strip()
        sev       = self._alarm_sev_combo.currentText()
        if not name or not var or not val_text:
            return
        try:
            value  = float(val_text)
            value2 = float(val2_text) if val2_text else None
        except ValueError:
            return
        self._monitor_alarms.append({
            "name": name, "variable": var, "condition": cond,
            "value": value, "value2": value2, "severity": sev,
        })
        self._alarm_name_ed.clear()
        self._alarm_val_ed.clear()
        self._alarm_val2_ed.clear()
        self._alarm_form.setVisible(False)
        self._refresh_alarms_table()

    def _delete_alarm(self, idx):
        if 0 <= idx < len(self._monitor_alarms):
            self._monitor_alarms.pop(idx)
        self._refresh_alarms_table()

    def _refresh_alarms_table(self):
        cols = ["Name", "Variable", "Condition", "Value / Range", "Severity", "Status"]
        tbl  = self._alarms_table
        tbl.setUpdatesEnabled(False)
        tbl.setAlternatingRowColors(True)
        tbl.setShowGrid(False)
        tbl.setWordWrap(False)
        tbl.verticalHeader().setVisible(False)
        tbl.horizontalHeader().setStretchLastSection(True)
        tbl.setColumnCount(len(cols) + 1)
        tbl.setRowCount(len(self._monitor_alarms))
        tbl.setHorizontalHeaderLabels(cols + [""])
        for ri, alarm in enumerate(self._monitor_alarms):
            val_str = str(alarm["value"])
            if alarm.get("value2") is not None:
                val_str += f" – {alarm['value2']}"
            sev    = alarm["severity"]
            values = [alarm["name"], alarm["variable"], alarm["condition"],
                      val_str, sev, "Inactive (not connected)"]
            for ci, v in enumerate(values):
                item = QTableWidgetItem(str(v))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if ci == 4:
                    color = {"Info": "#60a5fa", "Warning": "#f2bd52",
                             "Critical": "#ff6f7d"}.get(sev, "#94a3b8")
                    item.setForeground(QColor(color))
                tbl.setItem(ri, ci, item)
            del_btn = QPushButton("✕")
            del_btn.setObjectName("tabClose")
            del_btn.setFixedSize(24, 24)
            del_btn.clicked.connect(lambda _=False, i=ri: self._delete_alarm(i))
            tbl.setCellWidget(ri, len(cols), del_btn)
        tbl.resizeColumnsToContents()
        tbl.setUpdatesEnabled(True)
