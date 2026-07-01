"""Custom frameless window title bar with status pills and window controls."""
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QWidget

from common import LOGO_PATH, _render_svg, _svg_icon


class TitleBar(QWidget):
    def __init__(self, main_window):
        super().__init__(main_window)
        self._win = main_window
        self.setObjectName("titleBar")
        self.setFixedHeight(40)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

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

        self.center_btn = QPushButton("⌕   DeepVac Dashboard")
        self.center_btn.setObjectName("titleCenter")
        self.center_btn.setFixedSize(320, 26)
        lay.addWidget(self.center_btn)

        lay.addStretch(1)

        self.status_system  = self._status_pill("check-circle", "System Offline",  "#ef4444", "statusText")
        lay.addWidget(self.status_system)
        lay.addSpacing(4)

        self.status_chamber = self._status_pill("broadcast",    "Chamber Offline", "#ef4444", "statusText")
        lay.addWidget(self.status_chamber)
        lay.addSpacing(4)

        self.btn_bell = QPushButton()
        self.btn_bell.setObjectName("titleIconBtn")
        self.btn_bell.setFixedSize(32, 40)
        self.btn_bell.setFocusPolicy(Qt.NoFocus)
        self.btn_bell.setToolTip("Notifications")
        self.btn_bell.setIcon(_svg_icon("bell", "#94a3b8", 16))
        self.btn_bell.setIconSize(QSize(16, 16))
        lay.addWidget(self.btn_bell)
        lay.addSpacing(4)

        self.btn_min   = QPushButton("─")
        self.btn_max   = QPushButton("□")
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

    def _status_pill(self, icon_name, label_text, icon_color, label_obj):
        pill = QFrame()
        pill.setObjectName("titleStatusPill")
        pill.setFixedHeight(28)
        pl = QHBoxLayout(pill)
        pl.setContentsMargins(8, 0, 8, 0)
        pl.setSpacing(5)
        icon = QLabel()
        icon.setPixmap(_render_svg(icon_name, icon_color, 14))
        icon.setFixedSize(14, 14)
        icon._icon_name = icon_name
        txt = QLabel(label_text)
        txt.setObjectName(label_obj)
        pl.addWidget(icon)
        pl.addWidget(txt)
        pill._icon = icon
        pill._txt  = txt
        return pill

    def set_system_status(self, ok, message=None):
        color = "#22c55e" if ok else "#ef4444"
        self.status_system._icon.setPixmap(_render_svg("check-circle", color, 14))
        self.status_system._txt.setText(message or ("System OK" if ok else "System Error"))

    def set_chamber_status(self, online):
        color = "#22c55e" if online else "#ef4444"
        self.status_chamber._icon.setPixmap(_render_svg("broadcast", color, 14))
        self.status_chamber._txt.setText("Chamber Online" if online else "Chamber Offline")

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
