"""Tab bar, editor groups, and the split-pane editor area."""
import json

from PySide6.QtCore import QByteArray, QMimeData, QPoint, QSize, Qt, Signal
from PySide6.QtGui import QDrag
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMenu, QPushButton,
    QScrollArea, QSplitter, QStackedWidget, QVBoxLayout, QWidget,
)

from common import _TAB_MIME, _new_gid, _svg_icon
from chart_widget import CrosshairSyncHub


class TabBar(QWidget):
    tab_activated          = Signal(int)
    tab_closed             = Signal(int)
    split_right_requested  = Signal(int)
    move_to_group_requested = Signal(int, int)  # tab_index, group_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("tabBar")
        self.setFixedHeight(35)
        self._entries = []
        self._active  = -1
        self._drag_start = None
        self._drag_idx   = -1
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

        frame.mousePressEvent   = lambda e, k=key: self._press(e, k)
        frame.mouseMoveEvent    = lambda e, k=key: self._move(e, k)
        frame.mouseReleaseEvent = lambda _ev: setattr(self, "_drag_start", None)
        frame.contextMenuEvent  = lambda e, k=key: self._ctx(e, k)
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
            self._drag_idx   = i

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
        menu   = QMenu(self)
        a_sr   = menu.addAction("Split Right")
        moves  = []
        if callable(self.get_other_groups):
            others = self.get_other_groups()
            if others:
                menu.addSeparator()
                for g in others:
                    act = menu.addAction(f"Move to Pane {g.group_id}")
                    moves.append((act, g.group_id))
        menu.addSeparator()
        a_cl   = menu.addAction("Close")
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

    def active_index(self): return self._active

    def active_key(self):
        if 0 <= self._active < len(self._entries):
            return self._entries[self._active]["key"]
        return None

    def count(self): return len(self._entries)

    def key_at(self, index):
        if 0 <= index < len(self._entries):
            return self._entries[index]["key"]
        return None

    def all_keys(self): return [e["key"] for e in self._entries]


class EditorGroup(QWidget):
    active_changed   = Signal(object)          # RunTabPage or None
    split_requested  = Signal(object, str, str) # self, direction, key
    group_empty      = Signal(object)           # self
    tab_received     = Signal(object, str, str) # self, key, id (from drop)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.group_id = _new_gid()
        self._pages   = {}   # key -> RunTabPage
        self.setAcceptDrops(True)
        self.setMinimumSize(300, 300)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self._drop_bar = QFrame()
        self._drop_bar.setObjectName("dropIndicator")
        self._drop_bar.setFixedHeight(3)
        self._drop_bar.setVisible(False)
        lay.addWidget(self._drop_bar)

        self.tab_bar = TabBar()
        self.tab_bar.get_other_groups = None
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

    def has_key(self, key):   return key in self._pages
    def all_keys(self):       return list(self._pages.keys())

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
        raw  = bytes(event.mimeData().data(_TAB_MIME))
        info = json.loads(raw.decode())
        self.tab_received.emit(self, info["key"], info["id"])
        event.acceptProposedAction()


class EditorArea(QWidget):
    active_page_changed = Signal(object)  # RunTabPage or None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("editorArea")
        self._groups       = []
        self._pages        = {}   # key -> (RunTabPage, EditorGroup)
        self._active_group = None
        self._sync_hub     = CrosshairSyncHub()

        self._root_splitter = QSplitter(Qt.Horizontal)
        self._root_splitter.setHandleWidth(4)
        self._root_splitter.setObjectName("editorSplitter")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._root_splitter, 1)

        first = self._make_group()
        self._root_splitter.addWidget(first)

        self._sync_btn = QPushButton(self)
        self._sync_btn.setObjectName("editorSyncButton")
        self._sync_btn.setCheckable(True)
        self._sync_btn.setFixedSize(28, 28)
        self._sync_btn.setIcon(_svg_icon("layout-split", "#94a3b8", 14))
        self._sync_btn.setIconSize(QSize(14, 14))
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

    def all_groups(self): return list(self._groups)

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
        if direction.startswith("move:"):
            target_gid = int(direction.split(":")[1])
            target = next((g for g in self._groups if g.group_id == target_gid), None)
            if target and target is not source_group:
                taken_page, taken_id = source_group.take_page(key)
                if taken_page:
                    self._pages[key] = (taken_page, target)
                    target.add_page(key, taken_id, taken_page)
                    self._active_group = target
            return

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
                remaining = [parent.widget(i) for i in range(parent.count())
                             if parent.widget(i) is not group]
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
