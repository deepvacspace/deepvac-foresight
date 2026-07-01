"""Shared constants, pure utilities, and SVG icon helpers used across all modules."""
import csv
import json
from pathlib import Path

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtGui import QIcon, QPainter, QPixmap, QTextDocument
from PySide6.QtPrintSupport import QPrinter

HERE = Path(__file__).resolve().parent
LOGO_PATH = str(HERE / "logo.png")
ICON_PATH  = str(HERE / "icon.png")

COLORS = [
    "#8bd66f", "#4f7cff", "#51d6c7", "#f2bd52",
    "#ff6f7d", "#b792ff", "#f48fb1", "#7ec8ff",
]

RULE_COLOR_OPTIONS = [
    ("Blue",   "#60a5fa"),
    ("Green",  "#8bd66f"),
    ("Amber",  "#f2bd52"),
    ("Red",    "#ff6f7d"),
    ("Purple", "#b792ff"),
    ("Teal",   "#51d6c7"),
]

_TAB_MIME = "application/deepvac-tab"
_gid = 0


def _new_gid() -> int:
    global _gid
    _gid += 1
    return _gid


def fmt(value, digits: int = 3) -> str:
    if value in (None, ""):
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 1000:
        return f"{number:,.1f}"
    return f"{number:,.{digits}f}".rstrip("0").rstrip(".")


def csv_escape(value) -> str:
    if value is None:
        return ""
    text = str(value)
    if any(ch in text for ch in [",", '"', "\n", "\r"]):
        return '"' + text.replace('"', '""') + '"'
    return text


def _html_to_pdf(html_content: str, output_path: str) -> None:
    printer = QPrinter()
    printer.setOutputFormat(QPrinter.PdfFormat)
    printer.setOutputFileName(output_path)
    doc = QTextDocument()
    doc.setHtml(html_content)
    doc.print_(printer)


def _render_svg(name: str, color: str, size: int = 20) -> QPixmap:
    path = HERE / "icons" / f"{name}.svg"
    try:
        svg_bytes = (
            path.read_text(encoding="utf-8")
            .replace("currentColor", color)
            .encode()
        )
        renderer = QSvgRenderer(QByteArray(svg_bytes))
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        return pixmap
    except Exception:
        return QPixmap()


def _svg_icon(name: str, color: str = "#94a3b8", size: int = 20) -> QIcon:
    px = _render_svg(name, color, size)
    return QIcon(px) if not px.isNull() else QIcon()


def _nav_icon(name: str, muted: str, accent: str, size: int = 22) -> QIcon:
    icon = QIcon()
    icon.addPixmap(_render_svg(name, muted, size), QIcon.Normal, QIcon.Off)
    icon.addPixmap(_render_svg(name, accent, size), QIcon.Normal, QIcon.On)
    return icon
