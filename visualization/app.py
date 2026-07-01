"""Entry point — splash screen and main() only."""
import sys

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen

from common import LOGO_PATH, ICON_PATH
from main_window import DeepVacDesktop


class LoadingSplash(QSplashScreen):
    def drawContents(self, painter):
        painter.setPen(QColor("#f8fafc"))
        painter.setFont(QFont("Segoe UI", 11, 600))
        painter.drawText(QRectF(0, 224, 520, 54),
                         Qt.AlignCenter | Qt.TextWordWrap, self.message())


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
    app.setWindowIcon(QIcon(ICON_PATH))

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
