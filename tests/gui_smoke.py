"""Headless GUI smoke test: open the sample folder, render, change look, screenshot."""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from photodesk.gui.app import MainWindow  # noqa: E402


def wait(app, cond, timeout=60):
    end = time.time() + timeout
    while not cond():
        if time.time() > end:
            raise TimeoutError
        app.processEvents()
        time.sleep(0.02)


def main(folder, shot):
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1600, 950)
    w.show()
    w.open_folder(folder)
    wait(app, lambda: w.view.pixmap is not None)
    first = w.view.pixmap.toImage()
    w.panel.look.setCurrentIndex(w.panel.look.findData("DCSPortraitLook.icm"))
    wait(app, lambda: w.view.pixmap.toImage() != first)
    app.processEvents()
    wait(app, lambda: not w.sheet.item(0).icon().isNull())
    w.grab().save(shot)
    print("ok", w.statusBar().currentMessage(), w.panel.source.text())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
