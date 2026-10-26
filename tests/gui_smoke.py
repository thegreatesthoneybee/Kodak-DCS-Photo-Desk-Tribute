"""Headless GUI smoke test against the original resources and a folder with the sample raw.

usage: python tests/gui_smoke.py <folder> <screenshot.png>
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt, QTimer  # noqa: E402
from PySide6.QtGui import QFontDatabase  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

from photodesk.gui import res as R  # noqa: E402
from photodesk.gui.app import MainWindow  # noqa: E402


def wait(app, cond, timeout=90):
    end = time.time() + timeout
    while not cond():
        if time.time() > end:
            raise TimeoutError
        app.processEvents()
        time.sleep(0.02)


def auto_close(app, action):
    """Run action() on the next modal dialog that opens."""
    def tick():
        w = app.activeModalWidget()
        if isinstance(w, QDialog):
            action(w)
        else:
            QTimer.singleShot(50, tick)
    QTimer.singleShot(50, tick)


def main(folder, shot):
    app = QApplication(sys.argv)
    for f in ("C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/arial.ttf",
              "/System/Library/Fonts/Supplemental/Arial.ttf"):
        if os.path.exists(f):
            fam = QFontDatabase.applicationFontFamilies(QFontDatabase.addApplicationFont(f))
            if fam:
                app.setFont(fam[0])
                break
    res = R.Resources.load()
    w = MainWindow(res)
    w.resize(1500, 950)
    w.show()
    menus = [a.text() for a in w.menuBar().actions()]
    sw = w.open_folder(folder)
    entry = sw.sheet.entries[0]
    wait(app, lambda: entry.thumb is not None)
    sw.view.setCurrentRow(0)
    sw.view.item(0).setSelected(True)
    app.processEvents()
    assert w.controls.b[1173].currentData() == "preset"
    # Edit through the panel and the menu, then undo/redo.
    w.acts[32788].trigger()                    # Lighting > Tungsten
    assert entry.settings.lighting == "Tungsten"
    undo_text = w.acts[57643].text()
    w.acts[57643].trigger()
    assert entry.settings.lighting == "Daylight"
    w.acts[57644].trigger()
    assert entry.settings.lighting == "Tungsten"
    w.acts[57643].trigger()
    w.acts[32855].trigger()                    # Tag
    assert entry.settings.tagged
    w.acts[32838].trigger()                    # Rotate CW
    assert entry.settings.rotation == 1
    w.acts[57643].trigger()
    # Sharpening dialog with preview, choose High, OK.
    auto_close(app, lambda d: (d.b[9].click(), d.b[1].click()))
    w.acts[33149].trigger()
    assert entry.settings.sharpen_amount == 10.0
    # Open the full image rendering and use the crop tool programmatically.
    iw = w.open_image(entry, sw)
    wait(app, lambda: iw.view.image is not None, 180)
    w.set_crop([entry], (0.1, 0.1, 0.6, 0.6))
    assert entry.settings.crop == (0.1, 0.1, 0.6, 0.6)
    w.toggle_dock(w.hist_dock)
    app.processEvents()
    w.mdi.tileSubWindows()
    app.processEvents()
    w.grab().save(shot)
    print("ok", menus, undo_text, w.st_count.text(), w.st_zoom.text())
    for e in sw.sheet.entries:
        e.dirty = False
    w.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
