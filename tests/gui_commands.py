"""Headless exercise of every menu command and dialog of the main window.

usage: python tests/gui_commands.py <folder with a DCS raw> <screenshot dir>
Each modal dialog that opens is screenshotted and accepted; any Python
exception raised by a command fails the run.
"""
import os
import sys
import time
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtGui import QFontDatabase  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox  # noqa: E402

from photodesk.gui import res as R  # noqa: E402
from photodesk.gui.app import MainWindow, install_qt_translation  # noqa: E402

SKIP = {57665, 57607, 57609, 57606, 32904, 33108, 57601, 57667, 32870, 32867, 1055, 1056}
SKIP |= set(range(33158, 33168))      # web links


def main(folder, outdir):
    app = QApplication(sys.argv)
    for f in ("C:/Windows/Fonts/malgun.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        if os.path.exists(f):
            app.setFont(QFontDatabase.applicationFontFamilies(QFontDatabase.addApplicationFont(f))[0])
            break
    errors = []
    sys.excepthook = lambda *exc: errors.append("".join(traceback.format_exception(*exc)))
    res = R.Resources.load()
    install_qt_translation(app, res.language)
    w = MainWindow(res)
    w.resize(1400, 900)
    w.show()
    sw = w.open_folder(folder)
    e = sw.sheet.entries[0]
    end = time.time() + 60
    while e.thumb is None and time.time() < end:
        app.processEvents()
    sw.view.setCurrentRow(0)
    sw.view.item(0).setSelected(True)
    w.copy_settings()
    shots = []

    def handle():
        d = app.activeModalWidget()
        if isinstance(d, (QDialog, QMessageBox)) and not isinstance(d, QFileDialog):
            name = f"{len(shots):02d}_{type(d).__name__}.png"
            d.grab().save(os.path.join(outdir, name))
            shots.append((name, d.windowTitle()))
            if isinstance(d, QMessageBox):
                d.done(QMessageBox.Ok)
            elif hasattr(d, "b") and d.b.get(1) is not None and d.b[1].isEnabled():
                d.b[1].click()
            else:
                d.accept()
        elif isinstance(d, QFileDialog):
            d.reject()
        QTimer.singleShot(150, handle)
    QTimer.singleShot(150, handle)
    ran = []
    for cmd, act in sorted(w.acts.items()):
        if cmd in SKIP or not act.isEnabled():
            continue
        before = len(errors)
        act.trigger()
        for _ in range(20):
            app.processEvents()
        ran.append(cmd)
        if len(errors) > before:
            errors[-1] = f"command {cmd} {act.text()}:\n{errors[-1]}"
        if w.mdi.activeSubWindow() is None and w.sheets:
            w.mdi.setActiveSubWindow(w.sheets[0])
        if w.sheets and not w.sheets[0].view.selectedItems():
            w.sheets[0].view.item(0).setSelected(True)
    # Dialogs whose commands are disabled for the sample camera (White Point is Pro 14n/SLR only).
    assert not w.acts[33152].isEnabled()
    before = len(errors)
    w.white_point_dialog()
    for _ in range(20):
        app.processEvents()
    if len(errors) > before:
        errors[-1] = f"white point dialog:\n{errors[-1]}"
    w.grab().save(os.path.join(outdir, "main.png"))
    for ent in sw.sheet.entries:
        ent.dirty = False
    print("commands:", len(ran), "dialogs:", len(shots))
    for s in shots:
        print(" ", *s)
    if errors:
        print("\n".join(errors))
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
