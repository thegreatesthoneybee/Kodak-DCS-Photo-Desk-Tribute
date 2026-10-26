"""Headless GUI smoke test against the original resources and a folder with the sample raw.

usage: python tests/gui_smoke.py <folder> <screenshot.png>
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Keep the user's Photo Desk settings (window layout, recent files) untouched.
os.environ.setdefault("PHOTODESK_SETTINGS", os.path.join(os.environ.get("TEMP", "/tmp"), "photodesk_test.ini"))

from PySide6.QtCore import QEvent, Qt, QTimer  # noqa: E402
from PySide6.QtGui import QFocusEvent, QFontDatabase  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

from photodesk.gui import res as R  # noqa: E402
from photodesk.gui.app import MainWindow, install_qt_translation  # noqa: E402


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
    for f in os.listdir(folder):      # start from camera settings
        if f.endswith(".photodesk.json"):
            os.remove(os.path.join(folder, f))
    res = R.Resources.load()
    install_qt_translation(app, res.language)
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
    before = entry.thumb
    w.acts[32788].trigger()                    # Lighting > Tungsten
    assert entry.settings.lighting == "Tungsten"
    wait(app, lambda: entry.thumb is not before)   # thumbnail re-rendered in the background
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
    w.mdi.setActiveSubWindow(iw)
    wait(app, lambda: iw.view.crop == entry.settings.crop, 180)
    w.set_tool("crop")                         # status bar shows the crop size in Crop mode
    cw, ch = iw.view.crop_pixels()
    assert w.st_size.text().count(str(cw)) and w.st_size.text().count(str(ch)), w.st_size.text()
    w.set_tool("select")
    w.toggle_dock(w.hist_dock)
    app.processEvents()
    # Save As (Standard JPEG) through the export path used by File > Save As.
    out = os.path.join(os.environ.get("TEMP", "/tmp"), "pd_smoke_out")
    os.makedirs(out, exist_ok=True)
    from pathlib import Path
    from photodesk import export
    auto_close(app, lambda d: None)
    dest = export.output_path(entry.path, out, "jpeg")
    if dest.exists():
        dest.unlink()
    paths = w._export([entry], [dest], "jpeg")
    assert paths and Path(paths[0]).stat().st_size > 10000, paths
    # Drop the exported JPEG onto a second contact sheet; the drop menu offers Copy Here first.
    from PySide6.QtCore import QMimeData, QPointF, QUrl
    from PySide6.QtGui import QDropEvent
    from PySide6.QtWidgets import QMenu
    import shutil
    drop_dir = Path(out) / "drop"
    shutil.rmtree(drop_dir, ignore_errors=True)
    drop_dir.mkdir()
    sw2 = w.open_folder(drop_dir)
    shown = []

    def pick():
        m = app.activePopupWidget()
        if not isinstance(m, QMenu):
            QTimer.singleShot(50, pick)
            return
        shown.append([x.text() for x in m.actions()])
        from PySide6.QtTest import QTest
        m.setActiveAction(m.actions()[0])
        QTest.keyClick(m, Qt.Key_Return)
    QTimer.singleShot(50, pick)
    md = QMimeData()
    md.setUrls([QUrl.fromLocalFile(str(paths[0]))])
    sw2.view.dropEvent(QDropEvent(QPointF(10, 10), Qt.CopyAction, md, Qt.RightButton, Qt.NoModifier))
    assert shown and shown[0][0] == "&Copy Here", shown
    assert (drop_dir / Path(paths[0]).name).exists() and Path(paths[0]).exists()
    assert sw2.view.count() == 1
    # Shift + left drag moves without a menu.
    moved_src = Path(out) / "moved.jpg"
    shutil.copy(paths[0], moved_src)
    md2 = QMimeData()
    md2.setUrls([QUrl.fromLocalFile(str(moved_src))])
    sw2.view.dropEvent(QDropEvent(QPointF(10, 10), Qt.MoveAction, md2, Qt.LeftButton, Qt.ShiftModifier))
    assert not moved_src.exists() and (drop_dir / "moved.jpg").exists() and sw2.view.count() == 2
    wait(app, lambda: all(e.thumb is not None for e in sw2.sheet.entries))
    # A file written into the folder by another program shows up on its own (folder watcher).
    shutil.copy(paths[0], drop_dir / "external.jpg")
    wait(app, lambda: sw2.view.count() == 3, 30)
    # Annotation goes to the focused (yellow) thumbnail only, even with several selected.
    w.mdi.setActiveSubWindow(sw2)
    app.processEvents()
    sw2.view.selectAll()
    sw2.view.setCurrentRow(1, sw2.view.selectionModel().SelectionFlag.Select)
    app.processEvents()
    focused = w.focused()
    assert len(w.selection()) == 3 and focused is not None
    w.info.note.setPlainText("note")
    w.info.eventFilter(w.info.note, QFocusEvent(QEvent.FocusOut))
    assert [e.settings.annotation for e in sw2.sheet.entries].count("note") == 1
    assert focused.settings.annotation == "note"
    for e in sw2.sheet.entries:
        e.dirty = False
    # Scrolling Hand on a contact sheet scrolls instead of selecting.
    from PySide6.QtGui import QMouseEvent
    w.set_tool("hand")
    sw2.view.clearSelection()
    vp = sw2.view.viewport()
    for typ, y in ((QEvent.MouseButtonPress, 50), (QEvent.MouseMove, 10), (QEvent.MouseButtonRelease, 10)):
        app.sendEvent(vp, QMouseEvent(typ, QPointF(20, y), QPointF(20, y), Qt.LeftButton,
                                      Qt.LeftButton if typ != QEvent.MouseButtonRelease else Qt.NoButton,
                                      Qt.NoModifier))
    assert not sw2.view.selectedItems()
    w.set_tool("select")
    w.mdi.setActiveSubWindow(sw)
    w.mdi.tileSubWindows()
    app.processEvents()
    w.grab().save(shot)
    print("ok", menus, undo_text, w.st_count.text(), w.st_zoom.text())
    for e in sw.sheet.entries:
        e.dirty = False
    w.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
