"""Qt GUI: contact sheet, image review with adjustments, and batch export."""
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QSettings, QSize, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QAction, QIcon, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QProgressDialog, QPushButton, QScrollArea, QSlider, QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

from .. import __version__, camera, export, pipeline, profiles
from ..cli import RAW_SUFFIXES

THUMB = 180
PREVIEW_LONG_EDGE = 1600


def to_qimage(img):
    """float32 or uint8 HxWx3 RGB -> QImage (owns a copy of the data)."""
    if img.dtype != np.uint8:
        img = np.round(np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    return QImage(img.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class Task(QRunnable):
    class _Signals(QObject):
        done = Signal(object, object)   # (token, result)
        failed = Signal(object, str)

    def __init__(self, token, fn, *args):
        super().__init__()
        self.token, self.fn, self.args = token, fn, args
        self.signals = Task._Signals()

    def run(self):
        try:
            result = self.fn(*self.args)
        except Exception as e:  # surfaced to the UI via the failed signal
            self.signals.failed.emit(self.token, str(e))
        else:
            self.signals.done.emit(self.token, result)


def _thumb_job(path):
    t = pipeline.thumbnail(path)
    return pipeline.resize_long_edge(t, THUMB * 2)


def _decode_job(path):
    raw = pipeline.decode(path, half_size=True)
    return raw, pipeline.load_settings(path, raw.make)


def _render_job(raw, settings, space):
    img = pipeline.render(raw, settings, space)
    return pipeline.resize_long_edge(img, PREVIEW_LONG_EDGE)


class ImageView(QScrollArea):
    """Zoomable image display; emits picked image coordinates when picking is enabled."""
    picked = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.label = QLabel(alignment=Qt.AlignCenter)
        self.label.setMinimumSize(1, 1)
        self.setWidget(self.label)
        self.setWidgetResizable(True)
        self.setAlignment(Qt.AlignCenter)
        self.pixmap = None
        self.fit = True
        self.picking = False
        self.label.mousePressEvent = self._click

    def set_image(self, qimage):
        self.pixmap = QPixmap.fromImage(qimage)
        self._update()

    def set_fit(self, fit):
        self.fit = fit
        self.setWidgetResizable(fit)
        self._update()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._update()

    def _update(self):
        if self.pixmap is None:
            self.label.clear()
            return
        if self.fit:
            self.label.setPixmap(self.pixmap.scaled(self.viewport().size(), Qt.KeepAspectRatio,
                                                    Qt.SmoothTransformation))
        else:
            self.label.setPixmap(self.pixmap)
            self.label.adjustSize()

    def _click(self, e):
        if not self.picking or self.pixmap is None:
            return
        shown = self.label.pixmap()
        ox = (self.label.width() - shown.width()) / 2
        oy = (self.label.height() - shown.height()) / 2
        x = (e.position().x() - ox) / shown.width()
        y = (e.position().y() - oy) / shown.height()
        if 0 <= x <= 1 and 0 <= y <= 1:
            self.picked.emit(x, y)


class AdjustPanel(QWidget):
    changed = Signal()
    pick_wb = Signal(bool)

    def __init__(self):
        super().__init__()
        form = QFormLayout(self)
        self.illuminant = QComboBox()
        self.illuminant.addItems(camera.ILLUMINANTS)
        self.look = QComboBox()
        self.wb_mode = QComboBox()
        for key, label in (("daylight", "Daylight"), ("as_shot", "As Shot"), ("auto", "Auto"),
                           ("custom", "Custom (Click Gray)")):
            self.wb_mode.addItem(label, key)
        self.pick = QPushButton("Pick Neutral...")
        self.pick.setCheckable(True)
        self.exposure = QDoubleSpinBox(minimum=-4, maximum=4, singleStep=0.1, decimals=2, suffix=" EV")
        self.sharpen = self._slider(0, 200)
        self.noise_luma = self._slider(0, 100)
        self.noise_chroma = self._slider(0, 100)
        self.source = QLabel()
        self.source.setWordWrap(True)
        form.addRow("Illuminant", self.illuminant)
        form.addRow("Look", self.look)
        form.addRow("White Balance", self.wb_mode)
        form.addRow("", self.pick)
        form.addRow("Exposure", self.exposure)
        form.addRow("Sharpening", self.sharpen)
        form.addRow("Noise (Luma)", self.noise_luma)
        form.addRow("Noise (Color)", self.noise_chroma)
        form.addRow("Color Source", self.source)
        self.rotate_l = QPushButton("Rotate Left")
        self.rotate_r = QPushButton("Rotate Right")
        row = QHBoxLayout()
        row.addWidget(self.rotate_l)
        row.addWidget(self.rotate_r)
        form.addRow(row)
        self.rotation = 0
        self.wb_custom = (1.0, 1.0, 1.0)
        self._loading = False
        for w in (self.illuminant, self.look, self.wb_mode):
            w.currentIndexChanged.connect(self._emit)
        self.exposure.valueChanged.connect(self._emit)
        for s in (self.sharpen, self.noise_luma, self.noise_chroma):
            s.valueChanged.connect(self._emit)
        self.rotate_l.clicked.connect(lambda: self._rotate(-1))
        self.rotate_r.clicked.connect(lambda: self._rotate(1))
        self.pick.toggled.connect(self.pick_wb)

    def _slider(self, lo, hi):
        s = QSlider(Qt.Horizontal, minimum=lo, maximum=hi)
        s.setTracking(False)
        return s

    def set_looks(self, looks):
        self.look.blockSignals(True)
        self.look.clear()
        for name, desc in looks:
            self.look.addItem(desc, name)
        self.look.blockSignals(False)

    def _rotate(self, d):
        self.rotation = (self.rotation + d) % 4
        self._emit()

    def _emit(self):
        if not self._loading:
            self.changed.emit()

    def load(self, s):
        self._loading = True
        self.illuminant.setCurrentText(s.illuminant)
        i = self.look.findData(s.look)
        self.look.setCurrentIndex(i if i >= 0 else 0)
        self.wb_mode.setCurrentIndex(max(0, self.wb_mode.findData(s.wb_mode)))
        self.wb_custom = tuple(s.wb_custom)
        self.exposure.setValue(s.exposure)
        self.sharpen.setValue(round(s.sharpen * 100))
        self.noise_luma.setValue(round(s.noise_luma * 100))
        self.noise_chroma.setValue(round(s.noise_chroma * 100))
        self.rotation = s.rotation
        self._loading = False

    def settings(self):
        return pipeline.Settings(
            illuminant=self.illuminant.currentText(),
            look=self.look.currentData() or pipeline.DEFAULT_LOOK,
            wb_mode=self.wb_mode.currentData(),
            wb_custom=self.wb_custom,
            exposure=self.exposure.value(),
            sharpen=self.sharpen.value() / 100,
            noise_luma=self.noise_luma.value() / 100,
            noise_chroma=self.noise_chroma.value() / 100,
            rotation=self.rotation,
        )


class ExportDialog(QDialog):
    def __init__(self, parent, store):
        super().__init__(parent)
        self.setWindowTitle("Save As")
        self.store = store
        form = QFormLayout(self)
        self.folder = QPushButton(store.value("export/dir", str(Path.home())))
        self.folder.clicked.connect(self._choose)
        self.format = QComboBox()
        self.format.addItems(export.FORMATS)
        self.format.setCurrentText(store.value("export/format", "JPEG"))
        self.space = QComboBox()
        self.space.addItems(list(pipeline.OUTPUT_SPACES))
        self.space.setCurrentText(store.value("export/space", "sRGB"))
        self.quality = QSpinBox(minimum=10, maximum=100, value=int(store.value("export/quality", 92)))
        self.resize = QCheckBox("Limit long edge")
        self.long_edge = QSpinBox(minimum=100, maximum=20000, value=int(store.value("export/long_edge", 3000)))
        self.resize.setChecked(store.value("export/resize", "false") == "true")
        form.addRow("Folder", self.folder)
        form.addRow("Format", self.format)
        form.addRow("Color Space", self.space)
        form.addRow("JPEG Quality", self.quality)
        form.addRow(self.resize, self.long_edge)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _choose(self):
        d = QFileDialog.getExistingDirectory(self, "Output Folder", self.folder.text())
        if d:
            self.folder.setText(d)

    def accept(self):
        s = self.store
        s.setValue("export/dir", self.folder.text())
        s.setValue("export/format", self.format.currentText())
        s.setValue("export/space", self.space.currentText())
        s.setValue("export/quality", self.quality.value())
        s.setValue("export/long_edge", self.long_edge.value())
        s.setValue("export/resize", "true" if self.resize.isChecked() else "false")
        super().accept()

    def options(self):
        return dict(out_dir=self.folder.text(), fmt=self.format.currentText(), space=self.space.currentText(),
                    quality=self.quality.value(), long_edge=self.long_edge.value() if self.resize.isChecked() else None)


def _export_job(path, settings, opts):
    return export.process_file(path, opts["out_dir"], opts["fmt"], opts["space"], opts["quality"],
                               opts["long_edge"], settings)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"PhotoDesk {__version__}")
        self.store = QSettings("PhotoDesk", "PhotoDesk")
        self.pool = QThreadPool.globalInstance()
        self.folder = None
        self.raw = None
        self.current = None
        self.render_token = 0
        self.clipboard = None

        self.sheet = QListWidget()
        self.sheet.setViewMode(QListWidget.IconMode)
        self.sheet.setIconSize(QSize(THUMB, THUMB))
        self.sheet.setResizeMode(QListWidget.Adjust)
        self.sheet.setSelectionMode(QListWidget.ExtendedSelection)
        self.sheet.setUniformItemSizes(True)
        self.sheet.currentItemChanged.connect(self._open_item)

        self.view = ImageView()
        self.view.picked.connect(self._picked)
        self.panel = AdjustPanel()
        self.panel.changed.connect(self._settings_changed)
        self.panel.pick_wb.connect(self._pick_mode)
        self.info = QLabel()
        self.info.setWordWrap(True)
        side = QWidget()
        sv = QVBoxLayout(side)
        sv.addWidget(self.panel)
        sv.addWidget(self.info)
        sv.addStretch()

        split = QSplitter()
        split.addWidget(self.sheet)
        split.addWidget(self.view)
        split.addWidget(side)
        split.setSizes([420, 900, 320])
        self.setCentralWidget(split)
        self.status = self.statusBar()
        self.save_timer = QTimer(self, singleShot=True, interval=400, timeout=self._persist)
        self._menus()
        self._refresh_looks()
        self.resize(1600, 950)
        last = self.store.value("last_folder")
        if last and Path(last).is_dir():
            self.open_folder(last)

    # ---------------------------------------------------------------- menus
    def _menus(self):
        def act(menu, text, slot, shortcut=None):
            a = QAction(text, self)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            a.triggered.connect(slot)
            menu.addAction(a)
            return a

        m = self.menuBar().addMenu("&File")
        act(m, "Open Folder...", self._choose_folder, QKeySequence.Open)
        act(m, "Save As...", self._export, QKeySequence.Save)
        act(m, "Install Kodak Profiles...", self._install_profiles)
        m.addSeparator()
        act(m, "Quit", self.close, QKeySequence.Quit)
        m = self.menuBar().addMenu("&Edit")
        act(m, "Copy Settings", self._copy, "Ctrl+Shift+C")
        act(m, "Paste Settings to Selected", self._paste, "Ctrl+Shift+V")
        act(m, "Reset Settings", self._reset)
        m = self.menuBar().addMenu("&View")
        fit = act(m, "Fit to Window", lambda checked: self.view.set_fit(checked), "Ctrl+0")
        fit.setCheckable(True)
        fit.setChecked(True)
        m = self.menuBar().addMenu("&Help")
        act(m, "About", lambda: QMessageBox.about(
            self, "PhotoDesk",
            f"PhotoDesk {__version__}\nKodak DCS raw processor using LibRaw and the Kodak "
            "DCS Photo Desk 4.3 color profiles."))

    # -------------------------------------------------------------- profiles
    def _refresh_looks(self):
        if profiles.is_installed():
            self.panel.set_looks(pipeline.available_looks())
            self.panel.setEnabled(True)
        else:
            self.panel.setEnabled(False)
            self.status.showMessage("Kodak profiles missing: File > Install Kodak Profiles...")

    def _install_profiles(self):
        path, _ = QFileDialog.getOpenFileName(self, "Photo_Desk_V_4_3_0.exe", str(Path.home()),
                                              "Photo Desk installer (*.exe)")
        if not path:
            return
        try:
            dest, names = profiles.install_from_installer(path)
        except Exception as e:
            QMessageBox.critical(self, "Install Profiles", str(e))
            return
        pipeline.color_lut.cache_clear()
        self._refresh_looks()
        QMessageBox.information(self, "Install Profiles", f"Installed {len(names)} profiles to\n{dest}")
        if self.raw:
            self._render()

    # --------------------------------------------------------- contact sheet
    def _choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Open Folder", self.folder or str(Path.home()))
        if d:
            self.open_folder(d)

    def open_folder(self, folder):
        self.folder = folder
        self.store.setValue("last_folder", folder)
        self.sheet.clear()
        files = sorted(p for p in Path(folder).iterdir()
                       if p.suffix.lower() in RAW_SUFFIXES and p.is_file() and not p.name.startswith("._"))
        for p in files:
            item = QListWidgetItem(p.name)
            item.setData(Qt.UserRole, str(p))
            item.setSizeHint(QSize(THUMB + 16, THUMB + 32))
            self.sheet.addItem(item)
            task = Task(item, _thumb_job, p)
            task.signals.done.connect(self._thumb_ready)
            self.pool.start(task)
        self.status.showMessage(f"{len(files)} images in {folder}")
        if files:
            self.sheet.setCurrentRow(0)

    def _thumb_ready(self, item, img):
        try:
            item.setIcon(QIcon(QPixmap.fromImage(to_qimage(img))))
        except RuntimeError:  # item deleted after folder change
            pass

    # ---------------------------------------------------------------- review
    def _open_item(self, item, _prev=None):
        if item is None:
            return
        path = item.data(Qt.UserRole)
        self.current = path
        self.raw = None
        self.status.showMessage(f"Loading {Path(path).name}...")
        task = Task(path, _decode_job, path)
        task.signals.done.connect(self._decoded)
        task.signals.failed.connect(self._failed)
        self.pool.start(task)

    def _decoded(self, path, result):
        if path != self.current:
            return
        self.raw, settings = result
        self.panel.load(settings)
        m = self.raw.metadata
        src = camera.source_profile(self.raw.model, settings.illuminant)
        self.panel.source.setText(src or "LibRaw matrix (no Kodak source profile for this camera)")
        self.info.setText(
            f"{m['make']} {m['model']}\nISO {m['iso']:.0f}  f/{m['aperture']:.1f}  "
            f"1/{1 / m['shutter']:.0f}s  {m['focal_length']:.0f}mm\n{m['timestamp'] or ''}"
            if m["shutter"] else f"{m['make']} {m['model']}")
        self._render()

    def _failed(self, token, message):
        self.status.showMessage(f"Error: {message}")

    def _settings_changed(self):
        if not self.raw:
            return
        s = self.panel.settings()
        src = camera.source_profile(self.raw.model, s.illuminant)
        self.panel.source.setText(src or "LibRaw matrix (no Kodak source profile for this camera)")
        self._render()
        self.save_timer.start()

    def _render(self):
        if not self.raw or not profiles.is_installed():
            return
        self.render_token += 1
        token = (self.current, self.render_token)
        task = Task(token, _render_job, self.raw, self.panel.settings(), "sRGB")
        task.signals.done.connect(self._rendered)
        task.signals.failed.connect(self._failed)
        self.status.showMessage("Rendering...")
        self.pool.start(task)

    def _rendered(self, token, img):
        if token != (self.current, self.render_token):
            return
        self.view.set_image(to_qimage(img))
        self.status.showMessage(Path(self.current).name)

    def _persist(self):
        if self.current and self.raw:
            pipeline.save_settings(self.current, self.panel.settings())

    def _pick_mode(self, on):
        self.view.picking = on
        self.view.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)

    def _picked(self, x, y):
        s = self.panel.settings()
        cam = self.raw.camera_rgb
        h, w = cam.shape[:2]
        # Map the clicked point in the oriented preview back to sensor coordinates.
        turns = ({0: 0, 3: 2, 5: 3, 6: 1}.get(self.raw.flip, 0) + s.rotation) % 4
        for _ in range(turns):
            x, y = y, 1 - x
        cx, cy, r = int(x * w), int(y * h), max(3, w // 200)
        try:
            wb = pipeline.wb_from_patch(cam, max(0, cx - r), max(0, cy - r), cx + r, cy + r)
        except ValueError as e:
            self.status.showMessage(str(e))
            return
        self.panel.wb_custom = wb
        self.panel.pick.setChecked(False)
        self.panel.wb_mode.setCurrentIndex(self.panel.wb_mode.findData("custom"))
        self._settings_changed()

    # ------------------------------------------------------ copy / paste
    def _copy(self):
        if self.raw:
            self.clipboard = self.panel.settings()
            self.status.showMessage("Settings copied")

    def _paste(self):
        if not self.clipboard:
            return
        for item in self.sheet.selectedItems():
            path = item.data(Qt.UserRole)
            s = pipeline.Settings.from_json(self.clipboard.to_json())
            if self.clipboard.wb_mode == "custom":
                s.wb_custom = self.clipboard.wb_custom
            pipeline.save_settings(path, s)
            if path == self.current:
                self.panel.load(s)
                self._render()
        self.status.showMessage(f"Pasted settings to {len(self.sheet.selectedItems())} images")

    def _reset(self):
        if self.raw:
            self.panel.load(pipeline.default_settings(self.raw.make))
            self._settings_changed()

    # -------------------------------------------------------------- export
    def _export(self):
        items = self.sheet.selectedItems()
        if not items:
            return
        if not profiles.is_installed():
            QMessageBox.warning(self, "Save As", "Install the Kodak profiles first.")
            return
        self._persist()
        dlg = ExportDialog(self, self.store)
        if dlg.exec() != QDialog.Accepted:
            return
        opts = dlg.options()
        progress = QProgressDialog("Saving...", "Cancel", 0, len(items), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        state = {"done": 0, "errors": []}

        def finished(path, _result=None):
            state["done"] += 1
            progress.setValue(state["done"])

        def failed(path, msg):
            state["errors"].append(f"{Path(path).name}: {msg}")
            finished(path)

        export_pool = QThreadPool(self)
        export_pool.setMaxThreadCount(2)
        for item in items:
            path = item.data(Qt.UserRole)
            raw_make, _ = camera.read_make_model(path)
            task = Task(path, _export_job, path, pipeline.load_settings(path, raw_make), opts)
            task.signals.done.connect(finished)
            task.signals.failed.connect(failed)
            export_pool.start(task)
        while state["done"] < len(items) and not progress.wasCanceled():
            QApplication.processEvents()
            export_pool.waitForDone(50)
        if progress.wasCanceled():
            export_pool.clear()
            export_pool.waitForDone()
        progress.close()
        if state["errors"]:
            QMessageBox.warning(self, "Save As", "\n".join(state["errors"]))
        else:
            self.status.showMessage(f"Saved {state['done']} images to {opts['out_dir']}")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("PhotoDesk")
    w = MainWindow()
    if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir():
        w.open_folder(sys.argv[1])
    w.show()
    return app.exec()
