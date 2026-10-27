"""PhotoDesk (Tribute) main window (after Kodak DCS Photo Desk 4.3): MDI contact sheets and full image renderings.

Menus, toolbar, accelerators, panels, dialogs, bitmaps, cursors and strings
come from the original PhotoDsk.exe resources (parsed, never executed) and the
official translation table, so the interface matches Photo Desk 4.3.
"""
import functools
import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QEvent, QFileSystemWatcher, QItemSelectionModel, QPoint, QRect, QSettings, QSize, Qt, QThreadPool, QTimer, QUrl
from PySide6.QtGui import QAction, QActionGroup, QColor, QGuiApplication, QIcon, QImage, QKeySequence, QPalette, QPixmap
from PySide6.QtWidgets import (
    QApplication, QDockWidget, QFileDialog, QInputDialog, QLabel, QMainWindow, QMdiArea, QMdiSubWindow, QMenu,
    QMessageBox, QStyleFactory, QToolBar, QToolButton,
)

from .. import APP_NAME, ORG_NAME, __version__, camera, export, pipeline, profiles, wbfile
from . import dialogs as D
from . import res as R
from .model import PASTE_GROUPS, Sheet, is_image, paste_into
from .panels import CROP_ITEMS, CloseUpWindow, HistogramPanel, ImageControlsPanel, ImageInfoPanel
from .views import ContactSheetView, ImageView, Task, ThumbIcons, qimage_array, to_qimage

PREVIEW_EDGE = 1400            # long edge of the "faster" full-image preview
THUMB_EDGE = 256
ZOOMS = {32842: 25, 32853: 50, 32852: 100, 32841: 200}
SHEET_ZOOMS = (50, 100, 200)      # contact sheets: 50/100/200 % (25 % is for full image renderings)
TOOLS = {32847: "select", 32848: "crop", 32846: "click", 33178: "closeup", 33177: "hand"}
# Edits that change the balance method: the White Point follows the new balance's temperature.
BALANCE_KINDS = ("balance", "click_balance", "lighting")
BALANCE_CMD = {32812: "preset", 32816: "click", 32814: "custom", 32815: "auto"}
LIGHTING_CMD = dict(zip((32787, 32788, 32789, 32790), pipeline.LIGHTINGS))
CROP_CMD = dict(zip(range(33110, 33118), CROP_ITEMS))
LOOK_BASE = 45000


def apply_appearance(app, appearance):
    style = QStyleFactory.create("Fusion")
    if style:
        app.setStyle(style)
    palette = app.style().standardPalette()
    if appearance == "dark":
        colors = {
            QPalette.ColorRole.Window: QColor(53, 53, 53),
            QPalette.ColorRole.WindowText: QColor(235, 235, 235),
            QPalette.ColorRole.Base: QColor(35, 35, 35),
            QPalette.ColorRole.AlternateBase: QColor(53, 53, 53),
            QPalette.ColorRole.ToolTipBase: QColor(245, 245, 220),
            QPalette.ColorRole.ToolTipText: QColor(20, 20, 20),
            QPalette.ColorRole.Text: QColor(235, 235, 235),
            QPalette.ColorRole.Button: QColor(53, 53, 53),
            QPalette.ColorRole.ButtonText: QColor(235, 235, 235),
            QPalette.ColorRole.BrightText: QColor(255, 80, 80),
            QPalette.ColorRole.Highlight: QColor(42, 130, 218),
            QPalette.ColorRole.HighlightedText: QColor(255, 255, 255),
            QPalette.ColorRole.PlaceholderText: QColor(150, 150, 150),
        }
        for role, color in colors.items():
            palette.setColor(role, color)
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(130, 130, 130))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(130, 130, 130))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(130, 130, 130))
    app.setPalette(palette)


def _dark_icon(icon):
    pixmap = icon.pixmap(32, 32)
    image = pixmap.toImage().convertToFormat(QImage.Format.Format_ARGB32)
    for y in range(image.height()):
        for x in range(image.width()):
            color = image.pixelColor(x, y)
            if color.alpha() and max(color.red(), color.green(), color.blue()) < 150 and \
                    max(color.red(), color.green(), color.blue()) - min(color.red(), color.green(), color.blue()) < 32:
                value = 170 + round(max(color.red(), color.green(), color.blue()) * 0.25)
                image.setPixelColor(x, y, QColor(value, value, value, color.alpha()))
    return QIcon(QPixmap.fromImage(image))
# Left out of the menus and toolbar: Help Topics (57667) and the context help button (57669), as
# no help is shipped, and the Kodak web links of the Help menu (33158-33167). The Help menu keeps
# only About Photo Desk.
NO_HELP = (57667, 57669) + tuple(range(33158, 33168))
# The Korean table translates About Photo Desk as "포토 데스크 도움말" (help); it opens the program information.
LABEL_OVERRIDES = {("Korean", 57664): "프로그램 정보(&A)..."}
MENU_CONTEXTS = ("File Menu", "Edit Menu", "View Menu", "Image Menu", "Info Menu", "Window Menu", "Help Menu")


def build_menu(parent, res, items, handlers, context=R.MENU_CONTEXT):
    """QMenu from resource menu items; handlers maps command id -> callable or QAction."""
    menu = QMenu(parent)
    for it in items:
        if it[0] == "sep":
            menu.addSeparator()
        elif it[0] == "popup":
            sub = build_menu(menu, res, it[2], handlers, context)
            sub.setTitle(res.tr(it[1], context))
            menu.addMenu(sub)
        else:
            h = handlers.get(it[1])
            if isinstance(h, QAction):
                menu.addAction(h)
            else:
                act = menu.addAction(res.tr(it[2], context).split("\t")[0])
                act.setEnabled(h is not None)
                if h is not None:
                    act.triggered.connect(lambda _=False, f=h: f())
    return menu


# ------------------------------------------------------------------ worker jobs
def _thumb_job(path, settings, pdir, raw_file):
    if not raw_file:
        img, _ = export.load_finished(path, THUMB_EDGE)
        img = pipeline.resize_long_edge(img, THUMB_EDGE)
        return None, None, pipeline.apply_crop(pipeline.orient(img, 0, settings.rotation), None)
    raw = pipeline.decode(path, half_size=True)
    small = pipeline.reduced(raw, THUMB_EDGE)
    img = pipeline.render(small, settings.copy(crop=None), profile_directory=pdir, quality="faster")
    return small, raw.metadata, img


def _finished_job(path, rotation):
    img, _ = export.load_finished(path)
    return pipeline.orient(img, 0, rotation)


def _render_job(raw, settings, pdir, quality, edge, space):
    src = pipeline.reduced(raw, edge) if quality == "faster" else raw
    return pipeline.render(src, settings.copy(crop=None), space, pdir, quality=quality)


def _decode_job(path):
    return pipeline.decode(path, half_size=True)


def _region_job(raw, settings, rect, pdir, space):
    return pipeline.render_region(raw, settings, rect, space, pdir)


def _export_job(path, dest, file_type, opts, settings, pdir):
    return export.process_file(path, dest, file_type, opts, settings, pdir)


# ------------------------------------------------------------------ MDI children
class DocumentSubWindow(QMdiSubWindow):
    def __init__(self):
        super().__init__()
        self.floating = False

    def event(self, event):
        handled = super().event(event)
        if event.type() == QEvent.Type.WindowActivate and self.floating:
            self.main._document_activated(self)
        return handled


class SheetWindow(DocumentSubWindow):
    """A contact sheet. A standalone sheet is never shown: it holds an image opened by itself
    (File > Open on a file) and goes away with that image's window."""

    def __init__(self, main, sheet, standalone=False):
        super().__init__()
        self.main, self.sheet, self.standalone = main, sheet, standalone
        self.view = ContactSheetView(main.icons)
        z = main.q.value("LastThumbnailViewSize", 100, int)
        self.view.set_zoom(z if z in SHEET_ZOOMS else 100)
        self.view.set_show_info(main.q.value("Thumbnail Info State", True, bool))
        self.setWidget(self.view)
        self.setWindowTitle(str(sheet.folder))
        self.setWindowIcon(main.res.app_icon())
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.view.populate(sheet.entries)
        self.view.itemSelectionChanged.connect(main.selection_changed)
        self.view.currentItemChanged.connect(lambda *_: main.selection_changed())
        # Double-clicking a thumbnail shows it in the sheet's floating image window.
        self.view.itemDoubleClicked.connect(lambda it: main.open_image(it.data(Qt.UserRole), self))
        self.viewer = None
        self.view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.view.customContextMenuRequested.connect(
            lambda pos: main.context_menu(194, self.view.viewport().mapToGlobal(pos)))
        self.view.viewport().installEventFilter(self)
        self.view.files_dropped.connect(lambda paths, pos, mode: main.drop_files(self, paths, pos, mode))
        self.images = []
        # Files copied into the folder by other programs (e.g. Camera Manager) appear automatically.
        self.watcher = QFileSystemWatcher([str(sheet.folder)], self)
        self._reload_timer = QTimer(self, singleShot=True, interval=500, timeout=self.reload)
        self.watcher.directoryChanged.connect(lambda _: self._reload_timer.start())

    def reload(self, replaced=()):
        """Pick up files added or removed in the folder, keeping the selection; ``replaced`` files
        were overwritten in place and get new thumbnails."""
        before = [e.path for e in self.view.entries()]
        self.sheet.refresh()
        replaced = {Path(p) for p in replaced}
        for e in self.sheet.entries:
            if e.path in replaced:
                e.thumb = e.small = None
                e.settings = pipeline.load_settings(e.path)
        if [e.path for e in self.sheet.entries] == before and not replaced:
            return
        selected = {id(e) for e in self.view.selected_entries()}
        current = self.view.focused_entry()
        self.view.blockSignals(True)
        self.view.populate(self.sheet.entries)
        for i, e in enumerate(self.sheet.entries):
            if id(e) in selected:
                self.view.item(i).setSelected(True)
            if e is current:
                self.view.setCurrentRow(i, QItemSelectionModel.NoUpdate)
        self.view.blockSignals(False)
        for e in self.sheet.entries:
            if e.thumb is None:
                self.main._load_thumb(self, e)
        self.main.selection_changed()

    def eventFilter(self, obj, ev):
        t = ev.type()
        if self.main.tool == "hand":            # Scrolling Hand: drag the contact sheet
            bar = self.view.verticalScrollBar()
            if t == ev.Type.MouseButtonPress and ev.button() == Qt.LeftButton:
                self._hand = (ev.position().y(), bar.value())
                return True
            if t == ev.Type.MouseMove and getattr(self, "_hand", None):
                bar.setValue(int(self._hand[1] - (ev.position().y() - self._hand[0])))
                return True
            if t == ev.Type.MouseButtonRelease and getattr(self, "_hand", None):
                self._hand = None
                return True
        if t == ev.Type.MouseButtonPress and ev.button() == Qt.LeftButton:
            tool = self.main.tool
            if tool in ("click", "closeup"):
                hit = self.view.thumb_at(ev.position().toPoint())
                if hit:
                    e, x, y = hit
                    if tool == "click":
                        self.main.click_balance(e, x, y, thumb=True)
                    else:
                        self.main.show_closeup(e, x, y)
                    return True
        return False

    def closeEvent(self, ev):
        if not self.main._closing_documents and not self.main.confirm_save(self.sheet.dirty()):
            ev.ignore()
            return
        for w in list(self.images):
            w.close()
        if self in self.main.sheets:
            self.main.sheets.remove(self)
        self.main._document_closed(self)
        super().closeEvent(ev)


class ImageWindow(DocumentSubWindow):
    """Full image rendering of one entry."""

    def __init__(self, main, sheet_win, entry):
        super().__init__()
        self.main, self.sheet_win, self.entry = main, sheet_win, entry
        cursors = main.cursors
        self.view = ImageView(cursors)
        self.view.set_tool(main.tool)
        self.setWidget(self.view)
        self.setWindowTitle(entry.path.name)
        self.setWindowIcon(main.res.app_icon())
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.view.clicked.connect(self._clicked)
        self.view.crop_changed.connect(lambda c: main.set_crop([self.entry], c))
        self.view.hover.connect(main.show_pixel)
        self.view.crop_size.connect(main.show_size)
        self.view.zoom_changed.connect(lambda z: main.show_zoom(z) if main.active_image() is self else None)
        self.view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.view.customContextMenuRequested.connect(lambda pos: main.context_menu(195, self.view.mapToGlobal(pos)))
        self.token = 0
        self.latest = [0]               # newest render token, read by render tasks to skip stale work
        self.rgb = None
        sheet_win.images.append(self)

    def set_entry(self, entry):
        """Show another image of the same contact sheet in this window."""
        old = self.entry
        self.entry = entry
        self.setWindowTitle(entry.path.name)
        self.rgb = None
        self.token += 1
        self.latest[0] = self.token
        if not any(w.entry is old for w in self.sheet_win.images):
            old.raw = None

    def _clicked(self, x, y):
        tool = self.main.tool
        if tool == "click":
            self.main.click_balance(self.entry, x, y)
        elif tool == "closeup":
            self.main.show_closeup(self.entry, x, y)

    def closeEvent(self, ev):
        sw = self.sheet_win
        if sw.standalone and not self.main._closing_documents:
            # No contact sheet behind this image: ask to save its changes here.
            if not self.main.confirm_save(sw.sheet.dirty()):
                ev.ignore()
                return
            if sw in self.main.sheets:
                self.main.sheets.remove(sw)
            sw.deleteLater()
        elif (not self.main._closing_documents and self.entry.dirty
              and not any(w is not self and w.entry is self.entry for w in sw.images)):
            # Full image rendering opened from a contact sheet: ask to save this image's changes.
            if not self.main.confirm_save([self.entry]):
                ev.ignore()
                return
        if self in sw.images:
            sw.images.remove(self)
        if sw.viewer is self:
            sw.viewer = None
            self.main.q.setValue("Image Viewer Geometry", self.geometry())
        if not any(w.entry is self.entry for w in sw.images):
            self.entry.raw = None
        self.main._document_closed(self)
        super().closeEvent(ev)


# ------------------------------------------------------------------ main window
class MainWindow(QMainWindow):
    def __init__(self, res):
        super().__init__()
        self.res = res
        # PHOTODESK_SETTINGS: an .ini file instead of the user's settings (used by the GUI tests).
        ini = os.environ.get("PHOTODESK_SETTINGS")
        self.q = QSettings(ini, QSettings.IniFormat) if ini else QSettings(ORG_NAME, APP_NAME)
        self.appearance = self.q.value("Appearance", "light")
        apply_appearance(QApplication.instance(), self.appearance)
        camera.set_unlock_all(self.q.value("Unlock All Camera Features", True, bool))
        self._light_icons = {}
        self.pdir = str(profiles.profile_dir())
        # Interactive renders (image windows, close-ups, review) use the global pool; contact sheet
        # thumbnails get their own pool so a large folder never queues ahead of what the user is viewing.
        # Each render already uses every core (icc.trilinear), so a few workers are enough.
        self.pool = QThreadPool.globalInstance()
        self.thumb_pool = QThreadPool(self)
        self.thumb_pool.setMaxThreadCount(max(1, min(4, (os.cpu_count() or 2) // 2)))
        # Image window previews: one worker, so while a slider moves only the newest render runs
        # (queued older ones are skipped as stale); each render uses every core itself.
        self.render_pool = QThreadPool(self)
        self.render_pool.setMaxThreadCount(1)
        self.icons = ThumbIcons(res)
        self.cursors = {"click": res.cursor(148), "hand": res.cursor(191), "closeup": res.cursor(30978),
                        "crop": Qt.CrossCursor, 221: res.cursor(221), 222: res.cursor(222),
                        223: res.cursor(223), 224: res.cursor(224)}
        self.sheets = []
        self.tool = "select"
        self.quality = self.q.value("Review Render Mode", "better")
        self.clipboard = None
        self.clipboard_serial = ""
        self.crop_mode = self.q.value("Crop Mode", "freeform")
        self.crop_other = (self.q.value("Constrained Crop Width X 10000", 40000, int) / 10000,
                           self.q.value("Constrained Crop Height X 10000", 60000, int) / 10000)
        self.opts = export.OutputOptions.from_json(self.q.value("OutputOptions", {}) or {})
        self.preview_override = None
        self.closeups = []
        self._active_window = None
        self._closing_documents = False
        self.player = None
        self.looks = pipeline.available_looks(self.pdir) if profiles.is_installed() else []
        self.icons.look_labels = {name: pipeline.look_label(desc) for name, desc in self.looks}
        self.setWindowTitle(f"{APP_NAME} v{__version__}")
        self.setWindowIcon(res.app_icon())
        self.mdi = QMdiArea()
        self.mdi.setBackground(self.palette().color(QPalette.ColorRole.Dark))
        self.setCentralWidget(self.mdi)
        self.mdi.subWindowActivated.connect(self._mdi_window_activated)
        self._actions()
        self._menus()
        self._toolbar()
        self._panels()
        self._status()
        self._editor_label()
        self.update_commands()
        self.restoreGeometry(self.q.value("MainFrame Position", b""))
        self.restoreState(self.q.value("ControlBar States", b""))
        histogram_size = self.q.value("Histogram Dock Size", QSize(), QSize)
        if histogram_size.isValid():
            self.hist_dock.resize(max(histogram_size.width(), self.hist_dock.minimumWidth()),
                                  max(histogram_size.height(), self.hist_dock.minimumHeight()))
        self._update_appearance_icons()

    # ---------------------------------------------------------- resources
    def _menu_texts(self):
        texts = {}

        def walk(items):
            for it in items:
                if it[0] == "popup":
                    walk(it[2])
                elif it[0] == "item" and it[1] not in NO_HELP:
                    texts.setdefault(it[1], it[2])
        walk(self.res.menu(128))
        return texts

    def _actions(self):
        res = self.res
        texts = self._menu_texts()
        accels = {}
        for flags, key, cmd in res.pe.accelerators(128):
            if not flags & 1:
                continue
            mods = Qt.KeyboardModifier(0)
            if flags & 4:
                mods |= Qt.ShiftModifier
            if flags & 8:
                mods |= Qt.ControlModifier
            if flags & 0x10:
                mods |= Qt.AltModifier
            qkey = _vk_to_qt(key)
            if qkey is not None:
                accels.setdefault(cmd, []).append(QKeySequence(mods.value | qkey))
        handlers = self._handlers()
        self.acts = {}
        bar_icons = {cmd: icon for cmd, icon in res.toolbar() if cmd}
        for cmd, text in texts.items():
            label, _ = R.strip_accel(LABEL_OVERRIDES.get((res.language, cmd)) or res.tr(text, MENU_CONTEXTS))
            act = QAction(label, self)
            prompt, tip = res.tip(cmd)
            if (res.language, cmd) in LABEL_OVERRIDES:
                tip = label.replace("&", "").rstrip(".")
            act.setStatusTip(prompt)
            act.setToolTip(tip or label.replace("&", ""))
            if cmd in accels:
                act.setShortcuts(accels[cmd])
            if cmd in bar_icons:
                act.setIcon(bar_icons[cmd])
            fn = handlers.get(cmd)
            if fn is None:
                act.setEnabled(False)
                act.setData("unimplemented")
            else:
                act.triggered.connect(lambda _=False, f=fn: f())
            self.acts[cmd] = act
            self.addAction(act)
        for group_cmds in (ZOOMS, TOOLS, BALANCE_CMD, LIGHTING_CMD, CROP_CMD, {33129: 0, 33128: 0},
                           {32824: 0, 32825: 0}):
            grp = QActionGroup(self)
            for cmd in group_cmds:
                if cmd in self.acts:
                    self.acts[cmd].setCheckable(True)
                    grp.addAction(self.acts[cmd])
        for cmd in (32836, 32857, 33104, 32840, 59392, 59393, 32855):
            self.acts[cmd].setCheckable(True)
        self.acts[33128 if self.quality == "better" else 33129].setChecked(True)
        self.acts[32824].setChecked(True)
        self.acts[TOOL_CMD["select"]].setChecked(True)
        self.acts[{v: k for k, v in CROP_CMD.items()}.get(self.crop_mode, 33110)].setChecked(True)
        self.acts[32840].setChecked(self.q.value("Thumbnail Info State", True, bool))

    def _handlers(self):
        h = {
            57601: self.file_open, 32868: self.open_selected, 57602: self.close_active, 57603: self.save,
            57604: lambda: self.save_as(selected=False), 32861: lambda: self.save_as(selected=True),
            57607: self.print_, 57609: self.print_preview, 57606: self.print_setup,
            32904: self.choose_editor, 33108: self.send_to_editor, 32905: self.camera_manager,
            33106: self.rename, 32870: self.delete, 57665: self.close,
            57643: self.undo, 57644: self.redo, 57634: self.copy_image, 32865: self.copy_settings,
            32866: self.paste_settings, 33196: self.paste_special, 57642: self.select_all,
            32858: self.deselect_all, 32859: self.select_tagged, 32860: self.select_inverse,
            33109: self.preferences, 33130: self.printing_preferences,
            32824: lambda: self.sort("name"), 32825: lambda: self.sort("date"),
            33179: lambda: self.zoom_step(1), 33180: lambda: self.zoom_step(-1), 32849: self.zoom_fit,
            32844: self.zoom_other, 32836: lambda: self.toggle_dock(self.controls_dock),
            32857: lambda: self.toggle_dock(self.info_dock), 33104: lambda: self.toggle_dock(self.hist_dock),
            32840: self.toggle_thumb_info, 59392: lambda: self.toolbar.setVisible(not self.toolbar.isVisible()),
            59393: self.toggle_status_bar,
            33188: self.review, 33129: lambda: self.set_quality("faster"), 33128: lambda: self.set_quality("better"),
            32838: lambda: self.rotate(1), 32837: lambda: self.rotate(-1), 32806: self.exposure_dialog,
            33187: self.exposure_tone_dialog, 33131: self.auto_exposure, 33152: self.white_point_dialog,
            33183: self.color_balance_dialog, 33197: self.lens_dialog, 33150: self.noise_dialog,
            33149: self.sharpen_dialog, 33118: self.crop_other_dialog, 32807: self.original_settings,
            32803: self.auto_scene, 32867: self.save_balance_file, 32855: lambda: self.tag(True),
            32856: lambda: self.tag(False), 32854: self.iptc_dialog, 1055: self.play_sounds,
            1056: self.stop_sounds, 57650: self.mdi.cascadeSubWindows, 57651: self.mdi.tileSubWindows,
            57649: self.mdi.tileSubWindows, 57664: self.about,
        }
        for cmd, z in ZOOMS.items():
            h[cmd] = functools.partial(self.set_zoom, z)
        for cmd, tool in TOOLS.items():
            h[cmd] = functools.partial(self.set_tool, tool)
        for cmd, bal in BALANCE_CMD.items():
            h[cmd] = functools.partial(self.set_balance, bal)
        for cmd, lt in LIGHTING_CMD.items():
            h[cmd] = functools.partial(self.edit, "lighting", lighting=lt)
        for cmd, key in CROP_CMD.items():
            h[cmd] = functools.partial(self.set_crop_mode, key)
        return h

    def _menus(self):
        res = self.res
        bar = self.menuBar()
        bar.clear()
        self.recent_menu = None
        for top in res.menu(128):
            menu = self._submenu(top[1], top[2])
            if top[1] == "&Window":
                menu.addSeparator()
                labels = ("별도 창으로 분리", "앱 창에 다시 넣기") if res.language == "Korean" else (
                    "Float Active Window", "Move into Main Window")
                self.float_window_action = QAction(labels[0], self)
                self.reattach_window_action = QAction(labels[1], self)
                self.float_window_action.triggered.connect(
                    lambda _=False: self.set_window_floating(self.active_document(), True))
                self.reattach_window_action.triggered.connect(
                    lambda _=False: self.set_window_floating(self.active_document(), False))
                menu.addAction(self.float_window_action)
                menu.addAction(self.reattach_window_action)
            bar.addMenu(menu)

    def _submenu(self, title, items):
        res = self.res
        menu = QMenu(res.tr(title, MENU_CONTEXTS), self)
        items = [it for it in items if not (it[0] == "item" and it[1] in NO_HELP)]
        # No leading, trailing or doubled separators once items are left out.
        items = [it for i, it in enumerate(items)
                 if it[0] != "sep" or (0 < i < len(items) - 1 and items[i - 1][0] != "sep")]
        for it in items:
            if it[0] == "sep":
                menu.addSeparator()
            elif it[0] == "popup":
                if any(x[0] == "item" and x[1] == LOOK_BASE for x in it[2]):
                    sub = QMenu(res.tr(it[1], MENU_CONTEXTS), menu)
                    self.look_menu = sub
                    self._fill_looks()
                    menu.addMenu(sub)
                else:
                    menu.addMenu(self._submenu(it[1], it[2]))
            elif it[1] == 57616:
                # MFC ID_FILE_MRU_FILE1: the placeholder item is replaced inline by "&1 path" entries.
                self.recent_menu = menu
                self.recent_anchor = menu.addSeparator()
                self.recent_anchor.setVisible(False)
                self.recent_acts = []
                self._fill_recent()
            else:
                menu.addAction(self.acts[it[1]])
        return menu

    def _fill_looks(self):
        self.look_menu.clear()
        grp = QActionGroup(self)
        self.look_acts = {}
        for name, desc in self.looks:
            act = QAction(pipeline.look_label(desc), self)     # owned by the window, shared by context menus
            self.look_menu.addAction(act)
            act.setCheckable(True)
            grp.addAction(act)
            act.triggered.connect(lambda _=False, n=name: self.edit("look", look=n))
            self.look_acts[name] = act

    def _fill_recent(self):
        if self.recent_menu is None:
            return
        for act in self.recent_acts:
            self.recent_menu.removeAction(act)
        recent = self.q.value("Recent File List", []) or []
        if isinstance(recent, str):
            recent = [recent]
        self.recent_acts = []
        if not recent:
            self.acts[57616].setEnabled(False)
            self.recent_acts.append(self.acts[57616])
        for i, path in enumerate(recent[:4]):
            act = QAction(f"&{i + 1} {path}", self.recent_menu)
            act.triggered.connect(lambda _=False, p=path: self.open_path(p))
            self.recent_acts.append(act)
        for act in self.recent_acts:
            self.recent_menu.insertAction(self.recent_anchor, act)

    def _add_recent(self, path):
        recent = self.q.value("Recent File List", []) or []
        if isinstance(recent, str):
            recent = [recent]
        recent = [str(path)] + [p for p in recent if p != str(path)]
        self.q.setValue("Recent File List", recent[:4])
        self._fill_recent()

    def _toolbar(self):
        tb = self.toolbar = QToolBar(self.res.tr("Toolbar", ("View Menu", "Common")), self)
        tb.setObjectName("Toolbar")
        tb.setIconSize(QSize(22, 22))
        for cmd, _ in self.res.toolbar():
            if cmd in NO_HELP:
                continue
            if cmd == 0:
                tb.addSeparator()
            else:
                tb.addAction(self.acts[cmd])
        self.addToolBar(tb)
        tb.visibilityChanged.connect(lambda v: self.acts[59392].setChecked(v))

    def _panels(self):
        res = self.res
        self.controls = ImageControlsPanel(res, self.acts)
        self.controls.set_looks(self.looks)
        self.controls.command.connect(lambda c: self.acts[c].trigger())
        self.controls.edit.connect(lambda kind, f: self.edit(kind, **f))
        self.controls.crop_mode.connect(self.set_crop_mode)
        self.controls.set_crop_mode(self.crop_mode, self.crop_other)
        self.info = ImageInfoPanel(res)
        self.info.command.connect(lambda c: self.acts[c].trigger())
        self.info.annotation.connect(lambda e, t: self.edit("annotation", entries=[e], annotation=t))
        self.info.play.connect(self._play)
        self.info.stop.connect(self.stop_sounds)
        self.hist = HistogramPanel(res, self)
        self.hist.set_appearance(self.appearance)
        ctx = ("Image Controls Panel", "Image Info Panel", "Histogram Panel", "View Menu")
        self.controls_dock = self._dock(res.tr("Image Controls", ctx), self.controls, Qt.RightDockWidgetArea, 32836)
        self.info_dock = self._dock(res.tr("Image Info", ctx), self.info, Qt.RightDockWidgetArea, 32857)
        self.hist_dock = self._dock(res.s(59956), self.hist, Qt.RightDockWidgetArea, 33104)
        self.hist_dock.setFloating(True)
        self.hist_dock.hide()
        self.tabifyDockWidget(self.controls_dock, self.info_dock)
        self.controls_dock.raise_()

    def _dock(self, title, widget, area, cmd):
        d = QDockWidget(title, self)
        d.setObjectName(title)
        d.setWidget(widget)
        d.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(area, d)
        d.visibilityChanged.connect(lambda v: self.acts[cmd].setChecked(d.isVisible()))
        self.acts[cmd].setChecked(True)
        return d

    def _update_appearance_icons(self):
        targets = [*self.acts.values(), *self.findChildren(QToolButton)]
        for target in targets:
            icon = target.icon()
            if icon.isNull():
                continue
            if target not in self._light_icons:
                self._light_icons[target] = icon
            original = self._light_icons[target]
            target.setIcon(_dark_icon(original) if self.appearance == "dark" else original)

    def set_appearance(self, appearance):
        self.appearance = appearance if appearance in ("light", "dark") else "light"
        self.q.setValue("Appearance", self.appearance)
        apply_appearance(QApplication.instance(), self.appearance)
        self.hist.set_appearance(self.appearance)
        self.mdi.setBackground(self.palette().color(QPalette.ColorRole.Dark))
        self._update_appearance_icons()

    def _status(self):
        sb = self.statusBar()
        self.st_count, self.st_pixel, self.st_xy, self.st_size, self.st_zoom = (QLabel() for _ in range(5))
        fm = sb.fontMetrics()
        for lbl, sample in ((self.st_count, self.res.s(59145)), (self.st_pixel, self.res.s(59143)),
                            (self.st_xy, self.res.s(59144)), (self.st_size, "width: 00000, height: 00000"),
                            (self.st_zoom, self.res.s(59149))):
            lbl.setMinimumWidth(fm.horizontalAdvance(sample) + 8)
            lbl.setFrameShape(QLabel.Panel)
            lbl.setFrameShadow(QLabel.Sunken)
            sb.addPermanentWidget(lbl)
        sb.showMessage(self.res.tr("Ready", ("Common", "Common Strings")))
        visible = self.q.value("Show Status Bar", True, bool)
        sb.setVisible(visible)
        self.acts[59393].setChecked(visible)

    def toggle_status_bar(self):
        visible = self.acts[59393].isChecked()
        self.statusBar().setVisible(visible)
        self.q.setValue("Show Status Bar", visible)

    # ---------------------------------------------------------- state
    def active_sheet(self):
        w = self.active_document()
        if isinstance(w, SheetWindow):
            return w
        if isinstance(w, ImageWindow):
            return w.sheet_win
        return next((sw for sw in self.sheets if not sw.standalone), None)

    def active_image(self):
        w = self.active_document()
        return w if isinstance(w, ImageWindow) else None

    def active_document(self):
        if self._active_window is not None and self.is_open_document(self._active_window):
            return self._active_window
        # currentSubWindow stays set while another top-level window (popup, dialog) has focus.
        return self.mdi.currentSubWindow()

    def is_open_document(self, window):
        if getattr(window, "floating", False):
            return any(window is sheet or window in sheet.images for sheet in self.sheets)
        return window in self.mdi.subWindowList()

    def _mdi_window_activated(self, window):
        if window is not None:
            self._active_window = window
        self.selection_changed()

    def _document_activated(self, window):
        self._active_window = window
        self.selection_changed()

    def _document_closed(self, window):
        if self._active_window is window:
            self._active_window = self.mdi.activeSubWindow()
            if self._active_window is window:
                self._active_window = None
        self.selection_changed()

    def set_window_floating(self, window, floating):
        if window is None or window.floating == floating:
            return
        if floating:
            window._mdi_geometry = window.geometry()
            global_pos = window.mapToGlobal(QPoint(0, 0))
            size = window.size()
            self.mdi.removeSubWindow(window)
            window.setParent(None, Qt.Window)
            window.floating = True
            window.setWindowState(Qt.WindowNoState)
            window.setGeometry(QRect(global_pos, size))
        else:
            window.hide()
            window.floating = False
            self.mdi.addSubWindow(window)
            window.setGeometry(window._mdi_geometry)
        self._active_window = window
        window.show()
        window.raise_()
        window.activateWindow()
        if not floating:
            self.mdi.setActiveSubWindow(window)
        self.selection_changed()

    def selection(self):
        """Selected entries: the image window's entry, or the contact sheet selection."""
        iw = self.active_image()
        if iw:
            return [iw.entry]
        sw = self.active_sheet()
        return sw.view.selected_entries() if sw else []

    def focused(self):
        iw = self.active_image()
        if iw:
            return iw.entry
        sw = self.active_sheet()
        return sw.view.focused_entry() if sw else None

    def selection_changed(self):
        e = self.focused()
        self.controls.set_entry(e)
        self.info.set_entry(e)
        sw = self.active_sheet()
        if sw:
            n = len(sw.sheet.entries)
            self.st_count.setText(self.res.s(59148).replace("%d", str(n), 1)
                                  .replace("%d", str(len(sw.view.selected_entries())), 1))
        else:
            self.st_count.setText("")
        iw = self.active_image()
        if iw:
            self.show_size(None)
            self.show_zoom(iw.view.zoom)
        elif sw:
            self.st_size.setText("")
            self.show_zoom(sw.view.zoom)
        self._update_histogram()
        self.update_commands()

    def update_commands(self):
        sel = self.selection()
        has = bool(sel)
        raw = has and all(e.is_raw for e in sel)
        sheet = self.active_sheet() is not None
        focus = self.focused()
        enable = {
            32868: has, 57602: self.active_document() is not None, 57603: has, 57604: has, 32861: has,
            57607: sheet, 57609: sheet, 33108: has, 33106: has, 32870: has and self.active_image() is None,
            57634: focus is not None, 32865: focus is not None, 32866: has and self.clipboard is not None,
            33196: has and self.clipboard is not None, 57642: sheet, 32858: sheet, 32859: sheet, 32860: sheet,
            32824: sheet, 32825: sheet, 33188: has, 32838: has, 32837: has, 32855: has, 32856: has, 32854: has,
            32848: has, 32846: raw, 33178: has, 32806: raw, 33187: raw,
            33152: raw and all(e.caps.white_point for e in sel), 33183: raw, 33150: raw,
            33149: raw, 32807: raw, 32812: raw, 32816: raw and any(e.settings.click_wb != (1.0, 1.0, 1.0) for e in sel),
            32814: raw and all("custom" in e.caps.balances for e in sel),
            32815: raw and all("auto" in e.caps.balances for e in sel),
            32803: raw and all(e.caps.auto_scene for e in sel), 33131: raw and all(e.caps.auto_scene for e in sel),
            33197: raw and all(e.caps.lens_optimization for e in sel),
            32867: len(sel) == 1 and raw and sel[0].settings.balance == "click",
            1055: focus is not None and bool(focus.sounds), 1056: self.player is not None,
            33179: sheet, 33180: sheet, 32849: self.active_image() is not None, 32844: self.active_image() is not None,
        }
        for cmd in list(LIGHTING_CMD) + [32812]:
            enable[cmd] = raw
        image = self.active_image() is not None
        for cmd, z in ZOOMS.items():
            enable[cmd] = image or (sheet and z in SHEET_ZOOMS)
        for cmd, on in enable.items():
            act = self.acts.get(cmd)
            if act is not None and act.data() != "unimplemented":
                act.setEnabled(bool(on))
        active_window = self.active_document()
        self.float_window_action.setEnabled(active_window is not None and not active_window.floating)
        self.reattach_window_action.setEnabled(active_window is not None and active_window.floating)
        if self.look_menu is not None:
            self.look_menu.setEnabled(raw)
            for name, act in self.look_acts.items():
                act.setEnabled(raw and all(name in dict(e.caps.looks(self.looks)) for e in sel))
        s = self.active_sheet()
        undo_id = s.sheet.undo_text_id() if s else 62091
        redo_id = s.sheet.redo_text_id() if s else 62092
        self.acts[57643].setText(R.strip_accel(self.res.s(undo_id, context=("Edit Menu",)))[0])
        self.acts[57644].setText(R.strip_accel(self.res.s(redo_id, context=("Edit Menu",)))[0])
        self.acts[57643].setEnabled(bool(s and s.sheet.undo_stack))
        self.acts[57644].setEnabled(bool(s and s.sheet.redo_stack))
        if focus is not None:
            st = focus.settings
            self.acts[32855].setChecked(st.tagged)
            if focus.is_raw:
                self.acts[{v: k for k, v in BALANCE_CMD.items()}[st.balance]].setChecked(True)
                self.acts[{v: k for k, v in LIGHTING_CMD.items()}[st.lighting]].setChecked(True)
                if st.look in self.look_acts:
                    self.look_acts[st.look].setChecked(True)

    # ---------------------------------------------------------- files
    def file_open(self):
        start = self.q.value("LastFileOpenDialogPath", str(Path.home()))
        dlg = QFileDialog(self, self.res.s(59501), start)
        dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)
        dlg.setNameFilters([APP_NAME + " (*.dcr *.DCR *.tif *.TIF *.tiff *.jpg *.JPG *.jpeg *.kdc *.KDC)",
                            "All files (*.*)"])
        bar = R.QWidget(dlg)                      # template 130: "Open Folder As Contact Sheet"
        folder_btn = R.build(self.res, bar, 130)[1000]
        folder_btn.setText(self.res.s(59502))
        dlg.layout().addWidget(bar, dlg.layout().rowCount(), 0, 1, -1)

        def as_sheet():
            cur = Path(dlg.directory().absolutePath())
            sel = dlg.selectedFiles()
            if sel and Path(sel[0]).is_dir():
                cur = Path(sel[0])
            dlg.done(2)
            self.open_path(cur)
        folder_btn.clicked.connect(as_sheet)
        dlg.directoryEntered.connect(lambda d: folder_btn.setText(
            self.res.s(59504).replace("%s", Path(d).name or d)))
        if dlg.exec() == QFileDialog.Accepted and dlg.selectedFiles():
            self.open_path(dlg.selectedFiles()[0])
        self.q.setValue("LastFileOpenDialogPath", dlg.directory().absolutePath())

    def open_path(self, path):
        p = Path(path)
        if p.is_dir():
            return self.open_folder(p)
        if p.is_file() and is_image(p):
            # A file opens as a full image rendering only (no contact sheet of its folder); if its
            # folder is already open as a contact sheet, the image belongs to that sheet.
            for sw in self.sheets:
                entry = sw.sheet.find(p)
                if entry and (not sw.standalone or sw.sheet.files == {p}):
                    self.open_image(entry, sw)
                    return sw
            sw = SheetWindow(self, Sheet(p.parent, files=[p]), standalone=True)
            self.sheets.append(sw)
            self._add_recent(p)
            self._load_thumb(sw, sw.sheet.entries[0])
            self.open_image(sw.sheet.entries[0], sw)
            return sw
        D.message(self, self.res, self.res.s(60000).replace("%s", str(p)))

    def open_folder(self, folder):
        folder = Path(folder)
        for sw in self.sheets:
            if sw.sheet.folder == folder and not sw.standalone:
                if sw.floating:
                    self._document_activated(sw)
                    sw.raise_()
                    sw.activateWindow()
                else:
                    self.mdi.setActiveSubWindow(sw)
                return sw
        sheet = Sheet(folder)
        if self.q.value("Last Thumbnail Sort Order", "name") == "date":
            sheet.sort_by("date")
            self.acts[32825].setChecked(True)
        sw = SheetWindow(self, sheet)
        self.sheets.append(sw)
        self.mdi.addSubWindow(sw)
        sw.showMaximized() if not self.mdi.subWindowList()[:-1] else sw.show()
        self._add_recent(folder)
        for e in sheet.entries:
            self._load_thumb(sw, e)
        self.selection_changed()
        return sw

    def _load_thumb(self, sw, e):
        task = Task(e, _thumb_job, e.path, e.settings, self.pdir, e.is_raw)

        def done(entry, result):
            entry.small, entry.meta, img = result
            entry.thumb = to_qimage(img)
            s = entry.settings
            if entry.small is not None and s.white_point == s.white_point_ref:
                # Show the loaded balance's temperature (the render is the same: no edit, not dirty).
                s.white_point = s.white_point_ref = pipeline.balance_temperature(entry.small, s)
            sw.view.refresh_entry(entry)
            if entry is self.focused():
                self.info.set_entry(entry)
                self.controls.set_entry(entry)
                self._update_histogram()
        task.signals.done.connect(done)
        task.signals.failed.connect(lambda entry, msg: self.statusBar().showMessage(
            self.res.s(60000).replace("%s", entry.path.name) + f"  ({msg})"))
        self.thumb_pool.start(task)

    def refresh_thumb(self, entry):
        """Re-render a thumbnail after its settings changed (in the background; the newest edit wins)."""
        for sw in self.sheets:
            if entry in sw.sheet.entries:
                if entry.small is not None:
                    entry.thumb_gen += 1
                    gen = entry.thumb_gen
                    task = Task((entry, gen, sw), pipeline.render, entry.small,
                                entry.settings.copy(crop=None), pipeline.DISPLAY_SPACE, self.pdir, "faster",
                                stale=lambda e=entry, g=gen: e.thumb_gen != g)
                    task.signals.done.connect(self._thumb_rendered)
                    self.thumb_pool.start(task, 1)          # ahead of first-time thumbnail loads
                elif not entry.is_raw and entry.thumb is not None:
                    self._load_thumb(sw, entry)
                sw.view.refresh_entry(entry)

    def _thumb_rendered(self, token, img):
        entry, gen, sw = token
        if gen != entry.thumb_gen:
            return
        entry.thumb = to_qimage(img)
        if sw in self.sheets:
            sw.view.refresh_entry(entry)
        if entry is self.focused():
            self._update_histogram()

    def open_selected(self):
        sw = self.active_sheet()
        sel = self.selection()
        for i, e in enumerate(sel):
            if i == 0 or sw.standalone:
                self.open_image(e, sw)
            elif not any(w.entry is e for w in sw.images):
                # More than one image: the others get floating windows of their own.
                w = ImageWindow(self, sw, e)
                self._float_new(w, offset=24 * i)
                self._load_render(w)

    def _activate_image(self, w):
        if w.floating:
            self._document_activated(w)
            w.raise_()
            w.activateWindow()
        else:
            self.mdi.setActiveSubWindow(w)

    def _float_new(self, w, offset=0):
        """Show a new image window as a floating top-level window that can leave the main frame."""
        w.floating = True
        w.setParent(None, Qt.Window)
        w._mdi_geometry = QRect(0, 0, self.mdi.width() * 2 // 3, self.mdi.height() * 2 // 3)
        geo = self.q.value("Image Viewer Geometry")
        if isinstance(geo, QRect) and geo.isValid():
            w.setGeometry(geo.translated(offset, offset))
        else:
            top_left = self.mdi.mapToGlobal(QPoint(0, 0))
            w.setGeometry(QRect(top_left + QPoint(40 + offset, 40 + offset),
                                QSize(max(480, self.mdi.width() * 2 // 3), max(360, self.mdi.height() * 2 // 3))))
        w.show()
        self._active_window = w
        w.raise_()
        w.activateWindow()

    def _show_in_viewer(self, sheet_win, entry):
        """Show ``entry`` in the contact sheet's floating image window (reused for every image)."""
        w = sheet_win.viewer
        if w is not None and w not in sheet_win.images:
            w = sheet_win.viewer = None
        if w is None:
            w = sheet_win.viewer = ImageWindow(self, sheet_win, entry)
            self._float_new(w)
        else:
            if w.entry is not entry:
                if (w.entry.dirty and not any(o is not w and o.entry is w.entry for o in sheet_win.images)
                        and not self.confirm_save([w.entry])):
                    return w
                w.set_entry(entry)
            if w.isMinimized():
                w.showNormal()
            self._activate_image(w)
            if w.rgb is not None:
                return w
        self._load_render(w)
        self.selection_changed()
        return w

    def open_image(self, entry, sheet_win):
        for w in sheet_win.images:
            if w.entry is entry:
                self._activate_image(w)
                return w
        if not sheet_win.standalone:
            return self._show_in_viewer(sheet_win, entry)
        w = ImageWindow(self, sheet_win, entry)
        self.mdi.addSubWindow(w)
        w.resize(self.mdi.width() * 2 // 3, self.mdi.height() * 2 // 3)
        w.show()
        self._load_render(w)
        return w

    def _load_render(self, w):
        entry = w.entry
        if entry.is_raw and entry.raw is None:
            task = Task(entry, _decode_job, entry.path)

            def decoded(e, raw):
                e.meta = raw.metadata
                shown = [iw for iw in w.sheet_win.images if iw.entry is e]
                if not shown:
                    return              # the viewer moved on to another image meanwhile
                e.raw = raw
                for iw in shown:
                    self.render_window(iw)
            task.signals.done.connect(decoded)
            task.signals.failed.connect(lambda e, msg: D.message(self, self.res, self.res.s(60002) + "\n" + msg))
            self.pool.start(task)
        else:
            self.render_window(w)
        return w

    def render_window(self, w, settings=None):
        e = w.entry
        s = settings or e.settings
        w.token += 1
        token = (w, w.token)
        if e.is_raw and e.raw is None:
            return
        if not e.is_raw:
            task = Task(token, _finished_job, e.path, s.rotation)
        else:
            space = pipeline.resolve_space(self.opts.working_space, self.pdir)
            latest = w.latest            # plain list shared with the worker (no Qt object access off-thread)
            latest[0] = w.token
            task = Task(token, _render_job, e.raw, s, self.pdir, self.quality, PREVIEW_EDGE, space,
                        stale=lambda: latest[0] != token[1])
        task.signals.done.connect(self._show_render)
        task.signals.failed.connect(lambda t, msg: D.message(self, self.res, self.res.s(60002) + "\n" + msg))
        self.statusBar().showMessage(self.res.tr("Rendering...", ("Common", "Common Strings")))
        self.render_pool.start(task)

    def _show_render(self, token, img):
        w, n = token
        if n != w.token or not self.is_open_document(w):
            return
        e = w.entry
        meta = e.meta or {}
        fw, fh = meta.get("width") or img.shape[1], meta.get("height") or img.shape[0]
        if img.shape[0] > img.shape[1] != (fh > fw):
            fw, fh = fh, fw
        if e.is_raw and ((e.raw.flip in (5, 6)) + e.settings.rotation) % 2:
            fw, fh = min(fw, fh) if img.shape[1] < img.shape[0] else max(fw, fh), 0
            fh = round(fw * img.shape[0] / img.shape[1])
        else:
            fh = round(fw * img.shape[0] / img.shape[1])
        w.view.set_image(to_qimage(img), (fw, fh), e.settings.crop)
        w.view.ratio = self._crop_ratio()
        w.rgb = w.view.array
        self.statusBar().clearMessage()
        self.selection_changed()

    def close_active(self):
        w = self.active_document()
        if w:
            w.close()

    def confirm_save(self, entries):
        """Ask to save changed settings before closing (Yes / Yes to All / No); False cancels."""
        save_all = False
        read_only = []
        for e in entries:
            if not save_all:
                dlg = D.YesToAllDialog(self.res, self, self.res.s(62151).replace("%s", e.path.name))
                r = dlg.exec()
                if r == D.YesToAllDialog.YES_ALL:
                    save_all = True
                elif r != D.OK:
                    e.dirty = False
                    continue
            if e.is_raw and e.read_only:
                read_only.append(e)
            elif e.is_raw:
                pipeline.save_settings(e.path, e.settings)
            e.dirty = False                 # finished files: changes are dropped (Save As keeps them)
        if read_only:                       # e.g. on a CD: the Save As dialog box appears
            self._save_as_entries(read_only, True)
        return True

    def save(self):
        sw = self.active_sheet()
        if sw:
            read_only = sw.sheet.save(self.selection() or sw.sheet.entries)
            if read_only:
                self._save_as_entries(read_only, True)
            self.statusBar().showMessage(self.res.tr("Saved", ("Common", "Common Strings")), 3000)

    def save_as(self, selected):
        entries = self.selection()
        if entries:
            self._save_as_entries(entries, selected)

    def _save_as_entries(self, entries, selected):
        res = self.res
        title = res.s(62014 if selected or len(entries) > 1 else 62013)
        types = res.s(62008).split("|")
        filters = [f"{types[0]}", f"{types[2]}", f"{types[4]}"]
        start = self.q.value("LastFileSaveAsDialogPath", str(entries[0].path.parent))
        dlg = QFileDialog(self, title, start)
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        dlg.setNameFilters(filters)
        last_type = self.q.value("Save As File Type", 0, int)
        dlg.selectNameFilter(filters[min(last_type, 2)])
        multi = len(entries) > 1
        if multi:
            dlg.setFileMode(QFileDialog.Directory)
            dlg.setOption(QFileDialog.ShowDirsOnly, False)
        else:
            dlg.selectFile(entries[0].path.stem)
        # Rename Setup, Open With, Options... and Save Files In Folder (template 166) below the file chooser.
        setup = R.QWidget(dlg)
        b = R.build(res, setup, 166)
        b[1063].setChecked(False)
        b[1064].setText(self.q.value("Save As Basename", "image"))
        b[1065].setText(str(self.q.value("Save As Num Digits", 4)))
        b[1066].setText(str(self.q.value("Save As Start From", 1)))
        for cid in (1063, 1068, 1064, 1065, 1066, 1067):
            b[cid].setVisible(multi)
        in_folder = []
        if multi:
            def folder_label(d):
                b[1067].setText(res.s(59510).replace("%s", Path(d).name or d))
            folder_label(start)
            dlg.directoryEntered.connect(folder_label)

            def save_in_folder():
                in_folder.append(Path(dlg.directory().absolutePath()))
                dlg.done(QFileDialog.Accepted)
            b[1067].clicked.connect(save_in_folder)
        editor = self.q.value("TIFF Open With", "")
        b[1088].setEnabled(bool(editor))
        b[1088].setText(res.s(59514).replace("%s", Path(editor).stem) if editor else b[1088].text())
        b[1375].clicked.connect(self.preferences)
        dlg.layout().addWidget(setup, dlg.layout().rowCount(), 0, 1, -1)
        if dlg.exec() != QFileDialog.Accepted:
            return
        kind = filters.index(dlg.selectedNameFilter()) if dlg.selectedNameFilter() in filters else 0
        file_type = ("tiff", "jpeg", "asis")[kind]
        self.q.setValue("Save As File Type", kind)
        if in_folder:
            target = in_folder[0]
        elif dlg.selectedFiles():
            target = Path(dlg.selectedFiles()[0])
        else:
            return
        if multi:
            out_dir = target if target.is_dir() else target.parent
            if b[1063].isChecked():
                try:
                    digits, start = int(b[1065].text()), int(b[1066].text())
                except ValueError:
                    D.message(self, res, res.s(60018))
                    return
                names = export.sequence_names(b[1064].text().strip() or "image", digits, start, len(entries))
                self.q.setValue("Save As Basename", b[1064].text().strip())
                self.q.setValue("Save As Num Digits", digits)
                self.q.setValue("Save As Start From", start)
            else:
                names = [e.path.stem for e in entries]
            dests = [export.output_path(e.path, out_dir, file_type, n) for e, n in zip(entries, names)]
        else:
            out_dir = target.parent
            dests = [export.output_path(entries[0].path, out_dir, file_type, target.stem)]
        self.q.setValue("LastFileSaveAsDialogPath", str(out_dir))
        self._export(entries, dests, file_type, open_with=editor if b[1088].isChecked() else None)

    def _export(self, entries, dests, file_type, open_with=None):
        res = self.res
        replace_all = False
        jobs = []
        for e, d in zip(entries, dests):
            if d.exists() and d.resolve() != e.path.resolve() and not replace_all:
                r = D.YesToAllDialog(res, self, res.s(59954).replace("%s", d.name)).exec()
                if r == D.YesToAllDialog.YES_ALL:
                    replace_all = True
                elif r != D.OK:
                    continue
            jobs.append((e, d))
        if not jobs:
            return []
        if file_type == "tiff" and self.opts.tiff_linear and not all(e.is_raw for e, _ in jobs):
            D.message(self, res, res.s(59521), "info")
        progress = D.ProgressDialog(res, self, res.s(59513) if file_type == "tiff" else res.s(62013), len(jobs))
        progress.show()
        state = {"done": 0, "errors": [], "paths": []}
        pool = QThreadPool(self)
        pool.setMaxThreadCount(max(1, min(4, os.cpu_count() or 1)))

        def finished(token, result=None):
            state["done"] += 1
            if result is not None:
                state["paths"].append(result)
            progress.set(state["done"])

        def failed(token, msg):
            state["errors"].append(res.s(60006).replace("%s", token[0].path.name, 1).replace("%s", str(token[1]), 1)
                                   + f"\n{msg}")
            finished(token)
        for e, d in jobs:
            t = Task((e, d), _export_job, e.path, d, file_type, self.opts, e.settings, self.pdir)
            t.signals.done.connect(finished)
            t.signals.failed.connect(failed)
            pool.start(t)
        while state["done"] < len(jobs) and not progress.canceled:
            QApplication.processEvents()
            pool.waitForDone(30)
        if progress.canceled:
            pool.clear()
            pool.waitForDone()
        progress.close()
        if state["errors"]:
            D.message(self, res, "\n\n".join(state["errors"]))
        for sw in self.sheets:
            if any(Path(p).parent == sw.sheet.folder for p in state["paths"]):
                sw.reload()
        if open_with:
            for p in state["paths"]:
                launch(open_with, p)
        return state["paths"]

    def choose_editor(self):
        flt = self.res.s(62039).split("|")
        path, _ = QFileDialog.getOpenFileName(self, self.res.s(59515), "",
                                              f"{flt[0]} ({flt[1]})" if sys.platform == "win32" else "")
        if path:
            self.q.setValue("TIFF Open With", path)
            self.q.setValue("JPEG Open With", path)
            self._editor_label()

    def _editor_label(self):
        """File menu and toolbar show the chosen editor: "Send to <editor>"."""
        editor = self.q.value("TIFF Open With", "")
        if editor:
            text = self.res.s(59516).replace("%s", Path(editor).stem, 1).replace("%s", "", 1)
            self.acts[33108].setText(R.strip_accel(text)[0].rstrip("\t"))

    def send_to_editor(self):
        editor = self.q.value("TIFF Open With", "")
        if not editor:
            self.choose_editor()
            editor = self.q.value("TIFF Open With", "")
            if not editor:
                return
        folder = self.q.value("Send Files To Path", "")
        if not folder:
            D.message(self, self.res, self.res.s(59518))
            return
        if not Path(folder).is_dir():
            D.message(self, self.res, self.res.s(59519))
            return
        entries = self.selection()
        for e in entries:                     # finished JPEG/TIFF files open unmodified
            if not e.is_raw:
                launch(self.q.value("JPEG Open With", editor) if e.file_type == "J" else editor, e.path)
        raws = [e for e in entries if e.is_raw]
        if raws:
            dests = [export.output_path(e.path, folder, "tiff") for e in raws]
            self._export(raws, dests, "tiff", open_with=editor)

    def camera_manager(self):
        """Run KODAK PROFESSIONAL DCS Camera Manager if it is installed (a separate Kodak program)."""
        exe = self.q.value("Camera Manager", "")
        if not exe or not Path(exe).exists():
            flt = self.res.s(62039).split("|")
            exe, _ = QFileDialog.getOpenFileName(self, self.res.tip(32905)[1], "",
                                                 f"{flt[0]} ({flt[1]})" if sys.platform == "win32" else "")
            if not exe:
                return
            self.q.setValue("Camera Manager", exe)
        try:
            subprocess.Popen(["open", "-a", exe] if exe.endswith(".app") else [exe])
        except OSError:
            D.message(self, self.res, self.res.s(60000).replace("%s", exe))

    def rename(self):
        sw = self.active_sheet()
        entries = self.selection()
        if not sw or not entries:
            return
        if any(e.read_only for e in entries):
            D.message(self, self.res, self.res.s(60050))
            return
        dlg = D.RenameDialog(self.res, self, entries, self.q)
        if dlg.exec() != D.QDialog.Accepted:
            return
        many = len(entries) > 1

        def on_error(e, kind):
            sid = {"exists": 60052, "denied": 60050, "failed": 60049}[kind]
            if not many:
                D.message(self, self.res, self.res.s(sid))
                return False
            sid = {60052: 60053, 60050: 60054, 60049: 60055}[sid]
            return D.message(self, self.res, self.res.s(sid), "question",
                             QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes
        sw.sheet.rename(entries, dlg.names, on_error)
        sw.view.populate(sw.sheet.entries)
        for w in sw.images:
            w.setWindowTitle(w.entry.path.name)
        self.selection_changed()

    def delete(self):
        sw = self.active_sheet()
        entries = self.selection()
        if not sw or not entries:
            return
        if D.message(self, self.res, self.res.tip(32870)[0] + "?", "question",
                     QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        from PySide6.QtCore import QFile
        for w in list(sw.images):
            if w.entry in entries:
                w.close()
        gone = sw.sheet.delete(entries, lambda p: QFile.moveToTrash(str(p)))
        if len(gone) < len(entries):
            D.message(self, self.res, self.res.s(60038).replace("%s", str(sw.sheet.folder)))
        sw.view.populate(sw.sheet.entries)
        self.selection_changed()

    def drop_files(self, sw, paths, global_pos, mode="menu"):
        """Files dropped on a contact sheet. mode: copy / move / auto (move on the same drive, copy
        across drives) / menu (right-button drag: drop menu 169 Copy Here / Move Here / Cancel)."""
        paths = [p for p in paths if is_image(Path(p))]
        if not paths:
            return
        if not os.access(sw.sheet.folder, os.W_OK):     # e.g. a folder on a CD
            D.message(self, self.res, self.res.s(60037))
            return
        if mode == "menu":
            menu = QMenu(self)
            chosen = {}
            for it in self.res.menu(169)[0][2]:
                if it[0] == "sep":
                    menu.addSeparator()
                elif it[0] == "item":
                    chosen[menu.addAction(self.res.tr(it[2], ("Drop Menu", "Common")))] = it[1]
            cmd = chosen.get(menu.exec(global_pos))
            if cmd not in (32863, 32862):
                return
        elif mode == "auto":
            same = os.path.splitdrive(str(Path(paths[0]).resolve()))[0].lower() == \
                os.path.splitdrive(str(sw.sheet.folder.resolve()))[0].lower()
            if sys.platform != "win32":
                same = os.stat(paths[0]).st_dev == os.stat(sw.sheet.folder).st_dev
            cmd = 32862 if same else 32863
        else:
            cmd = 32862 if mode == "move" else 32863
        replace_all = False

        def replace(dest):
            nonlocal replace_all
            if replace_all:
                return True
            r = D.YesToAllDialog(self.res, self, self.res.s(59954).replace("%s", dest.name)).exec()
            replace_all = r == D.YesToAllDialog.YES_ALL
            return r in (D.OK, D.YesToAllDialog.YES_ALL)

        existing = {e.path for e in sw.sheet.entries}
        done = []
        try:
            done = sw.sheet.transfer(paths, cmd == 32862, replace)
        except OSError as err:
            D.message(self, self.res, str(err))
        for other in self.sheets:
            if other is not sw and cmd == 32862:
                other.reload()
        sw.reload([p for p in done if p in existing])

    # ---------------------------------------------------------- printing
    def _print_image(self, printer):
        from PySide6.QtGui import QPainter
        p = QPainter(printer)
        iw = self.active_image()
        page = printer.pageLayout().paintRectPixels(printer.resolution())
        if iw and iw.view.image is not None:
            img = iw.view.image
            if iw.entry.settings.crop:
                c = iw.entry.settings.crop
                img = img.copy(round(c[0] * img.width()), round(c[1] * img.height()),
                               round((c[2] - c[0]) * img.width()), round((c[3] - c[1]) * img.height()))
            sz = img.size().scaled(page.size(), Qt.KeepAspectRatio)
            p.drawImage(R.QRect(0, 0, sz.width(), sz.height()), img)
        else:
            sw = self.active_sheet()
            if sw is None:
                p.end()
                return
            size = self.q.value("Contact Sheet Printing Size", "view")
            per_short_side = {"small": 10, "medium": 4, "large": 2}.get(size)
            view_cell = int(sw.view.cell_size().width() * printer.resolution() / 96)
            print_contact_sheet(p, printer, page, sw.sheet.entries, per_short_side, view_cell)
        p.end()

    def _printer(self):
        from PySide6.QtPrintSupport import QPrinter
        if not hasattr(self, "printer"):
            self.printer = QPrinter(QPrinter.HighResolution)
        return self.printer

    def print_(self):
        from PySide6.QtPrintSupport import QPrintDialog
        dlg = QPrintDialog(self._printer(), self)
        if dlg.exec() == QPrintDialog.Accepted:
            self._print_image(self._printer())

    def print_preview(self):
        from PySide6.QtPrintSupport import QPrintPreviewDialog
        dlg = QPrintPreviewDialog(self._printer(), self)
        dlg.paintRequested.connect(self._print_image)
        dlg.exec()

    def print_setup(self):
        from PySide6.QtPrintSupport import QPageSetupDialog
        QPageSetupDialog(self._printer(), self).exec()

    # ---------------------------------------------------------- edit
    def edit(self, kind, entries=None, **fields):
        if kind == "balance_capture":
            return self.balance_at_capture(entries)
        sw = self.active_sheet()
        entries = entries if entries is not None else self.selection()
        if not sw or not entries:
            return
        if kind not in ("tag", "untag", "annotation", "iptc", "rotate", "crop"):
            entries = [e for e in entries if e.is_raw]     # processing options need DCS raw files
        if kind in BALANCE_KINDS:
            def change(e, s):
                for k, v in fields.items():
                    setattr(s, k, v)
                self._set_balance_temperature(e, s)
            changed = sw.sheet.apply(entries, kind, change)
        else:
            changed = sw.sheet.set_fields(entries, kind, **fields)
        self.after_edit(changed)

    def _set_balance_temperature(self, e, s):
        """White Point slider shows the colour temperature of the new balance (and starts from it)."""
        raw = e.raw or e.small
        if raw is None:
            try:
                raw = e.small = pipeline.reduced(pipeline.decode(e.path, half_size=True), THUMB_EDGE)
            except Exception:
                return
        s.white_point = s.white_point_ref = pipeline.balance_temperature(raw, s)

    def after_edit(self, changed):
        for e in changed:
            self.refresh_thumb(e)
            for sw in self.sheets:
                for w in sw.images:
                    if w.entry is e:
                        self.render_window(w)
            for cu in self.closeups:
                if cu.entry is e:
                    self._update_closeup(cu)
        self.selection_changed()

    def undo(self):
        sw = self.active_sheet()
        if sw:
            changed = sw.sheet.undo()
            if any(e.path.parent == sw.sheet.folder and e not in sw.view.entries() for e in changed) or \
                    [e.path for e in sw.view.entries()] != [e.path for e in sw.sheet.entries]:
                sw.view.populate(sw.sheet.entries)      # rename undone: names/order changed
            self.after_edit(changed)

    def redo(self):
        sw = self.active_sheet()
        if sw:
            changed = sw.sheet.redo()
            if [e.path for e in sw.view.entries()] != [e.path for e in sw.sheet.entries]:
                sw.view.populate(sw.sheet.entries)
            self.after_edit(changed)

    def copy_image(self):
        iw = self.active_image()
        e = self.focused()
        if iw and iw.view.image is not None:
            img = iw.view.image
        elif e is not None and e.thumb is not None:
            img = e.thumb
        else:
            return
        c = e.settings.crop if e else None
        if c:
            img = img.copy(round(c[0] * img.width()), round(c[1] * img.height()),
                           round((c[2] - c[0]) * img.width()), round((c[3] - c[1]) * img.height()))
        QGuiApplication.clipboard().setImage(img)

    def copy_settings(self):
        e = self.focused()
        if e:
            self.clipboard = e.settings.copy()
            self.clipboard_serial = e.serial
            self.update_commands()

    def paste_settings(self):
        sw = self.active_sheet()
        if sw and self.clipboard:
            fields = tuple(f for cid, fs in PASTE_GROUPS.items() if cid not in (1350, 1351) for f in fs)
            self.after_edit(sw.sheet.paste(self.selection(), self.clipboard, fields, "paste_special",
                                           self.clipboard_serial))

    def paste_special(self):
        sw = self.active_sheet()
        entries = self.selection()
        if not (sw and self.clipboard and entries):
            return
        originals = {id(e): e.settings for e in entries}

        def preview(fields):
            for e in entries:
                e.settings = originals[id(e)]
                if fields:
                    s = e.settings.copy()
                    paste_into(e, s, self.clipboard, fields, self.clipboard_serial)
                    e.settings = s
            self._preview_refresh(entries)
        dlg = D.PasteSpecialDialog(self.res, self, PASTE_GROUPS, preview)
        ok = dlg.exec() == D.QDialog.Accepted
        fields = dlg.fields()
        for e in entries:
            e.settings = originals[id(e)]
        if ok and fields:
            self.after_edit(sw.sheet.paste(entries, self.clipboard, fields, source_serial=self.clipboard_serial))
        else:
            self._preview_refresh(entries)

    def _preview_refresh(self, entries):
        for e in entries:
            self.refresh_thumb(e)
            for sw in self.sheets:
                for w in sw.images:
                    if w.entry is e:
                        self.render_window(w)

    def select_all(self):
        sw = self.active_sheet()
        if sw:
            sw.view.selectAll()

    def deselect_all(self):
        sw = self.active_sheet()
        if sw:
            sw.view.clearSelection()

    def select_tagged(self):
        sw = self.active_sheet()
        if sw:
            for i in range(sw.view.count()):
                it = sw.view.item(i)
                it.setSelected(it.data(Qt.UserRole).settings.tagged)

    def select_inverse(self):
        sw = self.active_sheet()
        if sw:
            for i in range(sw.view.count()):
                it = sw.view.item(i)
                it.setSelected(not it.isSelected())

    def preferences(self):
        spaces = pipeline.working_spaces(self.pdir)
        dlg = D.PreferencesDialog(self.res, self, self.opts, self.q.value("Send Files To Path", ""), spaces,
                                 self.appearance, camera.unlock_all())
        if dlg.exec() == D.QDialog.Accepted:
            self.opts = dlg.opts
            self.q.setValue("OutputOptions", self.opts.to_json())
            self.q.setValue("Send Files To Path", dlg.send_folder)
            if dlg.appearance != self.appearance:
                self.set_appearance(dlg.appearance)
            if dlg.unlock_all != camera.unlock_all():
                self.set_unlock_all(dlg.unlock_all)
            for sw in self.sheets:
                for w in sw.images:
                    self.render_window(w)

    def set_unlock_all(self, on):
        """Lift (or restore) the per-camera-model limits on processing options."""
        camera.set_unlock_all(on)
        self.q.setValue("Unlock All Camera Features", bool(on))
        for sw in self.sheets:
            for e in sw.sheet.entries:
                if e.is_raw:
                    self.refresh_thumb(e)      # lens optimization etc. now (no longer) apply
            sw.view.viewport().update()
        self.selection_changed()

    def printing_preferences(self):
        dlg = D.PrintingPreferencesDialog(self.res, self, self.q.value("Contact Sheet Printing Size", "view"),
                                          self.q.value("Allow Docking", True, bool))
        if dlg.exec() == D.QDialog.Accepted:
            self.q.setValue("Contact Sheet Printing Size", dlg.size)
            self.q.setValue("Allow Docking", dlg.docking)
            for d in (self.controls_dock, self.info_dock, self.hist_dock):
                d.setAllowedAreas(Qt.AllDockWidgetAreas if dlg.docking else Qt.NoDockWidgetArea)
                if not dlg.docking:
                    d.setFloating(True)

    # ---------------------------------------------------------- view
    def sort(self, key):
        sw = self.active_sheet()
        if sw:
            sw.sheet.sort_by(key)
            sw.view.populate(sw.sheet.entries)
            self.q.setValue("Last Thumbnail Sort Order", key)

    def set_zoom(self, z):
        iw = self.active_image()
        if iw:
            iw.view.set_zoom(z)
        elif self.active_sheet() and z in SHEET_ZOOMS:
            self.active_sheet().view.set_zoom(z)
            self.q.setValue("LastThumbnailViewSize", z)
        else:
            return
        self.show_zoom(z)

    def zoom_step(self, d):
        iw = self.active_image()
        steps = [6, 12, 25, 33, 50, 67, 100, 200, 300, 400, 800, 1600]
        if iw:
            cur = iw.view.zoom
            nxt = [s for s in steps if s > cur] if d > 0 else [s for s in steps if s < cur]
            if nxt:
                self.set_zoom(nxt[0] if d > 0 else nxt[-1])
        elif self.active_sheet():
            cur = self.active_sheet().view.zoom
            i = min(max(0, SHEET_ZOOMS.index(cur) + d if cur in SHEET_ZOOMS else 1), len(SHEET_ZOOMS) - 1)
            self.set_zoom(SHEET_ZOOMS[i])

    def zoom_fit(self):
        iw = self.active_image()
        if iw:
            iw.view.set_zoom(fit=True)

    def zoom_other(self):
        iw = self.active_image()
        if iw:
            dlg = D.ScaleViewDialog(self.res, self, iw.view.zoom)
            if dlg.exec() == D.QDialog.Accepted:
                iw.view.set_zoom(dlg.zoom)

    def show_size(self, crop_px=None):
        """Width/height pane: the crop rectangle in Crop mode, otherwise the image size."""
        iw = self.active_image()
        if iw is None:
            return
        if crop_px is None and self.tool == "crop":
            crop_px = iw.view.crop_pixels()
        w, h = crop_px if crop_px else iw.view.full_size
        self.st_size.setText(self.res.s(59151).replace("%d", str(w), 1).replace("%d", str(h), 1))

    def show_zoom(self, z):
        self.st_zoom.setText(self.res.s(59150).replace("%d%%", f"{z}%"))
        for cmd, zz in ZOOMS.items():
            if zz == z:
                self.acts[cmd].setChecked(True)

    def show_pixel(self, info):
        if info is None:
            self.st_pixel.setText("")
            self.st_xy.setText("")
            return
        x, y, (r, g, b) = info
        self.st_pixel.setText(self.res.s(59147).replace("%d", str(r), 1).replace("%d", str(g), 1)
                              .replace("%d", str(b), 1))
        self.st_xy.setText(self.res.s(59146).replace("%d", str(x), 1).replace("%d", str(y), 1))

    def toggle_dock(self, dock):
        dock.setVisible(not dock.isVisible())
        if dock.isVisible():
            self._place_floating(dock)
            dock.raise_()
            if dock is self.hist_dock:
                self._update_histogram()

    def _place_floating(self, dock):
        """A floating panel with no remembered position goes inside the frame's top-right, not the screen origin."""
        if dock.isFloating() and dock.isVisible() and not self.frameGeometry().intersects(dock.frameGeometry()):
            g = self.geometry()
            dock.move(g.right() - dock.width() - 20, g.top() + 80)

    def showEvent(self, event):
        super().showEvent(event)
        for d in (self.controls_dock, self.info_dock, self.hist_dock):
            self._place_floating(d)

    def toggle_thumb_info(self):
        on = self.acts[32840].isChecked()
        self.q.setValue("Thumbnail Info State", on)
        for sw in self.sheets:
            sw.view.set_show_info(on)

    def set_quality(self, q):
        self.quality = q
        self.q.setValue("Review Render Mode", q)
        for sw in self.sheets:
            for w in sw.images:
                self.render_window(w)

    def review(self):
        entries = self.selection()
        if not entries:
            return

        def render(entry, quality, size, callback):
            if not entry.is_raw:
                img, _ = export.load_finished(entry.path, max(size.width(), size.height()))
                callback(entry, to_qimage(pipeline.apply_crop(pipeline.orient(img, 0, entry.settings.rotation),
                                                              entry.settings.crop)))
                return
            if quality == "faster" and entry.small is not None:
                callback(entry, to_qimage(pipeline.render(entry.small, entry.settings, profile_directory=self.pdir,
                                                          quality="faster")))
                return

            def go():
                raw = entry.raw or pipeline.decode(entry.path, half_size=True)
                return pipeline.render(pipeline.reduced(raw, max(size.width(), size.height())), entry.settings,
                                       profile_directory=self.pdir, quality="better")
            t = Task(entry, go)
            t.signals.done.connect(lambda e, img: callback(e, to_qimage(img)))
            self.pool.start(t)
        dlg = D.ReviewDialog(self.res, self, entries, render,
                             lambda e, on: self.edit("tag" if on else "untag", [e], tagged=on), self.quality)
        dlg.exec()

    def _update_histogram(self):
        if not self.hist_dock.isVisible():
            return
        iw = self.active_image()
        if iw and iw.rgb is not None:
            img = iw.rgb
            c = iw.entry.settings.crop
            if c:
                img = pipeline.apply_crop(img, c)
            self.hist.set_image(img)
            return
        e = self.focused()
        self.hist.set_image(qimage_array(e.thumb) if e is not None and e.thumb is not None else None)

    # ---------------------------------------------------------- tools
    def set_tool(self, tool):
        self.tool = tool
        self.acts[TOOL_CMD[tool]].setChecked(True)
        self.controls.set_tool(TOOL_CMD[tool])
        for sw in self.sheets:
            sw.view.viewport().setCursor(self.cursors.get(tool, Qt.ArrowCursor) if tool in ("click", "closeup", "hand")
                                         else Qt.ArrowCursor)
            for w in sw.images:
                w.view.set_tool(tool)
        self.show_size()

    def _crop_ratio(self):
        if self.crop_mode == "other":
            return self.crop_other
        return pipeline.CROP_RATIOS.get(self.crop_mode)

    def set_crop_mode(self, key):
        self.crop_mode = key
        self.q.setValue("Crop Mode", key)
        if key in CROP_ITEMS:
            self.acts[{v: k for k, v in CROP_CMD.items()}[key]].setChecked(True)
        self.controls.set_crop_mode(key, self.crop_other)
        for sw in self.sheets:
            for w in sw.images:
                w.view.ratio = self._crop_ratio()
        self.set_tool("crop")

    def crop_other_dialog(self):
        dlg = D.CropRatioDialog(self.res, self, self.crop_other)
        if dlg.exec() == D.QDialog.Accepted:
            self.crop_other = dlg.ratio
            self.q.setValue("Constrained Crop Width X 10000", int(dlg.ratio[0] * 10000))
            self.q.setValue("Constrained Crop Height X 10000", int(dlg.ratio[1] * 10000))
            self.acts[33118].setText(R.strip_accel(self.res.s(59152).replace("%s", f"{dlg.ratio[0]:g}", 1)
                                                   .replace("%s", f"{dlg.ratio[1]:g}", 1))[0])
            self.set_crop_mode("other")

    def set_crop(self, entries, crop):
        sw = self.active_sheet()
        if sw:
            self.after_edit(sw.sheet.set_fields(entries, "crop", crop=crop))

    def click_balance(self, entry, x, y, thumb=False):
        """Neutralise the clicked point; applied to every selected raw image."""
        if not entry.is_raw:
            return
        raw = entry.raw if entry.raw is not None and not thumb else entry.small
        if raw is None:
            return
        h, w = raw.camera_rgb.shape[:2]
        turns = ({0: 0, 3: 2, 5: 3, 6: 1}.get(raw.flip, 0) + entry.settings.rotation) % 4
        for _ in range(turns):
            x, y = y, 1 - x
        try:
            wb = pipeline.click_wb(raw.camera_rgb, min(w - 1, int(x * w)), min(h - 1, int(y * h)),
                                   radius=max(1, w // 400))
        except pipeline.ClickBalanceError as err:
            D.message(self, self.res, self.res.tr(str(err), ("Pop-Up Message",)))
            return
        # Applied to the selected images captured on the same camera (same serial number) as the
        # clicked image, which may itself be an unselected reference (gray card) shot.
        entries = [e for e in self.selection() if e.is_raw and e.serial == entry.serial] or [entry]
        self.edit("click_balance", entries, balance="click", click_wb=wb)

    def show_closeup(self, entry, x, y):
        cu = next((c for c in self.closeups if not c.pinned), None)
        if cu is None:
            cu = CloseUpWindow(self.res)
            cu.pin.connect(lambda c=cu: self._pin(c))
            cu.show_parent.connect(lambda c=cu: self._closeup_parent(c))
            cu.changed.connect(lambda c=cu: self._update_closeup(c))
            cu.destroyed.connect(lambda _=None, c=cu: self.closeups.remove(c) if c in self.closeups else None)
            cu.setAttribute(Qt.WA_DeleteOnClose)
            self.closeups.append(cu)
        cu.entry = entry
        cu.center = (x, y)
        cu.set_title()
        cu.show()
        cu.raise_()
        if entry.is_raw and entry.raw is None:
            t = Task(entry, _decode_job, entry.path)
            t.signals.done.connect(lambda e, raw: (setattr(e, "raw", raw), setattr(e, "meta", raw.metadata),
                                                   self._update_closeup(cu)))
            self.pool.start(t)
        else:
            self._update_closeup(cu)

    def _pin(self, cu):
        cu.pinned = True
        cu.set_title()

    def _closeup_parent(self, cu):
        for sw in self.sheets:
            if cu.entry in sw.sheet.entries:
                if sw.standalone:
                    self.mdi.setActiveSubWindow(sw.images[0]) if sw.images else None
                    return
                self.mdi.setActiveSubWindow(sw)
                for i in range(sw.view.count()):
                    if sw.view.item(i).data(Qt.UserRole) is cu.entry:
                        sw.view.setCurrentRow(i)

    def _update_closeup(self, cu):
        e = cu.entry
        if e is None:
            return
        if not e.is_raw:
            img, _ = export.load_finished(e.path)
            img = pipeline.orient(img, 0, e.settings.rotation)
            cu.full_size = (img.shape[1], img.shape[0])
            x0, y0, x1, y1 = cu.region(cu.full_size)
            region = pipeline.apply_crop(img, (x0, y0, x1, y1))
        elif e.raw is None:
            return
        else:
            h, w = e.raw.camera_rgb.shape[:2]
            fw, fh = (e.meta or {}).get("width") or w, (e.meta or {}).get("height") or h
            turns = ({0: 0, 3: 2, 5: 3, 6: 1}.get(e.raw.flip, 0) + e.settings.rotation) % 4
            cu.full_size = (fh, fw) if turns % 2 else (fw, fh)
            rect = cu.region(cu.full_size)
            space = pipeline.resolve_space(self.opts.working_space, self.pdir)
            t = Task(cu, _region_job, e.raw, e.settings, rect, self.pdir, space)
            t.signals.done.connect(lambda c, img: self._closeup_ready(c, img))
            self.pool.start(t)
            return
        self._closeup_ready(cu, region)

    def _closeup_ready(self, cu, img):
        if cu not in self.closeups:
            return
        z = cu.zoom() / 100
        region_w = cu.region(cu.full_size)
        tw = max(1, round((region_w[2] - region_w[0]) * cu.full_size[0] * z))
        th = max(1, round((region_w[3] - region_w[1]) * cu.full_size[1] * z))
        q = to_qimage(img).scaled(tw, th, Qt.IgnoreAspectRatio,
                                  Qt.SmoothTransformation if z < 1 else Qt.FastTransformation)
        cu.view.set_image(q)

    # ---------------------------------------------------------- image commands
    def rotate(self, d):
        sw = self.active_sheet()
        entries = self.selection()
        if sw and entries:
            def change(e, s):
                s.rotation = (s.rotation + d) % 4
                s.crop = pipeline.rotate_crop(s.crop, d % 4)
            self.after_edit(sw.sheet.apply(entries, "rotate", change))

    def _raw_selection(self):
        return [e for e in self.selection() if e.is_raw]

    def _adjust(self, dialog_cls, kind, *args, **kw):
        sw = self.active_sheet()
        entries = self._raw_selection()
        if not sw or not entries:
            return
        focus = self.focused() if self.focused() in entries else entries[0]
        originals = {id(e): e.settings for e in entries}
        fields_holder = {}

        def preview(values):
            for e in entries:
                e.settings = originals[id(e)] if values is None else originals[id(e)].copy(**values)
            self._preview_refresh(entries)
        mixed = any(any(getattr(e.settings, f) != getattr(focus.settings, f) for f in dialog_cls.fields)
                    for e in entries) if dialog_cls.fields else False
        capture = pipeline.capture_settings(focus.path)
        dlg = dialog_cls(self.res, self, focus.settings, preview, *args, mixed=mixed, capture=capture, **kw) \
            if dialog_cls is not D.NoiseDialog else dialog_cls(self.res, self, focus.settings, preview, focus.caps,
                                                               mixed=mixed, capture=capture,
                                                               presets=pipeline.noise_presets(focus.path))
        ok = dlg.exec() == D.QDialog.Accepted
        values = dlg.values()
        for e in entries:
            e.settings = originals[id(e)]
        if ok:
            self.edit(kind, entries, **values)
        else:
            self._preview_refresh(entries)

    def exposure_dialog(self):
        e = (self._raw_selection() or [None])[0]
        if e:
            self._adjust(D.ExposureDialog, "exposure", e.caps.exposure_range, lambda: self._auto_ev(e))

    def _auto_ev(self, e):
        raw = e.raw or e.small
        return pipeline.auto_exposure(raw, e.settings) if raw is not None else 0.0

    def exposure_tone_dialog(self):
        e = (self._raw_selection() or [None])[0]
        if not e:
            return

        def hist(ev):
            raw = e.small or e.raw
            if raw is None:
                return None
            return pipeline.ev_histogram(raw, e.settings.copy(exposure=ev), lo=-12, hi=6)
        self._adjust(D.ExposureToneDialog, "exposure_tone", e.caps.exposure_range, lambda: self._auto_ev(e), hist)

    def auto_exposure(self):
        sw = self.active_sheet()
        if sw:
            def change(e, s):
                raw = e.raw or e.small
                if raw is not None:
                    s.exposure = pipeline.auto_exposure(raw, s)
            self.after_edit(sw.sheet.apply(self._raw_selection(), "auto_exposure", change))

    def white_point_dialog(self):
        self._adjust(D.WhitePointDialog, "white_point")

    def color_balance_dialog(self):
        self._adjust(D.ColorBalanceDialog, "color_balance")

    def lens_dialog(self):
        if all(e.caps.lens_optimization for e in self._raw_selection()):
            self._adjust(D.LensDialog, "lens")

    def noise_dialog(self):
        self._adjust(D.NoiseDialog, "noise")

    def sharpen_dialog(self):
        self._adjust(D.SharpenDialog, "sharpening")

    def set_balance(self, bal):
        entries = [e for e in self._raw_selection() if bal in e.caps.balances]
        if bal == "click":
            entries = [e for e in entries if e.settings.click_wb != (1.0, 1.0, 1.0)]
        self.edit("balance", entries, balance=bal)

    def balance_at_capture(self, entries=None):
        """Balance > At Capture: balance method, lighting and white point as the camera set them."""
        sw = self.active_sheet()
        entries = [e for e in (entries if entries is not None else self.selection()) if e.is_raw]
        if not sw or not entries:
            self.selection_changed()
            return

        def change(e, s):
            cap = pipeline.capture_settings(e.path)
            for f in ("balance", "lighting"):
                setattr(s, f, getattr(cap, f))
            self._set_balance_temperature(e, s)
        changed = sw.sheet.apply(entries, "balance", change)
        self.after_edit(changed)
        if not changed:
            self.selection_changed()        # already at capture: show the method again, not "At Capture"

    def original_settings(self):
        sw = self.active_sheet()
        if sw:
            def change(e, s):
                cap = pipeline.capture_settings(e.path)
                for f in ("sharpen_amount", "sharpen_radius", "sharpen_threshold", "balance", "lighting",
                          "color_balance", "lens_optimization"):
                    setattr(s, f, getattr(cap, f))
                self._set_balance_temperature(e, s)
                if e.caps.pro_slr:
                    for f in ("rotation", "look", "noise_type", "noise_radius", "noise_strength", "noise_luma"):
                        setattr(s, f, getattr(cap, f))
            self.after_edit(sw.sheet.apply(self._raw_selection(), "original", change))

    def auto_scene(self):
        sw = self.active_sheet()
        if sw:
            def change(e, s):
                s.balance = "auto"
                self._set_balance_temperature(e, s)
                raw = e.raw or e.small
                if raw is not None:
                    s.exposure = pipeline.auto_exposure(raw, s)
            self.after_edit(sw.sheet.apply([e for e in self._raw_selection() if e.caps.auto_scene],
                                           "auto_scene", change))

    def save_balance_file(self):
        e = self.focused()
        res = self.res
        if e is None or not e.is_raw:
            D.message(self, res, res.s(60030))
            return
        if e.settings.balance != "click":
            D.message(self, res, res.s(60031))
            return
        if not e.caps.balance_file:
            D.message(self, res, res.s(60027))
            return
        flt = res.s(62009).split("|")
        path, _ = QFileDialog.getSaveFileName(self, res.tr("Save Balance File", MENU_CONTEXTS),
                                              str(e.path.parent / res.s(62157)), f"{flt[0]} ({flt[1]})")
        if not path:
            return
        p = Path(path)
        if p.suffix.lower() != ".wb":
            p = p.with_suffix(".wb")
        if len(p.stem) > 8:
            D.message(self, res, res.s(60036).replace("%s", p.name))
            return
        try:
            wbfile.write(p, e.path, e.settings.click_wb)
        except (OSError, wbfile.WBError):
            D.message(self, res, res.s(60026))

    def tag(self, on):
        self.edit("tag" if on else "untag", self.selection(), tagged=on)

    def iptc_dialog(self):
        sw = self.active_sheet()
        entries = self.selection()
        if not sw or not entries:
            return
        dlg = D.IPTCDialog(self.res, self, entries, self.q)
        if dlg.exec() == D.QDialog.Accepted:
            data = {id(e): d for e, d in zip(entries, dlg.data)}
            self.after_edit(sw.sheet.apply(entries, "iptc", lambda e, s: setattr(s, "iptc", data[id(e)])))

    def _play(self, paths):
        from PySide6.QtMultimedia import QSoundEffect
        self.stop_sounds()
        queue = list(paths)

        def nxt():
            if not queue:
                self.player = None
                self.update_commands()
                return
            self.player = QSoundEffect(self)
            self.player.setSource(QUrl.fromLocalFile(str(queue.pop(0))))
            self.player.playingChanged.connect(lambda: None if self.player is None or self.player.isPlaying()
                                               else nxt())
            self.player.play()
        nxt()
        self.update_commands()

    def play_sounds(self):
        e = self.focused()
        if e and e.sounds:
            self._play(e.sounds)

    def stop_sounds(self):
        if self.player is not None:
            p, self.player = self.player, None
            p.stop()
        self.update_commands()

    def about(self):
        D.AboutDialog(self.res, self).exec()

    def context_menu(self, rid, global_pos):
        items = self.res.menu(rid)[0][2]
        handlers = dict(self.acts)
        menu = QMenu(self)
        self._fill_context(menu, items, handlers)
        menu.exec(global_pos)

    def _fill_context(self, menu, items, handlers):
        for it in items:
            if it[0] == "sep":
                menu.addSeparator()
            elif it[0] == "popup":
                if any(x[0] == "item" and x[1] == LOOK_BASE for x in it[2]):
                    menu.addMenu(self.look_menu)
                else:
                    sub = menu.addMenu(self.res.tr(it[1], MENU_CONTEXTS))
                    self._fill_context(sub, it[2], handlers)
            elif it[1] in handlers:
                menu.addAction(handlers[it[1]])

    def closeEvent(self, ev):
        for sw in list(self.sheets):
            if not self.confirm_save(sw.sheet.dirty()):
                ev.ignore()
                return
        self._closing_documents = True
        for sw in list(self.sheets):
            sw.close()
        self.mdi.subWindowActivated.disconnect()
        if self.hist_dock.isFloating():
            self.q.setValue("Histogram Dock Size", self.hist_dock.size())
        self.q.setValue("MainFrame Position", self.saveGeometry())
        self.q.setValue("ControlBar States", self.saveState())
        for cu in list(self.closeups):
            cu.close()
        self.q.sync()
        super().closeEvent(ev)


TOOL_CMD = {v: k for k, v in TOOLS.items()}


def print_contact_sheet(p, printer, page, entries, per_short_side, view_cell):
    """Contact sheet print: thumbnails about ``per_short_side`` across the short side of the paper
    (or the on-screen size), each with a thin border, its crop outline, and number and name below."""
    from PySide6.QtGui import QPen
    short = min(page.width(), page.height())
    cell = max(32, short // per_short_side if per_short_side else view_cell)
    text_h = max(12, cell // 10)
    cols = max(1, page.width() // cell)
    rows = max(1, page.height() // (cell + text_h))
    font = p.font()
    font.setPixelSize(max(8, text_h * 2 // 3))
    p.setFont(font)
    for i, e in enumerate(entries):
        if i and i % (cols * rows) == 0:
            printer.newPage()
        k = i % (cols * rows)
        x0, y0 = (k % cols) * cell, (k // cols) * (cell + text_h)
        area = R.QRect(x0 + cell // 20, y0 + cell // 20, cell - cell // 10, cell - cell // 10)
        if e.thumb is not None:
            sz = e.thumb.size().scaled(area.size(), Qt.KeepAspectRatio)
            r = R.QRect(area.left() + (area.width() - sz.width()) // 2,
                        area.top() + (area.height() - sz.height()) // 2, sz.width(), sz.height())
            p.drawImage(r, e.thumb)
            p.setPen(QPen(Qt.black, max(1, cell // 200)))
            p.drawRect(r)
            c = e.settings.crop
            if c:
                p.setPen(QPen(Qt.black, max(1, cell // 150), Qt.DashLine))
                p.drawRect(R.QRect(r.left() + round(c[0] * r.width()), r.top() + round(c[1] * r.height()),
                                   round((c[2] - c[0]) * r.width()), round((c[3] - c[1]) * r.height())))
        p.setPen(Qt.black)
        p.drawText(R.QRect(x0, y0 + cell, cell, text_h), Qt.AlignHCenter | Qt.AlignTop,
                   f"{i + 1}  {e.path.name}")


def launch(app, path):
    if sys.platform == "darwin" and str(app).endswith(".app"):
        subprocess.Popen(["open", "-a", str(app), str(path)])
    else:
        subprocess.Popen([str(app), str(path)])


_VK = {0x08: Qt.Key_Backspace, 0x09: Qt.Key_Tab, 0x0D: Qt.Key_Return, 0x1B: Qt.Key_Escape, 0x20: Qt.Key_Space,
       0x21: Qt.Key_PageUp, 0x22: Qt.Key_PageDown, 0x23: Qt.Key_End, 0x24: Qt.Key_Home, 0x25: Qt.Key_Left,
       0x26: Qt.Key_Up, 0x27: Qt.Key_Right, 0x28: Qt.Key_Down, 0x2D: Qt.Key_Insert, 0x2E: Qt.Key_Delete,
       0xBB: Qt.Key_Plus, 0xBD: Qt.Key_Minus, 0x6B: Qt.Key_Plus, 0x6D: Qt.Key_Minus}


def _vk_to_qt(vk):
    if 0x30 <= vk <= 0x39 or 0x41 <= vk <= 0x5A:
        return vk
    if 0x70 <= vk <= 0x87:
        return Qt.Key_F1 + (vk - 0x70)
    k = _VK.get(vk)
    return k.value if k is not None else None


QT_LOCALES = {"English": "en", "Chinese": "zh_CN", "French": "fr", "German": "de", "Italian": "it",
              "Japanese": "ja", "Korean": "ko", "Portugese": "pt_BR", "Spanish": "es",
              "Traditional Chinese": "zh_TW"}


def install_qt_translation(app, language):
    """Standard dialogs (file chooser, print, colour) speak the Photo Desk UI language, like the
    system common dialogs of the original."""
    from PySide6.QtCore import QLibraryInfo, QLocale, QTranslator
    code = QT_LOCALES.get(language, "en")
    QLocale.setDefault(QLocale(code))
    tr = QTranslator(app)
    if tr.load(f"qtbase_{code}", QLibraryInfo.path(QLibraryInfo.TranslationsPath)):
        app.installTranslator(tr)


def _migrate_settings(settings, old=None):
    """Carry preferences over from releases that stored them under Kodak/DCS Photo Desk."""
    if settings.allKeys():
        return
    old = old if old is not None else QSettings("Kodak", "DCS Photo Desk")
    keys = old.allKeys()
    if not keys:
        return
    for key in keys:
        settings.setValue(key, old.value(key))
    settings.sync()
    old.clear()
    old.sync()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setOrganizationName(ORG_NAME)
    ini = os.environ.get("PHOTODESK_SETTINGS")
    settings = QSettings(ini, QSettings.IniFormat) if ini else QSettings(ORG_NAME, APP_NAME)
    if not ini:
        _migrate_settings(settings)
    apply_appearance(app, settings.value("Appearance", "light"))
    try:
        res = R.Resources.load()
    except R.MissingResources:
        installer, _ = QFileDialog.getOpenFileName(None, "Photo_Desk_V_4_3_0.exe", "", "Photo Desk installer (*.exe)")
        if not installer:
            QMessageBox.critical(None, APP_NAME,
                                 f"{APP_NAME} is not properly installed.\nThe Kodak profiles and interface files are missing.\n"
                                 "Select Photo_Desk_V_4_3_0.exe to install the Kodak profiles and interface.")
            return 1
        profiles.install_from_installer(installer)
        res = R.Resources.load()
    install_qt_translation(app, res.language)
    splash = D.Splash(res)
    splash.show()
    app.processEvents()
    w = MainWindow(res)
    w.resize(1400, 900)
    w.show()
    QTimer.singleShot(1500, splash.close)
    for arg in sys.argv[1:]:
        if Path(arg).exists():
            w.open_path(arg)
    return app.exec()
