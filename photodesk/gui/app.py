"""DCS Photo Desk main window: MDI contact sheets and full image renderings.

Menus, toolbar, accelerators, panels, dialogs, bitmaps, cursors and strings
come from the original PhotoDsk.exe resources (parsed, never executed) and the
official translation table, so the interface matches Photo Desk 4.3.
"""
import functools
import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QSettings, QSize, Qt, QThreadPool, QTimer, QUrl
from PySide6.QtGui import QAction, QActionGroup, QDesktopServices, QGuiApplication, QIcon, QImage, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QDockWidget, QFileDialog, QInputDialog, QLabel, QMainWindow, QMdiArea, QMdiSubWindow, QMenu,
    QMessageBox, QToolBar,
)

from .. import __version__, camera, export, pipeline, profiles
from . import dialogs as D
from . import res as R
from .model import PASTE_GROUPS, Sheet, is_image
from .panels import CROP_ITEMS, CloseUpWindow, HistogramPanel, ImageControlsPanel, ImageInfoPanel
from .views import ContactSheetView, ImageView, Task, ThumbIcons, qimage_array, to_qimage

PREVIEW_EDGE = 1400            # long edge of the "faster" full-image preview
THUMB_EDGE = 256
ZOOMS = {32842: 25, 32853: 50, 32852: 100, 32841: 200}
TOOLS = {32847: "select", 32848: "crop", 32846: "click", 33178: "closeup", 33177: "hand"}
BALANCE_CMD = {32812: "preset", 32816: "click", 32814: "custom", 32815: "auto"}
LIGHTING_CMD = dict(zip((32787, 32788, 32789, 32790), pipeline.LIGHTINGS))
CROP_CMD = dict(zip(range(33110, 33118), CROP_ITEMS))
LOOK_BASE = 45000
HELP_URLS = {33158: 62044, 33159: 62048, 33160: 62049, 33161: 62050, 33162: 62051, 33163: 62052,
             33164: 62053, 33165: 62054, 33167: 62061}
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
        img, _ = export.load_finished(path)
        img = pipeline.resize_long_edge(img, THUMB_EDGE)
        return None, None, pipeline.apply_crop(pipeline.orient(img, 0, settings.rotation), None)
    raw = pipeline.decode(path, half_size=True)
    small = pipeline.reduced(raw, THUMB_EDGE)
    img = pipeline.render(small, settings.copy(crop=None), profile_directory=pdir, quality="faster")
    return small, raw.metadata, img


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
class SheetWindow(QMdiSubWindow):
    def __init__(self, main, sheet):
        super().__init__()
        self.main, self.sheet = main, sheet
        self.view = ContactSheetView(main.icons)
        self.view.set_zoom(main.q.value("LastThumbnailViewSize", 100, int))
        self.view.set_show_info(main.q.value("Thumbnail Info State", True, bool))
        self.setWidget(self.view)
        self.setWindowTitle(str(sheet.folder))
        self.setWindowIcon(main.res.app_icon())
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.view.populate(sheet.entries)
        self.view.itemSelectionChanged.connect(main.selection_changed)
        self.view.currentItemChanged.connect(lambda *_: main.selection_changed())
        self.view.itemDoubleClicked.connect(lambda it: main.open_image(it.data(Qt.UserRole), self))
        self.view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.view.customContextMenuRequested.connect(
            lambda pos: main.context_menu(194, self.view.viewport().mapToGlobal(pos)))
        self.view.viewport().installEventFilter(self)
        self.view.files_dropped.connect(lambda paths, pos, mode: main.drop_files(self, paths, pos, mode))
        self.images = []

    def eventFilter(self, obj, ev):
        if ev.type() == ev.Type.MouseButtonPress and ev.button() == Qt.LeftButton:
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
        if not self.main.confirm_save(self.sheet.dirty()):
            ev.ignore()
            return
        for w in list(self.images):
            w.close()
        self.main.sheets.remove(self)
        super().closeEvent(ev)


class ImageWindow(QMdiSubWindow):
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
        self.view.zoom_changed.connect(lambda z: main.show_zoom(z) if main.mdi.activeSubWindow() is self else None)
        self.view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.view.customContextMenuRequested.connect(lambda pos: main.context_menu(195, self.view.mapToGlobal(pos)))
        self.token = 0
        self.rgb = None
        sheet_win.images.append(self)

    def _clicked(self, x, y):
        tool = self.main.tool
        if tool == "click":
            self.main.click_balance(self.entry, x, y)
        elif tool == "closeup":
            self.main.show_closeup(self.entry, x, y)

    def closeEvent(self, ev):
        if self in self.sheet_win.images:
            self.sheet_win.images.remove(self)
        if not any(w.entry is self.entry for w in self.sheet_win.images):
            self.entry.raw = None
        super().closeEvent(ev)


# ------------------------------------------------------------------ main window
class MainWindow(QMainWindow):
    def __init__(self, res):
        super().__init__()
        self.res = res
        self.q = QSettings("Kodak", "DCS Photo Desk")
        self.pdir = str(profiles.profile_dir())
        self.pool = QThreadPool.globalInstance()
        self.icons = ThumbIcons(res)
        self.cursors = {"click": res.cursor(148), "hand": res.cursor(191), "closeup": res.cursor(30978),
                        "crop": Qt.CrossCursor, 221: res.cursor(221), 222: res.cursor(222),
                        223: res.cursor(223), 224: res.cursor(224)}
        self.sheets = []
        self.tool = "select"
        self.quality = self.q.value("Review Render Mode", "better")
        self.clipboard = None
        self.crop_mode = self.q.value("Crop Mode", "freeform")
        self.crop_other = (self.q.value("Constrained Crop Width X 10000", 40000, int) / 10000,
                           self.q.value("Constrained Crop Height X 10000", 60000, int) / 10000)
        self.opts = export.OutputOptions.from_json(self.q.value("OutputOptions", {}) or {})
        self.preview_override = None
        self.closeups = []
        self.player = None
        self.looks = pipeline.available_looks(self.pdir) if profiles.is_installed() else []
        self.setWindowTitle(res.tr("DCS Photo Desk", ("About Box", "Common")))
        self.setWindowIcon(res.app_icon())
        self.mdi = QMdiArea()
        self.mdi.setBackground(Qt.gray)
        self.setCentralWidget(self.mdi)
        self.mdi.subWindowActivated.connect(lambda _: self.selection_changed())
        self._actions()
        self._menus()
        self._toolbar()
        self._panels()
        self._status()
        self.update_commands()
        self.restoreGeometry(self.q.value("MainFrame Position", b""))
        self.restoreState(self.q.value("ControlBar States", b""))

    # ---------------------------------------------------------- resources
    def _menu_texts(self):
        texts = {}

        def walk(items):
            for it in items:
                if it[0] == "popup":
                    walk(it[2])
                elif it[0] == "item":
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
            label, _ = R.strip_accel(res.tr(text, MENU_CONTEXTS))
            act = QAction(label, self)
            prompt, tip = res.tip(cmd)
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
        if 57669 not in self.acts:   # context help button on the toolbar
            act = QAction(self)
            act.setIcon(bar_icons.get(57669, QIcon()))
            act.setToolTip(res.tip(57669)[1] or "Help")
            act.triggered.connect(self.help_topics)
            self.acts[57669] = act
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
            59393: lambda: self.statusBar().setVisible(not self.statusBar().isVisible()),
            33188: self.review, 33129: lambda: self.set_quality("faster"), 33128: lambda: self.set_quality("better"),
            32838: lambda: self.rotate(1), 32837: lambda: self.rotate(-1), 32806: self.exposure_dialog,
            33187: self.exposure_tone_dialog, 33131: self.auto_exposure, 33152: self.white_point_dialog,
            33183: self.color_balance_dialog, 33197: self.lens_dialog, 33150: self.noise_dialog,
            33149: self.sharpen_dialog, 33118: self.crop_other_dialog, 32807: self.original_settings,
            32803: self.auto_scene, 32867: self.save_balance_file, 32855: lambda: self.tag(True),
            32856: lambda: self.tag(False), 32854: self.iptc_dialog, 1055: self.play_sounds,
            1056: self.stop_sounds, 57650: self.mdi.cascadeSubWindows, 57651: self.mdi.tileSubWindows,
            57649: self.mdi.tileSubWindows, 57667: self.help_topics, 57664: self.about,
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
        for cmd, sid in HELP_URLS.items():
            h[cmd] = functools.partial(self.open_url, sid)
        return h

    def _menus(self):
        res = self.res
        bar = self.menuBar()
        bar.clear()
        self.recent_menu = None
        for top in res.menu(128):
            menu = self._submenu(top[1], top[2])
            bar.addMenu(menu)

    def _submenu(self, title, items):
        res = self.res
        menu = QMenu(res.tr(title, MENU_CONTEXTS), self)
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
                self.recent_menu = QMenu(res.tr(it[2], MENU_CONTEXTS), menu)
                menu.addMenu(self.recent_menu)
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
        self.recent_menu.clear()
        recent = self.q.value("Recent File List", []) or []
        if isinstance(recent, str):
            recent = [recent]
        for i, path in enumerate(recent[:4]):
            act = self.recent_menu.addAction(f"&{i + 1} {path}")
            act.triggered.connect(lambda _=False, p=path: self.open_path(p))
        self.recent_menu.setEnabled(bool(recent))

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
        w, h = self.res.pe.toolbar(128)[0]
        tb.setIconSize(QSize(w, h))
        for cmd, _ in self.res.toolbar():
            if cmd == 0:
                tb.addSeparator()
            else:
                tb.addAction(self.acts[cmd])
        self.addToolBar(tb)
        tb.visibilityChanged.connect(lambda v: self.acts[59392].setChecked(v))

    def _panels(self):
        res = self.res
        self.controls = ImageControlsPanel(res, self.acts)
        self.controls.set_looks([(n, pipeline.look_label(d)) for n, d in self.looks])
        self.controls.command.connect(lambda c: self.acts[c].trigger())
        self.controls.edit.connect(lambda kind, f: self.edit(kind, **f))
        self.controls.crop_mode.connect(self.set_crop_mode)
        self.controls.set_crop_mode(self.crop_mode, self.crop_other)
        self.info = ImageInfoPanel(res)
        self.info.command.connect(lambda c: self.acts[c].trigger())
        self.info.annotation.connect(lambda t: self.edit("annotation", annotation=t))
        self.info.play.connect(self._play)
        self.info.stop.connect(self.stop_sounds)
        self.hist = HistogramPanel(res)
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

    # ---------------------------------------------------------- state
    def active_sheet(self):
        w = self.mdi.activeSubWindow()
        if isinstance(w, SheetWindow):
            return w
        if isinstance(w, ImageWindow):
            return w.sheet_win
        return self.sheets[0] if self.sheets else None

    def active_image(self):
        w = self.mdi.activeSubWindow()
        return w if isinstance(w, ImageWindow) else None

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
            fw, fh = iw.view.full_size
            self.st_size.setText(self.res.s(59151).replace("%d", str(fw), 1).replace("%d", str(fh), 1))
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
            32868: has, 57602: self.mdi.activeSubWindow() is not None, 57603: has, 57604: has, 32861: has,
            57607: sheet, 57609: sheet, 33108: has, 33106: has, 32870: has and self.active_image() is None,
            57634: focus is not None, 32865: focus is not None, 32866: has and self.clipboard is not None,
            33196: has and self.clipboard is not None, 57642: sheet, 32858: sheet, 32859: sheet, 32860: sheet,
            32824: sheet, 32825: sheet, 33188: has, 32838: has, 32837: has, 32855: has, 32856: has, 32854: has,
            32848: has, 32846: raw, 33178: has, 32806: raw, 33187: raw, 33152: raw, 33183: raw, 33150: raw,
            33149: raw, 32807: raw, 32812: raw, 32816: raw and any(e.settings.click_wb != (1.0, 1.0, 1.0) for e in sel),
            32814: raw and all("custom" in e.caps.balances for e in sel),
            32815: raw and all("auto" in e.caps.balances for e in sel),
            32803: raw and not any(e.caps.pro_back for e in sel), 33131: raw and not any(e.caps.pro_back for e in sel),
            33197: raw and all(e.caps.lens_optimization for e in sel),
            32867: len(sel) == 1 and raw and sel[0].settings.balance == "click",
            1055: focus is not None and bool(focus.sounds), 1056: self.player is not None,
            33179: sheet, 33180: sheet, 32849: self.active_image() is not None, 32844: self.active_image() is not None,
        }
        for cmd in list(LIGHTING_CMD) + [32812]:
            enable[cmd] = raw
        for cmd in ZOOMS:
            enable[cmd] = sheet
        for cmd, on in enable.items():
            act = self.acts.get(cmd)
            if act is not None and act.data() != "unimplemented":
                act.setEnabled(bool(on))
        if self.look_menu is not None:
            self.look_menu.setEnabled(raw)
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
        dlg.setNameFilters(["DCS Photo Desk (*.dcr *.DCR *.tif *.TIF *.tiff *.jpg *.JPG *.jpeg *.kdc *.KDC)",
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
            sw = self.open_folder(p.parent)
            entry = sw.sheet.find(p)
            if entry:
                self.open_image(entry, sw)
            return sw
        D.message(self, self.res, self.res.s(60000).replace("%s", str(p)))

    def open_folder(self, folder):
        folder = Path(folder)
        for sw in self.sheets:
            if sw.sheet.folder == folder:
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
            sw.view.refresh_entry(entry)
            if entry is self.focused():
                self.info.set_entry(entry)
                self._update_histogram()
        task.signals.done.connect(done)
        task.signals.failed.connect(lambda entry, msg: self.statusBar().showMessage(
            self.res.s(60000).replace("%s", entry.path.name) + f"  ({msg})"))
        self.pool.start(task)

    def refresh_thumb(self, entry):
        """Re-render a thumbnail after its settings changed."""
        for sw in self.sheets:
            if entry in sw.sheet.entries:
                if entry.small is not None:
                    img = pipeline.render(entry.small, entry.settings.copy(crop=None), profile_directory=self.pdir,
                                          quality="faster")
                    entry.thumb = to_qimage(img)
                elif not entry.is_raw and entry.thumb is not None:
                    self._load_thumb(sw, entry)
                sw.view.refresh_entry(entry)

    def open_selected(self):
        sw = self.active_sheet()
        for e in self.selection():
            self.open_image(e, sw)

    def open_image(self, entry, sheet_win):
        for w in sheet_win.images:
            if w.entry is entry:
                self.mdi.setActiveSubWindow(w)
                return w
        w = ImageWindow(self, sheet_win, entry)
        self.mdi.addSubWindow(w)
        w.resize(self.mdi.width() * 2 // 3, self.mdi.height() * 2 // 3)
        w.show()
        if entry.is_raw and entry.raw is None:
            task = Task(entry, _decode_job, entry.path)

            def decoded(e, raw):
                e.raw, e.meta = raw, raw.metadata
                self.render_window(w)
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
        if not e.is_raw:
            img, _ = export.load_finished(e.path)
            img = pipeline.orient(img, 0, s.rotation)
            self._show_render(token, img)
            return
        if e.raw is None:
            return
        space = pipeline.resolve_space(self.opts.working_space, self.pdir)
        task = Task(token, _render_job, e.raw, s, self.pdir, self.quality, PREVIEW_EDGE, space)
        task.signals.done.connect(self._show_render)
        task.signals.failed.connect(lambda t, msg: D.message(self, self.res, self.res.s(60002) + "\n" + msg))
        self.statusBar().showMessage(self.res.tr("Rendering...", ("Common", "Common Strings")))
        self.pool.start(task)

    def _show_render(self, token, img):
        w, n = token
        if n != w.token or w not in self.mdi.subWindowList():
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
        w = self.mdi.activeSubWindow()
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
                sw.sheet.refresh()
                sw.view.populate(sw.sheet.entries)
                for e in sw.sheet.entries:
                    if e.thumb is None:
                        self._load_thumb(sw, e)
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
            self.acts[33108].setText(R.strip_accel(self.res.s(59516).replace("%s", Path(path).stem, 1)
                                                   .replace("%s", "", 1))[0])

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
        dests = [export.output_path(e.path, folder, "tiff") for e in entries]
        self._export(entries, dests, "tiff", open_with=editor)

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
        try:
            sw.sheet.rename(entries, dlg.names)
        except OSError as err:
            D.message(self, self.res, self.res.s(60049) + f"\n{err}")
            return
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

        try:
            sw.sheet.transfer(paths, cmd == 32862, replace)
        except OSError as err:
            D.message(self, self.res, str(err))
        for other in self.sheets:
            if other is not sw and cmd == 32862:
                other.sheet.refresh()
                other.view.populate(other.sheet.entries)
        sw.view.populate(sw.sheet.entries)
        for e in sw.sheet.entries:
            if e.thumb is None:
                self._load_thumb(sw, e)
        self.selection_changed()

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
            per_row = {"small": 10, "medium": 4, "large": 2}.get(size)
            dpi = printer.resolution()
            cell = page.width() // per_row if per_row else int(sw.view.cell_size().width() * dpi / 96)
            cols = max(1, page.width() // cell)
            x = y = 0
            for e in sw.sheet.entries:
                if e.thumb is not None:
                    th = e.thumb.scaled(cell - 10, cell - 10, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    p.drawImage(x * cell + 5, y * cell + 5, th)
                    p.drawText(R.QRect(x * cell, y * cell + cell - 12, cell, 12), Qt.AlignCenter, e.path.name)
                x += 1
                if x >= cols:
                    x, y = 0, y + 1
                    if (y + 1) * cell > page.height():
                        printer.newPage()
                        y = 0
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
        sw = self.active_sheet()
        entries = entries if entries is not None else self.selection()
        if not sw or not entries:
            return
        if kind not in ("tag", "untag", "annotation", "iptc", "rotate", "crop"):
            entries = [e for e in entries if e.is_raw]     # processing options need DCS raw files
        changed = sw.sheet.set_fields(entries, kind, **fields)
        self.after_edit(changed)

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
            self.update_commands()

    def paste_settings(self):
        sw = self.active_sheet()
        if sw and self.clipboard:
            fields = tuple(f for cid, fs in PASTE_GROUPS.items() if cid not in (1350, 1351) for f in fs)
            self.after_edit(sw.sheet.paste([e for e in self.selection() if e.is_raw] or self.selection(),
                                           self.clipboard, fields, "paste_special"))

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
                    for f in fields:
                        setattr(s, f, getattr(self.clipboard, f))
                    e.settings = s
            self._preview_refresh(entries)
        dlg = D.PasteSpecialDialog(self.res, self, PASTE_GROUPS, preview)
        ok = dlg.exec() == D.QDialog.Accepted
        fields = dlg.fields()
        for e in entries:
            e.settings = originals[id(e)]
        if ok and fields:
            self.after_edit(sw.sheet.paste(entries, self.clipboard, fields))
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
        dlg = D.PreferencesDialog(self.res, self, self.opts, self.q.value("Send Files To Path", ""), spaces)
        if dlg.exec() == D.QDialog.Accepted:
            self.opts = dlg.opts
            self.q.setValue("OutputOptions", self.opts.to_json())
            self.q.setValue("Send Files To Path", dlg.send_folder)
            for sw in self.sheets:
                for w in sw.images:
                    self.render_window(w)

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
        elif self.active_sheet():
            self.active_sheet().view.set_zoom(z)
            self.q.setValue("LastThumbnailViewSize", z)
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
            order = [25, 50, 100, 200]
            i = min(max(0, order.index(cur) + d if cur in order else 2), 3)
            self.set_zoom(order[i])

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
                img, _ = export.load_finished(entry.path)
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
            sw.view.viewport().setCursor(self.cursors.get(tool, Qt.ArrowCursor) if tool in ("click", "closeup")
                                         else Qt.ArrowCursor)
            for w in sw.images:
                w.view.set_tool(tool)

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
        entries = [e for e in self.selection() if e.is_raw] or [entry]
        if entry not in entries:
            entries = [entry]
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
        dlg = dialog_cls(self.res, self, focus.settings, preview, *args, mixed=mixed, **kw) \
            if dialog_cls is not D.NoiseDialog else dialog_cls(self.res, self, focus.settings, preview, focus.caps,
                                                               mixed=mixed)
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

    def original_settings(self):
        sw = self.active_sheet()
        if sw:
            def change(e, s):
                cap = pipeline.at_capture(e.meta)
                for f in ("sharpen_amount", "sharpen_radius", "sharpen_threshold", "balance", "lighting",
                          "white_point", "color_balance"):
                    setattr(s, f, getattr(cap, f))
                if e.caps.pro_slr:
                    for f in ("rotation", "look", "noise_type", "noise_radius", "noise_strength", "noise_luma"):
                        setattr(s, f, getattr(cap, f))
            self.after_edit(sw.sheet.apply(self._raw_selection(), "original", change))

    def auto_scene(self):
        sw = self.active_sheet()
        if sw:
            def change(e, s):
                s.balance = "auto"
                raw = e.raw or e.small
                if raw is not None:
                    s.exposure = pipeline.auto_exposure(raw, s)
            self.after_edit(sw.sheet.apply([e for e in self._raw_selection() if not e.caps.pro_back],
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
        if e.caps.pro_back:
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
            write_wb_file(p, e.settings.click_wb)
        except OSError:
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

    # ---------------------------------------------------------- help
    def help_topics(self):
        chm = profiles.help_file(self.res.language)
        if chm is None:
            D.message(self, self.res, self.res.s(61703))
            return
        if sys.platform == "win32":
            os.startfile(str(chm))
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(chm)))

    def open_url(self, sid):
        QDesktopServices.openUrl(QUrl(self.res.s(sid)))

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
        self.mdi.subWindowActivated.disconnect()
        self.q.setValue("MainFrame Position", self.saveGeometry())
        self.q.setValue("ControlBar States", self.saveState())
        for cu in list(self.closeups):
            cu.close()
        super().closeEvent(ev)


TOOL_CMD = {v: k for k, v in TOOLS.items()}


def write_wb_file(path, wb):
    """DCS custom white balance file: camera RGB multipliers relative to green (Kodak .wb layout)."""
    import struct
    r, g, b = (float(v) / float(wb[1]) for v in wb)
    Path(path).write_bytes(b"KDKWB\x00\x01\x00" + struct.pack("<3f", r, 1.0, b))


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


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("DCS Photo Desk")
    app.setOrganizationName("Kodak")
    try:
        res = R.Resources.load()
    except R.MissingResources:
        installer, _ = QFileDialog.getOpenFileName(None, "Photo_Desk_V_4_3_0.exe", "", "Photo Desk installer (*.exe)")
        if not installer:
            QMessageBox.critical(None, "DCS Photo Desk",
                                 "Photo Desk is not properly installed.\nFiles needed by Photo Desk are missing.\n"
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
