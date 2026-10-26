"""Original Photo Desk UI resources for the Qt front end.

Everything visible in the original interface (menus, toolbar, dialogs and
panels laid out in dialog units, bitmaps, cursors, strings) is read from the
user's PhotoDsk.exe, which ``photodesk install-profiles`` stores next to the
Kodak profiles.  Strings go through the official translation table
(PhotoDeskUIStrings.u16) for the selected language.
"""
import functools
import io
import re

import numpy as np
from PIL import Image
from PySide6.QtCore import QDate, QRect, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QFont, QFontMetrics, QIcon, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDateEdit, QGroupBox, QLabel, QLineEdit, QListWidget,
    QProgressBar, QPushButton, QRadioButton, QSlider, QTextEdit, QToolButton, QVBoxLayout, QWidget,
)

from .. import profiles, ui_strings
from ..peres import PEResources

SS_TYPEMASK, SS_ICON, SS_BITMAP, SS_CENTER, SS_RIGHT = 0x1F, 0x03, 0x0E, 0x01, 0x02
ES_MULTILINE, ES_READONLY, ES_CENTER, ES_RIGHT = 0x4, 0x800, 0x1, 0x2
WS_VISIBLE, WS_DISABLED, TBS_VERT = 0x10000000, 0x08000000, 0x2
CBS_DROPDOWNLIST = 0x3
TRANSPARENT_KEY = (192, 192, 192)


# Translation-table LOCATIONs of each dialog template, menu and string group.
DIALOG_CONTEXT = {
    100: ("About Box",), 102: ("Progress dialog",), 103: ("Image Controls Panel",),
    129: ("Exposure Compensation Dialog",), 131: ("Scale View Dialog", "View Menu"),
    133: ("Save As Dialog", "Output Preferences Dialog"), 137: ("White Point Dialog",),
    153: ("Image Info Panel",), 166: ("Save As Dialog",), 197: ("Review Dialog",),
    200: ("Save As Dialog", "Output Preferences Dialog"), 213: ("Rename Dialog",),
    214: ("Output Preferences Dialog",), 250: ("Printing Preferences Dialog", "Output Preferences Dialog"),
    257: ("Close-Up Window",), 280: ("Color Balance Dialog",), 281: ("Exposure and Tone Dialog",),
    292: ("Paste Special Dialog",), 1121: ("IPTC Dialog",), 1123: ("Noise Reduction Dialog",),
    1124: ("Histogram Panel",), 1130: ("Constrained Crop Aspect Ratio Dialog",),
    1132: ("Sharpening Dialog",), 1133: ("Noise Reduction Dialog",), 1134: ("Noise Reduction Dialog",),
    1137: ("Lens Optimization Dialog",),
}
MENU_CONTEXT = ("File Menu", "Edit Menu", "View Menu", "Image Menu", "Info Menu", "Window Menu",
                "Help Menu")


class MissingResources(RuntimeError):
    pass


class Resources:
    def __init__(self, exe_bytes, table_bytes=None, language=None):
        self.pe = PEResources(exe_bytes)
        self.language = language or ui_strings.system_language()
        self.tr = ui_strings.Translator(table_bytes, self.language)
        self.strings = self.pe.strings()
        self._dlu = None

    @classmethod
    def load(cls, language=None):
        d = profiles.ui_dir()
        exe = d / "PhotoDsk.exe"
        if not exe.is_file():
            raise MissingResources(str(exe))
        table = d / "PhotoDeskUIStrings.u16"
        return cls(exe.read_bytes(), table.read_bytes() if table.is_file() else None, language)

    # ---------------------------------------------------------------- strings
    def s(self, sid, *args, context=("Pop-Up Message", "Common Strings", "Common")):
        """String-table entry, translated; '^', '%' and '#' placeholders filled from args."""
        text = self.tr(self.strings.get(sid, ""), context)
        return fill(text, *args) if args else text

    def tip(self, cmd):
        """(status bar prompt, tooltip) of a command, as MFC splits them on '\\n'."""
        text = self.strings.get(cmd, "")
        prompt, _, tooltip = text.partition("\n")
        return self.tr(prompt, MENU_CONTEXT), self.tr(tooltip or prompt, MENU_CONTEXT + ("Mac Toolbar",))

    # ---------------------------------------------------------------- images
    @functools.lru_cache(maxsize=None)
    def bitmap(self, rid, transparent=True):
        im = Image.open(io.BytesIO(self.pe.bitmap(rid))).convert("RGBA")
        if transparent:
            a = np.array(im)
            key = np.all(a[..., :3] == TRANSPARENT_KEY, -1)
            a[key, 3] = 0
            im = Image.fromarray(a)
        return pil_to_pixmap(im)

    def strip(self, rid, width, index, transparent=True):
        pm = self.bitmap(rid, transparent)
        return pm.copy(index * width, 0, width, pm.height())

    @functools.lru_cache(maxsize=None)
    def cursor(self, gid):
        data, hot = self.pe.cursor(gid)
        im = Image.open(io.BytesIO(data)).convert("RGBA")
        return QCursor(pil_to_pixmap(im), *hot)

    @functools.lru_cache(maxsize=None)
    def app_icon(self):
        im = Image.open(io.BytesIO(self.pe.icon(128)))
        icon = QIcon()
        for size in sorted(im.info.get("sizes", {im.size})):
            im.size = size
            icon.addPixmap(pil_to_pixmap(im.convert("RGBA")))
        return icon

    @functools.lru_cache(maxsize=None)
    def toolbar(self):
        """[(command id or 0 for separator, QIcon)] of the main toolbar."""
        (w, _), cmds = self.pe.toolbar(128)
        out, i = [], 0
        for cmd in cmds:
            if cmd == 0:
                out.append((0, None))
            else:
                out.append((cmd, QIcon(self.strip(128, w, i))))
                i += 1
        return out

    # ---------------------------------------------------------------- dialog units
    def dlu(self):
        """Pixels per dialog unit for the 8 pt dialog font (x, y)."""
        if self._dlu is None:
            fm = QFontMetrics(dialog_font())
            avg = fm.horizontalAdvance("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") / 52
            self._dlu = (round(avg + 0.5) / 4, fm.height() / 8)
        return self._dlu

    def rect(self, r):
        bx, by = self.dlu()
        x, y, w, h = r
        return QRect(round(x * bx), round(y * by), round(w * bx), round(h * by))

    def size(self, size):
        bx, by = self.dlu()
        return round(size[0] * bx), round(size[1] * by)

    def menu(self, rid):
        return self.pe.menu(rid)


def fill(text, *args):
    """Fill Photo Desk's '^', '%' and '#' placeholders (first, second, third argument)."""
    for mark, value in zip(("^", "%", "#"), args):
        text = text.replace(mark, str(value), 1)
    return text


def dialog_font():
    f = QFont(QApplication.font())
    f.setPointSizeF(max(8.0, f.pointSizeF() * 8 / 9))
    return f


def pil_to_pixmap(im):
    im = im.convert("RGBA")
    data = im.tobytes("raw", "RGBA")
    qim = QImage(data, im.width, im.height, 4 * im.width, QImage.Format_RGBA8888).copy()
    return QPixmap.fromImage(qim)


def strip_accel(text):
    """'&Open...\\tCtrl+O' -> ('&Open...', 'Ctrl+O')"""
    head, _, accel = text.partition("\t")
    return head, accel


def qt_shortcut(accel):
    """Windows accelerator text -> QKeySequence (Ctrl+F1, Alt+Ctrl+S, Ctrl++ ...)."""
    if not accel:
        return QKeySequence()
    parts = re.split(r"\+(?!$)", accel)
    key = parts[-1]
    mods = [p for p in parts[:-1]]
    key = {"Delete": "Del", "+": "+", "-": "-"}.get(key, key)
    return QKeySequence("+".join(mods + [key]))


# ------------------------------------------------------------------ widgets
class UpDown(QWidget):
    """msctls_updown32: two arrow buttons that step the buddy edit."""
    stepped = Signal(int)

    def __init__(self, parent):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        for arrow, step in ((Qt.UpArrow, 1), (Qt.DownArrow, -1)):
            b = QToolButton(self)
            b.setArrowType(arrow)
            b.setAutoRepeat(True)
            b.setFixedHeight(1)
            b.setSizePolicy(b.sizePolicy().horizontalPolicy(), b.sizePolicy().Expanding)
            b.clicked.connect(lambda _=False, s=step: self.stepped.emit(s))
            lay.addWidget(b)


class GraphFrame(QLabel):
    """Static used as a drawing surface (histograms, image areas)."""


class Built:
    """Widgets of a dialog template instantiated at their original positions."""

    def __init__(self):
        self.by_id = {}
        self.all = []

    def __getitem__(self, cid):
        return self.by_id[cid]

    def get(self, cid):
        return self.by_id.get(cid)

    def text(self, text):
        """First control whose original caption is text."""
        return next(w for c, w in self.all if c.text == text)


def build(res, parent, rid, skip=()):
    """Instantiate dialog template rid as child widgets of parent; returns Built."""
    dlg = res.pe.dialog(rid)
    parent.setFont(dialog_font())
    w, h = res.size(dlg.size)
    parent.setFixedSize(w, h) if parent.isWindow() else parent.setMinimumSize(w, h)
    ctx = DIALOG_CONTEXT.get(rid, ())
    if dlg.title and parent.isWindow():
        parent.setWindowTitle(res.tr(dlg.title, ctx))
    out = Built()
    groups = []
    for c in dlg.controls:
        if c.id in skip:
            continue
        wdg = _make(res, parent, c, ctx)
        if wdg is None:
            continue
        wdg.setGeometry(res.rect(c.rect))
        if not c.style & WS_VISIBLE:
            wdg.hide()
        if c.style & WS_DISABLED:
            wdg.setEnabled(False)
        if isinstance(wdg, QGroupBox):
            groups.append(wdg)
        if c.id != -1:
            out.by_id[c.id] = wdg
        out.all.append((c, wdg))
    for g in groups:
        g.lower()
    return out


def _make(res, parent, c, ctx):
    t = res.tr(c.text, ctx) if isinstance(c.text, str) else c.text
    cls = c.cls
    if cls == "Button":
        kind = c.button_kind
        if kind == "group":
            return QGroupBox(t, parent)
        if kind in ("check", "autocheck", "3state", "auto3state"):
            return QCheckBox(t, parent)
        if kind in ("radio", "autoradio"):
            return QRadioButton(t, parent)
        b = QPushButton(t, parent)
        if kind == "defpush":
            b.setDefault(True)
        if kind == "owner":
            b.setText("")
        return b
    if cls == "Static":
        lbl = GraphFrame(parent) if not t and c.style & SS_TYPEMASK == 0 else QLabel(parent)
        kind = c.style & SS_TYPEMASK
        if kind in (SS_ICON, SS_BITMAP) and isinstance(c.text, int):
            try:
                pm = res.bitmap(c.text) if kind == SS_BITMAP else res.app_icon().pixmap(32, 32)
            except KeyError:
                pm = res.app_icon().pixmap(32, 32)
            lbl.setPixmap(pm)
            return lbl
        if isinstance(t, str):
            lbl.setText(t)
        lbl.setWordWrap("\n" in (t or "") or c.rect[3] > 12)
        align = Qt.AlignTop
        if kind == SS_CENTER:
            align |= Qt.AlignHCenter
        elif kind == SS_RIGHT:
            align |= Qt.AlignRight
        lbl.setAlignment(align)
        return lbl
    if cls == "Edit":
        if c.style & ES_MULTILINE:
            e = QTextEdit(parent)
            e.setAcceptRichText(False)
        else:
            e = QLineEdit(parent)
            e.setAlignment(Qt.AlignRight if c.style & ES_RIGHT else Qt.AlignLeft)
        e.setReadOnly(bool(c.style & ES_READONLY))
        return e
    if cls == "ComboBox":
        cb = QComboBox(parent)
        cb.setEditable(c.style & 0x3 != CBS_DROPDOWNLIST and c.style & 0x3 != 0x1)
        r = res.rect(c.rect)
        cb.setFixedHeight(min(r.height(), QFontMetrics(dialog_font()).height() + 8))
        return cb
    if cls == "ListBox":
        return QListWidget(parent)
    if cls == "msctls_trackbar32":
        return QSlider(Qt.Vertical if c.style & TBS_VERT else Qt.Horizontal, parent)
    if cls == "msctls_updown32":
        return UpDown(parent)
    if cls == "msctls_progress32":
        return QProgressBar(parent)
    if cls == "SysDateTimePick32":
        d = QDateEdit(parent)
        d.setCalendarPopup(True)
        d.setDate(QDate.currentDate())
        return d
    return None


def fix_combo_geometry(built, res):
    """Combo boxes' template height includes the drop-down list; keep only the edit part."""
    for c, w in built.all:
        if isinstance(w, QComboBox):
            g = w.geometry()
            w.setGeometry(g.x(), g.y(), g.width(), w.height())


def color_icon(rgb, size=12):
    pm = QPixmap(size, size)
    pm.fill(QColor(*rgb))
    return QIcon(pm)
