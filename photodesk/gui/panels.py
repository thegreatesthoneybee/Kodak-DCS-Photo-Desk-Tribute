"""Image Controls, Image Info and Output Histogram panels and the Close-Up window."""
from PySide6.QtCore import QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QComboBox, QPushButton, QSlider, QTextEdit, QToolButton, QWidget

from .. import pipeline
from . import res as R
from .views import CloseUpView, HistogramWidget

TOOL_STRIP = (32847, 32848, 32846, 33178, 33177, 32855, 32856, 32838, 32837)
CROP_ITEMS = ("freeform", "square", "2x3l", "3x2p", "4x5l", "5x4p", "5x7l", "7x5p")
BALANCE_LABEL = {"preset": 33132, "click": 33133, "custom": 33134, "auto": 33135}
LIGHTING_LABEL = dict(zip(pipeline.LIGHTINGS, (33136, 33137, 33138, 33139)))
NOISE_LABEL = {"none": 33140, "basic": 33141, "advanced": 33142, "moire": 33143, "expert": 33146}


def bind(button, action):
    """Make a bitmap button run a menu command and follow its enabled/checked state."""
    button.clicked.connect(action.trigger)

    def sync():
        button.setEnabled(action.isEnabled())
        if button.isCheckable():
            button.setChecked(action.isChecked())
    action.changed.connect(sync)
    sync()


def icon_button(parent, rect, pixmap, tip, checkable=False):
    b = QToolButton(parent)
    b.setGeometry(rect)
    b.setIcon(QIcon(pixmap))
    b.setIconSize(pixmap.size())
    b.setAutoRaise(True)
    b.setToolTip(tip)
    b.setCheckable(checkable)
    return b


class ImageControlsPanel(QWidget):
    """Dialog template 103.  command(id) runs a menu command; edit(kind, fields) applies settings."""
    command = Signal(int)
    edit = Signal(str, dict)
    crop_mode = Signal(str)

    def __init__(self, res, actions):
        super().__init__()
        self.res = res
        b = self.b = R.build(res, self, 103)
        R.fix_combo_geometry(b, res)
        self._loading = False
        b[1237].setPixmap(res.bitmap(256, transparent=False).scaled(b[1237].size(), Qt.IgnoreAspectRatio,
                                                                    Qt.SmoothTransformation))
        # Tool strip (bitmap 146) in static 1040.
        strip = b[1040].geometry()
        self.tools = {}
        tool_count = len(TOOL_STRIP)
        for i, cmd in enumerate(TOOL_STRIP):
            act = actions.get(cmd)
            left = strip.x() + i * strip.width() // tool_count
            right = strip.x() + (i + 1) * strip.width() // tool_count
            btn = icon_button(self, QRect(left, strip.y(), right - left, strip.height()),
                              res.strip(146, 16, i), res.tip(cmd)[1], checkable=i < 5)
            btn.setIconSize(QSize(22, 22))
            if act is not None:
                bind(btn, act)
            self.tools[cmd] = btn
        # Use Original Camera Settings / Auto Scene Balance (bitmap 254) in static 1042.
        g = b[1042].geometry()
        for i, cmd in enumerate((32807, 32803)):
            btn = icon_button(self, QRect(g.x() + i * (g.width() // 2), g.y(), g.width() // 2, g.height()),
                              res.bitmap(254).copy(32 * i, 0, 32, 32), res.tip(cmd)[1])
            if actions.get(cmd):
                bind(btn, actions[cmd])
            self.tools[cmd] = btn
        g = b[1039].geometry()
        self.auto_exp = icon_button(self, QRect(g.x(), g.y(), 22, 20), res.bitmap(253), res.tip(33131)[1])
        if actions.get(33131):
            bind(self.auto_exp, actions[33131])
        for cmd in (33187, 33183, 33197, 33150, 33149):
            b[cmd].clicked.connect(lambda _=False, c=cmd: self.command.emit(c))
        b[1172].setEditable(False)
        for i, key in enumerate(CROP_ITEMS):
            b[1172].addItem(res.s(33119 + i), key)
        b[1172].addItem(res.s(33118).split("\n")[-1], "other")
        b[1172].activated.connect(self._crop_chosen)
        self.ev = b[1009]
        self.ev.setTracking(False)
        self.ev.setTickPosition(QSlider.TicksBelow)
        self.ev.valueChanged.connect(self._ev)
        self.wp = b[1010]
        self.wp.setRange(*(v // 100 for v in pipeline.WHITE_POINT_RANGE))
        self.wp.setTracking(False)
        self.wp.valueChanged.connect(self._wp)
        for cid in (1173, 1174, 1175):
            b[cid].setEditable(False)
        b[1173].activated.connect(lambda _: self._emit("balance", balance=b[1173].currentData()))
        b[1174].activated.connect(lambda _: self._emit("lighting", lighting=b[1174].currentData()))
        b[1175].activated.connect(lambda _: self._emit("look", look=b[1175].currentData()))
        self.looks = []
        self.set_entry(None)

    def _emit(self, kind, **fields):
        if not self._loading:
            self.edit.emit(kind, fields)

    def _ev(self, v):
        self.b[1038].setText(f"{v / 10:+.1f}" if v else "0.0")
        self._emit("exposure", exposure=v / 10)

    def _wp(self, v):
        self.b[1041].setText(f"{v * 100}° K")
        self._emit("white_point", white_point=v * 100)

    def _crop_chosen(self, i):
        key = self.b[1172].itemData(i)
        if key == "other":
            self.command.emit(33118)
        else:
            self.crop_mode.emit(key)

    def set_crop_mode(self, key, other=None):
        combo = self.b[1172]
        if other:
            combo.setItemText(len(CROP_ITEMS), self.res.s(59153).replace("%s", f"{other[0]:g}", 1)
                              .replace("%s", f"{other[1]:g}", 1))
        combo.setCurrentIndex(len(CROP_ITEMS) if key == "other" else CROP_ITEMS.index(key))

    def set_looks(self, looks):
        """looks: [(file name, description)] of every installed Look."""
        self.looks = list(looks)
        self._fill_looks(self.looks)

    def _fill_looks(self, looks):
        c = self.b[1175]
        c.clear()
        for name, desc in looks:
            c.addItem(pipeline.look_label(desc), name)

    def set_tool(self, cmd):
        for c, btn in self.tools.items():
            if btn.isCheckable():
                btn.setChecked(c == cmd)

    def set_entry(self, entry):
        """Show the settings of the focused image (None: nothing selected)."""
        self._loading = True
        b, res = self.b, self.res
        raw_ok = entry is not None and entry.is_raw
        for cid in (1009, 1173, 1010, 1174, 1175, 33187, 33183, 33150, 33149, 33197):
            b[cid].setEnabled(raw_ok)
        if entry is None:
            self._loading = False
            return
        s, caps = entry.settings, entry.caps
        rng = caps.exposure_range
        self.ev.setRange(round(-rng * 10), round(rng * 10))
        self.ev.setTickInterval(round(rng * 5))
        self.ev.setValue(round(s.exposure * 10))
        b[1038].setText(f"{s.exposure:+.1f}" if s.exposure else "0.0")
        b[1098].setText(f"{-rng:.1f}")
        b[1099].setText(f"+{rng:.1f}")
        for cid, v in zip((1212, 1213, 1214), s.tone):
            b[cid].setText(f"{v:.1f}")
        combo = b[1173]
        combo.clear()
        for bal in caps.balances:
            combo.addItem(res.s(BALANCE_LABEL[bal]), bal)
        combo.setCurrentIndex(max(0, combo.findData(s.balance)))
        self.wp.setValue(s.white_point // 100)
        b[1041].setText(f"{s.white_point}° K")
        # The colour temperature scale is only for Pro 14n/SLR images with the Preset or Auto method.
        self.wp.setEnabled(raw_ok and caps.white_point and s.balance in ("preset", "auto"))
        combo = b[1174]
        combo.clear()
        for lt in pipeline.LIGHTINGS:
            combo.addItem(res.s(LIGHTING_LABEL[lt]), lt)
        combo.setCurrentIndex(max(0, combo.findData(s.lighting)))
        for cid, v in zip((1207, 1210, 1211), s.color_balance):
            b[cid].setText(str(v))
        b[1215].setText(str(s.lens_optimization) if caps.lens_optimization else "0")
        b[33197].setEnabled(raw_ok and caps.lens_optimization)
        self._fill_looks(caps.looks(self.looks))      # older cameras support only some Looks
        b[1175].setCurrentIndex(max(0, b[1175].findData(s.look)))
        b[1226].setText(res.s(NOISE_LABEL[s.noise_type]))
        t = s.noise_type
        kind = caps.noise_controls(t)
        if kind == "strength":
            labels = (res.s(62121), "")
            vals = (f"{s.noise_strength}%", "")
        elif kind == "threshold":
            labels = (res.s(60048), "")
            vals = (str(s.moire_threshold), "")
        elif kind == "none":
            labels = (res.tr("Radius:", ("Image Controls Panel",)), res.tr("Strength:", ("Image Controls Panel",)))
            vals = ("0", "0")
        else:
            labels = (res.tr("Radius:", ("Image Controls Panel",)), res.tr("Strength:", ("Image Controls Panel",)))
            vals = (str(s.noise_radius), f"{s.noise_strength}%")
        b[1191].setText(labels[0])
        b[1192].setText(labels[1])
        b[1208].setText(vals[0])
        b[1209].setText(vals[1])
        b[1204].setText(f"{s.sharpen_amount:.1f}")
        b[1205].setText(f"{s.sharpen_radius:.1f}")
        b[1206].setText(str(s.sharpen_threshold))
        self._loading = False


class ImageInfoPanel(QWidget):
    """Dialog template 153: IPTC button, sounds, annotation, camera settings."""
    command = Signal(int)
    annotation = Signal(object, str)   # (entry, text): annotations apply to the focused image only
    play = Signal(object)     # list of sound paths
    stop = Signal()

    def __init__(self, res):
        super().__init__()
        self.res = res
        b = self.b = R.build(res, self, 153)
        R.fix_combo_geometry(b, res)
        b[32854].clicked.connect(lambda: self.command.emit(32854))
        g = b[1060].geometry()
        self.play_btn = icon_button(self, QRect(g.x(), g.y(), 24, 22), res.strip(164, 16, 0), res.tip(1055)[1])
        self.stop_btn = icon_button(self, QRect(g.x() + 26, g.y(), 24, 22), res.strip(164, 16, 1), res.tip(1055)[1])
        self.play_btn.clicked.connect(self._play)
        self.stop_btn.clicked.connect(self.stop)
        b[1051].setEditable(False)
        self.note = b[1047]
        self.note.installEventFilter(self)
        self._entry = None
        self._sounds = []
        self.set_entry(None)

    def eventFilter(self, obj, ev):
        if obj is self.note and ev.type() == ev.Type.FocusOut and self._entry is not None:
            text = self.note.toPlainText()[:500]
            if text != self._entry.settings.annotation:
                self.annotation.emit(self._entry, text)
        return False

    def _play(self):
        i = self.b[1051].currentIndex()
        if not self._sounds:
            return
        self.play.emit(self._sounds if i == len(self._sounds) else [self._sounds[i]])

    def set_entry(self, entry):
        self._entry = entry
        b, res = self.b, self.res
        on = entry is not None
        for w in (b[32854], self.note):
            w.setEnabled(on)
        self._sounds = entry.sounds if on else []
        combo = b[1051]
        combo.clear()
        for snd in self._sounds:
            combo.addItem(snd.name)
        if self._sounds:
            combo.addItem(res.s(62005))
        for w in (combo, self.play_btn, self.stop_btn):
            w.setEnabled(bool(self._sounds))
        self.note.blockSignals(True)
        self.note.setPlainText(entry.settings.annotation if on else "")
        self.note.blockSignals(False)
        lst = b[1054]
        lst.clear()
        if not on:
            return
        s = res.s
        try:
            size_mb = entry.path.stat().st_size / 1e6
        except OSError:
            size_mb = 0
        if not entry.is_raw:
            lst.addItem(s(50008))
            lst.addItem(s(50012) if entry.file_type == "T" else s(50013))
        meta = entry.meta or {}
        if entry.model:
            lst.addItem(s(50000) + f"{entry.make} {entry.model}".strip())
        ts = meta.get("timestamp")
        if ts:
            lst.addItem(s(50004) + ts[:10])
            lst.addItem(s(50005) + ts[11:19])
        w, h = meta.get("width"), meta.get("height")
        if w and h:
            lst.addItem(s(50006) + str(w))
            lst.addItem(s(50007) + str(h))
            lst.addItem(s(50010).replace("%.1f", f"{w * h / 1e6:.1f}"))
        if entry.is_raw:
            lst.addItem(s(50011))
        lst.addItem(s(50009).replace("%.1f", f"{size_mb:.1f}"))
        if meta.get("iso"):
            lst.addItem(f"ISO:  {meta['iso']:.0f}")
        if meta.get("shutter"):
            sh = meta["shutter"]
            lst.addItem(f"Shutter:  {'1/%d' % round(1 / sh) if sh < 1 else '%g' % sh} s")
        if meta.get("aperture"):
            lst.addItem(f"Aperture:  f/{meta['aperture']:.1f}")
        if meta.get("focal_length"):
            lst.addItem(f"Focal Length:  {meta['focal_length']:.0f} mm")


class HistogramPanel(QWidget):
    """Dialog template 1124: output histogram with optional R, G, B."""

    def __init__(self, res, parent=None):
        super().__init__(parent)
        b = self.b = R.build(res, self, 1124)
        default_size = res.size(res.pe.dialog(1124).size)
        self.setMinimumSize(*default_size)
        self.resize(*default_size)
        self.graph = HistogramWidget()
        self.graph.setParent(self)
        self._graph_geometry = b[1119].geometry()
        self._graph_right_margin = self.width() - self._graph_geometry.x() - self._graph_geometry.width()
        self._graph_bottom_margin = self.height() - self._graph_geometry.y() - self._graph_geometry.height()
        self.graph.setGeometry(self._graph_geometry)
        self.graph.show()
        for cid, ch in ((1120, "r"), (1121, "g"), (1122, "b")):
            b[cid].toggled.connect(lambda on, c=ch: self._toggle(c, on))
        self.set_appearance("light")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        g = self._graph_geometry
        self.graph.setGeometry(
            g.x(), g.y(),
            max(g.width(), self.width() - g.x() - self._graph_right_margin),
            max(g.height(), self.height() - g.y() - self._graph_bottom_margin),
        )

    def set_appearance(self, appearance):
        if appearance == "dark":
            style = (
                "QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #c0c0c0; "
                "border-radius: 2px; background-color: #292929; } "
                "QCheckBox::indicator:checked { background-color: #3478d4; border-color: #a9ccff; }"
            )
        else:
            style = (
                "QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #666666; "
                "border-radius: 2px; background-color: #ffffff; } "
                "QCheckBox::indicator:checked { background-color: #2673d9; border-color: #174e9e; }"
            )
        for cid in (1120, 1121, 1122):
            self.b[cid].setStyleSheet(style)

    def _toggle(self, ch, on):
        self.graph.channels[ch] = on
        self.graph.update()

    def set_image(self, img_u8):
        self.graph.set_image(img_u8)
        for cid in (1120, 1121, 1122):
            self.b[cid].setEnabled(img_u8 is not None)


CLOSEUP_ZOOMS = (25, 33, 50, 67, 100, 200, 300, 400)


class CloseUpWindow(QWidget):
    """Close-up window (template 257): pin, show parent, zoom in/out and a zoom list."""
    pin = Signal()
    show_parent = Signal()
    changed = Signal()

    def __init__(self, res):
        super().__init__(None, Qt.Tool)
        self.res = res
        b = self.b = R.build(res, self, 257)
        R.fix_combo_geometry(b, res)
        self.setMinimumSize(0, 0)
        self.setMaximumSize(16777215, 16777215)
        self.view = CloseUpView()
        self.view.setParent(self)
        self.view.setGeometry(b[1239].geometry())
        self.view.show()
        self.view.dragged.connect(self._drag)
        g = b[1247].geometry()
        cmds = ((1249, self.pin.emit), (1250, self.show_parent.emit), (1251, lambda: self.step(1)),
                (1252, lambda: self.step(-1)))
        ctx = ("Close-Up Window",)
        tips = (res.tr("Pin to image ^", ctx).replace("^", "").strip(), res.tr("Show ^", ctx).replace("^", "").strip(),
                res.tr("Zoom In", ctx), res.tr("Zoom Out", ctx))
        self.buttons = []
        for i, ((_, fn), tip) in enumerate(zip(cmds, tips)):
            btn = icon_button(self, QRect(g.x() + i * 22, g.y(), 22, 22), res.strip(266, 16, i), tip)
            btn.clicked.connect(fn)
            self.buttons.append(btn)
        combo = b[1256]
        combo.setEditable(False)
        for z in CLOSEUP_ZOOMS:
            combo.addItem(res.tr(f"{z}%", ("Close-Up Window Zoom-menu",)), z)
        combo.setCurrentIndex(CLOSEUP_ZOOMS.index(100))
        combo.currentIndexChanged.connect(lambda _: self.changed.emit())
        self.entry = None
        self.center = (0.5, 0.5)
        self.pinned = False
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)
        self.resize(self.size().width() * 2, self.size().height() * 2)

    def _menu(self, pos):
        from .app import build_menu
        handlers = {1249: self.pin.emit, 1250: self.show_parent.emit, 1251: lambda: self.step(1),
                    1252: lambda: self.step(-1)}
        menu = build_menu(self, self.res, self.res.menu(199)[0][2], handlers, ("Close-Up Window",))
        menu.exec(self.mapToGlobal(pos))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        g = self.b[1239].geometry()
        self.view.setGeometry(g.x(), g.y(), self.width() - 2 * g.x(), self.height() - g.y() - g.x())
        self.changed.emit()

    def zoom(self):
        return self.b[1256].currentData()

    def step(self, d):
        c = self.b[1256]
        c.setCurrentIndex(min(max(0, c.currentIndex() + d), c.count() - 1))

    def set_title(self):
        name = self.entry.path.name if self.entry else ""
        self.setWindowTitle(self.res.tr("Close-up: %s", ("Close-Up Window",)).replace("%s", name))
        ctx = ("Close-Up Window",)
        if self.pinned:
            self.buttons[0].setToolTip(self.res.tr("Pinned to image %s. Pinned close-ups cannot be unpinned.", ctx)
                                       .replace("%s", name))
            self.buttons[0].setEnabled(False)
        else:
            self.buttons[0].setToolTip(self.res.tr("Pin to image %s", ctx).replace("%s", name))
        self.buttons[1].setToolTip(self.res.tr("Show %s", ctx).replace("%s", name))

    def region(self, full_size):
        """Normalised rect of the oriented image shown at the current zoom."""
        z = self.zoom() / 100
        fw, fh = full_size
        w = self.view.width() / z / fw
        h = self.view.height() / z / fh
        cx, cy = self.center
        cx = min(max(cx, w / 2), 1 - w / 2) if w < 1 else 0.5
        cy = min(max(cy, h / 2), 1 - h / 2) if h < 1 else 0.5
        self.center = (cx, cy)
        return (max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(1.0, cx + w / 2), min(1.0, cy + h / 2))

    def _drag(self, dx, dy):
        if self.entry is None or not getattr(self, "full_size", None):
            return
        z = self.zoom() / 100
        self.center = (self.center[0] + dx / z / self.full_size[0], self.center[1] + dy / z / self.full_size[1])
        self.changed.emit()
