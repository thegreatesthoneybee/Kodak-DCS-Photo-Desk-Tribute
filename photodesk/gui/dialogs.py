"""Photo Desk dialogs, laid out from the original dialog templates."""
import datetime
import os

from PySide6.QtCore import QDate, Qt, Signal
from PySide6.QtGui import QPalette, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QLabel, QLineEdit, QMessageBox, QPushButton, QSlider, QTextEdit, QWidget,
)

from .. import APP_NAME, AUTHOR, BASED_ON, __version__, export, iptc, pipeline
from . import res as R
from .views import EVHistogram, to_qimage

OK, CANCEL = 1, 2


def message(parent, res, text, kind="warning", buttons=QMessageBox.Ok):
    box = QMessageBox(parent)
    box.setWindowTitle(APP_NAME)
    box.setText(text)
    box.setIcon({"warning": QMessageBox.Warning, "info": QMessageBox.Information,
                 "question": QMessageBox.Question, "error": QMessageBox.Critical}[kind])
    box.setStandardButtons(buttons)
    for b in box.buttons():
        role = box.standardButton(b)
        label = {QMessageBox.Ok: "OK", QMessageBox.Cancel: "Cancel", QMessageBox.Yes: "&Yes",
                 QMessageBox.No: "&No"}.get(role)
        if label:
            b.setText(res.tr(label, ("Common", "Common Strings")))
    return box.exec()


class TemplateDialog(QDialog):
    """QDialog whose controls come from dialog template rid."""
    rid = None

    def __init__(self, res, parent=None):
        super().__init__(parent)
        self.res = res
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.b = R.build(res, self, self.rid)
        R.fix_combo_geometry(self.b, res)
        if self.b.get(OK):
            self.b[OK].clicked.connect(self.accept)
        if self.b.get(CANCEL):
            self.b[CANCEL].clicked.connect(self.reject)

    def tr(self, text):
        return self.res.tr(text, R.DIALOG_CONTEXT.get(self.rid, ()))


# Messages of the Auto buttons, which Photo Desk does not have (English unless translated here).
AUTO_MESSAGES = {
    "noise": {"English": "The image has too few even areas to measure the noise.",
              "Korean": "이미지에 노이즈를 잴 수 있는 고른 영역이 너무 적습니다."},
    "edges": {"English": "The image has too few edges to measure the sharpness.",
              "Korean": "이미지에 선명도를 잴 수 있는 윤곽이 너무 적습니다."},
}


def auto_message(res, key):
    texts = AUTO_MESSAGES[key]
    return texts.get(res.language, texts["English"])


def auto_button(dlg, res, below, context):
    """Auto push button in the free slot above the template's At Capture button ``below`` (where the
    Lens Optimization dialog has its Auto); Photo Desk's Sharpening and Noise Reduction dialogs have
    none, Auto there measures the image."""
    at = dlg.b[below].geometry()
    b = QPushButton(res.tr("Auto", context), dlg)
    b.setFont(dlg.b[below].font())
    b.setGeometry(at.x(), at.y() - (at.height() + res.size((0, 3))[1]), at.width(), at.height())
    b.show()
    return b


class Linked:
    """Trackbar + edit (+ up-down) bound to one value in [lo, hi] with given decimals."""

    def __init__(self, slider, edit=None, spin=None, lo=0, hi=100, decimals=0, step=None, on_change=None,
                 invalid=None):
        self.slider, self.edit, self.lo, self.hi, self.dec = slider, edit, lo, hi, decimals
        self.k = 10 ** decimals
        self.step = step or 1 / self.k
        self.on_change = on_change
        self.invalid = invalid
        slider.setRange(round(lo * self.k), round(hi * self.k))
        slider.setSingleStep(max(1, round(self.step * self.k)))
        slider.setPageStep(max(1, round((hi - lo) * self.k / 10)))
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(max(1, round((hi - lo) * self.k / 10)))
        slider.valueChanged.connect(lambda v: self._set(v / self.k, from_slider=True))
        if edit is not None:
            edit.editingFinished.connect(self._edited)
        if spin is not None:
            spin.stepped.connect(lambda s: self.set(self.value() + s * self.step, notify=True))
        self._value = lo
        self._quiet = False

    def value(self):
        return round(self._value, self.dec) if self.dec else int(round(self._value))

    def text(self, v):
        return f"{v:.{self.dec}f}" if self.dec else str(int(round(v)))

    def _edited(self):
        try:
            v = float(self.edit.text())
        except ValueError:
            v = None
        if v is None or not self.lo <= v <= self.hi:
            if self.invalid:
                self.invalid(self.lo, self.hi)
            self.edit.setText(self.text(self._value))
            return
        self.set(v, notify=True)

    def _set(self, v, from_slider=False):
        if self._quiet:
            return
        self._value = min(max(v, self.lo), self.hi)
        if self.edit is not None:
            self.edit.setText(self.text(self._value))
        if self.on_change:
            self.on_change()

    def set(self, v, notify=False):
        v = min(max(v, self.lo), self.hi)
        self._quiet = True
        self.slider.setValue(round(v * self.k))
        self._quiet = False
        self._value = v
        if self.edit is not None:
            self.edit.setText(self.text(v))
        if notify and self.on_change:
            self.on_change()


class AdjustDialog(TemplateDialog):
    """Processing option dialog with Preview and At Capture, applied to the selection.

    preview(values or None) is called while Preview is on; values() returns the
    Settings fields the dialog edits.
    """
    preview_id = None
    capture_id = None
    fields = ()

    def __init__(self, res, parent, settings, preview, mixed=False, capture=None):
        super().__init__(res, parent)
        self._preview = preview
        self.settings = settings
        self.capture = capture or pipeline.Settings()   # what the camera set at capture
        if self.capture_id and self.b.get(self.capture_id):
            self.b[self.capture_id].clicked.connect(self._at_capture)
        if self.preview_id:
            self.b[self.preview_id].toggled.connect(lambda _: self.changed())
        self.setup(settings)
        if mixed and self.b.get(1224):
            self.b[1224].setText(res.s(None) if False else self.tr("Slider does not reflect setting for all images."))

    def setup(self, s):
        raise NotImplementedError

    def values(self):
        raise NotImplementedError

    def changed(self):
        if self.preview_id and self.b[self.preview_id].isChecked():
            self._preview(self.values())
        else:
            self._preview(None)

    def _at_capture(self):
        self.setup(self.capture)
        self.changed()

    def invalid(self, lo, hi):
        message(self, self.res, self.res.s(62063, lo, hi))

    def done(self, r):
        self._preview(None)
        super().done(r)


def _ev_labels(b, rng):
    for cid, f in zip((1100, 1101, 1322, 1102, 1103), (-1, -0.5, 0, 0.5, 1)):
        v = f * rng
        b[cid].setText("0" if v == 0 else f"{v:+.1f}")


class ExposureDialog(AdjustDialog):
    rid, preview_id = 129, 1341
    fields = ("exposure",)

    def __init__(self, res, parent, settings, preview, rng, auto, mixed=False, capture=None):
        self.rng, self.auto = rng, auto
        super().__init__(res, parent, settings, preview, mixed, capture)

    def setup(self, s):
        if not hasattr(self, "ev"):
            self.ev = Linked(self.b[1001], None, None, -self.rng, self.rng, 1, on_change=self._ev_changed)
            _ev_labels(self.b, self.rng)
            self.b[33131].setText("")
            self.b[33131].setIcon(R.QIcon(self.res.bitmap(253)))
            self.b[33131].setToolTip(self.res.tip(33131)[1])
            self.b[33131].clicked.connect(self._auto)
        self.ev.set(s.exposure)
        self._ev_changed(notify=False)

    def _auto(self):
        value = self.auto()
        if value is not None:
            self.ev.set(value, notify=True)

    def _ev_changed(self, notify=True):
        v = self.ev.value()
        self.b[1002].setText(f"{v:+.1f}" if v else "0.0")
        if notify:
            self.changed()

    def values(self):
        return {"exposure": self.ev.value()}


class ExposureToneDialog(ExposureDialog):
    rid, preview_id, capture_id = 281, 1336, 1335
    fields = ("exposure", "tone")

    def __init__(self, res, parent, settings, preview, rng, auto, hist, mixed=False, capture=None, curve=None):
        self.hist_fn, self.curve_fn = hist, curve
        super().__init__(res, parent, settings, preview, rng, auto, mixed, capture)

    def setup(self, s):
        if not hasattr(self, "graph"):
            frame = self.b[1119]
            self.graph = EVHistogram()
            self.graph.setParent(self)
            self.graph.setGeometry(frame.geometry())
            self.graph.show()
            self.graph.moved.connect(self._marker)
            for cid in (1314, 1316, 1318):
                self.b[cid].editingFinished.connect(self._tone_edited)
        self.tone_settings = s
        super().setup(s)
        self.tone = list(s.tone)
        self._show_tone()

    def _ev_changed(self, notify=True):
        super()._ev_changed(False)
        if hasattr(self, "graph"):
            self.graph.set_hist(self.hist_fn(self.ev.value()))
        if notify:
            self.changed()

    def _show_tone(self):
        for cid, v in zip((1314, 1316, 1318), self.tone):
            self.b[cid].setText(f"{v:.1f}")
        self.graph.set_tone(self.tone)
        self.graph.set_curve(self.curve_fn(tuple(self.tone), self.tone_settings) if self.curve_fn else None)

    def _check(self, tone):
        err = pipeline.validate_tone(tone)
        if err:
            message(self, self.res, self.tr(err))
            return False
        return True

    def _tone_edited(self):
        try:
            tone = [float(self.b[c].text()) for c in (1314, 1316, 1318)]
        except ValueError:
            self._show_tone()
            return
        if tone != self.tone and self._check(tone):
            self.tone = tone
            self.changed()
        self._show_tone()

    def _marker(self, i, ev):
        tone = list(self.tone)
        tone[i] = ev
        if pipeline.validate_tone(tone) is None:
            self.tone = tone
            self._show_tone()
            self.changed()

    def values(self):
        return {"exposure": self.ev.value(), "tone": tuple(self.tone)}


class WhitePointDialog(AdjustDialog):
    rid, preview_id = 137, 1340
    fields = ("white_point",)

    def setup(self, s):
        if not hasattr(self, "wp"):
            lo, hi = pipeline.WHITE_POINT_RANGE
            self.wp = Linked(self.b[1212], None, None, lo // 100, hi // 100, 0, on_change=self._show)
            self.b[1217].setVisible(False)
        self.wp.set(s.white_point // 100)
        self._show(False)

    def _show(self, notify=True):
        self.b[1213].setText(f"{self.wp.value() * 100}° K")
        if notify:
            self.changed()

    def values(self):
        return {"white_point": self.wp.value() * 100}


class ColorBalanceDialog(AdjustDialog):
    rid, preview_id, capture_id = 280, 1337, 1271
    fields = ("color_balance",)

    def setup(self, s):
        if not hasattr(self, "ch"):
            self.ch = [Linked(self.b[sl], self.b[ed], self.b[sp], -100, 100, 0, on_change=self.changed,
                              invalid=self.invalid)
                       for sl, ed, sp in ((1262, 1277, 1291), (1263, 1278, 1290), (1264, 1279, 1289))]
        for c, v in zip(self.ch, s.color_balance):
            c.set(v)

    def values(self):
        return {"color_balance": tuple(c.value() for c in self.ch)}


class LensDialog(AdjustDialog):
    rid, preview_id, capture_id = 1137, 1362, 1335
    fields = ("lens_optimization",)

    def __init__(self, res, parent, settings, preview, auto, mixed=False, capture=None):
        self.auto = auto
        super().__init__(res, parent, settings, preview, mixed, capture)

    def setup(self, s):
        if not hasattr(self, "st"):
            self.st = Linked(self.b[1359], self.b[1360], self.b[1361], 0, 99, 0, on_change=self.changed,
                             invalid=self.invalid)
            # Auto measures the corner colour cast on the image (Photo Desk itself only restored the
            # camera's strength, the same as At Capture).
            self.b[1272].clicked.connect(lambda: self.st.set(self.auto(), notify=True))
        self.st.set(s.lens_optimization)

    def values(self):
        return {"lens_optimization": self.st.value()}


class SharpenDialog(AdjustDialog):
    rid, preview_id, capture_id = 1132, 1339, 1345
    fields = ("sharpen_amount", "sharpen_radius", "sharpen_threshold")

    def __init__(self, res, parent, settings, preview, auto, mixed=False, capture=None):
        self.auto = auto            # () -> (amount, radius, threshold) measured on the image, or None
        super().__init__(res, parent, settings, preview, mixed, capture)

    def setup(self, s):
        if not hasattr(self, "amount"):
            self.auto_btn = auto_button(self, self.res, self.capture_id, R.DIALOG_CONTEXT.get(self.rid, ()))
            self.auto_btn.clicked.connect(self._auto)
            self.amount = Linked(self.b[9401], self.b[9404], self.b[9408], 0, 20, 1, on_change=self.changed,
                                 invalid=self._invalid1)
            self.radius = Linked(self.b[9402], self.b[9405], self.b[9409], 0, 10, 1, on_change=self.changed,
                                 invalid=self._invalid1)
            self.thr = Linked(self.b[9403], self.b[9406], self.b[9410], 0, 255, 0, on_change=self.changed,
                              invalid=self.invalid)
            for cid, name in ((8, "none"), (7, "low"), (10, "medium"), (9, "high")):
                self.b[cid].clicked.connect(lambda _=False, n=name: self._preset(n))
        self.amount.set(s.sharpen_amount)
        self.radius.set(s.sharpen_radius)
        self.thr.set(s.sharpen_threshold)

    def _invalid1(self, lo, hi):
        message(self, self.res, self.res.s(62065).replace("%.1f", f"{lo:.1f}", 1).replace("%.1f", f"{hi:.1f}", 1))

    def _preset(self, name):
        self._set_values(pipeline.SHARPEN_PRESETS[name])

    def _auto(self):
        v = self.auto()
        if v is None:
            message(self, self.res, auto_message(self.res, "edges"), "info")
            return
        self._set_values(v)

    def _set_values(self, v):
        a, r, t = v
        self.amount.set(a)
        self.radius.set(r)
        self.thr.set(t)
        self.changed()

    def values(self):
        return {"sharpen_amount": self.amount.value(), "sharpen_radius": self.radius.value(),
                "sharpen_threshold": self.thr.value()}


NOISE_LABEL = {"none": 33140, "basic": 33141, "advanced": 33142, "moire": 33143, "expert": 33146}


# Radius tick statics of template 1133, left to right (1233 "1" ... 1234 "10"), and the hidden "4"
# (1245) Photo Desk shows at the right end for Advanced (radius 1..4).
RADIUS_TICKS = (1233, 1239, 1240, 1237, 1241, 1242, 1238, 1243, 1244, 1234)


class NoiseDialog(QDialog):
    """Noise Reduction as in Photo Desk 4.3.

    Pro Back / Pro SLR images (ProImage) always use template 1133: Basic (radius 2..6 in steps of
    2), Advanced (1..4) and Advanced with Moire Reduction (1..10) take Radius and Strength, Expert a
    single Strength; Low/Medium/High come from the file's preset tables. Older DCS images use
    template 1134: only Advanced with Moire Reduction has a setting, its Threshold.
    """
    fields = ("noise_type", "noise_radius", "noise_strength", "noise_luma", "moire_threshold")

    def __init__(self, res, parent, settings, preview, caps, auto, mixed=False, capture=None, presets=None):
        super().__init__(parent)
        self.res, self._preview, self.caps, self.mixed = res, preview, caps, mixed
        self.auto = auto            # (noise type) -> noise fields measured on the image, or None
        self.capture = capture or pipeline.Settings()
        self.presets = presets or pipeline.noise_presets()
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.values_ = {k: getattr(settings, k) for k in
                        ("noise_type", "noise_radius", "noise_strength", "noise_luma", "moire_threshold")}
        self.body = None
        self._build()

    def _template_for(self, t):
        if self.caps.pro_image:
            return 1133
        # Older DCS images: the threshold page, or (unlocked) radius/strength for the other types.
        return 1133 if self.caps.unlocked and self.caps.noise_controls(t) in ("radius", "strength") else 1134

    def _build(self):
        t = self.values_["noise_type"]
        rid = self._template_for(t)
        preview_on = self.body is not None and self.body.get(self._pid) and self.body[self._pid].isChecked()
        if self.body is not None:
            for w in [w for _, w in self.body.all] + [self.auto_btn]:
                w.hide()
                w.deleteLater()
        self.rid = rid
        self.b = self.body = R.build(self.res, self, rid)
        R.fix_combo_geometry(self.b, self.res)
        for _, w in self.b.all:
            w.show() if w.isVisibleTo(self) or True else None
        for c, w in self.b.all:
            if not c.style & R.WS_VISIBLE:
                w.hide()
        self._pid = 1343 if rid == 1133 else 1342
        combo = self.b[1219]
        combo.setEditable(False)
        for name in self.caps.noise_types:
            combo.addItem(self.res.s(NOISE_LABEL[name]), name)
        combo.setCurrentIndex(max(0, combo.findData(t)))
        combo.currentIndexChanged.connect(self._type_changed)
        self.b[OK].clicked.connect(self.accept)
        self.b[CANCEL].clicked.connect(self.reject)
        self.b[self._pid].setChecked(bool(preview_on))
        self.b[self._pid].toggled.connect(lambda _: self.changed())
        # Auto above At Capture (template 1133) or above Preview (1134, which has no At Capture).
        self.auto_btn = auto_button(self, self.res, 8 if rid == 1133 else self._pid, R.DIALOG_CONTEXT.get(rid, ()))
        self.auto_btn.clicked.connect(self._auto)
        self.auto_btn.setEnabled(self.caps.noise_controls(t) != "none")
        if rid == 1133:
            self._build_1133(t)
        else:
            self.first = None
            self.second = Linked(self.b[9403], self.b[9406], self.b[9410], 0, 100, 0, on_change=self._edited,
                                 invalid=self.invalid)
            self.second.set(self.values_["moire_threshold"])
            on = t == "moire"
            for cid in (9403, 9406, 9410):
                self.b[cid].setEnabled(on)
            if not on and t != "none":
                self.b[1224].setText(self.res.s(62046))
        if self.mixed and not self.b[1224].text():
            self.b[1224].setText(self.res.tr("Slider does not reflect setting for all images.",
                                             ("Noise Reduction Dialog", "Moiré Threshold Dialog")))
        self.setWindowTitle(self.res.tr("Noise Reduction", ("Noise Reduction Dialog",)))
        self.setFixedSize(*self.res.size(self.res.pe.dialog(rid).size))

    def _build_1133(self, t):
        b, res = self.b, self.res
        kind = self.caps.noise_controls(t)
        for cid in RADIUS_TICKS[1:-1] + (1245,):
            b[cid].hide()
        if kind == "strength":
            # Expert: one Strength slider (0..100 %); the radius row is hidden.
            b[1348].setText(res.s(62119))
            for cid in (1347, 9402, 9405, 9409) + RADIUS_TICKS:
                b[cid].hide()
            self.first = None
            self.second = Linked(b[9403], b[9406], b[9410], 0, 100, 0, on_change=self._edited,
                                 invalid=self.invalid)
            self.second.set(self.values_["noise_strength"])
        else:
            lo, hi = pipeline.NOISE_RADIUS_RANGE.get(t, (1, 10))
            step = 2 if t == "basic" else 1
            b[1347].setText(res.s(62118))
            b[1348].setText(res.s(62117))
            # Ticks: first and last radius of the type (Advanced shows its "4" at the right end).
            b[1233].setText(str(lo))
            b[1234].setText(str(hi))
            self.first = Linked(b[9402], b[9405], b[9409], lo, hi, 0, step=step, on_change=self._edited,
                                invalid=self.invalid)
            self.first.slider.setTickInterval(step)
            self.first.slider.setPageStep(step)
            self.first.set(self.values_["noise_radius"])
            self.second = Linked(b[9403], b[9406], b[9410], 0, 100, 0, on_change=self._edited,
                                 invalid=self.invalid)
            self.second.set(self.values_["noise_strength"])
            enabled = kind == "radius"
            for cid in (9402, 9403, 9405, 9406, 9409, 9410):
                b[cid].setEnabled(enabled)
        enabled = kind != "none"
        for cid, name in ((7, "low"), (10, "medium"), (9, "high")):
            b[cid].clicked.connect(lambda _=False, n=name: self._preset(n))
            b[cid].setEnabled(enabled)
        b[8].clicked.connect(self._at_capture)

    def invalid(self, lo, hi):
        message(self, self.res, self.res.s(62063, lo, hi))

    def _type_changed(self):
        t = self.b[1219].currentData()
        self.values_["noise_type"] = t
        if self.rid == 1133:
            # Photo Desk switches to the type's Medium preset (keeps it within the new range).
            pipeline.apply_noise_preset(self.values_, "medium", self.presets)
            self._build()
        elif self._template_for(t) != self.rid:
            self._build()
        else:
            on = t == "moire"
            for cid in (9403, 9406, 9410):
                self.b[cid].setEnabled(on)
            self.b[1224].setText("" if on or t == "none" else self.res.s(62046))
        self.changed()

    def _edited(self):
        t = self.values_["noise_type"]
        if self.rid == 1133:
            if self.first is None:
                self.values_["noise_strength"] = self.values_["noise_luma"] = self.second.value()
            else:
                r = self.first.value()
                if t == "basic":
                    r = min(6, max(2, 2 * round(r / 2)))
                    if r != self.first.value():
                        self.first.set(r)
                self.values_["noise_radius"], self.values_["noise_strength"] = r, self.second.value()
        else:
            self.values_["moire_threshold"] = self.second.value()
        self.changed()

    def _preset(self, name):
        pipeline.apply_noise_preset(self.values_, name, self.presets)
        self._build()
        self.changed()

    def _auto(self):
        v = self.auto(self.values_["noise_type"])
        if v is None:
            message(self, self.res, auto_message(self.res, "noise"), "info")
            return
        self.values_.update(v)
        self._build()
        self.changed()

    def _at_capture(self):
        s = self.capture
        for k in self.values_:
            self.values_[k] = getattr(s, k)
        self._build()
        self.changed()

    def changed(self):
        self._preview(self.values() if self.b[self._pid].isChecked() else None)

    def values(self):
        return dict(self.values_)

    def done(self, r):
        self._preview(None)
        super().done(r)


class CropRatioDialog(TemplateDialog):
    rid = 1130

    def __init__(self, res, parent, ratio):
        super().__init__(res, parent)
        w, h = ratio or (4, 6)
        self.b[1175].setText(f"{w:g}")
        self.b[1176].setText(f"{h:g}")

    def accept(self):
        try:
            w, h = float(self.b[1175].text()), float(self.b[1176].text())
        except ValueError:
            w = h = 0
        if not (0 < w <= 1000 and 0 < h <= 1000):
            message(self, self.res, self.res.s(62063, "0.01", "1000"))
            return
        self.ratio = (w, h)
        super().accept()


class ScaleViewDialog(TemplateDialog):
    rid = 131

    def __init__(self, res, parent, zoom):
        super().__init__(res, parent)
        self.b[1011].setText(str(zoom))

    def accept(self):
        try:
            z = int(self.b[1011].text())
        except ValueError:
            z = 0
        if not 1 <= z <= 1600:
            message(self, self.res, self.res.s(62063, 1, 1600))
            return
        self.zoom = z
        super().accept()


class PasteSpecialDialog(TemplateDialog):
    rid = 292

    def __init__(self, res, parent, groups, preview):
        super().__init__(res, parent)
        self.groups = groups
        self._preview = preview
        for cid in groups:
            self.b[cid].setChecked(True)
            self.b[cid].toggled.connect(self.changed)
        self.b[4].clicked.connect(lambda: self._all(True))
        self.b[6].clicked.connect(lambda: self._all(False))
        self.b[1349].toggled.connect(self.changed)

    def _all(self, on):
        for cid in self.groups:
            self.b[cid].setChecked(on)

    def fields(self):
        return tuple(f for cid, fs in self.groups.items() if self.b[cid].isChecked() for f in fs)

    def changed(self, *_):
        self._preview(self.fields() if self.b[1349].isChecked() else None)

    def done(self, r):
        self._preview(None)
        super().done(r)


class RenameDialog(TemplateDialog):
    rid = 213

    def __init__(self, res, parent, entries, qsettings):
        super().__init__(res, parent)
        self.entries, self.q = entries, qsettings
        multi = len(entries) > 1
        for cid in (1160, 1161, 1162, 1163, 1164, 1168, 1165):
            self.b[cid].setVisible(multi)
        if multi:
            self.b[1159].setText(qsettings.value("Save As Basename", "image"))
            self.b[1162].setText(str(qsettings.value("Save As Num Digits", 4)))
            self.b[1163].setText(str(qsettings.value("Save As Start From", 1)))
        else:
            self.b[1159].setText(entries[0].path.name)
        for cid in (1159, 1162, 1163):
            self.b[cid].textChanged.connect(self._update)
        self._update()

    def _validate(self):
        """Return (names, error) using the original's rules and messages."""
        s = self.res.s
        base = self.b[1159].text().strip()
        if not base:
            return None, s(60034)
        if any(c in base for c in '\\/:?"<>|'):
            return None, s(60017)
        if len(base) > 200:
            return None, s(60016)
        if len(self.entries) == 1:          # one file: the name is typed with its extension
            return [base], None
        if base.count("*") > 1:
            return None, s(60044)
        digits, start = self.b[1162].text().strip(), self.b[1163].text().strip()
        if not digits.isdigit():
            return None, s(60018)
        if int(digits) >= 10:
            return None, s(60019)
        if not start.isdigit():
            return None, s(60020)
        if len(start) >= 10:
            return None, s(60021)
        if len(str(int(start))) > int(digits):
            return None, s(60035)
        if len(str(int(start) + len(self.entries) - 1)) > int(digits):
            return None, s(62064)
        return export.sequence_names(base, int(digits), int(start), len(self.entries)), None

    def _update(self):
        names, err = self._validate()
        self.b[1167].setText(err.strip() if err else "")
        if names and len(self.entries) > 1:
            self.b[1165].setText(names[0] + self.entries[0].path.suffix)
        self.b[OK].setEnabled(names is not None)

    def accept(self):
        names, err = self._validate()
        if err:
            message(self, self.res, err)
            return
        if len(self.entries) == 1:
            new_ext = os.path.splitext(names[0])[1].lower()
            if new_ext != self.entries[0].path.suffix.lower() and message(
                    self, self.res, self.res.s(60051), "question",
                    QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
                return
        # Full file names: several files keep their own extensions.
        self.names = names if len(self.entries) == 1 else [n + e.path.suffix for n, e in zip(names, self.entries)]
        if len(self.entries) > 1:
            self.q.setValue("Save As Basename", self.b[1159].text().strip())
            self.q.setValue("Save As Num Digits", int(self.b[1162].text()))
            self.q.setValue("Save As Start From", int(self.b[1163].text()))
        super().accept()


class PreferencesDialog(TemplateDialog):
    """Edit > Preferences: RGB working space, send-to folder, export size, TIFF and JPEG options."""
    rid = 214
    APPEARANCE_LABELS = {
        "Chinese": ("外观", "浅色", "深色"),
        "French": ("Apparence", "Clair", "Sombre"),
        "German": ("Darstellung", "Hell", "Dunkel"),
        "Italian": ("Aspetto", "Chiaro", "Scuro"),
        "Japanese": ("外観", "ライト", "ダーク"),
        "Korean": ("화면 테마", "밝게", "어둡게"),
        "Portugese": ("Aparência", "Claro", "Escuro"),
        "Spanish": ("Apariencia", "Claro", "Oscuro"),
        "Traditional Chinese": ("外觀", "淺色", "深色"),
    }

    # Not in the original Photo Desk: lifts the per-camera-model limits on processing options.
    UNLOCK_LABELS = {
        "Chinese": "解除所有相机型号的功能限制",
        "French": "Débloquer toutes les fonctions pour tous les appareils",
        "German": "Alle Funktionen für alle Kameramodelle freischalten",
        "Italian": "Sblocca tutte le funzioni per tutti i modelli",
        "Japanese": "すべての機種で全機能を使用可能にする",
        "Korean": "기종 제한 해제 (모든 기종에서 모든 기능 사용)",
        "Portugese": "Desbloquear todas as funções para todos os modelos",
        "Spanish": "Desbloquear todas las funciones para todos los modelos",
        "Traditional Chinese": "解除所有相機型號的功能限制",
    }

    def __init__(self, res, parent, opts, send_folder, spaces, appearance="light", unlock_all=False):
        super().__init__(res, parent)
        b = self.b
        self.opts = export.OutputOptions.from_json(opts.to_json())
        self.spaces = spaces
        self.appearance = appearance
        self.unlock_all = unlock_all
        labels = self.APPEARANCE_LABELS.get(res.language, ("Appearance", "Light", "Dark"))
        extra_height = res.size((0, 48))[1]
        self.setFixedSize(self.width(), self.height() + extra_height)
        label = QLabel(labels[0], self)
        label.setGeometry(res.rect((14, 416, 68, 14)))
        self.appearance_combo = QComboBox(self)
        self.appearance_combo.setGeometry(res.rect((82, 414, 105, 16)))
        self.appearance_combo.addItem(labels[1], "light")
        self.appearance_combo.addItem(labels[2], "dark")
        self.appearance_combo.setCurrentIndex(max(0, self.appearance_combo.findData(appearance)))
        self.unlock_check = QCheckBox(
            self.UNLOCK_LABELS.get(res.language, "Unlock all features for every camera model"), self)
        self.unlock_check.setGeometry(res.rect((14, 434, 260, 14)))
        self.unlock_check.setChecked(unlock_all)
        b[1179].setEditable(False)
        for path, desc in spaces:
            b[1179].addItem(desc, path)
        current = pipeline.resolve_space(self.opts.working_space)
        idx = next((i for i, (p, _) in enumerate(spaces) if os.path.normcase(p) == os.path.normcase(current)), 0)
        b[1179].setCurrentIndex(idx)
        b[1130].setText(send_folder or "")
        b[1131].clicked.connect(self._choose)
        b[1338].setChecked(self.opts.resize)
        {"percent": b[1366], "fill": b[1365], "fit_trim": b[1363], "fit_fill": b[1364]}[self.opts.scale_style].setChecked(True)
        b[1368].setText(str(self.opts.percent))
        b[1369].setText(str(self.opts.width))
        b[1278].setText(str(self.opts.height))
        b[1339].setChecked(True)
        b[1082].setText(f"{self.opts.resolution:g}")
        b[1375].setEditable(False)
        b[1375].addItems([res.s(62159), res.s(62160)])
        b[1375].setCurrentIndex(1 if self.opts.resolution_unit == "cm" else 0)
        (b[1085] if self.opts.tiff_linear else b[1182]).setChecked(True)
        (b[1354] if self.opts.tiff_16bit else b[1353]).setChecked(True)
        b[1087].setChecked(self.opts.render_gps)
        b[1180].setChecked(self.opts.embed_tiff_profile)
        {"best": b[1006], "better": b[1007], "good": b[1008]}[self.opts.jpeg_quality].setChecked(True)
        b[1181].setChecked(self.opts.embed_jpeg_profile)
        self._ratio = self.opts.width / max(1, self.opts.height)
        self._busy = False
        b[1369].textEdited.connect(lambda _: self._pixels(1369))
        b[1278].textEdited.connect(lambda _: self._pixels(1278))
        b[1279].textEdited.connect(lambda _: self._print(1279))
        b[1280].textEdited.connect(lambda _: self._print(1280))
        b[1082].textEdited.connect(lambda _: self._pixels(1369))
        b[1375].currentIndexChanged.connect(self._units)
        b[1338].toggled.connect(self._enable)
        for cid in (1366, 1365, 1363, 1364):
            b[cid].toggled.connect(self._enable)
        self._units()
        self._pixels(1369, keep=True)
        self._enable()

    def _choose(self):
        d = QFileDialog.getExistingDirectory(self, self.res.s(59508), self.b[1130].text())
        if d:
            self.b[1130].setText(d)

    def _num(self, cid, default=0.0):
        try:
            return float(self.b[cid].text())
        except ValueError:
            return default

    def _dpu(self):
        return max(1.0, self._num(1082, 300))

    def _units(self):
        cm = self.b[1375].currentIndex() == 1
        for cid in (1374, 1376):
            self.b[cid].setVisible(not cm)
        for cid in (1392, 1393):
            self.b[cid].setVisible(cm)
        self._pixels(1369, keep=True)

    def _pixels(self, cid, keep=False):
        if self._busy:
            return
        self._busy = True
        w, h = self._num(1369), self._num(1278)
        if not keep and self.b[1339].isChecked() and w and h:
            if cid == 1369:
                h = round(w / self._ratio)
                self.b[1278].setText(str(int(h)))
            else:
                w = round(h * self._ratio)
                self.b[1369].setText(str(int(w)))
        if w and h:
            self._ratio = w / h
        self.b[1279].setText(f"{w / self._dpu():.3f}")
        self.b[1280].setText(f"{h / self._dpu():.3f}")
        self._busy = False

    def _print(self, cid):
        if self._busy:
            return
        v = self._num(cid)
        target = 1369 if cid == 1279 else 1278
        self.b[target].setText(str(int(round(v * self._dpu()))))
        self._pixels(target)

    def _enable(self):
        on = self.b[1338].isChecked()
        for cid in (1391, 1366, 1365, 1363, 1364, 1390):
            self.b[cid].setEnabled(on)
        self.b[1368].setEnabled(on and self.b[1366].isChecked())
        frame = on and not self.b[1366].isChecked()
        for cid in (1384, 1385, 1386, 1369, 1387, 1279, 1388, 1278, 1389, 1280, 1339, 1374, 1376, 1392, 1393):
            self.b[cid].setEnabled(frame)

    def accept(self):
        s = self.res.s
        b = self.b
        o = self.opts
        if b[1338].isChecked():
            if b[1366].isChecked():
                p = self._num(1368)
                if not 10 <= p <= 400:
                    message(self, self.res, s(62063, 10, 400))
                    return
                o.percent = int(p)
            else:
                w, h = self._num(1369), self._num(1278)
                if not (50 <= w <= 18000 and 50 <= h <= 18000):
                    message(self, self.res, s(62063, 50, 18000))
                    return
                o.width, o.height = int(w), int(h)
        r = self._num(1082)
        if not 1 <= r <= 4000:
            message(self, self.res, s(62063, 1, 4000))
            return
        o.resolution = r
        o.resolution_unit = "cm" if b[1375].currentIndex() == 1 else "in"
        if b[1179].currentData():
            o.working_space = pipeline.space_setting(b[1179].currentData())
        o.resize = b[1338].isChecked()
        o.scale_style = ("percent" if b[1366].isChecked() else "fill" if b[1365].isChecked()
                         else "fit_trim" if b[1363].isChecked() else "fit_fill")
        o.tiff_linear = b[1085].isChecked()
        o.tiff_16bit = b[1354].isChecked()
        o.render_gps = b[1087].isChecked()
        o.embed_tiff_profile = b[1180].isChecked()
        o.jpeg_quality = "best" if b[1006].isChecked() else "better" if b[1007].isChecked() else "good"
        o.embed_jpeg_profile = b[1181].isChecked()
        self.send_folder = b[1130].text().strip()
        self.appearance = self.appearance_combo.currentData()
        self.unlock_all = self.unlock_check.isChecked()
        super().accept()


class PrintingPreferencesDialog(TemplateDialog):
    rid = 250
    SIZES = {1183: "view", 1184: "small", 1185: "medium", 1186: "large"}

    def __init__(self, res, parent, size, docking):
        super().__init__(res, parent)
        for cid, name in self.SIZES.items():
            self.b[cid].setChecked(name == size)
        self.b[1259].setChecked(docking)

    def accept(self):
        self.size = next(n for c, n in self.SIZES.items() if self.b[c].isChecked())
        self.docking = self.b[1259].isChecked()
        super().accept()


class IPTCDialog(TemplateDialog):
    """IPTC / Job Tracker data of the selected images, one at a time."""
    rid = 1121
    FIELD_IDS = {
        "object_name": 1008, "city": 1010, "province_state": 1011, "country": 1012, "date_created": 1009,
        "category": 1021, "urgency": 1027, "supplemental_category": 1022, "byline": 1013,
        "byline_title": 1014, "credit": 1015, "original_transmission_reference": 1017, "source": 1016,
        "caption_writer": 1020, "keywords": 1023, "caption": 1028, "headline": 1019, "copyright": 1024,
        "special_instructions": 1025,
    }

    def __init__(self, res, parent, entries, qsettings):
        super().__init__(res, parent)
        self.entries, self.q = entries, qsettings
        self.data = [dict(e.settings.iptc) for e in entries]
        self.index = 0
        self.dirty = False
        urg = self.b[1027]
        urg.setEditable(False)
        for i, label in enumerate(iptc.URGENCY):
            urg.addItem(res.tr(label or "None", ("IPTC Dialog Urgency menu",)), str(i) if i else "")
        self.b[1009].setDisplayFormat("yyyy/MM/dd")
        self.b[1009].setSpecialValueText(" ")
        self.b[1009].setMinimumDate(QDate(1000, 1, 1))
        for key, cid in self.FIELD_IDS.items():
            w = self.b[cid]
            limit = next(n for k, _, n, _ in iptc.FIELDS if k == key)
            if isinstance(w, QLineEdit):
                w.setMaxLength(limit if key not in ("keywords", "supplemental_category") else 2000)
                w.textEdited.connect(self._touched)
            elif isinstance(w, QTextEdit):
                w.textChanged.connect(self._touched)
            elif isinstance(w, QComboBox):
                w.activated.connect(self._touched)
            else:
                w.dateChanged.connect(self._touched)
        self.b[OK].clicked.disconnect()
        self.b[OK].clicked.connect(self._done)
        self.b[1001].clicked.connect(self._load)
        self.b[1002].clicked.connect(self._save_as)
        self.b[1007].clicked.connect(self._clear)
        self.b[1006].clicked.connect(lambda: self._go(-1))
        self.b[1005].clicked.connect(lambda: self._go(1))
        self.b[1157].clicked.connect(self._get_last)
        self.b[1029].clicked.connect(self._apply_all)
        self.b[1004].clicked.connect(self._commit)
        self._show()

    def _touched(self, *_):
        self.dirty = True

    def fields(self):
        out = {}
        for key, cid in self.FIELD_IDS.items():
            w = self.b[cid]
            if isinstance(w, QLineEdit):
                v = w.text()
            elif isinstance(w, QTextEdit):
                v = w.toPlainText()
            elif isinstance(w, QComboBox):
                v = w.currentData() or ""
            else:
                v = "" if w.date() == w.minimumDate() else w.date().toString("yyyyMMdd")
            if v.strip():
                out[key] = v
        return out

    def set_fields(self, d):
        for key, cid in self.FIELD_IDS.items():
            w = self.b[cid]
            v = d.get(key, "")
            if isinstance(w, QLineEdit):
                w.setText(v)
            elif isinstance(w, QTextEdit):
                w.blockSignals(True)
                w.setPlainText(v)
                w.blockSignals(False)
            elif isinstance(w, QComboBox):
                w.setCurrentIndex(max(0, w.findData(v)))
            else:
                w.blockSignals(True)
                date = QDate.fromString(v, "yyyyMMdd") if v else w.minimumDate()
                w.setDate(date if date.isValid() else w.minimumDate())
                w.blockSignals(False)

    def _show(self):
        e = self.entries[self.index]
        self.setWindowTitle(self.res.s(59952).replace("%s", e.path.name))
        self.set_fields(self.data[self.index])
        self.b[1158].setText(self.res.s(59953).replace("%d", str(self.index + 1), 1).replace("%d", str(len(self.entries)), 1))
        self.b[1006].setEnabled(self.index > 0)
        self.b[1005].setEnabled(self.index < len(self.entries) - 1)
        if e.thumb is not None:
            frame = self.b[1026]
            frame.setPixmap(QPixmap.fromImage(e.thumb).scaled(frame.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            frame.setAlignment(Qt.AlignCenter)
        self.dirty = False

    def _commit(self):
        self.data[self.index] = self.fields()
        self.q.setValue("IPTC/Record 2", iptc.encode(self.data[self.index]).hex())
        self.dirty = False

    def _go(self, step):
        self._commit()
        self.index = min(max(0, self.index + step), len(self.entries) - 1)
        self._show()

    def _clear(self):
        self.set_fields({})
        self.dirty = True

    def _get_last(self):
        blob = self.q.value("IPTC/Record 2", "")
        if blob:
            self.set_fields(iptc.decode(bytes.fromhex(blob)))
            self.dirty = True

    def _apply_all(self):
        d = self.fields()
        self.data = [dict(d) for _ in self.entries]
        self.q.setValue("IPTC/Record 2", iptc.encode(d).hex())
        self.dirty = False

    def _load(self):
        if self.dirty and message(self, self.res, self.res.s(60013), "question",
                                  QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        flt = self.res.s(62067).split("|")
        path, _ = QFileDialog.getOpenFileName(self, "", "", f"{flt[0]} ({flt[1]});;{flt[2]} ({flt[3]})")
        if not path:
            return
        try:
            self.set_fields(iptc.load_ipt(path))
        except (OSError, ValueError):
            message(self, self.res, self.res.s(60011).replace("%s", path))
            return
        self.dirty = True

    def _save_as(self):
        flt = self.res.s(62067).split("|")
        path, _ = QFileDialog.getSaveFileName(self, "", "", f"{flt[0]} ({flt[1]})")
        if not path:
            return
        if not path.lower().endswith(".ipt"):
            path += ".ipt"
        try:
            iptc.save_ipt(path, self.fields())
        except OSError:
            message(self, self.res, self.res.s(60012).replace("%s", path))

    def _done(self):
        self._commit()
        self.accept()

    def reject(self):
        if self.dirty:
            r = message(self, self.res, self.res.s(59951), "question",
                        QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if r == QMessageBox.Cancel:
                return
            if r == QMessageBox.Yes:
                self._done()
                return
        super().reject()


class ReviewDialog(TemplateDialog):
    """Review Selection: step through the selected images with Faster/Better rendering and tagging."""
    rid = 197
    navigate = Signal(int)

    def __init__(self, res, parent, entries, render, set_tag, quality):
        super().__init__(res, parent)
        self.entries, self.render, self.set_tag = entries, render, set_tag
        self.index = 0
        (self.b[1079] if quality == "better" else self.b[1080]).setChecked(True)
        self.b[1080].toggled.connect(lambda _: self._show())
        self.b[1077].toggled.connect(self._tag)
        self.b[1071].clicked.connect(lambda: self.go(self.index - 1))
        self.b[1072].clicked.connect(lambda: self.go(self.index + 1))
        self.b[1078].clicked.connect(lambda: self.go(0))
        self.b[1074].clicked.connect(lambda: self.go(len(self.entries) - 1))
        self.b[OK].clicked.disconnect()
        self.b[OK].clicked.connect(self.accept)
        self.area = self.b[1070]
        self.area.setAlignment(Qt.AlignCenter)
        self.area.setBackgroundRole(QPalette.ColorRole.Dark)
        self.area.setAutoFillBackground(True)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)
        self._show()

    def quality(self):
        return "better" if self.b[1079].isChecked() else "faster"

    def _menu(self, pos):
        from .app import build_menu
        menu = build_menu(self, self.res, self.res.menu(198)[0][2], {
            1077: lambda: self.b[1077].toggle(), 1071: lambda: self.go(self.index - 1),
            1072: lambda: self.go(self.index + 1), 1078: lambda: self.go(0),
            1074: lambda: self.go(len(self.entries) - 1)}, ("Review Dialog",))
        menu.exec(self.mapToGlobal(pos))

    def go(self, i):
        if 0 <= i < len(self.entries):
            self.index = i
            self._show()

    def _tag(self, on):
        self.set_tag(self.entries[self.index], on)

    def _show(self):
        e = self.entries[self.index]
        self.setWindowTitle(self.res.s(59955).replace("%s", e.path.name))
        self.b[1077].blockSignals(True)
        self.b[1077].setChecked(e.settings.tagged)
        self.b[1077].blockSignals(False)
        self.b[1073].setText(self.res.s(59953).replace("%d", str(self.index + 1), 1)
                             .replace("%d", str(len(self.entries)), 1))
        for cid, on in ((1071, self.index > 0), (1078, self.index > 0),
                        (1072, self.index < len(self.entries) - 1), (1074, self.index < len(self.entries) - 1)):
            self.b[cid].setEnabled(on)
        self.area.setText("")
        self.render(e, self.quality(), self.area.size(), self._rendered)

    def _rendered(self, entry, qimage):
        if entry is self.entries[self.index]:
            self.area.setPixmap(QPixmap.fromImage(qimage).scaled(self.area.size(), Qt.KeepAspectRatio,
                                                                  Qt.SmoothTransformation))


class YesToAllDialog(TemplateDialog):
    rid = 165
    YES_ALL = 1061

    def __init__(self, res, parent, text):
        super().__init__(res, parent)
        self.setWindowTitle(APP_NAME)
        self.b[1062].setText(text)
        self.b[OK].clicked.disconnect()
        self.b[OK].clicked.connect(lambda: self.done(OK))
        self.b[CANCEL].clicked.disconnect()
        self.b[CANCEL].clicked.connect(lambda: self.done(CANCEL))
        self.b[self.YES_ALL].clicked.connect(lambda: self.done(self.YES_ALL))
        icon = self.b.get(-1)
        try:
            self.b.all[-1][1].setPixmap(res.bitmap(167))
        except KeyError:
            pass


class ProgressDialog(TemplateDialog):
    rid = 102

    def __init__(self, res, parent, title, total):
        super().__init__(res, parent)
        self.setWindowTitle(title)
        self.total = max(1, total)
        self.canceled = False
        self.b[1003].setRange(0, self.total)
        self.b[CANCEL].clicked.disconnect()
        self.b[CANCEL].clicked.connect(self._cancel)
        self.set(0)

    def _cancel(self):
        self.canceled = True

    def set(self, n):
        self.b[1003].setValue(n)
        self.b[1004].setText(f"{n * 100 // self.total} %")


class AboutDialog(TemplateDialog):
    rid = 100

    def __init__(self, res, parent):
        super().__init__(res, parent)
        self.setWindowTitle(f"About {APP_NAME}")
        self.setFixedHeight(self.height() + res.size((0, 26))[1])
        self.b.all[0][1].setPixmap(res.app_icon().pixmap(32, 32))
        self.b.all[1][1].setText(f"{APP_NAME} v{__version__} — Based on {BASED_ON}")
        self.b.all[1][1].setGeometry(res.rect((40, 8, 264, 10)))
        self.b.all[2][1].setText(f"Created by {AUTHOR}")
        self.b.all[2][1].setGeometry(res.rect((40, 24, 200, 9)))
        self.b[OK].setGeometry(res.rect((247, 49, 50, 14)))


class Splash(QDialog):
    """Start-up/about splash (template 1122, bitmap 193)."""

    def __init__(self, res, parent=None):
        super().__init__(parent, Qt.SplashScreen | Qt.FramelessWindowHint)
        self.b = R.build(res, self, 1122)
        btn = self.b[OK]
        btn.setFlat(True)
        btn.setIconSize(btn.size())
        btn.setIcon(R.QIcon(res.bitmap(193, transparent=False).scaled(btn.size(), Qt.IgnoreAspectRatio,
                                                                       Qt.SmoothTransformation)))
        btn.clicked.connect(self.accept)
        # The template's text is the original's "Version 4.3.0.13" and libtiff copyrights.
        self.b.all[1][1].setText(f"{APP_NAME} v{__version__}\nBased on {BASED_ON}\n\nCreated by {AUTHOR}")
        self.b.all[1][1].setAlignment(Qt.AlignCenter)
