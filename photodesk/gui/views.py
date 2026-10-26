"""Contact sheet, full image rendering, close-up and histogram widgets."""
import numpy as np
from PySide6.QtCore import QObject, QPoint, QPointF, QRect, QRectF, QRunnable, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QScrollArea, QStyle, QStyledItemDelegate, QWidget

from .. import pipeline

THUMB_100 = 128               # thumbnail long edge at the 100% contact sheet size
INFO_H = 17                   # height of a thumbnail info row


def to_qimage(img):
    """float32 or uint8 HxWx3 RGB -> QImage (owns a copy of the data)."""
    if img.dtype != np.uint8:
        img = np.round(np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    return QImage(img.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def qimage_array(qimg):
    q = qimg.convertToFormat(QImage.Format_RGB888)
    a = np.frombuffer(q.constBits(), np.uint8, q.sizeInBytes()).reshape(q.height(), q.bytesPerLine())
    return a[:, :3 * q.width()].reshape(q.height(), q.width(), 3).copy()


class Task(QRunnable):
    class _Signals(QObject):
        done = Signal(object, object)
        failed = Signal(object, str)

    def __init__(self, token, fn, *args):
        super().__init__()
        self.token, self.fn, self.args = token, fn, args
        self.signals = Task._Signals()

    def run(self):
        try:
            result = self.fn(*self.args)
        except Exception as e:  # reported to the UI through the failed signal
            self.signals.failed.emit(self.token, f"{type(e).__name__}: {e}")
        else:
            self.signals.done.emit(self.token, result)


# ----------------------------------------------------------------- contact sheet
class ThumbIcons:
    """Thumbnail info icons from the original bitmaps."""

    def __init__(self, res):
        s = res.strip
        self.balance = {b: s(136, 16, i) for i, b in enumerate(pipeline.BALANCES)}
        self.lighting = {l: s(135, 16, i) for i, l in enumerate(pipeline.LIGHTINGS)}
        self.look_product, self.look_portrait = s(139, 16, 0), s(139, 16, 2)
        self.noise, self.exposure = res.bitmap(190), res.bitmap(189)
        self.sharpen = res.bitmap(206).copy(0, 0, 16, 15)
        self.tag, self.lock, self.sound = res.bitmap(187), res.bitmap(188), res.bitmap(171)
        self.file = {"E": res.bitmap(226), "T": res.bitmap(227), "J": res.bitmap(228), "R": res.bitmap(229)}


class ThumbDelegate(QStyledItemDelegate):
    def __init__(self, sheet_view):
        super().__init__(sheet_view)
        self.view = sheet_view

    def sizeHint(self, option, index):
        return self.view.cell_size()

    def paint(self, p, option, index):
        v = self.view
        entry = index.data(Qt.UserRole)
        r = option.rect.adjusted(2, 2, -2, -2)
        selected = option.state & QStyle.State_Selected
        p.save()
        p.fillRect(r, QColor(10, 36, 106) if selected else QColor(192, 192, 192))
        focus = index.row() == v.currentRow()
        pen = QPen(QColor(255, 255, 0) if focus and selected else QColor(128, 128, 128), 3 if focus else 1)
        p.setPen(pen)
        p.drawRect(r.adjusted(1, 1, -1, -1))
        info = v.show_info and v.zoom >= 50
        top = r.top() + (INFO_H if info else 4)
        bottom = r.bottom() - (INFO_H if info else 4)
        area = QRect(r.left() + 6, top + 2, r.width() - 12, bottom - top - 4)
        if entry.thumb is not None:
            pm = entry.thumb.scaled(area.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = area.left() + (area.width() - pm.width()) // 2
            y = area.top() + (area.height() - pm.height()) // 2
            p.drawImage(x, y, pm)
            if entry.settings.crop:
                c = entry.settings.crop
                p.setPen(QPen(QColor(255, 255, 255), 1, Qt.DashLine))
                p.drawRect(QRectF(x + c[0] * pm.width(), y + c[1] * pm.height(),
                                  (c[2] - c[0]) * pm.width(), (c[3] - c[1]) * pm.height()))
        if info:
            self._info(p, r, entry, index.row() + 1, selected)
        p.restore()

    def _info(self, p, r, e, number, selected):
        ic = self.view.icons
        s = e.settings
        x, y = r.left() + 4, r.top() + 1
        icons = []
        if e.is_raw:
            icons.append(ic.balance[s.balance])
            icons.append(ic.lighting[s.lighting])
            icons.append(ic.look_portrait if "Portrait" in s.look else ic.look_product)
            if s.noise_type != "none":
                icons.append(ic.noise)
            if s.sharpen_amount > 0:
                icons.append(ic.sharpen)
        for pm in icons:
            p.drawPixmap(x, y, pm)
            x += 17
        p.setPen(QColor(255, 255, 255) if selected else QColor(0, 0, 0))
        if e.is_raw:
            p.drawPixmap(x, y, ic.exposure)
            p.drawText(QRect(x + 17, y, 30, 15), Qt.AlignVCenter, f"{s.exposure:.1f}")
        x = r.right() - 4
        for pm, on in ((ic.sound, bool(e.sounds)), (ic.lock, e.read_only), (ic.tag, s.tagged)):
            if on:
                x -= 17
                p.drawPixmap(x, y, pm)
        by = r.bottom() - INFO_H + 1
        p.drawText(QRect(r.left() + 4, by, 30, 15), Qt.AlignVCenter, str(number))
        p.drawText(QRect(r.left() + 30, by, r.width() - 60, 15), Qt.AlignCenter, e.path.name)
        p.drawPixmap(r.right() - 20, by, ic.file[e.file_type])


class ContactSheetView(QListWidget):
    """Thumbnails of a folder; zoom 25/50/100/200 %, optional thumbnail info."""

    def __init__(self, icons):
        super().__init__()
        self.icons = icons
        self.zoom = 100
        self.show_info = True
        self.setViewMode(QListWidget.IconMode)
        self.setResizeMode(QListWidget.Adjust)
        self.setMovement(QListWidget.Static)
        self.setSelectionMode(QListWidget.ExtendedSelection)
        self.setUniformItemSizes(True)
        self.setSpacing(2)
        self.setStyleSheet("QListWidget { background: rgb(128,128,128); }")
        self.setItemDelegate(ThumbDelegate(self))

    def cell_size(self):
        edge = THUMB_100 * self.zoom // 100
        info = INFO_H * 2 if self.show_info and self.zoom >= 50 else 8
        return QSize(edge + 16, edge + info + 8)

    def set_zoom(self, zoom):
        self.zoom = zoom
        self.setGridSize(self.cell_size() + QSize(4, 4))
        self.doItemsLayout()

    def set_show_info(self, on):
        self.show_info = on
        self.set_zoom(self.zoom)

    def populate(self, entries):
        self.clear()
        for e in entries:
            it = QListWidgetItem()
            it.setData(Qt.UserRole, e)
            it.setToolTip(e.path.name)
            self.addItem(it)
        self.set_zoom(self.zoom)

    def entries(self):
        return [self.item(i).data(Qt.UserRole) for i in range(self.count())]

    def selected_entries(self):
        rows = sorted(self.row(i) for i in self.selectedItems())
        return [self.item(r).data(Qt.UserRole) for r in rows]

    def focused_entry(self):
        it = self.currentItem()
        if it is not None and it.isSelected():
            return it.data(Qt.UserRole)
        sel = self.selected_entries()
        return sel[0] if sel else None

    def refresh_entry(self, entry):
        for i in range(self.count()):
            if self.item(i).data(Qt.UserRole) is entry:
                self.update(self.indexFromItem(self.item(i)))

    def thumb_at(self, pos):
        """(entry, normalised x, y within the thumbnail image) under a viewport position."""
        it = self.itemAt(pos)
        if it is None:
            return None
        e = it.data(Qt.UserRole)
        if e.thumb is None:
            return None
        r = self.visualItemRect(it).adjusted(2, 2, -2, -2)
        info = self.show_info and self.zoom >= 50
        top, bottom = r.top() + (INFO_H if info else 4), r.bottom() - (INFO_H if info else 4)
        area = QRect(r.left() + 6, top + 2, r.width() - 12, bottom - top - 4)
        sz = e.thumb.size().scaled(area.size(), Qt.KeepAspectRatio)
        x0 = area.left() + (area.width() - sz.width()) / 2
        y0 = area.top() + (area.height() - sz.height()) / 2
        nx, ny = (pos.x() - x0) / sz.width(), (pos.y() - y0) / sz.height()
        if 0 <= nx <= 1 and 0 <= ny <= 1:
            return e, nx, ny
        return None


# ----------------------------------------------------------------- image rendering
class ImageCanvas(QWidget):
    """Draws the rendered image at the view scale, plus the crop rectangle."""

    def __init__(self, view):
        super().__init__()
        self.view = view
        self.setMouseTracking(True)

    def paintEvent(self, e):
        v = self.view
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(128, 128, 128))
        if v.image is None:
            return
        target = v.image_rect()
        p.setRenderHint(QPainter.SmoothPixmapTransform, v.scale < 1)
        p.drawImage(target, v.image)
        crop = v.crop_rect_widget()
        if crop is not None:
            p.setPen(QPen(QColor(0, 0, 0), 1))
            p.drawRect(crop)
            p.setPen(QPen(QColor(255, 255, 255), 1, Qt.DashLine))
            p.drawRect(crop)

    def mousePressEvent(self, e):
        self.view.press(e)

    def mouseMoveEvent(self, e):
        self.view.move(e)

    def mouseReleaseEvent(self, e):
        self.view.release(e)


class ImageView(QScrollArea):
    """A full image rendering; shows the uncropped image with the crop rectangle.

    Signals carry normalised coordinates of the oriented, uncropped image.
    """
    clicked = Signal(float, float)               # select/click-balance/close-up tools
    crop_changed = Signal(object)                # new crop rect or None
    hover = Signal(object)                       # (x px, y px, (r, g, b)) or None
    zoom_changed = Signal(int)

    def __init__(self, cursors=None):
        super().__init__()
        self.canvas = ImageCanvas(self)
        self.setWidget(self.canvas)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("QScrollArea { background: rgb(128,128,128); }")
        self.image = None              # QImage of the uncropped oriented render
        self.array = None
        self.full_size = (1, 1)        # full image size (pixels) of the oriented, uncropped image
        self.fit = True
        self.zoom = 100
        self.scale = 1.0
        self.tool = "select"
        self.cursors = cursors or {}
        self.crop = None
        self.ratio = None              # (w, h) crop aspect constraint
        self._drag = None

    # --------------------------------------------------------- image
    def set_image(self, qimage, full_size, crop):
        self.image = qimage
        self.array = qimage_array(qimage)
        self.full_size = full_size
        self.crop = crop
        self._relayout()

    def set_crop(self, crop):
        self.crop = crop
        self.canvas.update()

    def set_zoom(self, zoom=None, fit=False):
        self.fit = fit
        if zoom:
            self.zoom = zoom
        self._relayout()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.fit:
            self._relayout()

    def _relayout(self):
        fw, fh = self.full_size
        if self.fit:
            vp = self.viewport().size()
            self.scale = max(0.01, min(vp.width() / fw, vp.height() / fh))
            self.zoom = max(1, round(self.scale * 100))
        else:
            self.scale = self.zoom / 100
        w, h = max(1, round(fw * self.scale)), max(1, round(fh * self.scale))
        vp = self.viewport().size()
        self.canvas.resize(max(w, vp.width()), max(h, vp.height()))
        self.canvas.update()
        self.zoom_changed.emit(self.zoom)

    def image_rect(self):
        fw, fh = self.full_size
        w, h = round(fw * self.scale), round(fh * self.scale)
        return QRect((self.canvas.width() - w) // 2, (self.canvas.height() - h) // 2, w, h)

    def to_norm(self, pos):
        r = self.image_rect()
        return (pos.x() - r.left()) / max(1, r.width()), (pos.y() - r.top()) / max(1, r.height())

    def crop_rect_widget(self):
        if not self.crop:
            return None
        r = self.image_rect()
        x0, y0, x1, y1 = self.crop
        return QRectF(r.left() + x0 * r.width(), r.top() + y0 * r.height(),
                      (x1 - x0) * r.width(), (y1 - y0) * r.height())

    # --------------------------------------------------------- tools
    def set_tool(self, tool):
        self.tool = tool
        self.canvas.setCursor(self.cursors.get(tool, Qt.ArrowCursor))

    def _crop_hit(self, pos):
        """'inside', an edge/corner code like 'l', 'tr', or None."""
        c = self.crop_rect_widget()
        if c is None:
            return None
        m = 4
        x, y = pos.x(), pos.y()
        if not c.adjusted(-m, -m, m, m).contains(QPointF(x, y)):
            return None
        code = ("l" if abs(x - c.left()) <= m else "r" if abs(x - c.right()) <= m else "")
        code = ("t" if abs(y - c.top()) <= m else "b" if abs(y - c.bottom()) <= m else "") + code
        return code or "inside"

    def press(self, e):
        if e.button() != Qt.LeftButton or self.image is None:
            return
        nx, ny = self.to_norm(e.position())
        if self.tool == "hand":
            self._drag = ("hand", e.globalPosition(), self.horizontalScrollBar().value(),
                          self.verticalScrollBar().value())
        elif self.tool == "crop":
            hit = self._crop_hit(e.position())
            if hit:
                self._drag = ("crop", hit, (nx, ny), self.crop)
            elif 0 <= nx <= 1 and 0 <= ny <= 1:
                self._drag = ("new", (nx, ny))
                if self.crop:
                    self.crop = None
                    self._clicked_outside = True
            else:
                if self.crop:
                    self.crop = None
                    self.crop_changed.emit(None)
                self.canvas.update()
        elif 0 <= nx <= 1 and 0 <= ny <= 1:
            self.clicked.emit(nx, ny)

    def move(self, e):
        nx, ny = self.to_norm(e.position())
        if self.image is not None and 0 <= nx < 1 and 0 <= ny < 1:
            h, w = self.array.shape[:2]
            rgb = self.array[min(h - 1, int(ny * h)), min(w - 1, int(nx * w))]
            self.hover.emit((int(nx * self.full_size[0]), int(ny * self.full_size[1]), tuple(int(v) for v in rgb)))
        else:
            self.hover.emit(None)
        if self.tool == "crop" and self._drag is None:
            hit = self._crop_hit(e.position())
            shape = {"inside": "hand", "l": 224, "r": 224, "t": 222, "b": 222, "tl": 223, "br": 223,
                     "tr": 221, "bl": 221}.get(hit, "crop")
            self.canvas.setCursor(self.cursors.get(shape, Qt.ArrowCursor))
        if self._drag is None:
            return
        kind = self._drag[0]
        if kind == "hand":
            _, start, h0, v0 = self._drag
            d = e.globalPosition() - start
            self.horizontalScrollBar().setValue(int(h0 - d.x()))
            self.verticalScrollBar().setValue(int(v0 - d.y()))
            return
        nx, ny = min(max(nx, 0.0), 1.0), min(max(ny, 0.0), 1.0)
        if kind == "new":
            ax, ay = self._drag[1]
            self.crop = self._constrain(ax, ay, nx, ny)
        else:
            _, hit, (sx, sy), c = self._drag
            x0, y0, x1, y1 = c
            if hit == "inside":
                dx = min(max(nx - sx, -x0), 1 - x1)
                dy = min(max(ny - sy, -y0), 1 - y1)
                self.crop = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
            else:
                if "l" in hit:
                    x0 = nx
                if "r" in hit:
                    x1 = nx
                if "t" in hit:
                    y0 = ny
                if "b" in hit:
                    y1 = ny
                ax = c[2] if "l" in hit else c[0]
                ay = c[3] if "t" in hit else c[1]
                if self.ratio:
                    bx = x0 if "l" in hit else x1
                    by = y0 if "t" in hit else y1
                    self.crop = self._constrain(ax, ay, bx, by)
                else:
                    self.crop = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        self.canvas.update()

    def _constrain(self, ax, ay, bx, by):
        if self.ratio:
            fw, fh = self.full_size
            target = self.ratio[0] / self.ratio[1]
            w, h = abs(bx - ax) * fw, abs(by - ay) * fh
            if w / max(h, 1e-9) > target:
                w = h * target
            else:
                h = w / target
            bx = ax + np.sign(bx - ax or 1) * w / fw
            by = ay + np.sign(by - ay or 1) * h / fh
            bx, by = min(max(bx, 0), 1), min(max(by, 0), 1)
        return (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))

    def release(self, e):
        drag, self._drag = self._drag, None
        if drag is None or drag[0] == "hand":
            return
        c = self.crop
        if c and ((c[2] - c[0]) * self.full_size[0] < 4 or (c[3] - c[1]) * self.full_size[1] < 4):
            c = None
            self.crop = None
        self.crop_changed.emit(c)
        self.canvas.update()


class HistogramWidget(QWidget):
    """Output histogram (luminance plus optional R, G, B) of the working-space image."""

    def __init__(self):
        super().__init__()
        self.hist = None
        self.channels = {"r": False, "g": False, "b": False}

    def set_image(self, img_u8):
        if img_u8 is None:
            self.hist = None
        else:
            small = img_u8[::2, ::2].reshape(-1, 3).astype(np.float32)
            lum = small @ np.array([0.299, 0.587, 0.114], np.float32)
            self.hist = [np.histogram(ch, 256, (0, 256))[0] for ch in (lum, small[:, 0], small[:, 1], small[:, 2])]
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(255, 255, 255))
        p.setPen(QColor(0, 0, 0))
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))
        if self.hist is None:
            return
        peak = max(1, max(h[1:255].max() for h in self.hist))
        w, h = self.width() - 2, self.height() - 2
        layers = [(self.hist[0], QColor(0, 0, 0))]
        for i, (k, col) in enumerate((("r", QColor(255, 0, 0)), ("g", QColor(0, 170, 0)), ("b", QColor(0, 0, 255)))):
            if self.channels[k]:
                layers.append((self.hist[i + 1], col))
        for hist, col in layers:
            p.setPen(col)
            for x in range(w):
                v = hist[x * 256 // w]
                bh = int(min(1.0, v / peak) * h)
                p.drawLine(1 + x, h, 1 + x, h - bh)


class EVHistogram(QWidget):
    """Exposure histogram of the Exposure and Tone dialog with shadow/midtone/highlight markers."""
    moved = Signal(int, float)        # marker index, EV

    LO, HI = -12.0, 6.0

    def __init__(self):
        super().__init__()
        self.hist = None
        self.tone = list(pipeline.TONE_DEFAULT)
        self._drag = None

    def set_hist(self, hist):
        self.hist = hist
        self.update()

    def set_tone(self, tone):
        self.tone = list(tone)
        self.update()

    def _x(self, ev):
        return 4 + (ev - self.LO) / (self.HI - self.LO) * (self.width() - 8)

    def _ev(self, x):
        return self.LO + (x - 4) / max(1, self.width() - 8) * (self.HI - self.LO)

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(255, 255, 255))
        p.setPen(QColor(0, 0, 0))
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))
        gh = self.height() - 14
        if self.hist is not None:
            hist, edges = self.hist
            peak = max(1, hist.max())
            p.setPen(QColor(64, 64, 64))
            for v, a, b in zip(hist, edges[:-1], edges[1:]):
                x0, x1 = int(self._x(a)), int(self._x(b))
                bh = int(v / peak * (gh - 4))
                for x in range(x0, max(x0 + 1, x1)):
                    p.drawLine(x, gh, x, gh - bh)
        for ev, col in zip(self.tone, (QColor(0, 0, 0), QColor(128, 128, 128), QColor(255, 255, 255))):
            x = self._x(ev)
            pts = [QPointF(x, gh + 1), QPointF(x - 5, gh + 12), QPointF(x + 5, gh + 12)]
            p.setBrush(col)
            p.setPen(QColor(0, 0, 0))
            p.drawPolygon(pts)

    def mousePressEvent(self, e):
        x = e.position().x()
        d = [abs(self._x(ev) - x) for ev in self.tone]
        i = int(np.argmin(d))
        if d[i] < 8:
            self._drag = i

    def mouseMoveEvent(self, e):
        if self._drag is not None:
            ev = round(self._ev(e.position().x()), 1)
            self.moved.emit(self._drag, ev)

    def mouseReleaseEvent(self, e):
        self._drag = None


class CloseUpView(QWidget):
    """Close-up window contents: a magnified region rendered at full quality."""
    dragged = Signal(float, float)    # move by normalised delta

    def __init__(self):
        super().__init__()
        self.image = None
        self._last = None
        self.setMinimumSize(200, 200)

    def set_image(self, qimage):
        self.image = qimage
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(128, 128, 128))
        if self.image is not None:
            x = (self.width() - self.image.width()) // 2
            y = (self.height() - self.image.height()) // 2
            p.drawImage(QPoint(x, y), self.image)

    def mousePressEvent(self, e):
        self._last = e.position()

    def mouseMoveEvent(self, e):
        if self._last is not None:
            d = e.position() - self._last
            self._last = e.position()
            self.dragged.emit(-d.x(), -d.y())

    def mouseReleaseEvent(self, e):
        self._last = None
