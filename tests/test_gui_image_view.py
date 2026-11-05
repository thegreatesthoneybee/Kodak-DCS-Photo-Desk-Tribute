"""Offscreen tests for ImageView viewport layout and zoom input."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QCoreApplication, QPoint, QPointF, Qt
from PySide6.QtGui import (QColor, QImage, QNativeGestureEvent, QPalette, QPointingDevice,
                           QWheelEvent)
from PySide6.QtWidgets import QApplication

from photodesk.gui.views import ImageView


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def make_view(qt_app, size=(240, 180), image_size=(500, 400)):
    view = ImageView()
    view.resize(*size)
    view.show()
    qt_app.processEvents()
    image = QImage(*image_size, QImage.Format.Format_RGB32)
    image.fill(QColor(180, 40, 20))
    view.set_image(image, image_size, None)
    view.set_zoom(100)
    qt_app.processEvents()
    return view


def send_native_zoom(view, global_pos, delta, target=None):
    target = target or view.canvas
    local = QPointF(target.mapFromGlobal(global_pos.toPoint()))
    scene = QPointF(view.mapFromGlobal(global_pos.toPoint()))
    event = QNativeGestureEvent(Qt.NativeGestureType.ZoomNativeGesture,
                                QPointingDevice.primaryPointingDevice(), 2,
                                local, scene, global_pos, delta, QPointF(), 1)
    QCoreApplication.sendEvent(target, event)


def send_wheel(view, global_pos, *, angle=0, pixels=0, modifiers=Qt.KeyboardModifier.NoModifier):
    local = QPointF(view.canvas.mapFromGlobal(global_pos.toPoint()))
    event = QWheelEvent(local, global_pos, QPoint(0, pixels), QPoint(0, angle),
                        Qt.MouseButton.NoButton, modifiers, Qt.ScrollPhase.NoScrollPhase, False)
    QCoreApplication.sendEvent(view.canvas, event)


def image_point_under(view, global_pos):
    viewport_pos = view.viewport().mapFromGlobal(global_pos.toPoint())
    canvas_pos = view.canvas.mapFrom(view.viewport(), viewport_pos)
    rect = view.image_rect()
    return ((canvas_pos.x() - rect.left()) / rect.width(),
            (canvas_pos.y() - rect.top()) / rect.height())


@pytest.mark.parametrize("zoom", [25, 100, 200])
def test_fixed_zoom_resize_and_palette_keep_viewport_margins_gray(qt_app, zoom):
    view = make_view(qt_app, size=(180, 140), image_size=(40, 24))
    try:
        view.set_zoom(zoom)
        view.resize(340, 240)
        qt_app.processEvents()

        assert view.canvas.width() >= view.viewport().width()
        assert view.canvas.height() >= view.viewport().height()
        shot = view.viewport().grab().toImage()
        assert shot.pixelColor(2, 2).rgba() == view.palette().color(QPalette.ColorRole.Dark).rgba()
        image_rect = view.image_rect()
        assert shot.pixelColor(image_rect.center().x(), image_rect.center().y()) == QColor(180, 40, 20)

        palette = view.palette()
        palette.setColor(QPalette.ColorRole.Dark, QColor(73, 81, 89))
        view.setPalette(palette)
        qt_app.processEvents()
        shot = view.viewport().grab().toImage()
        assert shot.pixelColor(2, 2) == QColor(73, 81, 89)
    finally:
        view.close()


def test_native_pinch_zoom_anchors_pointer_and_clamps_limits(qt_app):
    view = make_view(qt_app)
    try:
        global_pos = QPointF(view.viewport().mapToGlobal(QPoint(75, 55)))
        before = image_point_under(view, global_pos)
        send_native_zoom(view, global_pos, 0.5)
        qt_app.processEvents()
        assert view.zoom == 150
        assert image_point_under(view, global_pos) == pytest.approx(before, abs=0.01)

        view.set_zoom(1590)
        send_native_zoom(view, global_pos, 1.0)
        assert view.zoom == 1600
        view.set_zoom(2)
        send_native_zoom(view, global_pos, -0.99)
        assert view.zoom == 1
        send_native_zoom(view, global_pos, -1.0)
        assert view.zoom == 1
    finally:
        view.close()


def test_native_pinch_accumulates_tiny_deltas_and_handles_viewport_events(qt_app):
    view = make_view(qt_app)
    try:
        global_pos = QPointF(view.viewport().mapToGlobal(QPoint(75, 55)))
        for _ in range(100):
            send_native_zoom(view, global_pos, 0.001, view.viewport())
        assert view.zoom >= 110
        assert view.scale > 1.1

        view.set_zoom(100)
        assert view.scale == 1
        for _ in range(10):
            send_native_zoom(view, global_pos, 0.001, view.viewport())
        assert view.scale == pytest.approx(1.001 ** 10)
    finally:
        view.close()


def test_control_wheel_zooms_but_plain_wheel_still_scrolls(qt_app):
    view = make_view(qt_app)
    try:
        global_pos = QPointF(view.viewport().mapToGlobal(QPoint(80, 60)))
        send_wheel(view, global_pos, angle=120, modifiers=Qt.KeyboardModifier.ControlModifier)
        assert view.zoom == 120
        send_wheel(view, global_pos, pixels=60, modifiers=Qt.KeyboardModifier.ControlModifier)
        assert view.zoom > 120

        view.verticalScrollBar().setValue(0)
        send_wheel(view, global_pos, angle=-120)
        assert view.verticalScrollBar().value() > 0
    finally:
        view.close()


def test_pinch_from_fit_uses_exact_scale_and_invalid_events_do_not_change_it(qt_app):
    view = make_view(qt_app)
    try:
        view.set_zoom(fit=True)
        qt_app.processEvents()
        initial = view.scale
        global_pos = QPointF(view.viewport().mapToGlobal(QPoint(80, 60)))
        send_native_zoom(view, global_pos, 0.1, view.viewport())
        assert not view.fit
        assert view.scale == pytest.approx(initial * 1.1)
        for delta in (-1.0, -2.0, float("nan"), float("inf")):
            send_native_zoom(view, global_pos, delta, view.viewport())
            assert view.scale == pytest.approx(initial * 1.1)
    finally:
        view.close()


def test_fixed_zoom_resize_covers_viewport_after_scrollbars_disappear(qt_app):
    view = make_view(qt_app, size=(180, 140), image_size=(300, 200))
    try:
        assert view.horizontalScrollBar().maximum() > 0
        assert view.verticalScrollBar().maximum() > 0
        view.resize(600, 400)
        qt_app.processEvents()
        assert view.zoom == 100
        assert view.horizontalScrollBar().maximum() == 0
        assert view.verticalScrollBar().maximum() == 0
        shot = view.viewport().grab().toImage()
        gray = view.palette().color(QPalette.ColorRole.Dark).rgba()
        for x in (2, shot.width() - 3):
            for y in (2, shot.height() - 3):
                assert shot.pixelColor(x, y).rgba() == gray
    finally:
        view.close()
