"""Image-owned controls; use real Qt resources without starting RAW render workers."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent, QRect
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication, QDockWidget

from photodesk import camera, pipeline
from photodesk.gui.app import ImageWindow, MainWindow
from photodesk.gui.panels import ImageControlsPanel
from photodesk.gui.res import MissingResources, Resources


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window(qt_app, tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTODESK_SETTINGS", str(tmp_path / "settings.ini"))
    try:
        resources = Resources.load()
    except MissingResources as exc:
        pytest.skip(f"Photo Desk UI resources unavailable: {exc}")
    monkeypatch.setattr(camera, "read_make_model", lambda path: ("Kodak", "DCS Pro SLR/n"))
    monkeypatch.setattr(pipeline, "load_settings", lambda path: pipeline.Settings())
    monkeypatch.setattr(MainWindow, "_load_thumb", lambda *args: None)
    monkeypatch.setattr(MainWindow, "_load_render", lambda *args: None)
    monkeypatch.setattr(MainWindow, "refresh_thumb", lambda *args: None)
    monkeypatch.setattr(MainWindow, "render_window", lambda *args, **kwargs: None)
    monkeypatch.setattr(MainWindow, "confirm_save", lambda *args: True)
    folder = tmp_path / "photos"
    folder.mkdir()
    for name in ("a.dcr", "b.dcr", "c.jpg"):
        (folder / name).touch()
    monkeypatch.setattr(camera, "_unlock_all", camera.unlock_all())
    main = MainWindow(resources)
    main.resize(1500, 950)
    main.show()
    yield main, folder
    main.close()
    qt_app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_folder_has_no_adjustments_and_image_owns_toggleable_panel(window, qt_app):
    main, folder = window
    sheet = main.open_folder(folder)
    sheet.view.setCurrentRow(0)
    qt_app.processEvents()
    assert not main.findChildren(ImageControlsPanel)
    assert not main.acts[32836].isEnabled()
    assert not main.acts[32836].isChecked()

    image = main.open_image(sheet.sheet.entries[0], sheet)
    qt_app.processEvents()
    assert image.isAncestorOf(image.controls)
    assert image.controls.isVisible()
    assert image.controls.b[1214].text() == "3.5"
    assert image.controls_scroll.viewport().width() >= image.controls.width()
    assert image.controls_scroll.horizontalScrollBar().maximum() == 0
    assert image.controls_dock not in main.findChildren(QDockWidget)
    assert main.acts[32836].isEnabled()
    main.acts[32836].trigger()
    assert not image.controls.isVisible()
    assert not main.acts[32836].isChecked()
    main.acts[32836].trigger()
    assert image.controls.isVisible()

    main._document_activated(sheet)
    assert not main.acts[32836].isEnabled()
    assert not main.acts[32836].isChecked()
    assert image.controls.isVisible()
    main._activate_image(image)
    image.controls_dock.close()
    assert not main.acts[32836].isChecked()
    main.acts[32836].trigger()
    assert image.controls.isVisible()

    main.set_window_floating(image, False)
    qt_app.processEvents()
    assert image.controls_dock.parentWidget() is main
    assert image.controls_dock in main.tabifiedDockWidgets(main.info_dock)
    assert not image.isAncestorOf(image.controls)
    assert image.controls.isVisible()
    main.set_window_floating(image, True)
    qt_app.processEvents()
    assert image.isAncestorOf(image.controls) and image.controls.isVisible()
    image.close()
    qt_app.processEvents()
    assert not sheet.images
    assert not main.acts[32836].isEnabled()


def test_mdi_controls_follow_active_image_and_preserve_hidden_state(window, qt_app):
    main, folder = window
    sheet = main.open_folder(folder)
    first = main.open_image(sheet.sheet.entries[0], sheet)
    main.set_window_floating(first, False)
    second = ImageWindow(main, sheet, sheet.sheet.entries[1])
    main.mdi.addSubWindow(second)
    second.show()
    main._activate_image(second)
    qt_app.processEvents()
    assert second.controls_dock.parentWidget() is main
    assert second.controls.isVisible()
    assert first.controls_dock.parentWidget() is first.content
    assert not first.controls.isVisible()

    main._activate_image(first)
    qt_app.processEvents()
    assert first.controls_dock.parentWidget() is main
    assert first.controls.isVisible()
    assert not second.controls.isVisible()
    main.acts[32836].trigger()
    assert not first.controls_visible
    main.set_window_floating(first, True)
    assert first.controls_dock.parentWidget() is first.content
    assert not first.controls.isVisible()
    main.set_window_floating(first, False)
    assert first.controls_dock.parentWidget() is main
    assert not first.controls.isVisible()
    main.acts[32836].trigger()
    assert first.controls.isVisible()

    main.mdi.setActiveSubWindow(sheet)
    qt_app.processEvents()
    assert not first.controls.isVisible() and not second.controls.isVisible()
    assert not main.acts[32836].isEnabled()
    main._activate_image(second)
    second.close()
    qt_app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    main._activate_image(first)
    assert first.controls_dock.parentWidget() is main
    assert first.controls.isVisible()


def test_each_open_image_keeps_its_own_window_and_adjustment_panel(window, qt_app):
    main, folder = window
    sheet = main.open_folder(folder)
    first, second, finished = sheet.sheet.entries
    first.settings.exposure = 0.4
    second.settings.exposure = -0.6
    other = main.open_image(first, sheet)
    qt_app.processEvents()
    assert other.controls.ev.value() == 4
    viewer = main.open_image(second, sheet)
    assert viewer is not other
    assert viewer.controls.ev.value() == -6
    qt_app.processEvents()
    assert other.controls is not viewer.controls

    # Deliberately invoke a non-active window's panel: edits must not leak to the active entry.
    main._activate_image(other)
    viewer.controls.ev.setValue(12)
    assert second.settings.exposure == 1.2
    assert first.settings.exposure == 0.4
    main._activate_image(viewer)
    other.controls.tools[32838].click()
    assert first.settings.rotation == 1
    assert second.settings.rotation == 0

    finished_window = main.open_image(finished, sheet)
    qt_app.processEvents()
    assert len(sheet.images) == 3
    assert viewer.entry is second
    assert finished_window.entry is finished
    assert not finished_window.controls.ev.isEnabled()
    assert other.controls.ev.isEnabled()
    other.controls.tools[32838].click()
    assert first.settings.rotation == 2
    assert finished.settings.rotation == 0
    other.close()
    qt_app.processEvents()
    assert viewer in sheet.images
    assert viewer.controls.isVisible()


def test_open_selected_creates_multiple_windows_without_duplicate_views(window, qt_app):
    main, folder = window
    sheet = main.open_folder(folder)
    for i in range(sheet.view.count()):
        sheet.view.item(i).setSelected(True)
    main._document_activated(sheet)
    main.open_selected()
    qt_app.processEvents()
    assert len(sheet.images) == 3
    assert {id(w.entry) for w in sheet.images} == {id(e) for e in sheet.sheet.entries}
    assert all(w.isVisible() and w.floating for w in sheet.images)
    existing = sheet.images[0]
    existing.showMinimized()
    qt_app.processEvents()
    assert main.open_image(existing.entry, sheet) is existing
    qt_app.processEvents()
    assert not existing.isMinimized()
    assert len(sheet.images) == 3


def test_panel_edit_and_undo_stay_in_own_contact_sheet(window, qt_app):
    main, folder = window
    first_sheet = main.open_folder(folder)
    first = first_sheet.sheet.entries[0]
    image = main.open_image(first, first_sheet)
    other_folder = folder.parent / "other"
    other_folder.mkdir()
    (other_folder / "other.dcr").touch()
    second_sheet = main.open_folder(other_folder)
    second_sheet.view.setCurrentRow(0)
    second = second_sheet.sheet.entries[0]
    main._document_activated(second_sheet)
    qt_app.processEvents()
    image.controls.ev.setValue(8)
    assert first.settings.exposure == 0.8
    assert second.settings.exposure == 0
    assert len(first_sheet.sheet.undo_stack) == 1
    assert not second_sheet.sheet.undo_stack
    assert main.active_image() is image
    main.acts[57643].trigger()
    assert first.settings.exposure == 0
    assert image.controls.ev.value() == 0
    image.close()
    qt_app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    # Shared action refresh must not retain callbacks to deleted per-image widgets.
    main._document_activated(second_sheet)
    main.set_tool("crop")
    main.set_tool("select")
    main.update_commands()


def test_histogram_detaches_from_tabs_at_minimum_size_but_remains_resizable(window, qt_app):
    main, _ = window
    dock = main.hist_dock
    for _ in range(2):
        dock.setFloating(False)
        main.tabifyDockWidget(main.info_dock, dock)
        dock.show()
        dock.raise_()
        qt_app.processEvents()
        assert dock.height() == dock.minimumSizeHint().height()

        dock.setFloating(True)
        qt_app.processEvents()
        minimum = dock.minimumSizeHint().expandedTo(dock.minimumSize())
        assert dock.size() == minimum
        assert main.hist.isVisible()
        dock.resize(minimum.width() + 120, minimum.height() + 100)
        qt_app.processEvents()
        assert dock.width() > minimum.width()
        assert dock.height() > minimum.height()

    # A rapid re-dock must not apply the queued floating-size reset to the tab.
    dock.setFloating(False)
    dock.setFloating(True)
    dock.setFloating(False)
    main.tabifyDockWidget(main.info_dock, dock)
    dock.raise_()
    qt_app.processEvents()
    assert not dock.isFloating()
    assert dock.height() == dock.minimumSizeHint().height()


def test_right_tabs_scroll_when_window_is_short_and_recover_when_enlarged(window, qt_app):
    main, folder = window
    sheet = main.open_folder(folder)
    image = main.open_image(sheet.sheet.entries[0], sheet)
    main.set_window_floating(image, False)
    main.hist_dock.setFloating(False)
    main.tabifyDockWidget(main.info_dock, main.hist_dock)
    docks = (image.controls_dock, main.info_dock, main.hist_dock)
    main.resize(1000, 240)
    qt_app.processEvents()
    for dock in docks:
        dock.show()
        dock.raise_()
        qt_app.processEvents()
        scroll = dock.widget()
        panel = scroll.widget()
        if dock is main.hist_dock:
            assert scroll.viewport().width() >= main.hist.minimumWidth()
            assert scroll.viewport().height() >= main.hist.minimumHeight()
            assert not scroll.horizontalScrollBar().isVisible()
            assert not scroll.verticalScrollBar().isVisible()
            continue
        assert scroll.verticalScrollBar().maximum() > 0
        assert scroll.verticalScrollBar().isVisible()
        scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
        qt_app.processEvents()
        bottom = panel.mapTo(scroll.viewport(), panel.rect().bottomLeft())
        assert bottom.y() <= scroll.viewport().height()

    main.resize(1500, 1200)
    qt_app.processEvents()
    for dock in docks:
        dock.raise_()
        qt_app.processEvents()
        assert dock.widget().verticalScrollBar().maximum() == 0
        assert not dock.widget().verticalScrollBar().isVisible()


def test_docked_histogram_stays_visible_with_both_sidebar_tabs(window, qt_app):
    main, folder = window
    sheet = main.open_folder(folder)
    image = main.open_image(sheet.sheet.entries[0], sheet)
    main.set_window_floating(image, False)
    histogram = main.hist_dock
    histogram.setFloating(False)
    main.tabifyDockWidget(main.info_dock, histogram)
    histogram.show()
    histogram.raise_()
    qt_app.processEvents()
    qt_app.processEvents()
    assert not main.tabifiedDockWidgets(histogram)
    assert image.controls_dock in main.tabifiedDockWidgets(main.info_dock)
    for dock in (main.info_dock, image.controls_dock, main.info_dock):
        dock.raise_()
        qt_app.processEvents()
        assert not histogram.isFloating()
        assert not histogram.visibleRegion().isEmpty()
        assert not dock.visibleRegion().isEmpty()
        assert histogram.geometry().bottom() < dock.geometry().top()
    histogram.setFloating(True)
    qt_app.processEvents()
    assert histogram.size() == histogram.minimumSizeHint().expandedTo(histogram.minimumSize())


@pytest.mark.parametrize("saved_geometry", [None, QRect(40, 40, 480, 400)])
def test_open_from_contact_sheet_keeps_photo_visible_beside_controls(window, qt_app, saved_geometry):
    main, folder = window
    main.resize(800, 600)
    if saved_geometry is not None:
        main.q.setValue("Image Viewer Geometry", saved_geometry)
    sheet = main.open_folder(folder)
    image = main.open_image(sheet.sheet.entries[0], sheet)
    pixels = QImage(80, 60, QImage.Format_RGB32)
    pixels.fill(QColor("red"))
    image.view.set_image(pixels, (800, 600), None)
    qt_app.processEvents()
    for floating in (True, False, True):
        main.set_window_floating(image, floating)
        qt_app.processEvents()
        assert not image.view.visibleRegion().isEmpty()
        assert image.view.viewport().width() >= 250
        assert image.controls.isVisible()
        shot = image.view.viewport().grab().toImage()
        assert shot.pixelColor(shot.width() // 2, shot.height() // 2) == QColor("red")
        if floating:
            assert image.content.isAncestorOf(image.view)
            assert image.content.isAncestorOf(image.controls)
            assert image.width() <= image.screen().availableGeometry().width()


def test_main_and_floating_images_keep_independent_histograms(window, qt_app):
    main, folder = window
    sheet = main.open_folder(folder)
    main.hist_dock.setFloating(False)
    main.hist_dock.show()
    entries = sheet.sheet.entries
    for value, entry in zip((80, 100, 120), entries):
        thumbnail = QImage(60, 40, QImage.Format_RGB32)
        thumbnail.fill(QColor(value, value, value))
        entry.thumb = thumbnail
    sheet.view.setCurrentRow(0)
    qt_app.processEvents()
    main._update_histogram()
    assert main.hist_dock.parentWidget() is main
    assert main.hist.graph.hist[0][80] > 0

    first = main.open_image(entries[0], sheet)
    first.rgb = np.full((40, 60, 3), 160, np.uint8)
    first.update_histogram()
    second = ImageWindow(main, sheet, entries[1])
    second.rgb = np.full((40, 60, 3), 200, np.uint8)
    main._float_new(second)
    second.update_histogram()
    qt_app.processEvents()
    assert first.hist is not main.hist and first.hist is not second.hist
    assert second.hist is not main.hist
    assert main.hist_dock.parentWidget() is main
    assert first.hist_dock.parentWidget() is first.content
    assert second.hist_dock.parentWidget() is second.content
    assert not main.hist_dock.isHidden()
    assert not first.hist_dock.isHidden() and not second.hist_dock.isHidden()
    for image in (first, second):
        scroll = image.hist_dock.widget()
        assert image.hist_dock.height() == image.hist_dock.minimumSizeHint().height()
        assert image.hist_dock.geometry().bottom() < image.controls_dock.geometry().top()
        assert scroll.viewport().width() >= image.hist.minimumWidth()
        assert scroll.viewport().height() >= image.hist.minimumHeight()
        assert not scroll.horizontalScrollBar().isVisible()
        assert not scroll.verticalScrollBar().isVisible()
    assert first.hist.graph.hist[0][160] > 0
    assert second.hist.graph.hist[0][200] > 0
    assert main.hist.graph.hist[0][80] > 0
    assert main.hist.graph.hist[0][160] == 0 and main.hist.graph.hist[0][200] == 0

    # The menu action follows the active floating image without hiding the main histogram
    # or changing another image window's independent panel.
    main._activate_image(first)
    main.toggle_histogram()
    assert first.hist_dock.isHidden()
    assert not second.hist_dock.isHidden()
    assert not main.hist_dock.isHidden()
    main.toggle_histogram()
    assert not first.hist_dock.isHidden()

    # MDI images use the main histogram, while their local panel is hidden and restored on float.
    main.set_window_floating(first, False)
    qt_app.processEvents()
    assert first.hist_dock.isHidden()
    assert main.hist.graph.hist[0][160] > 0
    main.set_window_floating(first, True)
    qt_app.processEvents()
    assert not first.hist_dock.isHidden()
    assert main.hist_dock.parentWidget() is main
    assert main.hist.graph.hist[0][80] > 0

    # The actual MDI sheet selection remains the main histogram source while a floating viewer
    # is active, including when that selected thumbnail finishes an asynchronous render.
    sheet.view.setCurrentRow(1)
    qt_app.processEvents()
    assert main.hist.graph.hist[0][100] > 0
    entries[1].thumb_gen += 1
    main._thumb_rendered((entries[1], entries[1].thumb_gen, sheet),
                         np.full((40, 60, 3), 110, np.uint8))
    assert main.hist.graph.hist[0][110] > 0
    assert main.hist_dock.objectName() == main.res.s(59956)
    assert first.hist.graph.hist[0][160] > 0
    assert second.hist.graph.hist[0][200] > 0

    first.hist_dock.setFloating(True)
    qt_app.processEvents()
    minimum = first.hist_dock.minimumSizeHint().expandedTo(first.hist_dock.minimumSize())
    assert first.hist_dock.size() == minimum
    first.hist_dock.resize(minimum.width() + 120, minimum.height() + 100)
    qt_app.processEvents()
    assert first.hist_dock.width() > minimum.width()
    assert first.hist_dock.height() > minimum.height()
    assert not second.hist_dock.isFloating()

    second.close()
    qt_app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    main._activate_image(first)
    qt_app.processEvents()
    assert first.hist_dock.parentWidget() is first.content
    assert first.hist.graph.hist[0][160] > 0
    first.close()
    qt_app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert main.hist_dock.parentWidget() is main
    assert not main.hist_dock.isHidden()
    assert main.hist.graph.hist[0][110] > 0


def test_visible_histogram_restores_and_initializes_independent_image_panel(window, qt_app):
    main, folder = window
    main.hist_dock.setFloating(False)
    main.hist_dock.show()
    qt_app.processEvents()
    main.close()
    qt_app.processEvents()
    restored = MainWindow(main.res)
    try:
        restored.show()
        qt_app.processEvents()
        assert not restored.hist_dock.isHidden()
        assert not restored.hist_dock.isFloating()
        sheet = restored.open_folder(folder)
        image = restored.open_image(sheet.sheet.entries[0], sheet)
        qt_app.processEvents()
        assert not restored.hist_dock.visibleRegion().isEmpty()
        assert not image.hist_dock.visibleRegion().isEmpty()
        assert image.hist_dock is not restored.hist_dock
    finally:
        restored.close()
        qt_app.processEvents()
