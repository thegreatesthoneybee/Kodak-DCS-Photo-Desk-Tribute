"""Image-owned controls; use real Qt resources without starting RAW render workers."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
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


def test_each_image_panel_targets_its_owner_and_viewer_reuse_refreshes(window, qt_app):
    main, folder = window
    sheet = main.open_folder(folder)
    first, second, finished = sheet.sheet.entries
    first.settings.exposure = 0.4
    second.settings.exposure = -0.6
    viewer = main.open_image(first, sheet)
    qt_app.processEvents()
    assert viewer.controls.ev.value() == 4
    assert main.open_image(second, sheet) is viewer
    assert viewer.controls.ev.value() == -6
    other = ImageWindow(main, sheet, first)
    main._float_new(other)
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

    main.open_image(finished, sheet)
    qt_app.processEvents()
    assert viewer.entry is finished
    assert not viewer.controls.ev.isEnabled()
    assert other.controls.ev.isEnabled()
    other.controls.tools[32838].click()
    assert first.settings.rotation == 2
    assert finished.settings.rotation == 0
    other.close()
    qt_app.processEvents()
    assert viewer in sheet.images
    assert viewer.controls.isVisible()


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
