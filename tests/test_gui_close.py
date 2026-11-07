"""Close confirmation is transactional across dirty image entries."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from photodesk import camera, pipeline
from photodesk.gui import dialogs as D
from photodesk.gui import app as gui_app
from photodesk.gui.app import ImageWindow, MainWindow, SheetWindow
from photodesk.gui.model import Entry, Sheet
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
    monkeypatch.setattr(camera, "_unlock_all", camera.unlock_all())
    monkeypatch.setattr(pipeline, "load_settings", lambda path: pipeline.Settings())
    monkeypatch.setattr(MainWindow, "_load_thumb", lambda *args: None)
    monkeypatch.setattr(MainWindow, "_load_render", lambda *args, **kwargs: None)
    monkeypatch.setattr(MainWindow, "refresh_thumb", lambda *args: None)
    monkeypatch.setattr(MainWindow, "render_window", lambda *args, **kwargs: None)

    class FakeSaveChangesDialog:
        YES, NO, CANCEL, YES_ALL = 1, 7, 2, 1061
        answers = []
        prompts = []

        def __init__(self, res, parent, text, multiple=False):
            self.prompts.append((text, multiple))

        def exec(self):
            return self.answers.pop(0)

    monkeypatch.setattr(D, "SaveChangesDialog", FakeSaveChangesDialog)
    messages = []
    monkeypatch.setattr(D, "message", lambda parent, res, text, *args, **kwargs: messages.append(text))

    folder = tmp_path / "photos"
    folder.mkdir()
    for name in ("a.dcr", "b.dcr", "c.dcr", "finished.jpg"):
        (folder / name).touch()
    main = MainWindow(resources)
    main.resize(1200, 800)
    main.show()
    yield main, folder, FakeSaveChangesDialog, messages
    for sw in list(main.sheets):
        for entry in sw.sheet.entries:
            entry.dirty = False
    if main.isVisible():
        main.close()
    qt_app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _sheet(main, folder):
    return main.open_folder(folder)


def _image(main, sheet, entry):
    image = ImageWindow(main, sheet, entry)
    main.mdi.addSubWindow(image)
    image.show()
    return image


def _dirty(entry, exposure):
    entry.settings.exposure = exposure
    entry.dirty = True


def _state(entries):
    return [(entry.dirty, entry.settings.to_json()) for entry in entries]


@pytest.mark.parametrize("answers", [[7, 1, 2], [1, 7, 2], [7, 1, 0]])
def test_app_close_cancel_keeps_all_images_and_edits_after_prior_decisions(window, monkeypatch, qt_app, answers):
    main, folder, dialog, _ = window
    sheet = _sheet(main, folder)
    entries = sheet.sheet.entries[:3]
    images = [_image(main, sheet, entry) for entry in entries]
    for index, entry in enumerate(entries):
        _dirty(entry, (index + 1) / 10)
    before = _state(entries)
    dialog.answers[:] = answers
    saved, loaded = [], []
    monkeypatch.setattr(pipeline, "save_settings", lambda path, settings: saved.append(path))
    monkeypatch.setattr(pipeline, "load_settings", lambda path: loaded.append(path) or pipeline.Settings())

    assert not main.close()
    qt_app.processEvents()
    assert main.isVisible()
    assert sheet in main.sheets
    assert all(image in sheet.images for image in images)
    assert _state(entries) == before
    assert not saved and not loaded
    assert len(dialog.prompts) == len(answers)
    assert all(multiple for _, multiple in dialog.prompts)


@pytest.mark.parametrize("cancel_result", [2, 0])
def test_image_close_cancel_and_window_close_keep_its_entry_unchanged(window, qt_app, cancel_result):
    main, folder, dialog, _ = window
    sheet = _sheet(main, folder)
    entry = sheet.sheet.entries[0]
    image = _image(main, sheet, entry)
    _dirty(entry, 0.7)
    before = _state([entry])
    dialog.answers[:] = [cancel_result]

    assert not image.close()
    qt_app.processEvents()
    assert main.isVisible() and sheet in main.sheets and image in sheet.images
    assert _state([entry]) == before
    assert len(dialog.prompts) == 1
    assert dialog.prompts[0][1] is False


def test_app_close_yes_all_covers_dirty_entries_on_multiple_sheets_once(window, tmp_path, monkeypatch, qt_app):
    main, folder, dialog, _ = window
    first = _sheet(main, folder)
    second_folder = tmp_path / "other"
    second_folder.mkdir()
    (second_folder / "other.dcr").touch()
    second = _sheet(main, second_folder)
    entries = [first.sheet.entries[0], second.sheet.entries[0]]
    images = [_image(main, first, entries[0]), _image(main, second, entries[1])]
    for index, entry in enumerate(entries):
        _dirty(entry, (index + 1) / 10)
    saved = []
    monkeypatch.setattr(pipeline, "save_settings", lambda path, settings: saved.append(path))
    dialog.answers[:] = [1061]

    assert main.close()
    qt_app.processEvents()
    assert len(dialog.prompts) == 1 and dialog.prompts[0][1] is True
    assert saved == [entry.path for entry in entries]
    assert all(not entry.dirty for entry in entries)
    assert not main.sheets
    assert all(image not in sheet.images for image, sheet in zip(images, (first, second)))


def test_save_failure_aborts_close_and_keeps_dirty_settings(window, monkeypatch):
    main, folder, dialog, messages = window
    sheet = _sheet(main, folder)
    entry = sheet.sheet.entries[0]
    _dirty(entry, 0.8)
    before = _state([entry])
    dialog.answers[:] = [1]

    def fail_save(*args):
        raise OSError("disk is full")

    monkeypatch.setattr(pipeline, "save_settings", fail_save)
    assert not main.close()
    assert main.isVisible() and sheet in main.sheets
    assert _state([entry]) == before
    assert messages and "disk is full" in messages[0]


def test_read_only_raw_save_as_cancel_aborts_close(window, monkeypatch):
    main, folder, dialog, _ = window
    sheet = _sheet(main, folder)
    entry = sheet.sheet.entries[0]
    _dirty(entry, 0.8)
    before = _state([entry])
    dialog.answers[:] = [1]
    monkeypatch.setattr(Entry, "read_only", property(lambda self: True))

    class CancelFileDialog(QFileDialog):
        def exec(self):
            return QFileDialog.Rejected

    monkeypatch.setattr(gui_app, "QFileDialog", CancelFileDialog)

    assert not main.close()
    assert main.isVisible() and sheet in main.sheets
    assert _state([entry]) == before


def test_finished_image_save_as_failure_aborts_close(window, monkeypatch):
    main, folder, dialog, _ = window
    sheet = _sheet(main, folder)
    entry = next(e for e in sheet.sheet.entries if e.path.suffix == ".jpg")
    _dirty(entry, 0.8)
    before = _state([entry])
    dialog.answers[:] = [1]
    calls = []

    destination = folder / "export.tif"

    class AcceptedFileDialog(QFileDialog):
        def exec(self):
            return QFileDialog.Accepted

        def selectedFiles(self):
            return [str(destination)]

        def selectedNameFilter(self):
            return self.nameFilters()[0]

    monkeypatch.setattr(gui_app, "QFileDialog", AcceptedFileDialog)

    def failed_export(entries, dests, file_type, open_with=None, close_context=False):
        calls.append((list(entries), list(dests), close_context))
        return False

    monkeypatch.setattr(main, "_export", failed_export)

    assert not main.close()
    assert len(calls) == 1 and calls[0][0] == [entry] and calls[0][2] is True
    assert calls[0][1] == [destination]
    assert main.isVisible() and sheet in main.sheets
    assert _state([entry]) == before


def test_sheet_close_confirms_once_before_closing_children(window, monkeypatch, qt_app):
    main, folder, dialog, _ = window
    sheet = _sheet(main, folder)
    entries = sheet.sheet.entries[:2]
    images = [_image(main, sheet, entry) for entry in entries]
    for index, entry in enumerate(entries):
        _dirty(entry, (index + 1) / 10)
    saved = []
    monkeypatch.setattr(pipeline, "save_settings", lambda path, settings: saved.append(path))
    dialog.answers[:] = [1061]

    assert sheet.close()
    qt_app.processEvents()
    assert len(dialog.prompts) == 1 and dialog.prompts[0][1] is True
    assert saved == [entry.path for entry in entries]
    assert all(not entry.dirty for entry in entries)
    assert sheet not in main.sheets and not sheet.images
    assert all(image not in sheet.images for image in images)


def test_image_close_no_restores_settings_and_refreshes_surviving_sheet(window, monkeypatch, qt_app):
    main, folder, dialog, _ = window
    sheet = _sheet(main, folder)
    entry = sheet.sheet.entries[0]
    image = _image(main, sheet, entry)
    _dirty(entry, 0.9)
    dialog.answers[:] = [7]
    refreshed = []
    monkeypatch.setattr(main, "refresh_thumb", lambda changed: refreshed.append(changed))

    assert image.close()
    qt_app.processEvents()
    assert not entry.dirty
    assert entry.settings.exposure == 0
    assert not entry.settings_from_sidecar
    assert refreshed == [entry]


def test_app_close_discard_does_not_refresh_closing_sheet(window, monkeypatch, qt_app):
    main, folder, dialog, _ = window
    sheet = _sheet(main, folder)
    entry = sheet.sheet.entries[0]
    _dirty(entry, 0.9)
    dialog.answers[:] = [7]
    refreshed = []
    monkeypatch.setattr(main, "refresh_thumb", lambda changed: refreshed.append(changed))

    assert main.close()
    qt_app.processEvents()
    assert not entry.dirty and entry.settings.exposure == 0
    assert not refreshed


def test_image_close_only_prompts_for_its_entry(window, monkeypatch, qt_app):
    main, folder, dialog, _ = window
    sheet = _sheet(main, folder)
    image_entry, sibling_entry = sheet.sheet.entries[:2]
    image = _image(main, sheet, image_entry)
    _dirty(image_entry, 0.5)
    _dirty(sibling_entry, 0.6)
    saved = []
    monkeypatch.setattr(pipeline, "save_settings", lambda path, settings: saved.append(path))
    dialog.answers[:] = [1]

    assert image.close()
    qt_app.processEvents()
    assert len(dialog.prompts) == 1 and dialog.prompts[0][1] is False
    assert saved == [image_entry.path]
    assert not image_entry.dirty
    assert sibling_entry.dirty
    assert sheet in main.sheets


def test_delete_aborts_when_closing_selected_dirty_image_is_cancelled(window, monkeypatch):
    main, folder, dialog, _ = window
    sheet = _sheet(main, folder)
    entry = sheet.sheet.entries[0]
    image = _image(main, sheet, entry)
    _dirty(entry, 0.5)
    dialog.answers[:] = [D.SaveChangesDialog.CANCEL]
    monkeypatch.setattr(D, "message", lambda *args, **kwargs: QMessageBox.Yes)
    monkeypatch.setattr(main, "active_sheet", lambda: sheet)
    monkeypatch.setattr(main, "selection", lambda: [entry])
    deleted = []
    monkeypatch.setattr(sheet.sheet, "delete", lambda *args, **kwargs: deleted.append(args) or [])

    main.delete()

    assert not deleted
    assert entry.path.is_file()
    assert entry in sheet.sheet.entries
    assert entry.dirty
    assert image in sheet.images


def test_open_another_image_keeps_unsaved_window_without_prompt(window, monkeypatch):
    main, folder, dialog, _ = window
    sheet = _sheet(main, folder)
    discarded, next_entry = sheet.sheet.entries[:2]
    viewer = main.open_image(discarded, sheet)
    _dirty(discarded, 0.75)
    refreshed = []
    monkeypatch.setattr(main, "refresh_thumb", lambda changed: refreshed.append(changed))

    second = main.open_image(next_entry, sheet)
    assert second is not viewer
    assert viewer.entry is discarded and viewer.isVisible()
    assert second.entry is next_entry and second.isVisible()
    assert discarded.dirty and discarded.settings.exposure == 0.75
    assert not dialog.prompts
    assert refreshed == []


def test_standalone_backing_sheet_stays_while_another_image_is_open(window, qt_app):
    main, folder, dialog, _ = window
    standalone = SheetWindow(main, Sheet(folder, files=[folder / "a.dcr", folder / "b.dcr"]), standalone=True)
    main.sheets.append(standalone)
    first, sibling = standalone.sheet.entries
    first_image = _image(main, standalone, first)
    sibling_image = _image(main, standalone, sibling)
    _dirty(first, 0.5)
    _dirty(sibling, 0.6)
    dialog.answers[:] = [7]

    assert first_image.close()
    qt_app.processEvents()
    assert standalone in main.sheets
    assert sibling_image in standalone.images
    assert not first.dirty and sibling.dirty
    assert len(dialog.prompts) == 1


def test_app_close_without_dirty_entries_does_not_prompt(window, qt_app):
    main, folder, dialog, _ = window
    sheet = _sheet(main, folder)
    _image(main, sheet, sheet.sheet.entries[0])

    assert main.close()
    qt_app.processEvents()
    assert not dialog.prompts
    assert not main.sheets


@pytest.mark.parametrize("outcome", ["cancel", "failure"])
def test_close_export_partial_work_never_reports_success_or_reloads_sheet(window, monkeypatch, outcome):
    main, folder, _, messages = window
    sheet = _sheet(main, folder)
    entries = sheet.sheet.entries[:2]
    for e in entries:
        _dirty(e, 1.0)
    before = _state(entries)
    progresses, pools = [], []
    monkeypatch.setattr(sheet, "reload", lambda: pytest.fail("Close export must not reload dirty documents"))

    class CallbackSignal:
        def __init__(self):
            self.callbacks = []

        def connect(self, callback):
            self.callbacks.append(callback)

        def emit(self, *args):
            for callback in self.callbacks:
                callback(*args)

    class FakeTask:
        def __init__(self, token, *args):
            self.token = token
            self.signals = type("Signals", (), {})()
            self.signals.done = CallbackSignal()
            self.signals.failed = CallbackSignal()

    class FakeProgress:
        def __init__(self, *args):
            self.canceled = False
            self.modality = None
            progresses.append(self)

        def setWindowModality(self, modality):
            self.modality = modality

        def show(self):
            pass

        def set(self, count):
            if outcome == "cancel" and count == 1:
                self.canceled = True

        def close(self):
            pass

    class FakePool:
        def __init__(self, *args):
            self.started = 0
            self.cleared = False
            pools.append(self)

        def setMaxThreadCount(self, count):
            pass

        def start(self, task):
            self.started += 1
            if self.started == 1:
                task.signals.done.emit(task.token, task.token[1])
            elif outcome == "failure":
                task.signals.failed.emit(task.token, "write failed")

        def clear(self):
            self.cleared = True

        def waitForDone(self, *args):
            return True

    monkeypatch.setattr(gui_app, "Task", FakeTask)
    monkeypatch.setattr(gui_app, "QThreadPool", FakePool)
    monkeypatch.setattr(D, "ProgressDialog", FakeProgress)
    destinations = [folder / "first-export.tif", folder / "second-export.tif"]
    assert main._export(entries, destinations, "tiff", close_context=True) is False
    assert progresses[0].modality == Qt.ApplicationModal
    assert _state(entries) == before
    assert main.isVisible() and sheet in main.sheets
    if outcome == "cancel":
        assert pools[0].cleared
    else:
        assert any("write failed" in message for message in messages)
