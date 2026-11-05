"""Preferences radio groups use the original resources without shared app state."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QDialog

from photodesk import camera, export, pipeline, profiles
from photodesk.gui.dialogs import PreferencesDialog
from photodesk.gui.res import MissingResources, Resources


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def preferences(qt_app, tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTODESK_SETTINGS", str(tmp_path / "settings.ini"))
    monkeypatch.setattr(camera, "_unlock_all", False)
    try:
        resources = Resources.load()
    except MissingResources as exc:
        pytest.skip(f"Photo Desk UI resources unavailable: {exc}")

    opts = export.OutputOptions(
        resize=True,
        scale_style="percent",
        tiff_linear=False,
        tiff_16bit=False,
        jpeg_quality="best",
    )
    alternate_space = str(tmp_path / "alternate.icc")
    spaces = [(pipeline.resolve_space(opts.working_space), "Default"), (alternate_space, "Alternate")]
    dialog = PreferencesDialog(resources, None, opts, "", spaces)
    yield dialog, alternate_space
    dialog.close()
    dialog.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qt_app.processEvents()


def test_preference_radio_groups_are_independent_and_accept_all_options(preferences):
    dialog, alternate_space = preferences
    b = dialog.b

    assert b[1366].isChecked()  # Export size: percent.
    assert b[1182].isChecked()  # TIFF image data: render to RGB working space.
    assert b[1353].isChecked()  # TIFF depth: 8 bit.
    assert b[1006].isChecked()  # JPEG quality: best.

    b[1365].click()  # Export size: fill frame.
    b[1085].click()  # TIFF image data: linear.
    b[1354].click()  # TIFF depth: 16 bit.
    b[1008].click()  # JPEG quality: good.
    b[1179].setCurrentIndex(1)
    b[1375].setCurrentIndex(1)
    dialog.appearance_combo.setCurrentIndex(dialog.appearance_combo.findData("dark"))

    assert b[1365].isChecked()
    assert b[1085].isChecked()
    assert b[1354].isChecked()
    assert b[1008].isChecked()

    dialog.accept()

    assert dialog.result() == QDialog.Accepted
    assert dialog.opts.scale_style == "fill"
    assert dialog.opts.tiff_linear
    assert dialog.opts.tiff_16bit
    assert dialog.opts.jpeg_quality == "good"
    assert dialog.opts.working_space == alternate_space
    assert dialog.opts.resolution_unit == "cm"
    assert dialog.appearance == "dark"


def test_each_preference_radio_choice_preserves_other_sections(preferences):
    dialog, _ = preferences
    groups = ((1366, 1365, 1363, 1364), (1085, 1182), (1353, 1354), (1006, 1007, 1008))
    for group in groups:
        other_choices = {
            cid: dialog.b[cid].isChecked()
            for other in groups if other != group for cid in other
        }
        for selected in group:
            dialog.b[selected].click()
            assert [cid for cid in group if dialog.b[cid].isChecked()] == [selected]
            assert all(dialog.b[cid].isChecked() == checked for cid, checked in other_choices.items())


def test_portable_rgb_space_selection_survives_preferences_reopen(qt_app, tmp_path, monkeypatch):
    try:
        resources = Resources.load()
    except MissingResources as exc:
        pytest.skip(f"Photo Desk UI resources unavailable: {exc}")
    monkeypatch.setattr(profiles, "data_dir", lambda: tmp_path / "user-data")
    monkeypatch.setattr(profiles, "system_color_dirs_profiles", lambda: iter(()))
    spaces = pipeline.working_spaces(tmp_path / "empty-bundle")
    for index, (path, description) in enumerate(spaces):
        dialog = PreferencesDialog(resources, None, export.OutputOptions(), "", spaces)
        dialog.b[1179].setCurrentIndex(index)
        dialog.accept()
        saved = export.OutputOptions.from_json(dialog.opts.to_json())
        assert pipeline.resolve_space(saved.working_space) == path
        reopened = PreferencesDialog(resources, None, saved, "", spaces)
        assert reopened.b[1179].currentText() == description
        for widget in (dialog, reopened):
            widget.close()
            widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
