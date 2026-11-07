"""Save prompts must distinguish explicit No from Esc/window-close cancellation."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from photodesk.gui.dialogs import SaveChangesDialog, YesToAllDialog
from photodesk.gui.res import MissingResources, Resources


@pytest.fixture
def context():
    app = QApplication.instance() or QApplication([])
    try:
        resources = Resources.load()
    except MissingResources as exc:
        pytest.skip(str(exc))
    yield app, resources
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.mark.parametrize("multiple", [False, True])
def test_save_dialog_buttons_are_distinct_ordered_and_fit(context, multiple):
    app, res = context
    dialog = SaveChangesDialog(res, None, "Save image.DCR?", multiple=multiple)
    dialog.show()
    app.processEvents()
    expected = [dialog.YES, dialog.YES_ALL, dialog.NO, dialog.CANCEL] if multiple else [dialog.YES, dialog.NO, dialog.CANCEL]
    assert set(dialog.buttons) == set(expected)
    assert [key for key, button in sorted(dialog.buttons.items(), key=lambda item: item[1].x())] == expected
    buttons = [dialog.buttons[key] for key in expected]
    for button in buttons:
        assert button.isVisible()
        assert dialog.rect().contains(button.geometry())
    for left, right in zip(buttons, buttons[1:]):
        assert not left.geometry().intersects(right.geometry())
    assert dialog.cancel_button.isDefault()
    assert "&C" in dialog.cancel_button.text()
    assert dialog.b[dialog.YES_ALL].isHidden() == (not multiple)
    dialog.close()
    assert dialog.result() == dialog.CANCEL
    dialog.deleteLater()


@pytest.mark.parametrize("choice", [SaveChangesDialog.YES, SaveChangesDialog.NO, SaveChangesDialog.CANCEL, SaveChangesDialog.YES_ALL])
def test_explicit_buttons_return_their_own_outcomes(context, choice):
    _, res = context
    dialog = SaveChangesDialog(res, None, "Save image.DCR?", multiple=True)
    QTimer.singleShot(0, dialog.buttons[choice].click)
    assert dialog.exec() == choice
    dialog.deleteLater()


@pytest.mark.parametrize("action", ["escape", "close", "enter"])
def test_escape_close_and_default_enter_cancel(context, action):
    _, res = context
    dialog = SaveChangesDialog(res, None, "Save image.DCR?")
    if action == "close":
        QTimer.singleShot(0, dialog.close)
    else:
        key = Qt.Key_Escape if action == "escape" else Qt.Key_Return
        QTimer.singleShot(0, lambda: QTest.keyClick(dialog, key))
    assert dialog.exec() == dialog.CANCEL
    dialog.deleteLater()


def test_existing_overwrite_dialog_keeps_original_no_result(context):
    _, res = context
    dialog = YesToAllDialog(res, None, "Replace file?")
    QTimer.singleShot(0, dialog.b[2].click)
    assert dialog.exec() == 2
    dialog.deleteLater()
