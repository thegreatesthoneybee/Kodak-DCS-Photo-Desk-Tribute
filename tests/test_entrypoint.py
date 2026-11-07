"""The GUI has its own Windows taskbar identity, including pythonw launches."""
import ctypes
import subprocess
import sys
from types import SimpleNamespace

import pytest
from photodesk import __main__ as entrypoint


def test_windows_identity_is_set_before_gui_start(monkeypatch):
    events = []
    monkeypatch.setattr(sys, "argv", ["photodesk"])
    monkeypatch.setattr(entrypoint.multiprocessing, "freeze_support", lambda: events.append("freeze"))
    monkeypatch.setattr(entrypoint, "_set_windows_app_id", lambda: events.append("identity"))
    monkeypatch.setitem(sys.modules, "photodesk.gui.app", SimpleNamespace(main=lambda: events.append("gui") or 7))
    assert entrypoint.main() == 7
    assert events == ["freeze", "identity", "gui"]


def test_cli_does_not_change_application_identity(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["photodesk", "looks"])
    monkeypatch.setattr(entrypoint.multiprocessing, "freeze_support", lambda: None)
    def unexpected():
        pytest.fail("CLI must not change the GUI taskbar identity")
    monkeypatch.setattr(entrypoint, "_set_windows_app_id", unexpected)
    monkeypatch.setitem(sys.modules, "photodesk.cli", SimpleNamespace(main=lambda: 3))
    assert entrypoint.main() == 3


def test_other_platforms_do_not_load_windows_shell(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    def unexpected(*args):
        pytest.fail("Windows DLL loaded on another platform")
    monkeypatch.setattr(ctypes, "WinDLL", unexpected, raising=False)
    entrypoint._set_windows_app_id()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows shell API")
def test_actual_windows_process_receives_photodesk_identity():
    script = '''
import ctypes
from photodesk.__main__ import _set_windows_app_id
_set_windows_app_id()
get_id = ctypes.WinDLL("shell32").GetCurrentProcessExplicitAppUserModelID
get_id.argtypes = [ctypes.POINTER(ctypes.c_wchar_p)]
get_id.restype = ctypes.c_long
value = ctypes.c_wchar_p()
assert get_id(ctypes.byref(value)) == 0
try:
    assert value.value == "org.photodesk.PhotoDesk", value.value
finally:
    free = ctypes.WinDLL("ole32").CoTaskMemFree
    free.argtypes = [ctypes.c_void_p]
    free(value)
'''
    subprocess.run([sys.executable, "-c", script], check=True, timeout=20)
