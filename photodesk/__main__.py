import multiprocessing
import sys

COMMANDS = ("convert", "install-profiles", "looks", "-h", "--help")


def _set_windows_app_id():
    """Keep source launches out of pythonw.exe's taskbar group."""
    if sys.platform != "win32":
        return
    import ctypes
    set_id = ctypes.WinDLL("shell32").SetCurrentProcessExplicitAppUserModelID
    set_id.argtypes = [ctypes.c_wchar_p]
    set_id.restype = ctypes.c_long
    result = set_id("org.photodesk.PhotoDesk")
    if result < 0:
        raise OSError(result, "Could not set PhotoDesk's Windows application identity")


def main():
    multiprocessing.freeze_support()  # frozen builds: let worker processes run their task, not the app
    if len(sys.argv) > 1 and sys.argv[1] in COMMANDS:
        from photodesk.cli import main as cli_main
        return cli_main()
    _set_windows_app_id()
    from photodesk.gui.app import main as gui_main
    return gui_main()


if __name__ == "__main__":
    sys.exit(main())
