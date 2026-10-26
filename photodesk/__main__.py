import multiprocessing
import sys

COMMANDS = ("convert", "install-profiles", "looks", "-h", "--help")


def main():
    multiprocessing.freeze_support()  # frozen builds: let worker processes run their task, not the app
    if len(sys.argv) > 1 and sys.argv[1] in COMMANDS:
        from photodesk.cli import main as cli_main
        return cli_main()
    from photodesk.gui.app import main as gui_main
    return gui_main()


if __name__ == "__main__":
    sys.exit(main())
