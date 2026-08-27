import sys
import os
import io


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


if sys.platform == "win32":
    os.system("chcp 65001 > nul")

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import tkinter as tk
from src.gui.main_window import MainWindow
from src.app_controller import AppController


def main():
    root = tk.Tk()
    app = AppController()
    gui = MainWindow(root, app)
    root.mainloop()


if __name__ == "__main__":
    main()
