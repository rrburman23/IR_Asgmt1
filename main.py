"""
File Name: main.py
Description: Unified entry/router for the Tate Gallery AI Search System.
- Launches: PyQt GUI (default), CLI (--cli), test suite (--test), or IR eval.
- Handles OS signals for clean exits.

Packaging/performance:
- Resolve icon path relative to executable directory when frozen (taskbar icon).
"""

from __future__ import annotations

import os
import sys
import argparse
import unittest


def app_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


# Make sure stdout/stderr always have isatty(), even in a PyInstaller windowed exe.
class _SafeStream:
    def __init__(self, backing):
        self._backing = backing

    def write(self, data):
        if self._backing is not None:
            return self._backing.write(data)

    def flush(self):
        if self._backing is not None and hasattr(self._backing, "flush"):
            return self._backing.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        if self._backing is not None:
            return getattr(self._backing, name)
        raise AttributeError(name)


sys.stdout = _SafeStream(getattr(sys, "stdout", None))
sys.stderr = _SafeStream(getattr(sys, "stderr", None))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ICON_PATH = os.path.join(app_dir(), "icon.ico")

try:
    import ctypes

    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("tate.search.engine")
except Exception:
    pass

def launch_tests():
    import test_engine

    print("\n" + "=" * 60)
    print("TATE GALLERY SEARCH | AUTOMATED UNIT TESTS")
    print("=" * 60)
    suite = unittest.TestLoader().loadTestsFromModule(test_engine)
    unittest.TextTestRunner(verbosity=2).run(suite)


def launch_evaluation():
    from evaluate_engine import EvalConfig, run_evaluation
    import ingest_data

    run_evaluation(config=EvalConfig(data_path=ingest_data.OUTPUT_FILE))


def launch_cli():
    from hybrid_search import ArtGallerySearchEngine
    import ingest_data

    print("\n" + "=" * 60)
    print("TATE GALLERY SEARCH | CLI MODE")
    print("=" * 60)

    try:
        engine = ArtGallerySearchEngine(ingest_data.OUTPUT_FILE)
        print("\nReady! Commands: '/exit', '/test', '/evaluate', '/help'")
        print("-" * 60)
        while True:
            try:
                query = input("\nEnter search query: ").strip()
                if not query:
                    continue

                if query.startswith("/"):
                    cmd = query.lower()
                    if cmd in ["/exit", "/quit"]:
                        print("[INFO] Shutting down.")
                        break
                    if cmd == "/help":
                        print("\nCOMMANDS: /exit, /test, /evaluate, /help")
                        print("Search for anything else by typing without a slash.")
                        continue
                    if cmd == "/test":
                        launch_tests()
                        continue
                    if cmd in ["/evaluate", "/eval"]:
                        launch_evaluation()
                        continue
                    print(f"[ERROR] Unknown command: {query}")
                    continue

                response = engine.hybrid_search(query, top_k=5, page=1, per_page=5)
                results = response.get("results", [])
                if not results:
                    print("[INFO] No matching artworks found.")
                    continue

                suggestion = response.get("suggestion")
                if suggestion:
                    print(f"\n[!] Did you mean: '{suggestion}'?")

                print(f"\n--- TOP {len(results)} RESULTS ---")
                for res in results:
                    print(f"{res['Rank']}. {res['Title']}")
                    print(f"   Artist: {res['Artist']}")
                    print(f"   Logic:  {res['Reasons']}")
                    print("-" * 20)
            except KeyboardInterrupt:
                break
    except Exception as e:
        print(f"[CRITICAL] CLI Error: {e}")
        sys.exit(1)


def launch_gui():
    import signal

    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QTimer
    from PyQt6.QtGui import QIcon

    from gui import ArtSearchGUI

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(ICON_PATH))

    timer = QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    window = ArtSearchGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    from ingest_data import ensure_data_exists

    parser = argparse.ArgumentParser(description="Tate Gallery AI Search Engine")
    parser.add_argument("--cli", action="store_true", help="Start CLI")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--evaluate", action="store_true", help="Run eval")
    args = parser.parse_args()

    ensure_data_exists()

    if args.test:
        launch_tests()
    elif args.evaluate:
        launch_evaluation()
    elif args.cli:
        launch_cli()
    else:
        launch_gui()
