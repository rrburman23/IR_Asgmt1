"""
File Name: main.py
Description: Unified entry/router for the Tate Gallery AI Search System.
- Launches: PyQt GUI (default), CLI (--cli), test suite (--test), or IR eval.
- Handles OS signals for clean exits.
- Satisfies Pylance diagnostics for function exports.
"""

import os
import sys
import argparse
import signal
import unittest

# pylint: disable=no-name-in-module
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

# Local project imports
from ingest_data import ensure_data_exists, OUTPUT_FILE
from hybrid_search import ArtGallerySearchEngine
from gui import ArtSearchGUI

# Suppress TensorFlow/transformer logging before model load
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def launch_tests():
    """Run all automatic regression/unit/logic tests."""
    # pylint: disable=import-outside-toplevel
    import test_engine

    print("\n" + "=" * 60)
    print("TATE GALLERY SEARCH | AUTOMATED UNIT TESTS")
    print("=" * 60)
    suite = unittest.TestLoader().loadTestsFromModule(test_engine)
    unittest.TextTestRunner(verbosity=2).run(suite)


def launch_evaluation():
    """Runs strict IR evaluation on MRR and NDCG@10 metrics."""
    # pylint: disable=import-outside-toplevel
    from evaluate_engine import run_evaluation

    run_evaluation(OUTPUT_FILE)


def launch_cli():
    """Synchronous command-line interface with strict command handling."""
    print("\n" + "=" * 60)
    print("TATE GALLERY SEARCH | CLI MODE")
    print("=" * 60)

    try:
        engine = ArtGallerySearchEngine(OUTPUT_FILE)
        print("\nReady! Commands: '/exit', '/test', '/evaluate', '/help'")
        print("-" * 60)
        while True:
            try:
                query = input("\nEnter search query: ").strip()
                if not query:
                    continue

                # Strict Slash Command Handling
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

                # Standard Search (if no slash)
                results = engine.hybrid_search(query, top_k=5)
                if not results:
                    print("[INFO] No matching artworks found.")
                    continue

                if results[0].get("Suggestion"):
                    print(f"\n[!] Did you mean: '{results[0]['Suggestion']}'?")

                print(f"\n--- TOP {len(results)} RESULTS ---")
                for res in results:
                    print(f"{res['Rank']}. {res['Title']}")
                    print(f"   Artist: {res['Artist']}")
                    print(f"   Logic:  {res['Reasons']}")
                    print("-" * 20)
            except KeyboardInterrupt:
                break
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"[CRITICAL] CLI Error: {e}")
        sys.exit(1)


def launch_gui():
    """Launches the PyQt GUI with robust signal handling."""
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QApplication(sys.argv)

    timer = QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    window = ArtSearchGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
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
