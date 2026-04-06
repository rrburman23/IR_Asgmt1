"""
File Name: main.py
Description: The primary entry point and master router for the application.
             Launches the PyQt6 GUI (default), the text-based CLI (--cli),
             the automated unit test suite (--test), or the formal IR
             evaluation pipeline (--evaluate). Implements OS signal handling
             to ensure interrupts (Ctrl+C) work gracefully across all modes.
"""

import os
import sys
import argparse
import signal
import unittest

# Suppress TensorFlow logging warnings before any ML libraries are imported
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# PyQt6 imports for GUI mode
# pylint: disable=no-name-in-module
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

# Import data ingestion pipeline to verify data integrity before boot
from ingest_data import ensure_data_exists, OUTPUT_FILE
from hybrid_search import ArtGallerySearchEngine
from gui import ArtSearchGUI


# ==========================================
# 1. Automated Testing & Evaluation Routers
# ==========================================


def launch_tests():
    """
    Programmatically loads and executes the unit tests from test_engine.py.
    This bypasses the standard unittest CLI parsing to avoid conflicts with
    our own argparse implementation.
    """
    # Lazy load the testing module to preserve memory during standard usage
    import test_engine

    print("\n" + "=" * 60)
    print("TATE GALLERY SEARCH | AUTOMATED UNIT TESTS")
    print("=" * 60)

    # Load and execute the test suite programmatically
    suite = unittest.TestLoader().loadTestsFromModule(test_engine)
    unittest.TextTestRunner(verbosity=2).run(suite)


def launch_evaluation():
    """
    Executes the formal Information Retrieval evaluation pipeline, calculating
    Mean Reciprocal Rank (MRR) and NDCG@10 for predefined query judgments.
    """
    # pylint: disable=import-outside-toplevel
    # Lazy load the evaluation module
    from evaluate_engine import run_evaluation

    # The run_evaluation function handles its own clean terminal output
    run_evaluation(OUTPUT_FILE)


# ==========================================
# 2. Interactive Interfaces (CLI & GUI)
# ==========================================


def launch_cli():
    """
    Starts the terminal-based interactive search interface.
    Provides continuous querying and allows execution of tests/evaluations
    via slash commands (/test, /evaluate) directly from the prompt.
    """
    print("\n" + "=" * 60)
    print("TATE GALLERY SEARCH | CLI MODE")
    print("=" * 60)

    try:
        # Initialize the AI engine and load data/indexes
        engine = ArtGallerySearchEngine(OUTPUT_FILE)
        print("\nReady! Commands: '/exit', '/test', '/evaluate'")
        print("-" * 60)

        # Main interactive loop
        while True:
            try:
                query = input("\nEnter search query: ").strip()

                if not query:
                    continue

                # Check for administrative slash commands
                if query.lower() in ["/exit", "/quit", "exit"]:
                    break
                if query.lower() in ["/test", "test"]:
                    launch_tests()
                    continue
                if query.lower() in ["/evaluate", "/eval", "evaluate"]:
                    launch_evaluation()
                    continue

                # Execute standard hybrid search retrieval
                results = engine.hybrid_search(query, top_k=5)

                # Format output for terminal readability
                print(f"\n--- TOP {len(results)} RESULTS ---")
                for res in results:
                    print(f"{res['Rank']}. {res['Title']}")
                    print(f"   Artist: {res['Artist']}")
                    print(f"   Score:  {res['Score']}")
                    print(f"   Link:   {res['Thumbnail']}")
                    print("-" * 20)

            except KeyboardInterrupt:
                # Catch Ctrl+C gracefully in the terminal instead of crashing
                print("\n[INFO] Interrupt received. Shutting down...")
                break

    except Exception as e:  # pylint: disable=broad-except
        print(f"[CRITICAL] CLI Error: {e}")
        sys.exit(1)


def launch_gui():
    """
    Starts the PyQt6 Graphical Interface.
    Implements a QTimer heartbeat to ensure the Python interpreter
    can catch system signals (like SIGINT/Ctrl+C) while the C++ Qt loop runs.
    """
    # 1. Instruct Python to use the default OS behavior for interrupt signals
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)

    # 2. The 'Heartbeat' Timer
    # Python suspends execution when `app.exec()` runs. This timer wakes
    # the interpreter up every 500ms so it can detect if Ctrl+C was pressed.
    timer = QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    # 3. Initialize and display the main window
    window = ArtSearchGUI()
    window.show()

    # 4. Start the application event loop
    sys.exit(app.exec())


# ==========================================
# 3. Execution Routing
# ==========================================

if __name__ == "__main__":
    # Configure command-line arguments for the master router
    parser = argparse.ArgumentParser(description="Tate Gallery AI Search Engine")
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run the application in terminal mode instead of launching the GUI.",
    )
    parser.add_argument(
        "--test", action="store_true", help="Run the automated unit test suite."
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run the formal IR metrics evaluation pipeline.",
    )
    args = parser.parse_args()

    # Pre-flight check: Ensure dataset exists or download it automatically
    ensure_data_exists()

    # Route execution based on flags provided by the user
    # Priority order ensures tests/evaluations run first if multiple flags are passed
    if args.test:
        launch_tests()
    elif args.evaluate:
        launch_evaluation()
    elif args.cli:
        launch_cli()
    else:
        # Default behavior with no flags
        launch_gui()
