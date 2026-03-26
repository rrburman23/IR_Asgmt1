"""
File Name: main.py
Description: Primary entry point for the Art Gallery Search Engine.
             Acts as a router to launch either the GUI (default) or the CLI.
"""

# pylint: disable=no-name-in-module, import-outside-toplevel, wrong-import-position

import os
import sys
import argparse

# Suppress verbose TF INFO logs before anything else loads
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Import your ingestion pipeline to ensure data exists
from ingest_data import (
    download_dataset,
    process_and_filter,
    DATA_URL,
    RAW_FILE,
    OUTPUT_FILE,
)

def ensure_data_exists():
    """Checks if the dataset exists; if not, triggers the download pipeline."""
    if not os.path.exists(OUTPUT_FILE):
        print("[BOOT] Local document store not found. Initializing ETL pipeline...")
        try:
            download_dataset(DATA_URL, RAW_FILE)
            process_and_filter(RAW_FILE, OUTPUT_FILE)
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"[CRITICAL] Data ingestion failed: {e}")
            sys.exit(1)


def launch_cli():
    """Launches the Command Line Interface."""
    # We only import the engine if we are running the CLI to save memory
    from hybrid_search import ArtGallerySearchEngine

    print("\n" + "=" * 60)
    print("ART GALLERY SEARCH ENGINE (CLI MODE)")
    print("=" * 60)

    try:
        engine = ArtGallerySearchEngine(OUTPUT_FILE)
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"[CRITICAL] Engine initialization failed: {e}")
        sys.exit(1)

    print("\nSEARCH INTERFACE READY. Type '/exit' to quit.")
    print("-" * 60)

    while True:
        query = input("\nEnter search query: ").strip()

        if query.lower() in ["/exit", "/quit", "exit", "quit"]:
            print("[INFO] Shutting down engine...")
            break

        if not query:
            continue

        results = engine.hybrid_search(query, top_k=5)

        print(f"\n--- TOP {len(results)} RESULTS ---")
        for res in results:
            desc = (
                (res["Description"][:75] + "...")
                if len(res["Description"]) > 75
                else res["Description"]
            )
            print(f"{res['Rank']}. {res['Title']}")
            print(f"   Artist: {res['Artist']}")
            print(f"   Medium: {desc}")
            print(f"   Score:  {res['Score']}")
            if res.get("Thumbnail"):
                print(f"   Image:  {res['Thumbnail']}")
            print("-" * 20)


def launch_gui():
    """Launches the Graphical User Interface."""
    from gui import ArtSearchGUI
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = ArtSearchGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    # 1. Parse command-line arguments
    parser = argparse.ArgumentParser(description="Tate Gallery Search Engine")
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Launch the app in Command Line Interface mode instead of the GUI.",
    )
    args = parser.parse_args()

    # 2. Make sure we have data before starting anything
    ensure_data_exists()

    # 3. Route to the correct interface
    if args.cli:
        launch_cli()
    else:
        launch_gui()
