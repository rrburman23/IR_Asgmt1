"""
File Name: main.py
Description: Primary entry point for the Art Gallery Search Engine.
             Orchestrates the Ingestion, Testing, and Retrieval pipelines.
"""

# Suppress verbose TF INFO logs
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import os
import sys
from ingest_data import (
    download_dataset,
    process_and_filter,
    DATA_URL,
    RAW_FILE,
    OUTPUT_FILE,
)
from hybrid_search import ArtGallerySearchEngine


def bootstrap_system():
    """
    Ensures the environment is prepared before launching the engine.
    """
    print("=" * 60)
    print("ART GALLERY SEARCH ENGINE")
    print("=" * 60)

    # 1. Pipeline Check: Data Ingestion
    if not os.path.exists(OUTPUT_FILE):
        print("[BOOT] Local document store not found. Initializing ETL pipeline...")
        try:
            download_dataset(DATA_URL, RAW_FILE)
            process_and_filter(RAW_FILE, OUTPUT_FILE)
        except Exception as e:
            print(f"[CRITICAL] Data ingestion failed: {e}")
            sys.exit(1)
    else:
        print(f"[BOOT] Verified existing document store: {OUTPUT_FILE}")

    # 2. Start Engine
    try:
        engine = ArtGallerySearchEngine(OUTPUT_FILE)
        return engine
    except Exception as e:
        print(f"[CRITICAL] Engine initialization failed: {e}")
        sys.exit(1)


def run_interactive_session(engine):
    """
    Launches the Command Line Interface for end-users.
    """
    print("\n" + "-" * 60)
    print("SEARCH INTERFACE READY")
    print(
        "Type 'exit' to quit. Type 'evaluate' to run metrics (requires evaluate_engine.py)."
    )
    print("-" * 60)

    while True:
        query = input("\nEnter search query: ")
        query = " ".join(query.split())

        if query.lower() in ["exit", "quit"]:
            print("[INFO] Shutting down engine...")
            break

        if not query:
            continue

        # Option to trigger evaluation from main if you choose to import it
        if query.lower() == "evaluate":
            print("[INFO] Please run 'python evaluate_engine.py' for the full report.")
            continue

        results = engine.hybrid_search(query, top_k=5)

        print(f"\n--- TOP {len(results)} RESULTS ---")
        for res in results:
            desc = (
                res["Description"][:75] + "..."
                if len(res["Description"]) > 75
                else res["Description"]
            )

            print(f"{res['Rank']}. {res['Title'].strip().title()}")
            print(f"   Artist: {res['Artist'].strip().title()}")
            print(f"   Medium: {desc}")
            print(f"   Score:  {res['Score']}")
            print("-" * 20)


if __name__ == "__main__":
    # Initialize the system
    search_engine = bootstrap_system()

    # Launch CLI
    run_interactive_session(search_engine)
