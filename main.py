"""
File Name: main.py
Description: Primary entry point for the Art Gallery Search Engine.
             Orchestrates the Ingestion, Testing, and Retrieval pipelines.
"""

import os
import sys
import pandas as pd

from ingest_data import (
    download_dataset,
    process_and_filter,
    DATA_URL,
    RAW_FILE,
    OUTPUT_FILE,
)
from hybrid_search import ArtGallerySearchEngine
from evaluate_engine import run_evaluation

# Suppress verbose TF INFO logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


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
        except (ConnectionError, FileNotFoundError, KeyError, ValueError) as e:
            print(f"[CRITICAL] Data ingestion failed: {e}")
            sys.exit(1)
    else:
        print(f"[BOOT] Verified existing document store: {OUTPUT_FILE}")

    # 2. Start Engine
    try:
        engine = ArtGallerySearchEngine(OUTPUT_FILE)
        return engine
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"[CRITICAL] Engine initialization failed: {e}")
        sys.exit(1)


def run_interactive_session(engine):
    """
    Launches the Command Line Interface for end-users.
    """
    print("\n" + "-" * 60)
    print("SEARCH INTERFACE READY")
    print("Type '/exit' to quit. Type '/evaluate' to run metrics.")
    print("Optional: Use '| key:val' to apply metadata filters (e.g., 'landscape | year:1888').")
    print("-" * 60)

    while True:
        query = input("\nEnter search query: ")
        query = query.strip()

        if query.lower() in ["/exit", "/quit"]:
            print("[INFO] Shutting down engine...")
            break

        if not query:
            continue

        # Option to trigger evaluation from main
        if query.lower() == "/evaluate":
            run_evaluation()
            continue

        # Parse optional filters
        parts = query.split("|")
        search_query = " ".join(parts[0].split())
        
        filters = {}
        if len(parts) > 1:
            for part in parts[1:]:
                if ":" in part:
                    k, v = part.split(":", 1)
                    filters[k.strip().lower()] = v.strip()

        results = engine.hybrid_search(search_query, top_k=5, filters=filters)

        print(f"\n--- TOP {len(results)} RESULTS ---")
        for res in results:
            desc = (
                res["Description"][:75] + "..."
                if len(res["Description"]) > 75
                else res["Description"]
            )

            year_str = f" | Year: {res['Year']}" if pd.notna(res['Year']) else ""
            print(f"{res['Rank']}. {res['Title'].strip().title()}{year_str}")
            print(f"   Artist: {res['Artist'].strip().title()}")
            print(f"   Medium: {desc}")
            print(f"   Score:  {res['Score']:.4f}")
            print(f"   Why this result?: {res['Reasons']}")
            print("-" * 20)


if __name__ == "__main__":
    # Initialize the system
    search_engine = bootstrap_system()

    # Launch CLI
    run_interactive_session(search_engine)
