"""
File Name: ingest_data.py
Description: Automates the acquisition of the Tate Gallery metadata
             and filters the corpus to a 2,000-document subset.
"""

import pandas as pd
import requests
import os

# Configuration: Source URL for the Tate Gallery artwork metadata (GitHub)
DATA_URL = "https://github.com/tategallery/collection/raw/master/artwork_data.csv"
OUTPUT_FILE = "art_gallery_data.csv"


def download_dataset(url, filename):
    """
    Downloads the raw CSV dataset from the remote repository.
    """
    if os.path.exists(filename):
        print(f"[INFO] {filename} already exists. Skipping download.")
        return

    print(f"[INFO] Downloading dataset from {url}...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        print("[SUCCESS] Dataset downloaded successfully.")
    else:
        raise ConnectionError(
            f"Failed to download data. Status code: {response.status_code}"
        )


def process_and_filter(input_file, output_file, sample_size=2000):
    """
    Cleans the raw metadata and exports a balanced subset for indexing.
    Required fields: Title, Artist, Description (Medium/Inscription).
    """
    print("[INFO] Processing and filtering metadata...")

    # Load dataset - low_memory=False used to handle mixed types in large CSV
    df = pd.read_csv(input_file, low_memory=False)

    # Standardizing column names for internal consistency
    # We use 'medium' as a proxy for 'description' as per common IR practices
    required_columns = ["id", "artist", "title", "medium", "year"]
    df_subset = df[required_columns].copy()

    # Data Cleaning: Remove records with missing critical metadata
    df_subset.dropna(subset=["artist", "title", "medium"], inplace=True)

    # Deduplication: Ensure unique artwork entries
    df_subset.drop_duplicates(subset=["id"], inplace=True)

    # Sampling: Constrain the corpus to the project scope (approx. 2,000 docs)
    # A random_state is set for reproducibility during evaluation
    df_final = df_subset.sample(n=sample_size, random_state=42)

    # Export to project directory
    df_final.to_csv(output_file, index=False)
    print(f"[SUCCESS] Processed {len(df_final)} documents. Saved to {output_file}.")


if __name__ == "__main__":
    try:
        # Step 1: Physical acquisition of data
        download_dataset(DATA_URL, "raw_tate_data.csv")

        # Step 2: Logical filtering and normalization
        process_and_filter("raw_tate_data.csv", OUTPUT_FILE)

    except Exception as e:
        print(f"[ERROR] Ingestion pipeline failed: {e}")
