"""
File Name: ingest_data.py
Description: Automates the acquisition of the Tate Gallery metadata,
             standardizes text through aggressive normalization (punctuation
             and hyphen handling), and filters the corpus for search.
"""

import os
import re
import pandas as pd
import requests

# Source URL and local file naming conventions
DATA_URL = "https://github.com/tategallery/collection/raw/master/artwork_data.csv"
RAW_FILE = "raw_tate_data.csv"
OUTPUT_FILE = "art_gallery_data.csv"


def ensure_data_exists():
    """
    Ensures the processed dataset exists. Downloads and processes if missing.
    Useful for multi-device deployment.
    """
    if os.path.exists(OUTPUT_FILE):
        print(f"[INFO] {OUTPUT_FILE} already exists. Skipping ingestion.")
        return

    print("[INFO] Dataset not found. Starting ingestion pipeline...")
    try:
        download_dataset(DATA_URL, RAW_FILE)
        process_and_filter(RAW_FILE, OUTPUT_FILE)
        print("[SUCCESS] Dataset ready.")
    except (
        ConnectionError,
        FileNotFoundError,
        KeyError,
        pd.errors.EmptyDataError,
    ) as e:
        print(f"[ERROR] Ingestion pipeline failed: {e}")
        raise


def download_dataset(url, filename, timeout=30):
    """
    Downloads the raw CSV dataset with timeout and status validation.
    Uses streaming write for robustness with large files.
    """
    if os.path.exists(filename):
        print(f"[INFO] {filename} already exists. Skipping download.")
        return

    print(f"[INFO] Downloading dataset from {url}...")

    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("[SUCCESS] Dataset downloaded successfully.")

    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Network error during download: {e}") from e


def normalize_text(text):
    """
    Enhanced normalization: Strips hyphens and punctuation to
    ensure 'self-portrait' matches 'self portrait'. Suitable
    for both BM25 and semantic models.
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Replace hyphens with spaces to merge compound words correctly
    text = text.replace("-", " ")

    # Remove all punctuation except alphanumeric and spaces
    text = re.sub(r"[^\w\s]", "", text)

    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def process_and_filter(input_file, output_file, sample_size=70000):
    """
    Cleans metadata, normalizes text, and exports a balanced subset.
    Standardizes column names and removes records missing image URLs.
    """
    print("[INFO] Processing and filtering metadata...")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found.")

    # Load dataset with high-capacity settings
    df = pd.read_csv(input_file, low_memory=False)

    # Standardize column headers
    df.columns = df.columns.str.lower().str.strip()

    # Define the critical schema needed for the app
    required_columns = ["id", "artist", "title", "medium", "year", "thumbnailurl"]

    # Validate presence of required data
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Required columns {missing_cols} not found in dataset.")

    df = df[required_columns].copy()

    # Log initial dataset volume
    initial_size = len(df)

    # Strict cleaning: Remove duplicates and rows missing essential display data
    df.dropna(subset=["artist", "title", "medium", "thumbnailurl"], inplace=True)
    df.drop_duplicates(subset=["id"], inplace=True)

    print(f"[INFO] Records before cleaning: {initial_size}")
    print(f"[INFO] Records after cleaning:  {len(df)}")

    # Apply the punctuation-aware normalization
    for col in ["artist", "title", "medium"]:
        df[col] = df[col].astype(str).apply(normalize_text)

    # Deterministic sampling to ensure consistency across devices
    actual_sample_size = min(len(df), sample_size)
    if actual_sample_size < sample_size:
        print("[WARNING] Dataset too small for requested sample size.")

    df_final = df.sample(n=actual_sample_size, random_state=42)

    # Export to final search-ready CSV
    df_final.to_csv(output_file, index=False)
    print(f"[SUCCESS] Final dataset size: {len(df_final)}")
    print(f"[SUCCESS] Saved to {output_file}")


if __name__ == "__main__":
    try:
        download_dataset(DATA_URL, RAW_FILE)
        process_and_filter(RAW_FILE, OUTPUT_FILE)
    except Exception as e:  # pylint: disable=broad-except
        print(f"[ERROR] Main ingestion loop failed: {e}")
