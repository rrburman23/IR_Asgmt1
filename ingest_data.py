"""
File Name: ingest_data.py
Description: Automates the acquisition of the Tate Gallery metadata,
             standardizes/normalizes text, and filters the corpus.
"""

import pandas as pd
import requests
import os
import re

DATA_URL = "https://github.com/tategallery/collection/raw/master/artwork_data.csv"
RAW_FILE = "raw_tate_data.csv"
OUTPUT_FILE = "art_gallery_data.csv"


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
        raise ConnectionError(f"Network error during download: {e}")


def normalize_text(text):
    """
    Light text normalization suitable for BOTH BM25 and dense models.

    NOTE:
    - Lowercases text
    - Normalizes whitespace
    - Preserves punctuation (important for semantic models)
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def process_and_filter(input_file, output_file, sample_size=2000):
    """
    Cleans metadata, normalizes text, and exports a balanced subset.
    """

    print("[INFO] Processing and filtering metadata...")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found.")

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------
    df = pd.read_csv(input_file, low_memory=False)

    # --------------------------------------------------
    # Column normalization
    # --------------------------------------------------
    df.columns = df.columns.str.lower().str.strip()

    required_columns = ["id", "artist", "title", "medium", "year"]

    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in dataset.")

    df = df[required_columns].copy()

    # --------------------------------------------------
    # Data cleaning
    # --------------------------------------------------
    initial_size = len(df)

    df.dropna(subset=["artist", "title", "medium"], inplace=True)
    df.drop_duplicates(subset=["id"], inplace=True)

    # Optional diagnostic
    if df["id"].duplicated().any():
        print("[WARNING] Duplicate IDs detected after cleaning.")

    print(f"[INFO] Records before cleaning: {initial_size}")
    print(f"[INFO] Records after cleaning:  {len(df)}")

    # --------------------------------------------------
    # Text normalization
    # --------------------------------------------------
    for col in ["artist", "title", "medium"]:
        df[col] = df[col].astype(str).apply(normalize_text)

    # --------------------------------------------------
    # Robust sampling
    # --------------------------------------------------
    actual_sample_size = min(len(df), sample_size)

    if actual_sample_size < sample_size:
        print(
            f"[WARNING] Dataset size ({len(df)}) is smaller than "
            f"requested sample ({sample_size})."
        )

    df_final = df.sample(n=actual_sample_size, random_state=42)

    # --------------------------------------------------
    # Export
    # --------------------------------------------------
    df_final.to_csv(output_file, index=False)

    print(f"[SUCCESS] Final dataset size: {len(df_final)}")
    print(f"[SUCCESS] Saved to {output_file}")


if __name__ == "__main__":
    try:
        download_dataset(DATA_URL, RAW_FILE)
        process_and_filter(RAW_FILE, OUTPUT_FILE)

    except Exception as e:
        print(f"[ERROR] Ingestion pipeline failed: {e}")
