"""
File Name: ingest_data.py
Description: Automates the acquisition of the Tate Gallery metadata.
- Implements a Placeholder Policy: Artworks without images are NOT dropped.
- Standardizes text for search while protecting URL structures.
- Generates a searchable semantic blob for dense indexing.
"""

import os
import re
import pandas as pd
import requests

# Source and Output Configuration
DATA_URL = "https://github.com/tategallery/collection/raw/master/artwork_data.csv"
RAW_FILE = "raw_tate_data.csv"
OUTPUT_FILE = "art_gallery_data.csv"


def ensure_data_exists():
    """
    Ensures the processed dataset exists on disk.
    If missing, triggers the download and ETL pipeline.
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
    ) as exc:
        print(f"[ERROR] Ingestion pipeline failed: {exc}")
        raise


def download_dataset(url, filename, timeout=30):
    """
    Downloads the raw CSV dataset using a streaming byte-write for memory safety.
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
        print("[SUCCESS] Raw dataset downloaded.")
    except requests.exceptions.RequestException as exc:
        raise ConnectionError(f"Network error during download: {exc}") from exc


def normalize_text_for_ingest(text):
    """
    Internal normalization for document storage.
    Note: This is NOT applied to URLs.
    """
    if not isinstance(text, str):
        return ""
    # Standard lowercase and hyphen handling
    text = text.lower().replace("-", " ")
    # Strip everything except alpha-numeric and standard spaces
    text = re.sub(r"[^\w\s]", "", text)
    # Collapse multiple whitespaces
    return re.sub(r"\s+", " ", text).strip()


def process_and_filter(input_file, output_file, sample_size=70000):
    """
    Loads, cleans, and standardizes the Tate metadata.
    Placeholder Policy: Keeps records even if thumbnailurl is missing.
    """
    print("[INFO] Processing and filtering metadata...")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found.")

    # High-capacity load
    df = pd.read_csv(input_file, low_memory=False)

    # Standardize column headers
    df.columns = df.columns.str.lower().str.strip()

    # Define the schema required for the search application
    required_cols = ["id", "artist", "title", "medium", "year", "thumbnailurl"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing essential columns: {missing}")

    df = df[required_cols].copy()

    # Data Cleaning: deduplicate on Tate ID
    initial_count = len(df)
    df.drop_duplicates(subset=["id"], inplace=True)

    # Placeholder Logic: Fill missing strings instead of dropping rows
    for col in ["artist", "title", "medium", "thumbnailurl"]:
        df[col] = df[col].fillna("").astype(str)

    print(f"[INFO] Records before cleaning: {initial_count}")
    print(f"[INFO] Records preserved:       {len(df)}")

    # Deterministic sampling (up to 70k or actual size)
    actual_sample = min(len(df), sample_size)
    df_final = df.sample(n=actual_sample, random_state=42).copy()

    # Pre-calculate search fields to reduce runtime overhead
    # We do NOT normalize thumbnailurl here to protect punctuation in links
    for col in ["artist", "title", "medium"]:
        df_final[f"search_{col}"] = df_final[col].apply(normalize_text_for_ingest)

    # Create a semantic blob for the Dense Transformer
    df_final["semantic_blob"] = (
        "Title: "
        + df_final["title"]
        + ". Artist: "
        + df_final["artist"]
        + ". Medium: "
        + df_final["medium"]
        + "."
    )

    # Export to search-ready CSV
    df_final.to_csv(output_file, index=False)
    print(f"[SUCCESS] Final search corpus: {len(df_final)} records.")
    print(f"[SUCCESS] Saved to {output_file}")


if __name__ == "__main__":
    try:
        download_dataset(DATA_URL, RAW_FILE)
        process_and_filter(RAW_FILE, OUTPUT_FILE)
    except Exception as e:
        print(f"[FATAL] Ingestion failed: {e}")
