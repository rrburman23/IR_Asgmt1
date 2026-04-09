"""
File Name: ingest_data.py
Description: Download, clean, and preprocess the Tate Gallery dataset.
- Preserves raw text for GUI display (capitalization, punctuation).
- Creates separate normalized columns for BM25 search.
- Patches dead Tate URLs to the new Media CDN.
"""

import os
import re
import pandas as pd
import requests

DATA_URL = "https://github.com/tategallery/collection/raw/master/artwork_data.csv"
RAW_FILE = "raw_tate_data.csv"
OUTPUT_FILE = "art_gallery_data.csv"


def ensure_data_exists():
    """Download and process data unless already present."""
    if os.path.exists(OUTPUT_FILE):
        print(f"[INFO] {OUTPUT_FILE} already exists. Skipping ingestion.")
        return
    print("[INFO] Dataset not found. Starting ingestion pipeline...")
    try:
        download_dataset(DATA_URL, RAW_FILE)
        process_and_filter(RAW_FILE, OUTPUT_FILE)
        print("[SUCCESS] Dataset ready.")
    except Exception as e:
        print(f"[ERROR] Ingestion pipeline failed: {e}")
        raise


def download_dataset(url, filename, timeout=30):
    """Download the CSV from the Tate GitHub repo."""
    if os.path.exists(filename):
        return
    print(f"[INFO] Downloading dataset from {url}...")
    resp = requests.get(url, timeout=timeout, stream=True)
    resp.raise_for_status()
    with open(filename, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def normalize_text(text):
    """Aggressive normalization exclusively for hidden search fields."""
    if not isinstance(text, str):
        return ""
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def process_and_filter(input_file, output_file, sample_size=70000):
    """Clean Tate data, fix URLs, and preserve raw fields for the GUI."""
    print("[INFO] Processing and filtering metadata...")
    df = pd.read_csv(input_file, low_memory=False)
    df.columns = df.columns.str.lower().str.strip()

    # Keep extended metadata for the GUI
    keep = [
        "id",
        "title",
        "artist",
        "medium",
        "year",
        "datetext",
        "dimensions",
        "creditline",
        "thumbnailurl",
    ]
    for col in keep:
        if col not in df.columns:
            df[col] = ""

    df = df[keep].copy()
    df.drop_duplicates(subset=["id"], inplace=True)

    # 1. Fill NA but DO NOT normalize these display columns!
    for col in keep:
        df[col] = df[col].fillna("").astype(str)

    # 2. Dead Tate URLs point to the new CDN
    df["thumbnailurl"] = df["thumbnailurl"].str.replace(
        "http://www.tate.org.uk", "https://media.tate.org.uk"
    )

    # 3. Create HIDDEN normalized columns specifically for BM25/Dense Search
    df["search_title"] = df["title"].apply(normalize_text)
    df["search_artist"] = df["artist"].apply(normalize_text)
    df["search_medium"] = df["medium"].apply(normalize_text)

    # 4. Create Semantic Blob for BERT
    df["semantic_blob"] = (
        "Title: "
        + df["title"]
        + ". Artist: "
        + df["artist"]
        + ". Medium: "
        + df["medium"]
        + ". Year: "
        + df["year"]
        + "."
    )

    df_final = df.sample(n=min(len(df), sample_size), random_state=42).copy()
    df_final.to_csv(output_file, index=False)
    print(f"[SUCCESS] Final search corpus: {len(df_final)} records.")


if __name__ == "__main__":
    ensure_data_exists()
