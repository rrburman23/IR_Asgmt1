"""
File Name: ingest_data.py
Description: Download, clean, and preprocess the Tate Gallery dataset.
- Preserves raw text for GUI display (capitalization, punctuation).
- Creates separate normalized columns for BM25 search.
- Creates semantic_blob for dense retrieval.
- Adds description_chunks (JSON list) for chunked dense retrieval (max pooling).
- Patches dead Tate URLs to the new Media CDN.
"""

import os
import re
import json
import unicodedata
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
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def chunk_text_by_words(
    text: str, chunk_words: int = 40, overlap_words: int = 10
) -> list[str]:
    """
    Chunk text into overlapping word windows.
    Works well for short-ish 'semantic_blob' strings too.
    """
    if not isinstance(text, str):
        return []
    words = text.split()
    if not words:
        return []

    chunk_words = max(5, int(chunk_words))
    overlap_words = max(0, int(overlap_words))
    step = max(1, chunk_words - overlap_words)

    chunks: list[str] = []
    for start in range(0, len(words), step):
        end = start + chunk_words
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break

    # Ensure at least one chunk exists
    return chunks if chunks else [text]


def process_and_filter(input_file, output_file, sample_size=70000):
    """Clean Tate data, fix URLs, and preserve raw fields for the GUI."""
    print("[INFO] Processing and filtering metadata...")
    df = pd.read_csv(input_file, low_memory=False)
    df.columns = df.columns.str.lower().str.strip()

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

    for col in keep:
        df[col] = df[col].fillna("").astype(str)

    # Patch Tate URLs to CDN
    df["thumbnailurl"] = df["thumbnailurl"].str.replace(
        "http://www.tate.org.uk", "https://media.tate.org.uk"
    )

    # Hidden normalized columns for BM25
    df["search_title"] = df["title"].apply(normalize_text)
    df["search_artist"] = df["artist"].apply(normalize_text)
    df["search_medium"] = df["medium"].apply(normalize_text)

    # Semantic Blob (dense retrieval source)
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

    # NEW: description_chunks column (JSON list per row)
    # Chunk the semantic_blob; this supports chunked dense retrieval with max pooling.
    chunks = []
    for blob in df["semantic_blob"].tolist():
        chunk_list = chunk_text_by_words(blob, chunk_words=40, overlap_words=10)
        chunks.append(json.dumps(chunk_list, ensure_ascii=False))
    df["description_chunks"] = chunks

    df_final = df.sample(n=min(len(df), sample_size), random_state=42).copy()
    df_final.to_csv(output_file, index=False)
    print(f"[SUCCESS] Final search corpus: {len(df_final)} records.")
    print("[SUCCESS] Added column: description_chunks (JSON word-window chunks).")


if __name__ == "__main__":
    ensure_data_exists()
