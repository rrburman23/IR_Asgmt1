"""
File Name: ingest_data.py
Description: Automates the acquisition of the Tate Gallery metadata,
             standardizes/normalizes text, and filters the corpus.
             This version includes advanced NLP processing for sparse retrieval
             and text chunking for dense retrieval, as per the design spec.
"""

import os
import re
import json

import pandas as pd
import requests
import nltk
from unidecode import unidecode
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

DATA_URL = "https://github.com/tategallery/collection/raw/master/artwork_data.csv"
RAW_FILE = "raw_tate_data.csv"
OUTPUT_FILE = "art_gallery_data.csv"


def setup_nltk():
    """Downloads necessary NLTK data models if not already present."""
    try:
        nltk.data.find("tokenizers/punkt")
    except nltk.downloader.DownloadError:
        print("[INFO] NLTK 'punkt' tokenizer not found. Downloading...")
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/stopwords")
    except nltk.downloader.DownloadError:
        print("[INFO] NLTK 'stopwords' not found. Downloading...")
        nltk.download("stopwords")
    print("[INFO] NLTK resources are ready.")


# Initialize NLP tools once
setup_nltk()
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


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


def normalize_text_for_display(text):
    """Light text normalization for display purposes."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def process_text_for_sparse(text):
    """
    Full NLP pipeline for sparse retrieval (BM25F).
    - Removes accents
    - Lowercases
    - Tokenizes
    - Removes stop words
    - Applies Porter Stemming
    """
    if not isinstance(text, str):
        return ""
    text = unidecode(text)  # 1. Accent removal
    text = text.lower()  # 2. Lowercasing
    tokens = word_tokenize(text)  # 3. Tokenization

    # 4. Stop-word removal, non-alpha filtering, and stemming
    processed_tokens = [
        stemmer.stem(word)
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]
    return " ".join(processed_tokens)


def create_text_chunks(text, chunk_size=100, overlap=20):
    """
    Splits text into overlapping chunks for dense vector embedding.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    # The step is chunk_size - overlap to create the overlap
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i : i + chunk_size]
        if not chunk:
            continue
        chunks.append(" ".join(chunk))
        # Ensure the loop terminates correctly if the last chunk is full
        if i + chunk_size >= len(words):
            break
    return chunks


def process_and_filter(input_file, output_file, sample_size=2000):
    """
    Cleans metadata, engineers search features, and exports a balanced subset.
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
        df[col] = df[col].astype(str).apply(normalize_text_for_display)

    # --------------------------------------------------
    # Robust sampling
    # --------------------------------------------------
    actual_sample_size = min(len(df), sample_size)

    if actual_sample_size < sample_size:
        print(
            f"[WARNING] Dataset size ({len(df)}) is smaller than "
            f"requested sample ({sample_size})."
        )

    # Select final columns and export the sampled dataset
    final_columns = [
        "id",
        "artist",
        "title",
        "medium",
        "year",
        "processed_sparse_text",
        "description_chunks",
    ]
    df_final = df.sample(n=actual_sample_size, random_state=42)[final_columns]

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

    except (
        ConnectionError,
        FileNotFoundError,
        KeyError,
        pd.errors.EmptyDataError,
    ) as e:
        print(f"[ERROR] Ingestion pipeline failed: {e}")
