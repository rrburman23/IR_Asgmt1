"""
File Name: hybrid_search.py
Description: The core AI retrieval engine. Combines Lexical (BM25) and
             Semantic (Sentence Transformers) search.
             Utilizes CUDA for NVIDIA GPUs and
             Multi-processing for Ryzen CPUs.
"""

import os
import time
from typing import Optional
import pandas as pd
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Constants for caching to prevent re-running 1.2s encoding on every boot
VECTOR_CACHE = "embeddings.npy"


class ArtGallerySearchEngine:
    """
    Handles the indexing and retrieval of artwork metadata.
    """

    def __init__(self, data_path: str):
        """
        Loads the CSV and prepares the hardware for AI operations.
        """
        print("[INFO] Loading Document Store...")
        self.df = pd.read_csv(data_path)

        # Standardize data: Fill empty values and ensure everything is a string
        for col in ["title", "artist", "medium", "thumbnailurl"]:
            self.df[col] = self.df[col].fillna("").astype(str)

        # TITLE BOOSTING:
        # We repeat the title twice in the search string. This ensures that
        # a keyword match in the Title is mathematically more important than
        # a keyword match in the Medium/Description.
        self.combined_corpus = (
            (self.df["title"] + " ") * 2
            + "by "
            + self.df["artist"]
            + ". Medium: "
            + self.df["medium"]
        ).tolist()

        # Hardware Detection: Checks for NVIDIA GPU (CUDA)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[HARDWARE] Primary Device: {self.device.upper()}")
        if self.device == "cuda":
            print(f"[GPU] Detected: {torch.cuda.get_device_name(0)}")

        # Placeholders for search components
        self.bm25: Optional[BM25Okapi] = None
        self.dense_model: Optional[SentenceTransformer] = None
        self.document_embeddings: Optional[np.ndarray] = None

        # Trigger the indexing process
        self._build_indexes()

    def _build_indexes(self):
        """
        Constructs the indexes. BM25 is built on the CPU,
        while BERT embeddings are computed on the GPU.
        """
        # 1. Sparse Index (BM25): Lexical/Keyword-based matching
        print("[INFO] Building Sparse Index (BM25)...")
        tokenized_corpus = [doc.lower().split(" ") for doc in self.combined_corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 2. Dense Index (Sentence Transformers): Semantic/Meaning-based matching
        print("[INFO] Initializing Transformer Model...")
        self.dense_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

        # If we have a cache on disk, load it instantly
        if os.path.exists(VECTOR_CACHE):
            print(f"[CACHE] Loading pre-computed embeddings from {VECTOR_CACHE}...")
            self.document_embeddings = np.load(VECTOR_CACHE)
        else:
            # If no cache, use the GPU (or CPU) to compute embeddings and save them for next time
            print("[INFO] No cache found. Starting high-performance encoding...")
            start_time = time.perf_counter()

            if self.device == "cuda":
                # High batch size (128) is used to saturate the GPU's memory for speed
                self.document_embeddings = self.dense_model.encode(
                    self.combined_corpus,
                    batch_size=128,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                )
            else:
                # Fallback for CPU-only systems
                print("[CPU] Utilizing Multi-Process Pool for parallel encoding...")
                pool = self.dense_model.start_multi_process_pool()
                self.document_embeddings = self.dense_model.encode_multi_process(
                    self.combined_corpus, pool
                )
                self.dense_model.stop_multi_process_pool(pool)

            # Save the result so we never have to compute this again
            np.save(VECTOR_CACHE, self.document_embeddings)
            print(
                f"[SUCCESS] Encoding complete in {time.perf_counter() - start_time:.2f}s"
            )

    def search_sparse(self, query: str, top_k: int = 100):
        """Calculates keyword scores for a query."""
        assert self.bm25 is not None
        tokenized_query = query.lower().split(" ")
        scores = self.bm25.get_scores(tokenized_query)
        # Return the indices of the top 100 lexical matches
        return {idx: r for r, idx in enumerate(np.argsort(scores)[::-1][:top_k])}

    def search_dense(self, query: str, top_k: int = 100):
        """Calculates semantic similarity using dot product on the GPU."""
        assert self.dense_model is not None
        assert self.document_embeddings is not None
        # Convert the user query into a vector
        query_vec = self.dense_model.encode(
            [query], convert_to_numpy=True, device=self.device
        )
        # Perform matrix multiplication to find similarity scores
        scores = np.dot(self.document_embeddings, query_vec.T).flatten()
        # Return the indices of the top 100 semantic matches
        return {idx: r for r, idx in enumerate(np.argsort(scores)[::-1][:top_k])}

    def hybrid_search(self, query: str, top_k: int = 10, rrf_k: int = 60):
        """
        Uses Reciprocal Rank Fusion (RRF) to merge keyword and semantic results.
        """
        start_time = time.perf_counter()

        sparse_ranks = self.search_sparse(query)
        dense_ranks = self.search_dense(query)

        # RRF Formula: 1 / (k + rank)
        rrf_scores = {}
        for idx in range(len(self.df)):
            score = 0.0
            if idx in sparse_ranks:
                score += 1.0 / (rrf_k + sparse_ranks[idx])
            if idx in dense_ranks:
                score += 1.0 / (rrf_k + dense_ranks[idx])
            if score > 0:
                rrf_scores[idx] = score

        # Sort all artworks by their new fused score
        ranked_indices = sorted(
            rrf_scores, key=lambda idx: rrf_scores[idx], reverse=True
        )[:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices):
            doc = self.df.iloc[idx]

            # URL PATCHING:
            # Tate's raw data has dead links. We swap 'www.tate.org.uk'
            # for 'media.tate.org.uk' to hit the live CDN.
            raw_url = doc["thumbnailurl"]
            fixed_url = (
                raw_url.replace("http://www.tate.org.uk", "https://media.tate.org.uk")
                if "tate" in raw_url
                else raw_url
            )

            results.append(
                {
                    "Rank": rank + 1,
                    "id": doc["id"],
                    "Title": doc["title"].title(),  # Auto-format to proper Title Case
                    "Artist": doc["artist"].title(),
                    "Description": doc["medium"],
                    "Thumbnail": fixed_url,
                    "Score": round(rrf_scores[idx], 4),
                }
            )

        print(f"[TIMING] Total Retrieval: {time.perf_counter() - start_time:.4f}s")
        return results
