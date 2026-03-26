"""
File Name: hybrid_search.py
Description: Optimized retrieval system using BM25 and Sentence Transformers.
             Implements disk-based caching for AI embeddings to skip
             heavy re-computation on every startup.
"""

import os
import time
import pandas as pd
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Path for the cached AI vectors (saves ~20 seconds on startup)
CACHE_FILE = "embeddings.npy"


class ArtGallerySearchEngine:
    """
    Core engine for the Art Gallery Hybrid Search.
    Manages data ingestion, offline index construction (sparse and dense),
    and online query execution utilizing Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, data_path):
        """
        Initializes the search engine, loads the document store,
        and triggers the offline indexing phase.
        """
        print("[INFO] Loading Document Store...")
        self.df = pd.read_csv(data_path)

        # Ensure text columns are strings to prevent tokenization errors
        for col in ["title", "artist", "medium", "thumbnailurl"]:
            self.df[col] = self.df[col].fillna("").astype(str)

        # Create a rich, combined text field for both indexes
        self.combined_corpus = (
            self.df["title"]
            + " by "
            + self.df["artist"]
            + ". Medium: "
            + self.df["medium"]
        ).tolist()

        self.bm25 = None
        self.dense_model = None
        self.document_embeddings = None

        self._build_indexes()

    def _build_indexes(self):
        """
        Executes the offline indexing pipelines (Sparse and Dense).
        Uses a local .npy file to cache dense vectors.
        """
        # 1. Building the Sparse Index (Fast Lexical Search)
        print("[INFO] Building Sparse Index (BM25)...")
        tokenized_corpus = [doc.lower().split(" ") for doc in self.combined_corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 2. Building/Loading the Dense Index (Semantic Search)
        print("[INFO] Initializing Semantic Model...")
        self.dense_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Performance Check: Only compute embeddings if the cache doesn't exist
        if os.path.exists(CACHE_FILE):
            print(f"[CACHE] Loading pre-computed embeddings from {CACHE_FILE}...")
            self.document_embeddings = np.load(CACHE_FILE)
        else:
            print("[INFO] No cache found. Computing embeddings (Initial run)...")
            # Generate dense embeddings for all descriptions
            self.document_embeddings = self.dense_model.encode(
                self.combined_corpus, convert_to_numpy=True, show_progress_bar=True
            )
            # Save to disk to make the NEXT launch instant
            np.save(CACHE_FILE, self.document_embeddings)
            print(f"[SUCCESS] Embeddings cached to {CACHE_FILE}")

        print("[SUCCESS] All indexes built successfully.")

    def search_sparse(self, query, top_k=100):
        """Online Phase: Executes BM25 keyword matching."""
        tokenized_query = query.lower().split(" ")
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return {idx: rank for rank, idx in enumerate(top_indices)}

    def search_dense(self, query, top_k=100):
        """Online Phase: Executes Exact k-NN cosine similarity."""
        query_embedding = self.dense_model.encode([query], convert_to_numpy=True)
        # Compute exact cosine similarity via dot product
        scores = np.dot(self.document_embeddings, query_embedding.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return {idx: rank for rank, idx in enumerate(top_indices)}

    def hybrid_search(self, query, top_k=10, rrf_k=60):
        """
        Online Phase: Orchestrates dual-retrieval and fuses results via RRF.
        """
        start_time = time.perf_counter()

        sparse_ranks = self.search_sparse(query)
        dense_ranks = self.search_dense(query)

        # Reciprocal Rank Fusion (RRF) algorithm
        rrf_scores = {}
        for idx in range(len(self.df)):
            score = 0.0
            if idx in sparse_ranks:
                score += 1.0 / (rrf_k + sparse_ranks[idx])
            if idx in dense_ranks:
                score += 1.0 / (rrf_k + dense_ranks[idx])
            if score > 0:
                rrf_scores[idx] = score

        ranked_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

        # Format output for the UI
        results = []
        for rank, idx in enumerate(ranked_indices):
            doc = self.df.iloc[idx]
            results.append(
                {
                    "Rank": rank + 1,
                    "Title": doc["title"],
                    "Artist": doc["artist"],
                    "Description": doc["medium"],
                    "Thumbnail": doc["thumbnailurl"],
                    "Score": round(rrf_scores[idx], 4),
                }
            )

        print(f"[TIMING] Search completed in {time.perf_counter() - start_time:.4f}s")
        return results
