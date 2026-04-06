"""
File Name: hybrid_search.py
Description: Retrieval engine using BM25 and Sentence Transformers with Standard RRF.
             Handles data-patching for URLs and formatting for display.
"""

import os
import time
from typing import Optional
import pandas as pd
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Cache files to skip re-computation on startup
VECTOR_CACHE = "embeddings.npy"


class ArtGallerySearchEngine:
    """
    Hybrid Search Engine utilizing BM25 (Lexical) and BERT (Semantic) retrieval.
    Optimized for NVIDIA GPUs and Multi-core Ryzen CPUs.
    """

    def __init__(self, data_path: str):
        """
        Initializes the document store and prepares the hardware device.
        """
        print("[INFO] Loading Document Store...")
        self.df = pd.read_csv(data_path)

        # Standardize data and apply 'Title Boosting'
        for col in ["title", "artist", "medium", "thumbnailurl"]:
            self.df[col] = self.df[col].fillna("").astype(str)

        # We double the title to ensure lexical matches on title rank higher than matches in the medium/description.
        self.combined_corpus = (
            (self.df["title"] + " ") * 2
            + "by "
            + self.df["artist"]
            + ". Medium: "
            + self.df["medium"]
        ).tolist()

        # Hardware Detection: Prioritize CUDA for the 5070 Ti
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[HARDWARE] Primary Device: {self.device.upper()}")
        if self.device == "cuda":
            print(f"[GPU] Detected: {torch.cuda.get_device_name(0)}")

        self.bm25: Optional[BM25Okapi] = None
        self.dense_model: Optional[SentenceTransformer] = None
        self.document_embeddings: Optional[np.ndarray] = None
        self._build_indexes()

    def _build_indexes(self):
        """
        Constructs the search indexes. Uses GPU acceleration if available,
        otherwise utilizes a Multi-Process pool on the CPU.
        """
        # 1. Sparse Index (BM25)
        print("[INFO] Building Sparse Index (BM25)...")
        tokenized_corpus = [doc.lower().split(" ") for doc in self.combined_corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 2. Dense Index (Sentence Transformers)
        print("[INFO] Initializing Transformer Model...")
        self.dense_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

        if os.path.exists(VECTOR_CACHE):
            print(f"[CACHE] Loading pre-computed embeddings from {VECTOR_CACHE}...")
            self.document_embeddings = np.load(VECTOR_CACHE)
        else:
            print("[INFO] No cache found. Starting high-performance encoding...")
            start_time = time.perf_counter()

            if self.device == "cuda":
                # GPU PATH: High-throughput batch processing for RTX 5070 Ti
                self.document_embeddings = self.dense_model.encode(
                    self.combined_corpus,
                    batch_size=128,  # Large batches for 5070 Ti VRAM
                    show_progress_bar=True,
                    convert_to_numpy=True,
                )
            else:
                # CPU PATH: Multi-process pool for Ryzen 9700X
                print("[CPU] Utilizing Multi-Process Pool for parallel encoding...")
                pool = self.dense_model.start_multi_process_pool()
                self.document_embeddings = self.dense_model.encode_multi_process(
                    self.combined_corpus, pool
                )
                self.dense_model.stop_multi_process_pool(pool)

            np.save(VECTOR_CACHE, self.document_embeddings)
            print(
                f"[SUCCESS] Encoding complete in {time.perf_counter() - start_time:.2f}s"
            )

    def search_sparse(self, query: str, top_k: int = 100):
        """Performs lexical search using BM25."""
        assert self.bm25 is not None
        tokenized_query = query.lower().split(" ")
        scores = self.bm25.get_scores(tokenized_query)
        return {idx: r for r, idx in enumerate(np.argsort(scores)[::-1][:top_k])}

    def search_dense(self, query: str, top_k: int = 100):
        """Performs semantic vector search, utilizing the GPU for query encoding."""
        assert self.dense_model is not None
        assert self.document_embeddings is not None
        query_vec = self.dense_model.encode(
            [query], convert_to_numpy=True, device=self.device
        )
        scores = np.dot(self.document_embeddings, query_vec.T).flatten()
        return {idx: r for r, idx in enumerate(np.argsort(scores)[::-1][:top_k])}

    def hybrid_search(self, query: str, top_k: int = 10, rrf_k: int = 60):
        """
        Fuses Sparse and Dense results using Reciprocal Rank Fusion (RRF).
        Applies URL patching and formatting for output.
        """
        start_time = time.perf_counter()

        sparse_ranks = self.search_sparse(query)
        dense_ranks = self.search_dense(query)

        rrf_scores = {}
        for idx in range(len(self.df)):
            score = 0.0
            if idx in sparse_ranks:
                score += 1.0 / (rrf_k + sparse_ranks[idx])
            if idx in dense_ranks:
                score += 1.0 / (rrf_k + dense_ranks[idx])
            if score > 0:
                rrf_scores[idx] = score

        ranked_indices = sorted(
            rrf_scores, key=lambda idx: rrf_scores[idx], reverse=True
        )[:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices):
            doc = self.df.iloc[idx]

            # URL Patching: Convert dead links to live media CDN
            raw_url = doc["thumbnailurl"]
            fixed_url = (
                raw_url.replace("http://www.tate.org.uk", "https://media.tate.org.uk")
                if "tate" in raw_url
                else raw_url
            )

            results.append(
                {
                    "Rank": rank + 1,
                    "Title": doc["title"].title(),
                    "Artist": doc["artist"].title(),
                    "Description": doc["medium"],
                    "Thumbnail": fixed_url,
                    "Score": round(rrf_scores[idx], 4),
                }
            )

        print(f"[TIMING] Total Retrieval: {time.perf_counter() - start_time:.4f}s")
        return results

