"""
File Name: hybrid_search.py
Description: Retrieval engine using BM25 and Sentence Transformers with Standard RRF.
             Handles data-patching for URLs and formatting for display.
"""

import os
import time
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Path for the cached AI vectors
CACHE_FILE = "embeddings.npy"

class ArtGallerySearchEngine:
    """
    Core search engine that combines BM25 lexical search with
    Sentence Transformer semantic search using Reciprocal Rank Fusion.
    """

    def __init__(self, data_path):
        """
        Initializes the engine, cleans the dataframe, and builds indexes.
        """
        print("[INFO] Loading Document Store...")
        self.df = pd.read_csv(data_path)

        for col in ["title", "artist", "medium", "thumbnailurl"]:
            self.df[col] = self.df[col].fillna("").astype(str)

        # Title Boosting: Mentioning the title twice helps lexical relevance
        self.combined_corpus = (
            (self.df["title"] + " ") * 2
            + "by "
            + self.df["artist"]
            + ". "
            + "Medium: "
            + self.df["medium"]
        ).tolist()

        self.bm25 = None
        self.dense_model = None
        self.document_embeddings = None
        self._build_indexes()

    def _build_indexes(self):
        """Builds Sparse/Dense indexes or loads them from the disk cache."""
        print("[INFO] Building Sparse Index (BM25)...")
        tokenized_corpus = [doc.lower().split(" ") for doc in self.combined_corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

        print("[INFO] Initializing Semantic Model...")
        self.dense_model = SentenceTransformer("all-MiniLM-L6-v2")

        if os.path.exists(CACHE_FILE):
            print(f"[CACHE] Loading pre-computed embeddings from {CACHE_FILE}...")
            self.document_embeddings = np.load(CACHE_FILE)
        else:
            print("[INFO] Computing embeddings (Initial run)...")
            self.document_embeddings = self.dense_model.encode(
                self.combined_corpus, convert_to_numpy=True, show_progress_bar=True
            )
            np.save(CACHE_FILE, self.document_embeddings)

    def search_sparse(self, query, top_k=100):
        """
        Executes BM25 Lexical search and returns ranked indices.
        """
        tokenized_query = query.lower().split(" ")
        scores = self.bm25.get_scores(tokenized_query)
        return {idx: r for r, idx in enumerate(np.argsort(scores)[::-1][:top_k])}

    def search_dense(self, query, top_k=100):
        """
        Executes Semantic vector search using dot product similarity.
        """
        query_embedding = self.dense_model.encode([query], convert_to_numpy=True)
        scores = np.dot(self.document_embeddings, query_embedding.T).flatten()
        return {idx: r for r, idx in enumerate(np.argsort(scores)[::-1][:top_k])}

    def hybrid_search(self, query, top_k=10, rrf_k=60):
        """
        Fuses results using standard RRF logic.
        Patches URLs to live CDN and formats text for display.
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

        ranked_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices):
            doc = self.df.iloc[idx]

            # Patch dead URLs to live CDN and apply Title Case
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

        print(f"[TIMING] Search completed in: {time.perf_counter() - start_time:.4f}s")
        return results
