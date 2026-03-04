"""
File Name: Hybrid Search Engine core
Description: Implements a dual-pipeline retrieval system using BM25 for sparse
             lexical matching and Exact k-NN (Sentence Transformers) for dense
             semantic matching, fused via Reciprocal Rank Fusion (RRF).
"""

import os
import time
import pandas as pd
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Suppress verbose TF INFO logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


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
        self.df["title"] = self.df["title"].fillna("").astype(str)
        self.df["artist"] = self.df["artist"].fillna("").astype(str)
        self.df["medium"] = self.df["medium"].fillna("").astype(str)

        # Combine Title and Artist for the Exact-Match Sparse Index
        self.sparse_corpus = (self.df["title"] + " " + self.df["artist"]).tolist()

        # Use Medium/Description for the Semantic Dense Index
        self.dense_corpus = self.df["medium"].tolist()

        self.bm25 = None
        self.dense_model = None
        self.document_embeddings = None

        self._build_indexes()

    def _build_indexes(self):
        """
        Executes the offline indexing pipelines (Sparse and Dense).
        """
        print("[INFO] Building Sparse Index (BM25)...")
        # Tokenization: lowercasing and splitting by spaces
        tokenized_corpus = [doc.lower().split(" ") for doc in self.sparse_corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

        print("[INFO] Building Dense Vector Index (BERT)...")
        # Using a lightweight, highly efficient pre-trained transformer
        self.dense_model = SentenceTransformer("all-MiniLM-L6-v2")
        # Generate dense embeddings for all descriptions
        self.document_embeddings = self.dense_model.encode(
            self.dense_corpus, convert_to_numpy=True
        )
        print("[SUCCESS] All indexes built successfully.")

    def search_sparse(self, query, top_k=100):
        """
        Online Phase: Executes BM25 keyword matching.
        """
        tokenized_query = query.lower().split(" ")
        scores = self.bm25.get_scores(tokenized_query)
        # Get indices of top ranked documents
        top_indices = np.argsort(scores)[::-1][:top_k]
        return {idx: rank for rank, idx in enumerate(top_indices)}

    def search_dense(self, query, top_k=100):
        """
        Online Phase: Executes Exact k-NN cosine similarity on dense vectors.
        """
        query_embedding = self.dense_model.encode([query], convert_to_numpy=True)

        # Compute exact cosine similarity via dot product (vectors are normalized)
        # This satisfies the Exact k-NN requirement justified in the report
        scores = np.dot(self.document_embeddings, query_embedding.T).flatten()

        top_indices = np.argsort(scores)[::-1][:top_k]
        return {idx: rank for rank, idx in enumerate(top_indices)}

    def hybrid_search(self, query, top_k=10, rrf_k=60):
        """
        Online Phase: Orchestrates dual-retrieval and fuses results using RRF.
        Records and outputs execution latency.
        """
        start_time = time.perf_counter()

        print(f"\n[QUERY] '{query}'")

        sparse_ranks = self.search_sparse(query)
        dense_ranks = self.search_dense(query)

        # Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        for idx in range(len(self.df)):
            score = 0.0
            if idx in sparse_ranks:
                score += 1.0 / (rrf_k + sparse_ranks[idx])
            if idx in dense_ranks:
                score += 1.0 / (rrf_k + dense_ranks[idx])
            if score > 0:
                rrf_scores[idx] = score

        # Sort by fused score
        ranked_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

        # Format the output
        results = []
        for rank, idx in enumerate(ranked_indices):
            doc = self.df.iloc[idx]
            results.append(
                {
                    "id": doc["id"],
                    "Rank": rank + 1,
                    "Title": doc["title"],
                    "Artist": doc["artist"],
                    "Description": doc["medium"],
                    "Score": round(rrf_scores[idx], 4),
                }
            )

        latency = time.perf_counter() - start_time
        print(f"[TIMING] Retrieval and fusion completed in {latency:.4f} seconds.")

        return results


# ==========================================
# Interactive Command Line Interface
# ==========================================
def run_cli():
    """Runs the interactive Command Line Interface for the search engine."""
    print("\n" + "=" * 50)
    print("ART GALLERY SEARCH ENGINE INITIALIZING...")
    print("=" * 50)

    # Initialize the engine
    engine = ArtGallerySearchEngine("art_gallery_data.csv")
    print("\n[READY] Type your query below. Type 'exit' or 'quit' to stop.")

    while True:
        user_query = input("\nEnter search query: ").strip()

        if user_query.lower() in ["exit", "quit"]:
            print("Shutting down search engine.")
            break

        if not user_query:
            continue

        # Execute search and capture end-to-end interface latency
        cli_start_time = time.perf_counter()
        results = engine.hybrid_search(user_query, top_k=5)
        cli_latency = time.perf_counter() - cli_start_time

        # Display results utilizing standard search engine formatting
        print(f"\n--- Top {len(results)} Results ({cli_latency:.4f} seconds) ---")
        for res in results:
            desc = (
                res["Description"][:75] + "..."
                if len(res["Description"]) > 75
                else res["Description"]
            )
            print(f"{res['Rank']}. {res['Title']} | Artist: {res['Artist']}")
            print(f"   Score: {res['Score']} | Medium: {desc}\n")

if __name__ == "__main__":
    run_cli()
