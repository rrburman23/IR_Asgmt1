"""
File Name: hybrid_search.py
Description: Dual-path Hybrid Search Engine v3.2.

- Sparse: Fielded BM25Okapi on Title and Artist.
- Dense: multi-qa-MiniLM-L6-cos-v1 on Description and Medium.
- Fusion: Reciprocal Rank Fusion (RRF) with k=60.
"""

from __future__ import annotations
import os
import re
import time
import functools
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

try:
    import Levenshtein
except ImportError:
    Levenshtein = None

VECTOR_CACHE = "embeddings.npy"
DENSE_MODEL_ID = "multi-qa-MiniLM-L6-cos-v1"


def normalize_query(text: str) -> str:
    """Standardizes user queries to match indexed tokens."""
    if not isinstance(text, str):
        return ""
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


class ArtGallerySearchEngine:
    """Architectural Implementation of a Dual-Index Hybrid Search Engine."""

    def __init__(self, data_path: str):
        """Initializes store and hardware-specific indexes."""
        print("[INFO] Loading Document Store...")
        self.df = pd.read_csv(data_path)

        for col in self.df.columns:
            self.df[col] = self.df[col].fillna("").astype(str)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vocabulary: set[str] = set()
        self.bm25: Optional[BM25Okapi] = None
        self.dense_model: Optional[SentenceTransformer] = None
        self.document_embeddings: Optional[np.ndarray] = None

        self._build_indexes()

    def _build_indexes(self):
        """Constructs Sparse (BM25) and Dense (BERT) indexes."""
        # 1. Sparse Index
        print("[INFO] Building Fielded Sparse Index (Title/Artist)...")
        bm25_corpus = [
            (str(row["search_title"]) + " ") * 5 + (str(row["search_artist"]) + " ") * 3
            for _, row in self.df.iterrows()
        ]
        tokenized_corpus = [doc.split() for doc in bm25_corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 2. Dense Index
        print(f"[INFO] Initializing Transformer on {self.device.upper()}...")
        self.dense_model = SentenceTransformer(DENSE_MODEL_ID, device=self.device)

        if os.path.exists(VECTOR_CACHE):
            print(f"[CACHE] Loading dense embeddings from {VECTOR_CACHE}...")
            self.document_embeddings = np.load(VECTOR_CACHE)
        else:
            print("[INFO] Computing Semantic Embeddings...")
            semantic_corpus = self.df["semantic_blob"].tolist()
            self.document_embeddings = self.dense_model.encode(
                semantic_corpus,
                batch_size=128,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            if self.document_embeddings is not None:
                np.save(VECTOR_CACHE, self.document_embeddings)

        # 3. Spelling Vocabulary
        print("[INFO] Building Spelling Vocabulary...")
        target_cols = ["artist", "title", "medium"]
        for col in target_cols:
            texts = self.df[col].tolist()
            for text in texts:
                if isinstance(text, str) and text:
                    # FIX: Use raw strings for regex to avoid escape warnings
                    found = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
                    self.vocabulary.update(found)

        print("[SUCCESS] All search indexes are online.")

    def suggest_correction(self, query: str) -> Optional[str]:
        """Suggests a query correction using Levenshtein distance."""
        lev = Levenshtein
        if not self.vocabulary or lev is None:
            return None

        suggested, changed = [], False
        for word in query.split():
            clean = re.sub(r"[^a-zA-Z]", "", word.lower())
            if not clean or len(clean) < 4 or clean in self.vocabulary:
                suggested.append(word)
                continue

            dist_func = functools.partial(lev.distance, clean)
            closest = min(self.vocabulary, key=dist_func)

            if lev.distance(clean, closest) <= 1:
                suggested.append(closest)
                changed = True
            else:
                suggested.append(word)
        return " ".join(suggested) if changed else None

    def search_sparse(self, query: str, top_k: int = 100) -> Dict[int, int]:
        """Lexical search returning DocIndex:Rank."""
        if self.bm25 is None:
            return {}
        tokens = normalize_query(query).split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return {
            int(idx): rank for rank, idx in enumerate(top_indices) if scores[idx] > 0
        }

    def search_dense(self, query: str, top_k: int = 100) -> Dict[int, int]:
        """Semantic search returning DocIndex:Rank."""
        if self.dense_model is None or self.document_embeddings is None:
            return {}
        q_vec = self.dense_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )
        scores = np.dot(self.document_embeddings, q_vec.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return {
            int(idx): rank for rank, idx in enumerate(top_indices) if scores[idx] > 0.15
        }

    def hybrid_search(
        self, query: str, top_k: int = 10, k_rrf: int = 60
    ) -> List[Dict[str, Any]]:
        """Fuses dual-path results via RRF."""
        start_time = time.perf_counter()
        suggestion = self.suggest_correction(query)

        sparse_ranks = self.search_sparse(query, top_k=100)
        dense_ranks = self.search_dense(query, top_k=100)

        fused_scores: Dict[int, float] = {}
        for idx in set(sparse_ranks.keys()) | set(dense_ranks.keys()):
            s_rank = sparse_ranks.get(idx, 1000)
            d_rank = dense_ranks.get(idx, 1000)
            fused_scores[idx] = (1.0 / (k_rrf + s_rank)) + (1.0 / (k_rrf + d_rank))

        ranked_indices = sorted(
            fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True
        )[:top_k]

        results = []
        for idx in ranked_indices:
            doc = self.df.iloc[idx]
            reasons = []
            if idx in sparse_ranks and sparse_ranks[idx] < 20:
                reasons.append(f"Keyword (Rank {sparse_ranks[idx] + 1})")
            if idx in dense_ranks and dense_ranks[idx] < 20:
                reasons.append(f"AI Context (Rank {dense_ranks[idx] + 1})")

            raw_url = str(doc["thumbnailurl"])
            fixed_url = raw_url.replace(
                "http://www.tate.org.uk", "https://media.tate.org.uk"
            )

            results.append(
                {
                    "Rank": len(results) + 1,
                    "id": doc["id"],
                    "Title": str(doc["title"]).title(),
                    "Artist": str(doc["artist"]).title(),
                    "Medium": str(doc["medium"]),
                    "Description": f"A {doc['medium']} piece from the Tate collection.",
                    "Thumbnail": fixed_url,
                    "Score": round(fused_scores[idx], 4),
                    "Reasons": " + ".join(reasons) or "Hybrid Match",
                    "Suggestion": suggestion,
                }
            )

        print(f"[TIMING] Hybrid Retrieval: {time.perf_counter() - start_time:.4f}s")
        return results
