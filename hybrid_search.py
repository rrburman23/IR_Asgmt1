"""
File Name: hybrid_search.py
Description: Advanced Retrieval Engine v2.3.
             Improvements: query formatting, retrieval-optimized model,
             parameterized fusion, and strict input normalization.
"""

from __future__ import annotations
import os
import re
import time
import functools
from typing import Optional, Dict, Mapping, Any, List

import pandas as pd
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Optional import for spelling correction
try:
    import Levenshtein
except ImportError:
    Levenshtein = None

VECTOR_CACHE = "embeddings.npy"
DENSE_MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1" 


def process_text_for_sparse(text: str) -> str:
    """Normalize string for BM25 (lowercase, normalize hyphens, remove punctuation)."""
    if not isinstance(text, str):
        return ""
    text = text.lower().replace("-", " ")
    return re.sub(r"[^\w\s]", "", text)


def normalize_title(t: str) -> str:
    """Consistent normalization for title comparisons."""
    return str(t).strip().lower()


class ArtGallerySearchEngine:
    """
    Search Engine utilizing Fielded Sparse Retrieval
    and Chunked Semantic Dense Retrieval with Hardware Acceleration.
    Now with retrieval-optimized dense model and query alignment.
    """

    def __init__(self, data_path: str):
        print("[INFO] Loading Document Store...")
        self.df = pd.read_csv(data_path)
        # Strict field normalization
        for col in ["title", "artist", "medium", "thumbnailurl"]:
            self.df[col] = self.df[col].fillna("").astype(str).str.strip().str.lower()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vocabulary: set[str] = set()
        self.bm25: Optional[BM25Okapi] = None
        self.dense_model: Optional[SentenceTransformer] = None
        self.document_embeddings: Optional[np.ndarray] = None
        self._build_indexes()

    def _build_indexes(self):
        """Build BM25, dense model, and spelling index, all normalized."""
        print("[INFO] Building Sparse Index (Fielded BM25)...")
        weighted_corpus = [
            (row["title"] + " ") * 3 + (row["artist"] + " ") * 2 + row["medium"]
            for _, row in self.df.iterrows()
        ]
        tokenized_corpus = [
            process_text_for_sparse(doc).split() for doc in weighted_corpus
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)

        print(
            f"[INFO] Initializing Transformer ({DENSE_MODEL_NAME}) on {self.device.upper()}..."
        )
        self.dense_model = SentenceTransformer(DENSE_MODEL_NAME, device=self.device)

        if os.path.exists(VECTOR_CACHE):
            print(f"[CACHE] Loading embeddings from {VECTOR_CACHE}...")
            self.document_embeddings = np.load(VECTOR_CACHE)
        else:
            print("[INFO] Computing Semantic Embeddings...")
            semantic_corpus = [
                f"Title: {row['title']}. Artist: {row['artist']}. Medium: {row['medium']}."
                for _, row in self.df.iterrows()
            ]
            self.document_embeddings = self.dense_model.encode(
                semantic_corpus,
                batch_size=128,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            if self.document_embeddings is not None:
                np.save(VECTOR_CACHE, self.document_embeddings)

        print("[INFO] Building Spelling Vocabulary...")
        all_text_data = (
            self.df["artist"].tolist()
            + self.df["medium"].tolist()
            + self.df["title"].tolist()
        )
        for text in all_text_data:
            words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
            self.vocabulary.update(words)
        print("[SUCCESS] All indexes built successfully.")

    def suggest_correction(self, query: str) -> Optional[str]:
        lev = Levenshtein
        if not self.vocabulary or lev is None:
            return None
        suggested_query = []
        changed = False
        for word in query.split():
            clean_word = re.sub(r"[^a-zA-Z]", "", word.lower())
            if not clean_word or len(clean_word) < 3 or clean_word in self.vocabulary:
                suggested_query.append(word)
                continue
            dist_func = functools.partial(lev.distance, clean_word)
            closest_word = min(self.vocabulary, key=dist_func)
            if lev.distance(clean_word, closest_word) <= 2:
                suggested_query.append(closest_word)
                changed = True
            else:
                suggested_query.append(word)
        return " ".join(suggested_query) if changed else None

    def search_sparse(self, query: str, top_k: int = 100) -> Dict[int, float]:
        if self.bm25 is None:
            return {}
        qnorm = process_text_for_sparse(query)
        tokenized_query = qnorm.split()
        scores = self.bm25.get_scores(tokenized_query)
        scored_docs = {i: float(s) for i, s in enumerate(scores) if s > 0}
        sorted_docs = sorted(scored_docs.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_docs[:top_k])

    def search_dense(self, query: str, top_k: int = 100) -> Dict[int, float]:
        if self.dense_model is None or self.document_embeddings is None:
            return {}
        # Query formatting is key: structure like in corpus encoding
        qstr = f"Title: {query.strip().lower()}. Artist: . Medium: ."
        query_vec = self.dense_model.encode(
            [qstr],
            convert_to_numpy=True,
            device=self.device,
            normalize_embeddings=True,
        )
        scores = np.dot(self.document_embeddings, query_vec.T).flatten()
        scored_docs = {i: float(s) for i, s in enumerate(scores) if s > 0.1}
        sorted_docs = sorted(scored_docs.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_docs[:top_k])

    @staticmethod
    def _min_max_normalize(scores_dict: Mapping[int, Any]) -> Dict[int, float]:
        if not scores_dict:
            return {}
        vals = [float(v) for v in scores_dict.values()]
        min_v, max_v = min(vals), max(vals)
        if max_v == min_v:
            return {k: 1.0 for k in scores_dict}
        return {k: (float(v) - min_v) / (max_v - min_v) for k, v in scores_dict.items()}

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        fusion: str = "score",  # 'score' (weighted), or 'rrf'
        alpha: float = 0.6,  # Used only if fusion=='score'
        k_rrf: int = 60,  # Used only if fusion=='rrf'
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        start_time = time.perf_counter()
        suggestion = self.suggest_correction(query)
        sparse_scores = self.search_sparse(query, top_k=100)
        dense_scores = self.search_dense(query, top_k=100)

        if fusion == "score":
            # Weighted linear fusion of normalized scores
            norm_sparse = self._min_max_normalize(sparse_scores)
            norm_dense = self._min_max_normalize(dense_scores)
            fused_scores: Dict[int, float] = {}
            all_ids = set(norm_sparse) | set(norm_dense)
            for idx in all_ids:
                s_val = norm_sparse.get(idx, 0.0)
                d_val = norm_dense.get(idx, 0.0)
                fused_scores[idx] = (alpha * d_val) + ((1.0 - alpha) * s_val)
        else:
            # Reciprocal Rank Fusion (RRF)
            sparse_ranks = {
                doc_id: rank
                for rank, (doc_id, _) in enumerate(sparse_scores.items(), start=1)
            }
            dense_ranks = {
                doc_id: rank
                for rank, (doc_id, _) in enumerate(dense_scores.items(), start=1)
            }
            fused_scores: Dict[int, float] = {}
            all_ids = set(sparse_ranks.keys()) | set(dense_ranks.keys())
            for idx in all_ids:
                s_rank = sparse_ranks.get(idx, 1000)
                d_rank = dense_ranks.get(idx, 1000)
                fused_scores[idx] = (1.0 / (k_rrf + s_rank)) + (1.0 / (k_rrf + d_rank))

        ranked_indices = sorted(
            fused_scores, key=lambda x: fused_scores.get(x, 0.0), reverse=True
        )

        results: List[Dict[str, Any]] = []
        for idx in ranked_indices:
            doc = self.df.iloc[idx]
            if filters:
                if not all(
                    str(v).lower() in str(doc.get(k, "")).lower()
                    for k, v in filters.items()
                ):
                    continue
            # Transparency: top 20 (score/rank) for each branch
            reasons = []
            if idx in sparse_scores:
                reasons.append(f"Keyword (sparse={sparse_scores[idx]:.2f})")
            if idx in dense_scores:
                reasons.append(f"AI Context (dense={dense_scores[idx]:.2f})")
            raw_url = doc["thumbnailurl"]
            fixed_url = (
                raw_url.replace("http://www.tate.org.uk", "https://media.tate.org.uk")
                if "tate" in raw_url
                else raw_url
            )
            results.append(
                {
                    "Rank": len(results) + 1,
                    "id": doc["id"],
                    "Title": doc["title"].title(),
                    "Artist": doc["artist"].title(),
                    "Description": doc["medium"],
                    "Thumbnail": fixed_url,
                    "Score": round(float(fused_scores[idx]), 4),
                    "Reasons": " + ".join(reasons) or "Hybrid Match",
                    "Suggestion": suggestion,
                }
            )
            if len(results) == top_k:
                break
        print(f"[TIMING] Hybrid Retrieval: {time.perf_counter() - start_time:.4f}s")
        return results
