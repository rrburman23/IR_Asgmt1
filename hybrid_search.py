"""
hybrid_search.py

Core Hybrid Search Engine for the Tate dataset.

Features
--------
- Fielded BM25 with strong weighting for artist surname.
- Dense retrieval via SentenceTransformers using a chunked index with max pooling.
- Reciprocal Rank Fusion (RRF) over sparse + dense ranks.
- Intent-aware re-ranking (title/artist/medium/depiction heuristics).
- Server-side pagination.

Packaging / Performance (PyInstaller-friendly)
----------------------------------------------
- Resolves runtime files (CSV + .npy caches) relative to the executable directory
  when running as a frozen app.
- Lazy-builds BM25 (first sparse query) to reduce application startup time.
- Lazy-loads the dense model (first dense query) to reduce application startup time.
- Adds "intent gating" so dense retrieval is skipped for obviously lexical queries
  unless explicitly forced (reduces per-query latency on CPU laptops).
"""

from __future__ import annotations

import json
import os
import re
import string
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

try:
    import Levenshtein as _lev

    _LEV_AVAILABLE = True
except ImportError:  # pragma: no cover
    _lev = None
    _LEV_AVAILABLE = False


DENSE_MODEL_ID = "all-MiniLM-L6-v2"

# Chunked dense index caches (expected to be shipped alongside the exe).
CHUNK_EMB_FILENAME = "chunk_embeddings.npy"
CHUNK_MAP_FILENAME = "chunk_to_doc.npy"


def app_dir() -> str:
    """
    Return the directory where runtime assets are expected to live.

    - Frozen exe (PyInstaller): the directory containing the executable.
    - Normal Python: the directory containing this .py file.
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


BASE_DIR = app_dir()


def resolve_runtime_path(path_or_filename: str) -> str:
    """
    Resolve an absolute or relative path to a concrete runtime path.

    If a relative path is provided, it is resolved relative to BASE_DIR.
    """
    if not path_or_filename:
        return path_or_filename
    if os.path.isabs(path_or_filename):
        return path_or_filename
    return os.path.join(BASE_DIR, path_or_filename)


def normalize_query(text: str) -> str:
    """Normalize input for fair tokenization and case-insensitive matching."""
    if not isinstance(text, str):
        return ""
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_surname(artist: str) -> str:
    """
    Extract surname from Tate artist format.

    Example:
        'Turner, Joseph Mallord William' -> 'turner'
    """
    if not artist:
        return ""
    return artist.split(",")[0].strip().lower()


def normalize_artist_name(name: str) -> str:
    """Normalize artist names by removing punctuation and collapsing whitespace."""
    s = name.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s


class ArtGallerySearchEngine:
    """Hybrid search engine combining BM25 and dense retrieval with RRF fusion."""

    def __init__(self, data_path: str) -> None:
        data_path = resolve_runtime_path(data_path)

        print("[INFO] Loading Document Store...")
        self.df = pd.read_csv(data_path)
        for col in self.df.columns:
            self.df[col] = self.df[col].fillna("").astype(str)

        # Add surname field for fielded BM25.
        self.df["search_artist_surname"] = self.df["artist"].apply(extract_surname)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Spelling correction vocabulary.
        self.vocabulary: Set[str] = set()

        # Lazy-built indexes/models.
        self.bm25: Optional[BM25Okapi] = None
        self.dense_model: Optional[SentenceTransformer] = None

        # Dense (chunk) index; loaded from .npy caches when available.
        self.chunk_embeddings: Optional[np.ndarray] = None
        self.chunk_to_doc_idx: Optional[np.ndarray] = None

        # Legacy alias used by older code/tests (chunk embeddings).
        self.document_embeddings: Optional[np.ndarray] = None

        # Load caches now (fast). Do not load transformer model.
        self._load_chunk_index_only()

        # Build vocabulary now (moderate cost; helps suggest_correction).
        self._build_spelling_vocabulary()

        # Canonical artist mapping is cheap and helps surname queries.
        self.primary_artist_for_surname = self._build_canonical_artist_index()

        print("[SUCCESS] Engine online (BM25 and dense model are lazy-loaded).")

    # -------------------------------------------------------------------------
    # Lazy builders
    # -------------------------------------------------------------------------
    def _ensure_bm25(self) -> None:
        """Build BM25 on demand to reduce application startup time."""
        if self.bm25 is not None:
            return

        print("[INFO] Building BM25 index (lazy)...")
        bm25_corpus = [
            (
                (row.get("search_artist_surname", "") + " ") * 50
                + (row.get("search_artist", "") + " ") * 8
                + (row.get("search_title", "") + " ") * 3
                + (row.get("search_medium", "") + " ")
            )
            for _, row in self.df.iterrows()
        ]
        tokenized_corpus = [doc.split() for doc in bm25_corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("[SUCCESS] BM25 ready.")

    def _ensure_dense_model(self) -> None:
        """Load the SentenceTransformer model only when dense retrieval is needed."""
        if self.dense_model is not None:
            return
        print(f"[INFO] Initializing dense retriever on {self.device.upper()} (lazy)...")
        self.dense_model = SentenceTransformer(DENSE_MODEL_ID, device=self.device)

    # -------------------------------------------------------------------------
    # Dense chunk index loading/building
    # -------------------------------------------------------------------------
    def _load_chunk_index_only(self) -> None:
        """
        Load chunk embeddings and chunk-to-doc mapping.

        This is cheap relative to model loading. If the cache is missing,
        we fall back to rebuilding (slow and requires the dense model).
        """
        emb_path = resolve_runtime_path(CHUNK_EMB_FILENAME)
        map_path = resolve_runtime_path(CHUNK_MAP_FILENAME)

        if os.path.exists(emb_path) and os.path.exists(map_path):
            print(f"[CACHE] Loading chunk embeddings from {emb_path}...")
            self.chunk_embeddings = np.load(emb_path)
            self.chunk_to_doc_idx = np.load(map_path)

            # Backwards-compatible alias.
            self.document_embeddings = self.chunk_embeddings
            return

        print("[WARN] Dense cache not found; rebuilding dense chunks (slow).")
        self._build_chunk_index(emb_path=emb_path, map_path=map_path)

    def _build_chunk_index(self, emb_path: str, map_path: str) -> None:
        """
        Build dense chunk index from dataframe.

        WARNING: This is expensive and should not happen in a shipped build if
        the caches are present.
        """
        has_chunks = "description_chunks" in self.df.columns

        if not has_chunks:
            all_chunks = self.df.get(
                "semantic_blob", pd.Series([""] * len(self.df))
            ).tolist()
            chunk_to_doc_idx = np.arange(len(self.df), dtype=np.int32)
        else:
            all_chunks: List[str] = []
            mapping: List[int] = []
            for doc_idx, chunks_json in enumerate(
                self.df["description_chunks"].tolist()
            ):
                try:
                    chunks = json.loads(chunks_json) if chunks_json else []
                except Exception:  # pragma: no cover
                    chunks = []

                if not chunks:
                    chunks = [self.df.iloc[doc_idx].get("semantic_blob", "")]

                for chunk in chunks:
                    all_chunks.append(str(chunk))
                    mapping.append(doc_idx)

            chunk_to_doc_idx = np.array(mapping, dtype=np.int32)

        self._ensure_dense_model()
        assert self.dense_model is not None

        chunk_embeddings = self.dense_model.encode(
            all_chunks,
            batch_size=128,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        self.chunk_embeddings = chunk_embeddings
        self.chunk_to_doc_idx = chunk_to_doc_idx
        self.document_embeddings = chunk_embeddings

        np.save(emb_path, chunk_embeddings)
        np.save(map_path, chunk_to_doc_idx)

        print(
            f"[CACHE] Saved chunk embeddings to {emb_path} and mapping to {map_path}."
        )

    # -------------------------------------------------------------------------
    # Spelling vocabulary
    # -------------------------------------------------------------------------
    def _build_spelling_vocabulary(self) -> None:
        """Build vocabulary for Levenshtein-based spelling suggestions."""
        print("[INFO] Building Spelling Vocabulary...")
        title_artists = (
            self.df["search_title"].tolist() + self.df["search_artist"].tolist()
        )
        for text in title_artists:
            if text:
                self.vocabulary.update(
                    w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", text)
                )

    def suggest_correction(self, query: str) -> Optional[str]:
        """Suggest spelling correction using Levenshtein distance (if available)."""
        if not self.vocabulary or not _LEV_AVAILABLE or _lev is None:
            return None

        lev = _lev
        suggested: List[str] = []
        changed = False

        for word in query.split():
            clean = re.sub(r"[^a-zA-Z]", "", word.lower())
            if not clean or len(clean) < 4 or clean in self.vocabulary:
                suggested.append(word)
                continue

            closest = min(
                self.vocabulary, key=lambda w: (lev.distance(clean, w), -len(w))
            )
            if lev.distance(clean, closest) <= 2:
                suggested.append(closest)
                changed = True
            else:
                suggested.append(word)

        return " ".join(suggested) if changed else None

    # -------------------------------------------------------------------------
    # Query intent gating (latency win on CPU)
    # -------------------------------------------------------------------------
    @staticmethod
    def _looks_lexical(query: str) -> bool:
        """
        Heuristic to decide whether dense retrieval is likely unnecessary.

        Dense retrieval is O(num_chunks) with a full dot-product on CPU; skipping
        it for purely lexical queries drastically reduces latency.
        """
        q = query.strip()
        ql = q.lower()
        tokens = ql.split()

        if not q:
            return True

        # Slash commands are handled above engine; treat as lexical here.
        if q.startswith("/"):
            return True

        # Single token (surname, short keyword) is usually lexical.
        if len(tokens) <= 1:
            return True

        # Very short multi-token queries are often known-item/title style.
        if len(ql) <= 12:
            return True

        # Quoted strings are typically known-item.
        if '"' in q or "'" in q:
            return True

        # Catalog-like punctuation tends to be exact lexical lookups.
        if re.search(r"[\(\)\[\]:;]", q):
            return True

        return False

    # -------------------------------------------------------------------------
    # Retrieval primitives
    # -------------------------------------------------------------------------
    def search_sparse(self, query: str, top_k: int = 20000) -> Dict[int, int]:
        """BM25 search returning {row_index: rank} for positive-score docs."""
        self._ensure_bm25()
        assert self.bm25 is not None

        scores = self.bm25.get_scores(normalize_query(query).split())
        top_indices = np.argsort(scores)[::-1][:top_k]
        return {
            int(idx): rank for rank, idx in enumerate(top_indices) if scores[idx] > 0
        }

    def search_dense(self, query: str, top_k: int = 20000) -> Dict[int, int]:
        """
        Dense retrieval over chunk embeddings with max pooling back to documents.

        Returns
        -------
        Dict[int, int]
            {doc_row_index: rank} for docs above a cosine similarity threshold.
        """
        if self.chunk_embeddings is None or self.chunk_to_doc_idx is None:
            return {}

        self._ensure_dense_model()
        assert self.dense_model is not None

        q_vec = self.dense_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

        chunk_scores = np.dot(self.chunk_embeddings, q_vec.T).flatten()

        doc_scores = np.full(len(self.df), -1e9, dtype=np.float32)
        np.maximum.at(doc_scores, self.chunk_to_doc_idx, chunk_scores)

        top_indices = np.argsort(doc_scores)[::-1][:top_k]
        return {
            int(idx): rank
            for rank, idx in enumerate(top_indices)
            if doc_scores[idx] > 0.3
        }

    # -------------------------------------------------------------------------
    # Ranking helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _is_junk(title_lower: str, medium_lower: str, query_lower: str) -> bool:
        """Filter obvious metadata fragments and unclassified records."""
        if title_lower.strip().startswith("[") or re.search(r"\[.+\]", title_lower):
            return True
        for kw in ["inscription", "list of", "blank", "recto", "verso"]:
            if kw in title_lower and kw not in query_lower:
                return True
        if not medium_lower.strip():
            return True
        return False

    @staticmethod
    def _artist_word_match(query_words: set[str], artist_lower: str) -> bool:
        """Check if any whole-word query token appears in the artist name."""
        return any(
            re.search(rf"\b{re.escape(word)}\b", artist_lower) for word in query_words
        )

    def _build_canonical_artist_index(self) -> Dict[str, str]:
        """Map surname -> most common normalized full artist string in the dataset."""
        lookup: Dict[str, List[str]] = {}
        for artist in self.df["artist"]:
            surname = extract_surname(artist)
            lookup.setdefault(surname, []).append(normalize_artist_name(artist))

        result: Dict[str, str] = {}
        for surname, norm_names in lookup.items():
            most_common, _ = Counter(norm_names).most_common(1)[0]
            result[surname] = most_common
        return result

    def _is_canonical_artist(self, artist_raw: str, canonical: str) -> bool:
        """Return True if this record is the canonical artist for the surname."""
        return normalize_artist_name(artist_raw) == canonical

    def _deduplicate(self, ranked_indices: List[int]) -> List[int]:
        """Deduplicate results by (title, artist, year) fingerprint."""
        seen: Set[Tuple[str, str, str]] = set()
        deduped: List[int] = []

        for idx in ranked_indices:
            doc = self.df.iloc[idx]
            fp = (
                str(doc.get("title", "")).strip().lower(),
                str(doc.get("artist", "")).strip().lower(),
                str(doc.get("year", "")).strip(),
            )
            if fp in seen:
                continue
            seen.add(fp)
            deduped.append(idx)

        return deduped

    # -------------------------------------------------------------------------
    # Public search API
    # -------------------------------------------------------------------------
    def hybrid_search(
        self,
        query: str,
        top_k: int = 10000,
        k_rrf: int = 60,
        page: int = 1,
        per_page: int = 10,
        force_dense: bool = False,
    ) -> Dict[str, Any]:
        """
        Hybrid search with RRF fusion and intent-aware multipliers.

        Parameters
        ----------
        query:
            User query string.
        top_k:
            Candidate set size for sparse/dense rank extraction.
        k_rrf:
            RRF constant (larger => less steep rank contribution).
        page, per_page:
            Server-side pagination.
        force_dense:
            If True, dense retrieval always runs (useful for /evaluate).
        """
        suggestion = self.suggest_correction(query)

        # Intent gating: dense is expensive on CPU; skip for lexical queries
        # unless the caller forces dense (e.g., evaluation).
        use_dense = force_dense or (not self._looks_lexical(query))

        sparse_ranks = self.search_sparse(query, top_k=top_k)
        dense_ranks = self.search_dense(query, top_k=top_k) if use_dense else {}

        fused_scores: Dict[int, float] = {}

        query_lower = query.lower()
        query_words = set(query_lower.split())
        query_is_single_word = len(query_words) == 1

        lastname = query_lower.strip()
        canonical_artist = self.primary_artist_for_surname.get(lastname)

        # Soft multipliers (tie-breakers rather than overrides).
        canonical_boost = 3.0
        true_artist_boost = 1.5
        exact_title_boost = 3.0
        all_words_title_boost = 2.0
        strong_title_boost = 1.5
        tier1_medium_boost = 1.1
        tier2_medium_boost = 1.05
        depiction_penalty = 0.1
        subject_only_penalty = 0.5

        depiction_phrases = ["statue of", "portrait of", "bust of", "head of", "after"]
        tier1_media = ["oil", "canvas", "sculpture", "marble", "acrylic", "watercolour"]
        tier2_media = ["engraving", "etching", "lithograph"]

        candidate_indices = set(sparse_ranks.keys()) | set(dense_ranks.keys())

        for idx in candidate_indices:
            s_rank = sparse_ranks.get(idx, 25000)
            d_rank = dense_ranks.get(idx, 25000)

            doc = self.df.iloc[idx]
            title_lower = doc["title"].lower()
            artist_raw = doc["artist"]
            artist_lower = artist_raw.lower()
            medium_lower = doc["medium"].lower()

            if self._is_junk(title_lower, medium_lower, query_lower):
                fused_scores[idx] = -1.0
                continue

            base_score = (1.0 / (k_rrf + s_rank)) + (1.0 / (k_rrf + d_rank))
            multiplier = 1.0

            is_true_artist = self._artist_word_match(query_words, artist_lower)
            title_mentions_query = any(w in title_lower for w in query_words)
            is_depiction = any(d in title_lower for d in depiction_phrases)
            user_wants_depiction = any(d in query_lower for d in depiction_phrases)

            is_canonical = (
                query_is_single_word
                and canonical_artist
                and self._is_canonical_artist(artist_raw, canonical_artist)
            )

            norm_query = query_lower.strip()
            norm_title = title_lower.strip()

            all_words_in_title = len(query_words) >= 3 and all(
                w in norm_title for w in query_words
            )
            exact_title_match = norm_query == norm_title
            strong_title_overlap = (
                not exact_title_match
                and not all_words_in_title
                and sum(1 for w in query_words if w in norm_title)
                >= max(1, len(query_words) // 2)
            )

            # Artist boosts
            if is_canonical:
                multiplier *= canonical_boost
            elif is_true_artist:
                multiplier *= true_artist_boost

            # Title boosts
            if exact_title_match:
                multiplier *= exact_title_boost
            elif all_words_in_title:
                multiplier *= all_words_title_boost
            elif strong_title_overlap:
                multiplier *= strong_title_boost

            # Medium boosts
            if any(m in medium_lower for m in tier1_media):
                multiplier *= tier1_medium_boost
            elif any(m in medium_lower for m in tier2_media):
                multiplier *= tier2_medium_boost

            # Depiction penalty unless user explicitly wants depictions.
            if is_depiction and not user_wants_depiction:
                multiplier *= depiction_penalty

            # Subject-only penalty unless title match is strong.
            if (
                title_mentions_query
                and not is_true_artist
                and not exact_title_match
                and not all_words_in_title
                and not strong_title_overlap
            ):
                multiplier *= subject_only_penalty

            fused_scores[idx] = base_score * multiplier

        initial_ranking = sorted(
            (idx for idx in fused_scores if fused_scores[idx] >= 0),
            key=lambda i: fused_scores[i],
            reverse=True,
        )
        deduped_indices = self._deduplicate(initial_ranking)

        all_results: List[Dict[str, Any]] = []
        for idx in deduped_indices:
            doc = self.df.iloc[idx]
            reasons: List[str] = []

            if idx in sparse_ranks and sparse_ranks[idx] < 20:
                reasons.append(f"Keyword (Rank {sparse_ranks[idx] + 1})")
            if idx in dense_ranks and dense_ranks[idx] < 20:
                reasons.append(f"AI Context (Rank {dense_ranks[idx] + 1})")

            all_results.append(
                {
                    "Rank": len(all_results) + 1,
                    "id": doc["id"],
                    "Title": doc["title"],
                    "Artist": doc["artist"],
                    "Medium": doc["medium"],
                    "Year": doc["year"].replace("nan", "Unknown Date"),
                    "Dimensions": doc.get("dimensions", "").replace("nan", ""),
                    "CreditLine": doc.get("creditline", "").replace("nan", ""),
                    "Description": (
                        f"An archival {doc['medium']} piece from the Tate collection."
                        if doc["medium"]
                        else "An archival piece from the Tate collection."
                    ),
                    "Thumbnail": doc["thumbnailurl"],
                    "Score": round(fused_scores[idx], 4),
                    "Reasons": " + ".join(reasons) or "Hybrid Match",
                    "Suggestion": suggestion,
                }
            )

        # Canonical override for single-word surname searches.
        if query_is_single_word and canonical_artist:
            canonical_results = [
                r
                for r in all_results
                if normalize_artist_name(r["Artist"]) == canonical_artist
            ]
            other_results = [
                r
                for r in all_results
                if normalize_artist_name(r["Artist"]) != canonical_artist
            ]
            all_results = canonical_results + other_results

        # Pagination
        total_results = len(all_results)
        total_pages = max(1, (total_results + per_page - 1) // per_page)
        page = max(1, min(page, total_pages))
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page

        return {
            "results": all_results[start_idx:end_idx],
            "page": page,
            "per_page": per_page,
            "total_results": total_results,
            "total_pages": total_pages,
            "suggestion": suggestion,
        }
