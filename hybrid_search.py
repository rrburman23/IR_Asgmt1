"""
File Name: hybrid_search.py

Description:
Core Hybrid Search Engine for the Tate dataset.
- Fielded BM25 with surname strong weighting.
- Dense retrieval via SentenceTransformers (chunked dense with max pooling).
- RRF fusion with Canonical Artist tracking to mitigate IDF dilution.
- Supports server-side pagination.
"""

import os
import re
import string
import json
from collections import Counter
from typing import Optional, Dict, Any, List, Set, Tuple

import pandas as pd
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

try:
    import Levenshtein as _lev

    _LEV_AVAILABLE = True
except ImportError:
    _lev = None
    _LEV_AVAILABLE = False

# Legacy doc-level cache (no longer used for new builds, but kept for backward compat if needed)
VECTOR_CACHE = "embeddings.npy"

# Caches for chunked dense index
CHUNK_EMB_CACHE = "chunk_embeddings.npy"
CHUNK_MAP_CACHE = "chunk_to_doc.npy"

DENSE_MODEL_ID = "all-MiniLM-L6-v2"


def normalize_query(text: str) -> str:
    """Normalize input for case-insensitive search and fair tokenization."""
    if not isinstance(text, str):
        return ""
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_surname(artist: str) -> str:
    """
    Extract the surname from artist metadata.
    E.g., 'Turner, Joseph Mallord William' -> 'turner'
    """
    if not artist:
        return ""
    return artist.split(",")[0].strip().lower()


def normalize_artist_name(name: str) -> str:
    """Normalize artist names: lower, remove punctuation, collapse whitespace."""
    s = name.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s


class ArtGallerySearchEngine:
    """Hybrid Search Engine combining Lexical (BM25) and Semantic retrieval."""

    def __init__(self, data_path: str):
        print("[INFO] Loading Document Store...")
        self.df = pd.read_csv(data_path)
        for col in self.df.columns:
            self.df[col] = self.df[col].fillna("").astype(str)

        # Add explicit surname search field for BM25 anchoring
        self.df["search_artist_surname"] = self.df["artist"].apply(extract_surname)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vocabulary: Set[str] = set()
        self.bm25: Optional[BM25Okapi] = None
        self.dense_model: Optional[SentenceTransformer] = None

        # Dense indexes (chunked)
        # document_embeddings is kept as a backwards-compatible alias for tests/older code.
        self.document_embeddings: Optional[np.ndarray] = None
        self.chunk_embeddings: Optional[np.ndarray] = None
        self.chunk_to_doc_idx: Optional[np.ndarray] = None

        self._build_indexes()
        # Identify the dominant artist for every surname to handle ambiguous queries
        self.primary_artist_for_surname = self._build_canonical_artist_index()

    def _build_indexes(self) -> None:
        """Build fielded BM25, dense model, chunked index, and spelling vocabulary."""
        print("[INFO] Building Fielded Sparse Index (Artist Surname Priority)...")
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

        print(f"[INFO] Initializing dense retriever on {self.device.upper()}...")
        self.dense_model = SentenceTransformer(DENSE_MODEL_ID, device=self.device)

        # NEW: use chunked dense index instead of doc-level embeddings.npy
        self._load_or_build_chunk_index()

        print("[INFO] Building Spelling Vocabulary...")
        title_artists = (
            self.df["search_title"].tolist() + self.df["search_artist"].tolist()
        )
        for text in title_artists:
            if text:
                self.vocabulary.update(
                    w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", text)
                )
        print("[SUCCESS] All retrieval indexes are online.")

    def _load_or_build_chunk_index(self) -> None:
        """
        Build (or load) chunk-level dense index:
        - chunk_embeddings: embedding per chunk
        - chunk_to_doc_idx: maps each chunk -> parent doc row index

        Also sets:
        - document_embeddings alias -> chunk_embeddings for backward compatibility.
        """
        has_chunks = "description_chunks" in self.df.columns

        # 1) Load from cache if present
        if os.path.exists(CHUNK_EMB_CACHE) and os.path.exists(CHUNK_MAP_CACHE):
            print(f"[CACHE] Loading chunk embeddings from {CHUNK_EMB_CACHE}...")
            self.chunk_embeddings = np.load(CHUNK_EMB_CACHE)
            self.chunk_to_doc_idx = np.load(CHUNK_MAP_CACHE)

            # Backwards-compat alias (fixes tests / older code using document_embeddings)
            self.document_embeddings = self.chunk_embeddings
            return

        # 2) Build chunks and mapping from DataFrame
        if not has_chunks:
            print(
                "[WARN] description_chunks not found; falling back to semantic_blob as single chunk per doc."
            )
            all_chunks = self.df.get(
                "semantic_blob", pd.Series([""] * len(self.df))
            ).tolist()
            self.chunk_to_doc_idx = np.arange(len(self.df), dtype=np.int32)
        else:
            all_chunks: list[str] = []
            mapping: list[int] = []
            for doc_idx, chunks_json in enumerate(
                self.df["description_chunks"].tolist()
            ):
                try:
                    chunks = json.loads(chunks_json) if chunks_json else []
                except Exception:
                    chunks = []
                if not chunks:
                    chunks = [self.df.iloc[doc_idx].get("semantic_blob", "")]
                for c in chunks:
                    all_chunks.append(str(c))
                    mapping.append(doc_idx)
            self.chunk_to_doc_idx = np.array(mapping, dtype=np.int32)

        # 3) Compute embeddings for chunks
        print("[INFO] Computing chunk embeddings (first run)...")
        assert self.dense_model is not None

        self.chunk_embeddings = self.dense_model.encode(
            all_chunks,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        # Backwards-compat alias
        self.document_embeddings = self.chunk_embeddings

        # 4) Save caches
        np.save(CHUNK_EMB_CACHE, self.chunk_embeddings)
        np.save(CHUNK_MAP_CACHE, self.chunk_to_doc_idx)
        print(
            f"[CACHE] Saved chunk embeddings to {CHUNK_EMB_CACHE} and mapping to {CHUNK_MAP_CACHE}."
        )

    def _build_canonical_artist_index(self) -> Dict[str, str]:
        """Map each surname to the most common normalized full artist name in the corpus."""
        lookup: Dict[str, List[str]] = {}
        for artist in self.df["artist"]:
            surname = extract_surname(artist)
            norm_artist = normalize_artist_name(artist)
            lookup.setdefault(surname, []).append(norm_artist)

        result: Dict[str, str] = {}
        for surname, norm_names in lookup.items():
            most_common, _ = Counter(norm_names).most_common(1)[0]
            result[surname] = most_common
        return result

    def _deduplicate(self, ranked_indices: List[int]) -> List[int]:
        """Deduplicate results by Title, Artist, and Year to hide identical print runs."""
        seen: Set[Tuple[str, str, str]] = set()
        deduped: List[int] = []
        for idx in ranked_indices:
            doc = self.df.iloc[idx]
            title_norm = str(doc.get("title", "")).strip().lower()
            artist_norm = str(doc.get("artist", "")).strip().lower()
            year_norm = str(doc.get("year", "")).strip()
            fingerprint = (title_norm, artist_norm, year_norm)
            if fingerprint not in seen:
                seen.add(fingerprint)
                deduped.append(idx)
        return deduped

    def _is_canonical_artist(self, artist_raw: str, canonical: str) -> bool:
        """Check if the given record belongs to the dominant artist for that surname."""
        return normalize_artist_name(artist_raw) == canonical

    def suggest_correction(self, query: str) -> Optional[str]:
        """Suggest spelling correction using Levenshtein distance."""
        if not self.vocabulary or not _LEV_AVAILABLE or _lev is None:
            return None
        lev = _lev

        suggested, changed = [], False
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

    def search_sparse(self, query: str, top_k: int = 20000) -> Dict[int, int]:
        """Lexical BM25 search."""
        if self.bm25 is None:
            return {}
        scores = self.bm25.get_scores(normalize_query(query).split())
        top_indices = np.argsort(scores)[::-1][:top_k]
        return {
            int(idx): rank for rank, idx in enumerate(top_indices) if scores[idx] > 0
        }

    def search_dense(self, query: str, top_k: int = 20000) -> Dict[int, int]:
        """
        Chunked dense retrieval:
        - score each chunk vs query
        - max pool chunk scores back to document score
        - rank documents by pooled score
        Returns {doc_row_index: rank}.
        """
        if (
            self.dense_model is None
            or self.chunk_embeddings is None
            or self.chunk_to_doc_idx is None
        ):
            return {}

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
            if doc_scores[idx] > 0.25
        }

    @staticmethod
    def _is_junk(title_lower: str, medium_lower: str, query_lower: str) -> bool:
        """Filter out obvious metadata fragments and unclassified records."""
        if title_lower.strip().startswith("[") or re.search(r"\[.+\]", title_lower):
            return True
        for kw in ["inscription", "list of", "blank", "recto", "verso"]:
            if kw in title_lower and kw not in query_lower:
                return True
        if not medium_lower.strip():
            return True
        return False

    @staticmethod
    def _artist_word_match(query_words: set, artist_lower: str) -> bool:
        """Verify if any whole word in the query matches the artist name."""
        return any(
            re.search(rf"\b{re.escape(word)}\b", artist_lower) for word in query_words
        )

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10000,
        k_rrf: int = 60,
        page: int = 1,
        per_page: int = 10,
    ) -> Dict[str, Any]:
        """
        Paginated hybrid retrieval combining BM25, Dense vectors, and Intent Multipliers.
        """
        suggestion = self.suggest_correction(query)

        sparse_ranks = self.search_sparse(query, top_k=top_k)
        dense_ranks = self.search_dense(query, top_k=top_k)

        fused_scores: Dict[int, float] = {}
        query_lower = query.lower()
        query_words = set(query_lower.split())
        query_is_single_word = len(query_words) == 1

        lastname = query_lower.strip()
        canonical_artist = self.primary_artist_for_surname.get(lastname)

        # Multipliers and penalties to align math with human intent
        canonical_boost = 15.0
        true_artist_boost = 5.0
        tier1_medium_boost = 5.0
        tier2_medium_boost = 1.5
        depiction_penalty = 0.01
        subject_only_penalty = 0.1

        depiction_phrases = ["statue of", "portrait of", "bust of", "head of", "after"]
        tier1_media = ["oil", "canvas", "sculpture", "marble", "acrylic", "watercolour"]
        tier2_media = ["engraving", "etching", "lithograph"]

        # Phase 1: Score all candidates
        for idx in set(sparse_ranks.keys()) | set(dense_ranks.keys()):
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
            title_mentions_query = any(word in title_lower for word in query_words)
            is_depiction = any(d in title_lower for d in depiction_phrases)
            user_wants_depiction = any(d in query_lower for d in depiction_phrases)
            is_canonical = (
                query_is_single_word
                and canonical_artist
                and self._is_canonical_artist(artist_raw, canonical_artist)
            )

            # Apply Logic Multipliers
            if is_canonical:
                multiplier *= canonical_boost
            elif is_true_artist:
                multiplier *= true_artist_boost

            if any(m in medium_lower for m in tier1_media):
                multiplier *= tier1_medium_boost
            elif any(m in medium_lower for m in tier2_media):
                multiplier *= tier2_medium_boost

            if is_depiction and not user_wants_depiction:
                multiplier *= depiction_penalty

            if title_mentions_query and not is_true_artist:
                multiplier *= subject_only_penalty

            fused_scores[idx] = base_score * multiplier

        # Phase 2: Sort and Deduplicate
        initial_ranking = sorted(
            (idx for idx in fused_scores if fused_scores[idx] >= 0),
            key=lambda x: fused_scores[x],
            reverse=True,
        )
        deduped_indices = self._deduplicate(initial_ranking)

        # Phase 3: Build detailed result objects
        all_results: List[Dict[str, Any]] = []
        for idx in deduped_indices:
            doc = self.df.iloc[idx]
            reasons = []
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

        # Phase 4: Canonical Override (The "Hard Shield")
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

        # Phase 5: Pagination
        total_results = len(all_results)
        total_pages = max(1, (total_results + per_page - 1) // per_page)
        page = max(1, min(page, total_pages))

        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        results_page = all_results[start_idx:end_idx]

        return {
            "results": results_page,
            "page": page,
            "per_page": per_page,
            "total_results": total_results,
            "total_pages": total_pages,
            "suggestion": suggestion,
        }
