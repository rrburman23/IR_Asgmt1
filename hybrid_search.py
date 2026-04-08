"""
File Name: hybrid_search.py

Description:
- Hybrid retrieval/search over Tate artworks using fielded BM25 and dense retrieval.
- Strongly weighted surname anchor for IR robustness (IDF dilution mitigation).
- Canonical artist scoring (not brute force).
- Supports server-side pagination (page/per_page arguments to hybrid_search).
"""

import os
import re
import time
from typing import Optional, Dict, Any, List, Set, Tuple
from collections import Counter

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

VECTOR_CACHE = "embeddings.npy"
DENSE_MODEL_ID = "all-MiniLM-L6-v2"


def normalize_query(text: str) -> str:
    """
    Normalize input text for robust BM25 and semantic matching.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_surname(artist: str) -> str:
    """
    Extract and normalize the surname from artist metadata.
    E.g.: "Turner, Joseph Mallord William" -> "turner"
    """
    if not artist:
        return ""
    return artist.split(",")[0].strip().lower()


def normalize_artist_name(name: str) -> str:
    """
    Lowercase, remove punctuation, collapse whitespace.
    Used for comparing artists in deduplication/canonical logic.
    """
    import string

    s = name.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s


class ArtGallerySearchEngine:
    def __init__(self, data_path: str):
        print("[INFO] Loading Document Store...")
        self.df = pd.read_csv(data_path)
        for col in self.df.columns:
            self.df[col] = self.df[col].fillna("").astype(str)

        self.df["search_artist_surname"] = self.df["artist"].apply(extract_surname)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vocabulary: Set[str] = set()
        self.bm25: Optional[BM25Okapi] = None
        self.dense_model: Optional[SentenceTransformer] = None
        self.document_embeddings: Optional[np.ndarray] = None
        self._build_indexes()
        self.primary_artist_for_surname = self._build_canonical_artist_index()

        print(
            "Dominant artist per surname (example):",
            {
                k: v
                for k, v in self.primary_artist_for_surname.items()
                if k in ["turner", "blake", "black"]
            },
        )

    def _build_indexes(self):
        """
        Build weighted fielded BM25 and dense vector index.
        Surname gets maximal weight.
        """
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
        if os.path.exists(VECTOR_CACHE):
            print(f"[CACHE] Loading embeddings from {VECTOR_CACHE}...")
            self.document_embeddings = np.load(VECTOR_CACHE)
        else:
            print("[INFO] Computing semantic embeddings (first run)...")
            semantic_corpus = self.df["semantic_blob"].tolist()
            self.document_embeddings = self.dense_model.encode(
                semantic_corpus,
                batch_size=128,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            np.save(VECTOR_CACHE, self.document_embeddings)

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

    def _build_canonical_artist_index(self) -> Dict[str, str]:
        """
        For each surname, get the most common normalized full artist name.
        """
        lookup = {}
        for artist in self.df["artist"]:
            surname = extract_surname(artist)
            norm_artist = normalize_artist_name(artist)
            lookup.setdefault(surname, []).append(norm_artist)
        result = {}
        for surname, norm_names in lookup.items():
            most_common, _ = Counter(norm_names).most_common(1)[0]
            result[surname] = most_common
        return result

    def _deduplicate(self, ranked_indices: List[int]) -> List[int]:
        """
        Deduplicate results by (normalized title, normalized artist).
        """
        seen: Set[Tuple[str, str]] = set()
        deduped: List[int] = []
        for idx in ranked_indices:
            doc = self.df.iloc[idx]
            title_norm = doc["title"].strip().lower()
            artist_norm = doc["artist"].strip().lower()
            key = (title_norm, artist_norm)
            if key not in seen:
                seen.add(key)
                deduped.append(idx)
        return deduped

    def _is_canonical_artist(self, artist_raw: str, canonical: str) -> bool:
        """
        True if the given record is by the canonical artist for the surname.
        """
        return normalize_artist_name(artist_raw) == canonical

    def suggest_correction(self, query: str) -> Optional[str]:
        """
        Suggest spelling correction using Levenshtein distance.
        """
        if not self.vocabulary or not _LEV_AVAILABLE:
            return None
        lev = _lev
        if lev is None:
            return None
        suggested, changed = [], False
        for word in query.split():
            clean = re.sub(r"[^a-zA-Z]", "", word.lower())
            if not clean or len(clean) < 4 or clean in self.vocabulary:
                suggested.append(word)
                continue
            closest = min(
                self.vocabulary, key=lambda w: (lev.distance(clean, w), -len(w))
            )
            dist = lev.distance(clean, closest)
            if dist <= 2:
                suggested.append(closest)
                changed = True
            else:
                suggested.append(word)
        return " ".join(suggested) if changed else None

    def search_sparse(self, query: str, top_k: int = 20000) -> Dict[int, int]:
        """
        BM25 search on surname + artist + title + medium.
        """
        if self.bm25 is None:
            return {}
        scores = self.bm25.get_scores(normalize_query(query).split())
        top_indices = np.argsort(scores)[::-1][:top_k]
        return {
            int(idx): rank for rank, idx in enumerate(top_indices) if scores[idx] > 0
        }

    def search_dense(self, query: str, top_k: int = 20000) -> Dict[int, int]:
        """
        Dense retrieval (semantic search) on semantic_blob.
        """
        if self.dense_model is None or self.document_embeddings is None:
            return {}
        q_vec = self.dense_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )
        scores = np.dot(self.document_embeddings, q_vec.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return {
            int(idx): rank for rank, idx in enumerate(top_indices) if scores[idx] > 0.25
        }

    @staticmethod
    def _is_junk(title_lower: str, medium_lower: str, query_lower: str) -> bool:
        """
        Filter out junk/archival/fragment records.
        """
        if title_lower.strip().startswith("["):
            return True
        if re.search(r"\[.+\]", title_lower):
            return True
        for kw in ["inscription", "list of", "blank", "recto", "verso"]:
            if kw in title_lower and kw not in query_lower:
                return True
        if not medium_lower.strip():
            return True
        return False

    @staticmethod
    def _artist_word_match(query_words: set, artist_lower: str) -> bool:
        """
        True if any whole word in the query matches the artist.
        """
        return any(
            re.search(rf"\b{re.escape(word)}\b", artist_lower) for word in query_words
        )

    def hybrid_search(
        self,
        query: str,
        top_k: int = 100,
        k_rrf: int = 60,
        page: int = 1,
        per_page: int = 10,
    ) -> Dict[str, Any]:
        """
        Paginated hybrid IR (BM25 + dense + canonical artist boost).
        Returns paginated dict: results, page, per_page, total_results, total_pages, suggestion.
        """
        start_time = time.perf_counter()
        suggestion = self.suggest_correction(query)
        sparse_ranks = self.search_sparse(query, top_k=20000)
        dense_ranks = self.search_dense(query, top_k=20000)

        fused_scores: Dict[int, float] = {}
        query_lower = query.lower()
        query_words = set(query_lower.split())
        query_is_single_word = len(query_words) == 1
        lastname = query_lower.strip()
        canonical = self.primary_artist_for_surname.get(lastname)

        # Tuneable multipliers/penalties
        canonical_boost = 15.0
        true_artist_boost = 5.0
        tier1_medium_boost = 5.0
        tier2_medium_boost = 1.5
        depiction_penalty = 0.01
        subject_only_penalty = 0.1

        depiction_phrases = ["statue of", "portrait of", "bust of", "head of", "after"]
        tier1_media = ["oil", "canvas", "sculpture", "marble", "acrylic", "watercolour"]
        tier2_media = ["engraving", "etching", "lithograph"]

        # Score and collect all possible candidates
        for idx in set(sparse_ranks.keys()) | set(dense_ranks.keys()):
            s_rank = sparse_ranks.get(idx, 1000)
            d_rank = dense_ranks.get(idx, 1000)
            if s_rank > 500 and d_rank > 50:
                fused_scores[idx] = -1.0
                continue

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
                and canonical
                and self._is_canonical_artist(artist_raw, canonical)
            )

            # Scoring logic
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

        # Sort, deduplicate before pagination
        initial_ranking = sorted(
            (idx for idx in fused_scores if fused_scores[idx] >= 0),
            key=lambda x: fused_scores[x],
            reverse=True,
        )
        deduped_indices = self._deduplicate(initial_ranking)

        # Collect all results (to support correct paging + total_pages/total_results)
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
            if len(all_results) >= top_k:
                break
        
        # --- Canonical override for single-word surname queries ---
        if query_is_single_word and canonical:
            canonical_results = [
                r for r in all_results
                if normalize_artist_name(r["Artist"]) == canonical
            ]
            other_results = [
                r for r in all_results
                if normalize_artist_name(r["Artist"]) != canonical
            ]
            all_results = canonical_results + other_results

        # Pagination logic
        total_results = len(all_results)
        total_pages = max(1, (total_results + per_page - 1) // per_page)
        page = max(1, min(page, total_pages))
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        results_page = all_results[start_idx:end_idx]

        print(f"[TIMING] Hybrid Retrieval: {time.perf_counter() - start_time:.4f}s")
        return {
            "results": results_page,
            "page": page,
            "per_page": per_page,
            "total_results": total_results,
            "total_pages": total_pages,
            "suggestion": suggestion,
        }
