"""
File Name: Hybrid Search Engine core
Description: Implements a dual-pipeline retrieval system using BM25F for sparse
             lexical matching and Exact k-NN (Sentence Transformers) for dense
             semantic matching, fused via Reciprocal Rank Fusion (RRF).
"""

import os
import re
import json
import math
import time
from collections import Counter
import pandas as pd
import numpy as np

try:
    import Levenshtein
except ImportError:
    Levenshtein = None

from sentence_transformers import SentenceTransformer
from ingest_data import process_text_for_sparse

# Suppress verbose TF INFO logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class BM25F:
    """
    Custom BM25F implementation to support weighted fielded indexing.
    Weights term frequencies across fields before applying the non-linear
    saturation function.
    """
    def __init__(self, corpus_fields, field_weights, k1=1.5, b=0.75):
        self.field_weights = field_weights
        self.k1 = k1
        self.b = b
        self.doc_count = len(corpus_fields)

        self.doc_freqs = Counter()
        self.field_lengths = {f: [] for f in field_weights}
        self.avg_field_lengths = {}
        self.doc_term_freqs = []

        # Precompute frequencies and field lengths
        for doc in corpus_fields:
            doc_tfs = {}
            doc_unique_terms = set()
            for f, tokens in doc.items():
                self.field_lengths[f].append(len(tokens))
                freq = Counter(tokens)
                doc_tfs[f] = freq
                doc_unique_terms.update(tokens)
            self.doc_term_freqs.append(doc_tfs)
            for term in doc_unique_terms:
                self.doc_freqs[term] += 1

        # Calculate average lengths for normalization
        for f in self.field_weights:
            avg = sum(self.field_lengths[f]) / self.doc_count if self.doc_count else 1
            self.avg_field_lengths[f] = max(avg, 1)  # Prevent div by zero

    def get_idf(self, term):
        df = self.doc_freqs[term]
        return math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))

    def get_scores(self, query_tokens):
        scores = np.zeros(self.doc_count)
        for term in query_tokens:
            if term not in self.doc_freqs:
                continue
            idf = self.get_idf(term)

            for idx in range(self.doc_count):
                pseudo_tf = 0
                for f, weight in self.field_weights.items():
                    tf_f = self.doc_term_freqs[idx][f][term]
                    L_f = self.field_lengths[f][idx]
                    L_avg = self.avg_field_lengths[f]

                    # Length normalization per field
                    norm = (1 - self.b + self.b * (L_f / L_avg))
                    pseudo_tf += weight * (tf_f / norm)

                # Apply saturation after linearly combining field tf
                term_score = pseudo_tf / (self.k1 + pseudo_tf)
                scores[idx] += idf * term_score
        return scores


class ArtGallerySearchEngine:
    """
    Core engine for the Art Gallery Hybrid Search.
    Manages offline index construction (sparse and dense), and online query
    execution utilizing a Min-Max normalized Weighted Linear Combination.
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

        self.bm25 = None
        self.dense_model = None
        self.document_embeddings = None
        self.chunk_to_doc_idx = None
        self.vocabulary = set()

        self._build_indexes()

    def _build_indexes(self):
        """
        Executes the offline indexing pipelines (Sparse and Dense).
        """
        print("[INFO] Building Sparse Index (BM25F)...")
        # Prepare fielded corpus for BM25F using the shared NLP pipeline
        corpus_fields = []
        for _, row in self.df.iterrows():
            corpus_fields.append({
                'title': process_text_for_sparse(row['title']).split(),
                'artist': process_text_for_sparse(row['artist']).split(),
                'medium': process_text_for_sparse(row['medium']).split()
            })
        
        # Apply the architectural field boosts
        field_weights = {'title': 2.0, 'artist': 1.5, 'medium': 1.0}
        self.bm25 = BM25F(corpus_fields, field_weights)

        print("[INFO] Building Dense Vector Index (BERT) with Chunking...")
        # Using a lightweight, highly efficient pre-trained transformer
        self.dense_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        all_chunks = []
        self.chunk_to_doc_idx = []
        
        # Extract chunks and map them to their parent document IDs
        for doc_idx, chunks_json in enumerate(self.df["description_chunks"]):
            try:
                chunks = json.loads(chunks_json)
            except Exception:
                chunks = []
                
            if not chunks:
                chunks = [""] # Fallback to prevent dimension mismatch
                
            for chunk in chunks:
                all_chunks.append(chunk)
                self.chunk_to_doc_idx.append(doc_idx)

        # Generate dense embeddings for all chunks collectively
        self.document_embeddings = self.dense_model.encode(
            all_chunks, convert_to_numpy=True
        )
        self.chunk_to_doc_idx = np.array(self.chunk_to_doc_idx)
        
        print("[INFO] Building Vocabulary for Spelling Correction...")
        for text in self.df["artist"].tolist() + self.df["medium"].tolist():
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            self.vocabulary.update(words)
        print("[SUCCESS] All indexes built successfully.")

    def search_sparse(self, query, top_k=100):
        """
        Online Phase: Executes BM25F fielded keyword matching.
        """
        tokenized_query = process_text_for_sparse(query).split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        # Return a dictionary of {doc_id: score} for the top results
        return {idx: scores[idx] for idx in top_indices if scores[idx] > 0}

    def search_dense(self, query, top_k=100):
        """
        Online Phase: Executes Exact k-NN cosine similarity on dense vectors,
        utilizing Max Pooling to aggregate chunk scores to their parent
        documents.
        """
        try:
            query_embedding = self.dense_model.encode([query], convert_to_numpy=True)

            # Compute exact cosine similarity via dot product (vectors are normalized)
            chunk_scores = np.dot(self.document_embeddings, query_embedding.T).flatten()

            # Aggregate chunk scores back to document level using Max Pooling
            doc_scores = np.zeros(len(self.df))
            np.maximum.at(doc_scores, self.chunk_to_doc_idx, chunk_scores)

            top_indices = np.argsort(doc_scores)[::-1][:top_k]
            # Return a dictionary of {doc_id: score} for the top results
            return {idx: doc_scores[idx] for idx in top_indices if doc_scores[idx] > 0}
        except Exception as e:
            # Fail Soft Policy: Vector Search Failure fallback
            print(f"\n[WARNING] Vector search failed ({e}). Falling back to sparse-only.")
            return {}

    def suggest_correction(self, query):
        """
        Error Handling: Uses Levenshtein distance to suggest query corrections
        against the artist and medium vocabulary.
        """
        if not self.vocabulary or not Levenshtein:
            return None

        suggested_query = []
        changed = False
        
        for word in query.split():
            clean_word = re.sub(r'[^a-zA-Z]', '', word.lower())
            if not clean_word or len(clean_word) < 3:
                suggested_query.append(word)
                continue

            if clean_word not in self.vocabulary:
                # Find closest match within distance of 2
                closest_word = min(self.vocabulary, key=lambda w: Levenshtein.distance(clean_word, w))
                if Levenshtein.distance(clean_word, closest_word) <= 2:
                    suggested_query.append(closest_word)
                    changed = True
                else:
                    suggested_query.append(word)
            else:
                suggested_query.append(word)

        if changed:
            return " ".join(suggested_query)
        return None

    @staticmethod
    def _min_max_normalize(scores_dict):
        """Normalizes a dictionary of scores to a 0-1 range."""
        if not scores_dict:
            return {}

        scores = list(scores_dict.values())
        min_score, max_score = min(scores), max(scores)

        # Handle case where all scores are the same to avoid division by zero
        if max_score == min_score:
            return {doc_id: 1.0 for doc_id in scores_dict}

        return {
            doc_id: (score - min_score) / (max_score - min_score)
            for doc_id, score in scores_dict.items()
        }

    def hybrid_search(self, query, top_k=10, alpha=0.5, filters=None):
        """
        Online Phase: Orchestrates dual-retrieval and fuses results using
        Min-Max normalized Weighted Linear Combination, as per the report.
        Applies optional Post-Retrieval Filtering on metadata constraints.
        """
        start_time = time.perf_counter()

        filter_str = f" | [FILTERS] {filters}" if filters else ""
        print(f"\n[QUERY] '{query}'{filter_str}")

        # --- Error Handling: Spelling Correction ---
        suggestion = self.suggest_correction(query)
        if suggestion:
            print(f"        [DID YOU MEAN?] '{suggestion}'?")
        # -------------------------------------------

        # 1. Retrieve raw scores from both pipelines
        sparse_scores = self.search_sparse(query)
        dense_scores = self.search_dense(query)

        # 2. Normalize scores from each pipeline to a 0-1 range
        norm_sparse = self._min_max_normalize(sparse_scores)
        norm_dense = self._min_max_normalize(dense_scores)

        # 3. Fuse scores using Weighted Linear Combination
        fused_scores = {}
        all_ids = set(norm_sparse.keys()) | set(norm_dense.keys())
        for idx in all_ids:
            sparse_s = norm_sparse.get(idx, 0)
            dense_s = norm_dense.get(idx, 0)
            fused_scores[idx] = (alpha * dense_s) + ((1 - alpha) * sparse_s)

        # 4. Sort by fused score and format the output
        ranked_indices = sorted(fused_scores, key=fused_scores.get, reverse=True)
        results = []
        for idx in ranked_indices:
            doc = self.df.iloc[idx]

            # --- Post-Retrieval Filtering ---
            if filters:
                passes_filters = True
                for f_key, f_val in filters.items():
                    if f_key in doc:
                        # Simple case-insensitive substring match
                        if str(f_val).lower() not in str(doc[f_key]).lower():
                            passes_filters = False
                            break
                if not passes_filters:
                    continue
            # --------------------------------

            # --- System Transparency ("Why this result?") ---
            sparse_s = norm_sparse.get(idx, 0)
            dense_s = norm_dense.get(idx, 0)
            reasons = []
            if sparse_s > 0:
                reasons.append(f"Lexical Match (BM25F: {sparse_s:.2f})")
            if dense_s > 0:
                reasons.append(f"Semantic Match (BERT: {dense_s:.2f})")
            reason_str = " + ".join(reasons) if reasons else "Fallback"
            # ------------------------------------------------

            results.append({
                "id": doc["id"], "Rank": len(results) + 1,
                "Title": doc["title"], "Artist": doc["artist"],
                "Description": doc["medium"],
                "Year": doc.get("year", "N/A"),
                "Score": round(fused_scores[idx], 4),
                "Reasons": reason_str,
            })
            
            if len(results) == top_k:
                break

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
        user_input = input("\nEnter search query (optional: use '| key:val' for filters): ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Shutting down search engine.")
            break

        if not user_input:
            continue

        # Parse optional filters (e.g., "landscape | year:1888 | medium:oil")
        parts = user_input.split("|")
        query = parts[0].strip()
        
        filters = {}
        if len(parts) > 1:
            for part in parts[1:]:
                if ":" in part:
                    k, v = part.split(":", 1)
                    filters[k.strip().lower()] = v.strip()

        # Execute search and capture end-to-end interface latency
        cli_start_time = time.perf_counter()
        results = engine.hybrid_search(query, top_k=5, filters=filters)
        cli_latency = time.perf_counter() - cli_start_time

        # Display results utilizing standard search engine formatting
        print(f"\n--- Top {len(results)} Results ({cli_latency:.4f} seconds) ---")
        for res in results:
            desc = (
                res["Description"][:75] + "..."
                if len(res["Description"]) > 75
                else res["Description"]
            )
            year_str = f" | Year: {res['Year']}" if pd.notna(res['Year']) else ""
            print(f"{res['Rank']}. {res['Title']} | Artist: {res['Artist']}{year_str}")
            print(f"   Score: {res['Score']:.4f} | Medium: {desc}")
            print(f"   Why this result?: {res['Reasons']}\n")

if __name__ == "__main__":
    run_cli()
