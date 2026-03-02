"""
File Name: evaluate_engine.py
Description: Automates the formal offline evaluation of the hybrid search engine.
             Utilizes predefined Query Relevance Judgments (qrels) to compute
             Mean Reciprocal Rank (MRR) for exact known-item retrieval tasks,
             and Normalized Discounted Cumulative Gain (NDCG@K) for ad-hoc
             semantic retrieval tasks. Includes latency measurement.

Theoretical Foundations:
- MRR evaluates the system based on the rank of the first correct answer:
  MRR = 1 / Rank (averaged across all queries)

- NDCG evaluates the ranking quality by penalizing relevant documents that
  appear lower in the result set, normalized against an ideal ranking (IDCG):
  DCG  = Sum of (Relevance Score / log2(Rank + 1))
  NDCG = DCG / Ideal DCG
"""

# Suppress verbose TF INFO logs
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import numpy as np
from hybrid_search import ArtGallerySearchEngine


def calculate_mrr(ranked_ids, target_id):
    """Computes Reciprocal Rank (1/rank) for the first relevant document."""
    try:
        return 1.0 / (ranked_ids.index(target_id) + 1)
    except ValueError:
        return 0.0


def calculate_ndcg(ranked_ids, relevant_ids, k=10):
    """Computes NDCG@K using binary relevance."""
    dcg = sum(
        [
            1.0 / np.log2(i + 2)
            for i, doc_id in enumerate(ranked_ids[:k])
            if doc_id in relevant_ids
        ]
    )
    ideal_count = min(len(relevant_ids), k)
    idcg = sum([1.0 / np.log2(i + 2) for i in range(ideal_count)])
    return dcg / idcg if idcg > 0 else 0.0


def resolve_titles_to_ids(df, titles):
    """Maps normalized titles to unique document IDs for stable evaluation."""
    if isinstance(titles, str):
        titles = [titles]

    # Normalize titles in the dataframe and the input list
    df["title_normalized"] = df["title"].str.lower()
    normalized_titles = [t.lower() for t in titles]

    return df[df["title_normalized"].isin(normalized_titles)]["id"].unique().tolist()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("INITIALIZING SYSTEM EVALUATION PIPELINE")
    print("=" * 60)

    engine = ArtGallerySearchEngine("art_gallery_data.csv")

    # Engine Warm-up: Eliminates first-call latency spikes from BERT initialization
    _ = engine.hybrid_search("warm up", top_k=1)

    # Ground Truth Definitions
    known_item_qrels = {
        "a steamer off the coast": "a steamer off the coast",
        "matisse reclining nude": "reclining nude ii",
        "wyndham tryon fraga": "fraga",
    }

    semantic_qrels = {
        "a gloomy landscape with swirling clouds": [
            "clouds",
            "?a curtained bed, with a figure or figures reclining",
            "landscape with a peasant on a path",
        ],
        "a portrait of a woman": [
            "portrait of a woman",
            "head of a woman",
            "seated woman",
        ],
    }

    mrr_scores, ndcg_scores, latencies = [], [], []

    print("\n--- Phase 1: Known-Item Retrieval (MRR) ---")
    for query, target_title in known_item_qrels.items():
        # Resolve target title to ID for strict matching
        resolved = resolve_titles_to_ids(engine.df, target_title)
        if not resolved:
            print(f"[WARNING] Target '{target_title}' not found in corpus. Skipping.")
            continue
        target_id = resolved[0]

        start = time.perf_counter()
        raw_results = engine.hybrid_search(query, top_k=20)
        qlat = time.perf_counter() - start

        latencies.append(qlat)
        retrieved_ids = [res["id"] for res in raw_results if "id" in res]

        score = calculate_mrr(retrieved_ids, target_id)
        mrr_scores.append(score)
        print(f"Query: '{query}' ({qlat:.4f}s) | MRR: {score:.4f}")

    print("\n--- Phase 2: Semantic Retrieval (NDCG@10) ---")
    for query, rel_titles in semantic_qrels.items():
        relevant_ids = resolve_titles_to_ids(engine.df, rel_titles)

        if not relevant_ids:
            print(f"[WARNING] No relevant docs found for query '{query}'. Skipping.")
            continue

        start = time.perf_counter()
        raw_results = engine.hybrid_search(query, top_k=10)
        qlat = time.perf_counter() - start

        latencies.append(qlat)
        retrieved_ids = [res["id"] for res in raw_results]

        score = calculate_ndcg(retrieved_ids, relevant_ids, k=10)
        ndcg_scores.append(score)
        print(f"Query: '{query}' ({qlat:.4f}s) | NDCG@10: {score:.4f}")

    # Metrics Aggregation
    print("\n" + "=" * 60)
    print("OVERALL SYSTEM PERFORMANCE METRICS")
    print("=" * 60)
    print(f"Mean Reciprocal Rank (MRR):                 {np.mean(mrr_scores):.4f}")
    print(f"Mean NDCG@10:                               {np.mean(ndcg_scores):.4f}")
    print(
        f"Mean Query Latency:                         {np.mean(latencies) * 1000:.2f} ms"
    )
    print("=" * 60)
