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

import os
import time
import numpy as np

from hybrid_search import ArtGallerySearchEngine

# Suppress verbose TF INFO logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


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

    # Use a temporary copy to avoid SettingWithCopyWarning on the original engine DF
    temp_df = df.copy()
    temp_df["title_normalized"] = temp_df["title"].str.lower()
    normalized_titles = [t.lower() for t in titles]

    return (
        temp_df[temp_df["title_normalized"].isin(normalized_titles)]["id"]
        .unique()
        .tolist()
    )


def run_evaluation(data_path="art_gallery_data.csv"):
    """Primary entry point for executing the evaluation suite."""
    print("\n" + "=" * 60)
    print("INITIALIZING SYSTEM EVALUATION PIPELINE")
    print("=" * 60)

    # Initialize Engine
    try:
        engine = ArtGallerySearchEngine(data_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] Could not initialize engine: {e}")
        return

    # Engine Warm-up: Eliminates first-call latency spikes
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
        "impressionist landscapes": [
            "landscape", "impressionist", "water lilies", "garden"
        ],
        "religious figures and angels": [
            "angel", "christ", "madonna", "saints"
        ],
        "modernist cars": [], # Negative test (should yield poor matches)
    }

    mrr_scores, ndcg_scores, latencies = [], [], []

    print("\n--- Phase 1: Known-Item Retrieval (MRR) ---")
    for query, target_title in known_item_qrels.items():
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
        print(
            f"Query: '{query:<30}' | Latency: {qlat * 1000:6.2f}ms | MRR: {score:.4f}"
        )

    print("\n--- Phase 2: Semantic Retrieval (NDCG@10) ---")
    for query, rel_titles in semantic_qrels.items():
        # Negative test handling
        if not rel_titles:
            start = time.perf_counter()
            raw_results = engine.hybrid_search(query, top_k=10)
            qlat = time.perf_counter() - start
            latencies.append(qlat)
            print(f"Query: '{query:<30}' | Latency: {qlat * 1000:6.2f}ms | NEGATIVE TEST")
            continue

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
        print(
            f"Query: '{query:<30}' | Latency: {qlat * 1000:6.2f}ms | NDCG@10: {score:.4f}"
        )

    # Metrics Aggregation
    print("\n" + "=" * 60)
    print("OVERALL SYSTEM PERFORMANCE METRICS")
    print("=" * 60)
    if mrr_scores:
        print(f"Mean Reciprocal Rank (MRR):                 {np.mean(mrr_scores):.4f}")
    if ndcg_scores:
        print(f"Mean NDCG@10:                               {np.mean(ndcg_scores):.4f}")
    if latencies:
        print(
            f"Mean Query Latency:                         {np.mean(latencies) * 1000:.2f} ms"
        )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_evaluation()
