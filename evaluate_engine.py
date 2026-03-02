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

import time
import numpy as np
from hybrid_search import ArtGallerySearchEngine


def calculate_mrr(ranked_items, target_item):
    """
    Computes the Reciprocal Rank for a single known-item query.

    Parameters:
        ranked_items (list): An ordered list of retrieved document identifiers.
        target_item (str): The exact document identifier that satisfies the query.

    Returns:
        float: 1.0 / rank if the target is found, otherwise 0.0.
    """
    try:
        rank = ranked_items.index(target_item) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0


def calculate_ndcg(ranked_items, relevant_items, k=10):
    """
    Computes the Normalized Discounted Cumulative Gain at rank K (NDCG@K).

    Parameters:
        ranked_items (list): An ordered list of retrieved document identifiers.
        relevant_items (list): A collection of document identifiers deemed relevant to the query.
        k (int): The cutoff rank threshold for the evaluation.

    Returns:
        float: The NDCG@K score bounded between 0.0 and 1.0.
    """
    dcg = 0.0
    idcg = 0.0

    for i, doc_id in enumerate(ranked_items[:k]):
        if doc_id in relevant_items:
            rel_score = 1.0
            dcg += rel_score / np.log2(i + 2)

    ideal_relevant_count = min(len(relevant_items), k)
    for i in range(ideal_relevant_count):
        idcg += 1.0 / np.log2(i + 2)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


if __name__ == "__main__":
    print("\n============================================================")
    print("INITIALIZING SYSTEM EVALUATION PIPELINE")
    print("============================================================")

    engine = ArtGallerySearchEngine("art_gallery_data.csv")

    known_item_qrels = {
        "A Steamer off the Coast": "A Steamer off the Coast",
        "Matisse Reclining Nude": "Reclining Nude II",
        "Wyndham Tryon Fraga": "Fraga",
    }

    semantic_qrels = {
        "a gloomy landscape with swirling clouds": [
            "Clouds",
            "?A Curtained Bed, with a Figure or Figures Reclining",
            "Landscape with a Peasant on a Path",
        ],
        "a portrait of a woman": [
            "Portrait of a Woman",
            "Head of a Woman",
            "Seated Woman",
        ],
    }

    print("\n--- Phase 1: Evaluating Known-Item Retrieval (MRR) ---")
    mrr_scores = []
    mrr_latencies = []

    for query, target_title in known_item_qrels.items():
        start_time = time.perf_counter()
        raw_results = engine.hybrid_search(query, top_k=20)
        query_latency = time.perf_counter() - start_time
        mrr_latencies.append(query_latency)

        retrieved_titles = [res["Title"] for res in raw_results]

        score = calculate_mrr(retrieved_titles, target_title)
        mrr_scores.append(score)

        if target_title in retrieved_titles:
            rank_found = retrieved_titles.index(target_title) + 1
        else:
            rank_found = "Not Found (Top 20)"

        print(f"Query: '{query}' ({query_latency:.4f} seconds)")
        print(f"  -> Target: '{target_title}'")
        print(f"  -> Rank Found: {rank_found}")
        print(f"  -> Reciprocal Rank: {score:.4f}\n")

    mean_mrr = np.mean(mrr_scores) if mrr_scores else 0.0

    print("--- Phase 2: Evaluating Semantic Retrieval (NDCG@10) ---")
    ndcg_scores = []
    ndcg_latencies = []

    for query, relevant_titles in semantic_qrels.items():
        start_time = time.perf_counter()
        raw_results = engine.hybrid_search(query, top_k=10)
        query_latency = time.perf_counter() - start_time
        ndcg_latencies.append(query_latency)

        retrieved_titles = [res["Title"] for res in raw_results]

        score = calculate_ndcg(retrieved_titles, relevant_titles, k=10)
        ndcg_scores.append(score)

        items_retrieved = sum(1 for t in retrieved_titles if t in relevant_titles)

        print(f"Query: '{query}' ({query_latency:.4f} seconds)")
        print(f"  -> Expected Relevant Items: {len(relevant_titles)}")
        print(f"  -> Items Retrieved in Top 10: {items_retrieved}")
        print(f"  -> NDCG@10 Score: {score:.4f}\n")

    mean_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0

    all_latencies = mrr_latencies + ndcg_latencies
    mean_latency = np.mean(all_latencies) * 1000 if all_latencies else 0.0

    print("============================================================")
    print("OVERALL SYSTEM PERFORMANCE METRICS")
    print("============================================================")
    print(f"Mean Reciprocal Rank (MRR):                 {mean_mrr:.4f}")
    print(f"Normalized Discounted Cumulative Gain @ 10: {mean_ndcg:.4f}")
    print(f"Mean Query Latency:                         {mean_latency:.2f} ms")
    print("============================================================")
