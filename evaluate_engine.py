"""
File Name: evaluate_engine.py
Description: Advanced Evaluation Suite for the Tate Gallery Hybrid Search Engine.
             Computes MRR, NDCG@10, and p99 Latency.
             Optimized for the RTX 5070 Ti compute environment.

Refined Metric Definitions:
- MRR (Known-Item): Focuses on the 'Entry Point'—how quickly can a user find
  a specific, uniquely identified artwork?
- NDCG (Ad-hoc Semantic): Focuses on 'Discovery'—given a broad concept, how
  well does the engine cluster relevant results at the top of the list?
"""

import os
import time
from typing import List, Union
import numpy as np
import pandas as pd


from hybrid_search import ArtGallerySearchEngine

# Suppress TF and HF logs for clean evaluation output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def calculate_mrr(ranked_ids: list, target_id: int) -> float:
    """
    Calculates the Reciprocal Rank for a specific known-item.
    Formula: 1 / rank_i
    """
    try:
        return 1.0 / (ranked_ids.index(target_id) + 1)
    except ValueError:
        return 0.0


def calculate_ndcg(ranked_ids: list, relevant_ids: list, k: int = 10) -> float:
    """
    Calculates the Normalized Discounted Cumulative Gain at rank K.
    Uses binary relevance (1 for match, 0 for miss).
    """
    # Cumulative Gain with Logarithmic Discounting
    dcg = sum(
        [
            1.0 / np.log2(i + 2)
            for i, doc_id in enumerate(ranked_ids[:k])
            if doc_id in relevant_ids
        ]
    )

    # Ideal DCG calculation for normalization
    ideal_count = min(len(relevant_ids), k)
    idcg = sum([1.0 / np.log2(i + 2) for i in range(ideal_count)])

    return dcg / idcg if idcg > 0 else 0.0


def resolve_titles_to_ids(df: pd.DataFrame, titles: Union[str, List[str]]) -> list:
    """
    Converts human-readable titles into internal database IDs.
    Handles multi-title relevance sets for Phase 2 evaluation.
    """
    if isinstance(titles, str):
        titles = [titles]

    normalized_search = [t.lower().strip() for t in titles]

    # Efficient vectorized filtering
    mask = df["title"].str.lower().str.strip().isin(normalized_search)
    return df[mask]["id"].unique().tolist()


def run_evaluation(data_path: str = "art_gallery_data.csv"):
    """
    Core evaluation loop. Performs two-phase testing:
    Exact Match and Semantic Discovery.
    """
    print("\n" + "█" * 65)
    print("  TATE SEARCH EVALUATION v2.0 | HARDWARE: RTX 5070 Ti")
    print("█" * 65)

    # 1. Initialization
    try:
        engine = ArtGallerySearchEngine(data_path)
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"[FATAL] Engine Init Failed: {e}")
        return

    # Engine Warm-up (Ensures GPU kernels are resident in VRAM)
    _ = engine.hybrid_search("warm up", top_k=1)

    # 2. Ground Truth Configuration
    # Adjusted to match the specific high-frequency titles in your 10k slice
    all_known_items = {
        "a steamer off the coast": "a steamer off the coast",
        "river scene": "river scene",
        "self portrait": "self-portrait",
        "castle on a rock": "castle on rock",
    }

    semantic_qrels = {
        "mountain scenery": [
            "mountains",
            "cliffs",
            "rocks",
            "mountain scenery and buildings",
        ],
        "ocean and boats": [
            "shipping",
            "boats",
            "sailing vessel(s)",
            "coast scene",
            "on coast",
        ],
        "atmospheric studies": [
            "study of sky",
            "study of clouds",
            "study of a cloudy sky",
            "sky study",
        ],
    }

    # Filter targets against available data to ensure fair scoring
    available_titles = set(engine.df["title"].str.lower().str.strip().tolist())
    known_item_qrels = {
        q: t for q, t in all_known_items.items() if t.lower() in available_titles
    }

    mrr_scores, ndcg_scores, latencies = [], [], []

    # 3. Phase 1: Known-Item Retrieval (Precision Test)
    print(f"\n--- Phase 1: Known-Item Retrieval ({len(known_item_qrels)} Targets) ---")
    for query, target_title in known_item_qrels.items():
        resolved = resolve_titles_to_ids(engine.df, target_title)
        if not resolved:
            continue
        target_id = resolved[0]

        start = time.perf_counter()
        results = engine.hybrid_search(query, top_k=20)
        qlat = time.perf_counter() - start

        latencies.append(qlat)
        retrieved_ids = [res["id"] for res in results]

        score = calculate_mrr(retrieved_ids, target_id)
        mrr_scores.append(score)

        status = "PASSED" if score > 0.5 else "FAIL" if score == 0 else "WEAK"
        print(
            f"[{status:<6}] Query: '{query:<25}' | MRR: {score:.4f} | {qlat * 1000:5.2f}ms"
        )

    # 4. Phase 2: Semantic Discovery (Recall Test)
    print("\n--- Phase 2: Semantic Discovery (NDCG@10) ---")
    for query, rel_titles in semantic_qrels.items():
        rel_ids = resolve_titles_to_ids(engine.df, rel_titles)
        if not rel_ids:
            print(f"[SKIP] Query '{query}' - Ground truth not in 10k sample.")
            continue

        start = time.perf_counter()
        results = engine.hybrid_search(query, top_k=10)
        qlat = time.perf_counter() - start

        latencies.append(qlat)
        retrieved_ids = [res["id"] for res in results]

        score = calculate_ndcg(retrieved_ids, rel_ids, k=10)
        ndcg_scores.append(score)
        print(f"[INFO] Query: '{query:<25}' | NDCG: {score:.4f} | {qlat * 1000:5.2f}ms")

    # 5. Final Report Aggregation
    print("\n" + "=" * 65)
    print(" FINAL SYSTEM PERFORMANCE REPORT")
    print("=" * 65)
    print(f"MRR (Known-Item Accuracy):     {np.mean(mrr_scores):.4f}")
    print(f"NDCG@10 (Semantic Quality):    {np.mean(ndcg_scores):.4f}")
    print(f"Mean Latency (5070 Ti):        {np.mean(latencies) * 1000:.2f} ms")
    print(
        f"P95 Latency:                   {np.percentile(latencies, 95) * 1000:.2f} ms"
    )
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_evaluation()
