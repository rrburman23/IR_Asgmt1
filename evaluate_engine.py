"""
File Name: evaluate_engine.py
Description: Comprehensive evaluation suite for dual-index hybrid retrieval.
    - Updated for 69k Corpus: Handles duplicate titles and multi-ID ground truths.
    - MRR and NDCG@10 metrics.
"""

import time
from typing import List, Union

import numpy as np
import pandas as pd

from hybrid_search import ArtGallerySearchEngine


def color_text(text, color):
    """Colorizes output for CLI readability."""
    colors = {
        "green": "\033[92m",
        "red": "\033[91m",
        "yellow": "\033[93m",
        "cyan": "\033[96m",
        "bold": "\033[1m",
        "end": "\033[0m",
    }
    return f"{colors.get(color, '')}{text}{colors['end']}"


def calculate_mrr(ranked_ids: list, target_ids: list) -> float:
    """
    Computes Mean Reciprocal Rank.
    FIXED: Now checks if *any* valid target ID is in the ranked list,
    solving the duplicate title problem in large datasets.
    """
    for rank, doc_id in enumerate(ranked_ids):
        if doc_id in target_ids:
            return 1.0 / (rank + 1)
    return 0.0


def calculate_ndcg(ranked_ids: list, relevant_ids: list, k: int = 10) -> float:
    """Computes Normalized Discounted Cumulative Gain at rank k."""
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, doc_id in enumerate(ranked_ids[:k])
        if doc_id in relevant_ids
    )
    ideal_count = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_count))
    return dcg / idcg if idcg > 0 else 0.0


def calculate_concept_ndcg(
    retrieved_docs: list, relevant_concepts: list, k: int = 10
) -> float:
    """
    Computes NDCG by checking if the retrieved artwork's title or medium
    contains any of the target semantic concepts.
    """
    dcg = 0.0
    for i, doc in enumerate(retrieved_docs[:k]):
        # Combine title and medium to check for concept presence
        text_blob = f"{doc.get('Title', '')} {doc.get('Medium', '')}".lower()

        # If any of the relevant concepts are in the text blob, it's a hit!
        if any(concept.lower() in text_blob for concept in relevant_concepts):
            dcg += 1.0 / np.log2(i + 2)

    # Ideal DCG assumes all K spots are relevant
    idcg = sum(1.0 / np.log2(i + 2) for i in range(k))
    return dcg / idcg if idcg > 0 else 0.0


def resolve_titles_to_ids(df: pd.DataFrame, titles: Union[str, List[str]]) -> list:
    """Normalize titles and find ALL corresponding IDs in the DataFrame."""
    if isinstance(titles, str):
        titles = [titles]
    normalized = [str(t).strip().lower() for t in titles]
    mask = df["title"].str.strip().str.lower().isin(normalized)
    return [str(i) for i in df[mask]["id"].unique().tolist()]


def print_table(rows, headers):
    """Pretty-prints a result table."""
    col_widths = [
        max(len(str(row[i])) for row in rows + [headers]) + 2
        for i in range(len(headers))
    ]

    def fmt_row(row):
        return "|".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))

    print(color_text("-" * (sum(col_widths) + len(headers) - 1), "cyan"))
    print(color_text(fmt_row(headers), "bold"))
    print(color_text("-" * (sum(col_widths) + len(headers) - 1), "cyan"))
    for row in rows:
        print(fmt_row(row))
    print(color_text("-" * (sum(col_widths) + len(headers) - 1), "cyan"))


def run_evaluation(
    data_path: str = "art_gallery_data.csv",
    k_rrf: int = 60,
    top_k: int = 10,
):
    """Executes the IR evaluation suite."""
    print(color_text("═" * 70, "cyan"))
    print(
        color_text(
            "     TATE SEARCH EVALUATION — HYBRID SEARCH ENGINE     ", "bold"
        )
    )
    print(color_text("═" * 70, "cyan"))
    print(f"Fusion: Reciprocal Rank Fusion (RRF), k_rrf={k_rrf}\n")

    try:
        engine = ArtGallerySearchEngine(data_path)
    except Exception as e:
        print(color_text(f"[FATAL] Engine Init Failed: {e}", "red"))
        return

    _ = engine.hybrid_search("warm up", top_k=1, k_rrf=k_rrf)

    known_items = {
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

    mrr_scores, ndcg_scores, latencies = [], [], []
    rows_mrr, rows_ndcg = [], []

    # ---- Phase 1: Known-Item Retrieval (MRR) ----
    print(color_text("\nPhase 1: Known-Item Retrieval (MRR)", "bold"))
    for query, target_title in known_items.items():
        # Get ALL IDs that match this title
        target_ids = resolve_titles_to_ids(engine.df, target_title)
        if not target_ids:
            print(color_text(f"[SKIP] Query '{query}' - target not in data.", "yellow"))
            rows_mrr.append([query, target_title, "-", "-", "-", "SKIP"])
            continue

        start = time.perf_counter()
        results = engine.hybrid_search(query, top_k=top_k, k_rrf=k_rrf)
        qlat = (time.perf_counter() - start) * 1000
        latencies.append(qlat)

        retrieved_ids = [str(res["id"]) for res in results]
        score = calculate_mrr(retrieved_ids, target_ids)
        mrr_scores.append(score)

        if score > 0.5:
            status = color_text("PASS", "green")
        elif score == 0:
            status = color_text("FAIL", "red")
        else:
            status = color_text("WEAK", "yellow")

        rows_mrr.append(
            [
                query,
                target_title,
                f"{score:.4f}",
                f"{qlat:.2f}ms",
                retrieved_ids[:5] if retrieved_ids else [],
                status,
            ]
        )

    print_table(
        rows_mrr, ["Query", "Target", "MRR", "Latency", "Top-5 Retrieved IDs", "Status"]
    )

    # ---- Phase 2: Semantic Discovery (NDCG@10) ----
    print(color_text("\nPhase 2: Semantic Discovery (NDCG@10)", "bold"))
    for query, rel_titles in semantic_qrels.items():
        # Broaden the semantic net: gather all IDs that match these concepts
        rel_ids = resolve_titles_to_ids(engine.df, rel_titles)
        if not rel_ids:
            print(
                color_text(
                    f"[SKIP] Query '{query}' - no ground truth in data.", "yellow"
                )
            )
            rows_ndcg.append([query, rel_titles, "-", "-", "SKIP"])
            continue

        start = time.perf_counter()
        results = engine.hybrid_search(query, top_k=top_k, k_rrf=k_rrf)
        qlat = (time.perf_counter() - start) * 1000
        latencies.append(qlat)

        retrieved_ids = [str(res["id"]) for res in results]
        # score = calculate_ndcg(retrieved_ids, rel_ids, k=top_k)
        score = calculate_concept_ndcg(results, rel_titles, k=top_k)
        ndcg_scores.append(score)

        status = (
            color_text("Rel found", "green") if score > 0 else color_text("Miss", "red")
        )
        rows_ndcg.append([query, rel_titles, f"{score:.4f}", f"{qlat:.2f}ms", status])

    print_table(rows_ndcg, ["Query", "Relevant Titles", "NDCG@10", "Latency", "Status"])

    # ---- Final System Report ----
    print(color_text("\nFINAL SYSTEM PERFORMANCE REPORT", "bold"))
    print(color_text("=" * 60, "cyan"))
    mrr_mean = np.mean(mrr_scores) if mrr_scores else 0
    ndcg_mean = np.mean(ndcg_scores) if ndcg_scores else 0
    mean_latency = np.mean(latencies) if latencies else 0
    p95_latency = np.percentile(latencies, 95) if latencies else 0

    print(color_text(f"MRR (Known-Item Accuracy):     {mrr_mean:.4f}", "bold"))
    print(color_text(f"NDCG@10 (Semantic Quality):    {ndcg_mean:.4f}", "bold"))
    print(color_text(f"Mean Latency:                  {mean_latency:.2f} ms", "cyan"))
    print(color_text(f"P95 Latency:                   {p95_latency:.2f} ms", "cyan"))
    print(color_text("=" * 60, "cyan"))


if __name__ == "__main__":
    run_evaluation(k_rrf=60, top_k=10)
