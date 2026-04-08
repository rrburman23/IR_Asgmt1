"""
File Name: evaluate_engine.py
Description: Advanced Evaluation Suite for the Tate Gallery Hybrid Search Engine.
             Computes MRR, NDCG@10, and detailed latency metrics.
             Presents results in a colorized, formatted output table.
"""

import time
from typing import List, Union

import numpy as np
import pandas as pd

from hybrid_search import ArtGallerySearchEngine


# Colorized output helper
def color_text(text, color):
    """Wrap text with ANSI color/style escape codes for terminal output."""
    colors = {
        "green": "\033[92m",
        "red": "\033[91m",
        "yellow": "\033[93m",
        "cyan": "\033[96m",
        "bold": "\033[1m",
        "end": "\033[0m",
    }
    return f"{colors.get(color, '')}{text}{colors['end']}"


def calculate_mrr(ranked_ids: list, target_id: int) -> float:
    """Compute reciprocal rank for a known-item query."""
    try:
        return 1.0 / (ranked_ids.index(target_id) + 1)
    except ValueError:
        return 0.0


def calculate_ndcg(ranked_ids: list, relevant_ids: list, k: int = 10) -> float:
    """Calculate Normalized Discounted Cumulative Gain at rank K (binary rel)."""
    dcg = sum(
        [
            1.0 / np.log2(i + 2)
            for i, doc_id in enumerate(ranked_ids[:k])
            if doc_id in relevant_ids
        ]
    )
    # Ideal DCG: best possible ranking
    ideal_count = min(len(relevant_ids), k)
    idcg = sum([1.0 / np.log2(i + 2) for i in range(ideal_count)])
    return dcg / idcg if idcg > 0 else 0.0


def resolve_titles_to_ids(df: pd.DataFrame, titles: Union[str, List[str]]) -> list:
    """Normalize titles and find corresponding IDs in the DataFrame."""
    if isinstance(titles, str):
        titles = [titles]
    normalized = [str(t).strip().lower() for t in titles]
    mask = df["title"].str.strip().str.lower().isin(normalized)
    return df[mask]["id"].unique().tolist()


def print_table(rows, headers):
    """Pretty-print a table of results to the terminal."""
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
    fusion: str = "score",  # 'score' (weighted) or 'rrf'
    alpha: float = 0.6,  # For score fusion
    k_rrf: int = 60,  # For RRF fusion
):
    """
    Main evaluation: runs both known-item (MRR) and semantic (NDCG@10)
    and reports latency. Makes results easy to read.
    """
    # -------- Colorful Banner --------
    print(color_text("██" * 34, "cyan"))
    print(color_text("    TATE SEARCH EVALUATION (Hybrid Engine)    ", "bold"))
    print(color_text("██" * 34, "cyan"))
    print(
        f"Fusion: {fusion} (alpha={alpha})"
        if fusion == "score"
        else f"Fusion: {fusion} (k_rrf={k_rrf})"
    )
    print()

    # ---------- Init Engine ----------
    try:
        engine = ArtGallerySearchEngine(data_path)
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as e:
        print(color_text(f"[FATAL] Engine Init Failed: {e}", "red"))
        return

    # Engine warm-up for VRAM/TF/HF speeds
    _ = engine.hybrid_search("warm up", top_k=1)

    # --------- Ground Truth ----------
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
    available_titles = set(engine.df["title"].str.strip().str.lower().tolist())
    known_item_qrels = {
        q: t
        for q, t in all_known_items.items()
        if str(t).strip().lower() in available_titles
    }

    mrr_scores, ndcg_scores, latencies = [], [], []
    rows_mrr, rows_ndcg = [], []

    # ---- Phase 1: Known-Item MRR ----
    print(color_text("Phase 1: Known-Item Retrieval (MRR)", "bold"))
    for query, target_title in known_item_qrels.items():
        resolved = resolve_titles_to_ids(engine.df, target_title)
        if not resolved:
            msg = color_text(f"[SKIP] Query '{query}' - not found in data.", "yellow")
            print(msg)
            rows_mrr.append([query, target_title, "-", "-", "-", "SKIP"])
            continue
        target_id = resolved[0]

        start = time.perf_counter()
        results = engine.hybrid_search(
            query, top_k=20, fusion=fusion, alpha=alpha, k_rrf=k_rrf
        )
        qlat = (time.perf_counter() - start) * 1000  # ms

        latencies.append(qlat)
        retrieved_ids = [res["id"] for res in results]
        score = calculate_mrr(retrieved_ids, target_id)
        mrr_scores.append(score)
        # Status coloring
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
                retrieved_ids[:5],
                status,
            ]
        )

    print_table(
        rows_mrr, ["Query", "Target", "MRR", "Latency", "Top-5 Retrieved IDs", "Status"]
    )
    print()

    # -- Phase 2: Semantic Discovery NDCG@10 --
    print(color_text("Phase 2: Semantic Discovery (NDCG@10)", "bold"))
    for query, rel_titles in semantic_qrels.items():
        rel_ids = resolve_titles_to_ids(engine.df, rel_titles)
        if not rel_ids:
            msg = color_text(
                f"[SKIP] Query '{query}' - ground truth not in data.", "yellow"
            )
            print(msg)
            rows_ndcg.append([query, rel_titles, "-", "-", "SKIP"])
            continue

        start = time.perf_counter()
        results = engine.hybrid_search(
            query, top_k=10, fusion=fusion, alpha=alpha, k_rrf=k_rrf
        )
        qlat = (time.perf_counter() - start) * 1000
        latencies.append(qlat)
        retrieved_ids = [res["id"] for res in results]
        score = calculate_ndcg(retrieved_ids, rel_ids, k=10)
        ndcg_scores.append(score)
        status = (
            color_text("Rel found", "green") if score > 0 else color_text("Miss", "red")
        )
        rows_ndcg.append([query, rel_titles, f"{score:.4f}", f"{qlat:.2f}ms", status])

    print_table(rows_ndcg, ["Query", "Relevant Titles", "NDCG@10", "Latency", "Status"])

    # ---- Final System Report ----
    print(color_text("\nFINAL SYSTEM PERFORMANCE REPORT", "bold"))
    print(color_text("=" * 46, "cyan"))
    mrr_mean = np.mean(mrr_scores) if mrr_scores else 0
    ndcg_mean = np.mean(ndcg_scores) if ndcg_scores else 0
    mean_latency = np.mean(latencies) if latencies else 0
    p95_latency = np.percentile(latencies, 95) if latencies else 0

    print(color_text(f"MRR (Known-Item Accuracy):     {mrr_mean:.4f}", "bold"))
    print(color_text(f"NDCG@10 (Semantic Quality):    {ndcg_mean:.4f}", "bold"))
    print(color_text(f"Mean Latency:                  {mean_latency:.2f} ms", "cyan"))
    print(color_text(f"P95 Latency:                   {p95_latency:.2f} ms", "cyan"))
    print(color_text("=" * 46, "cyan"))


if __name__ == "__main__":
    # Run with score fusion; for RRF, use fusion="rrf", k_rrf=30 or 60
    # For example: run_evaluation(fusion='rrf', k_rrf=30)
    run_evaluation(fusion="score", alpha=0.6)
