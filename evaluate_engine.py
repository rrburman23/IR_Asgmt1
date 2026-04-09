"""
File Name: evaluate_engine.py
Description: Configurable evaluation suite for hybrid retrieval.

Why this version:
- Your result objects contain a synthetic Description, so semantic evaluation must
  use the corpus text in engine.df (semantic_blob/search_* fields) keyed by doc id.
- Configurable evaluation parameters via EvalConfig.

Metrics
Known-item:
- MRR@K
- Success@K (hit in top K)

Semantic discovery (binary relevance derived from concepts OR titles):
- NDCG@K (binary)
- Precision@K  (P@K = sum(rel[:K]) / K)
- MAP@K
- Success@K

Latency:
- Cold latency (ms): first execution of each query
- Warm latency (ms): immediate second execution of each query (more stable)
"""

from __future__ import annotations

import contextlib
import io
import time
from dataclasses import dataclass
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

from hybrid_search import ArtGallerySearchEngine

SemanticMode = Literal["concept", "titles"]


@dataclass(frozen=True)
class EvalConfig:
    data_path: str = "art_gallery_data.csv"
    top_k: int = 10
    k_rrf: int = 60

    # Pass/fail thresholds
    pass_mrr_threshold: float = 0.5
    pass_ndcg_threshold: float = 0.5

    # Semantic evaluation behavior
    semantic_mode: SemanticMode = "concept"
    semantic_min_concept_hits: int = 1
    semantic_min_hits_by_query: dict[str, int] | None = None

    # Which corpus fields to use for semantic relevance (must exist in CSV)
    semantic_text_fields: tuple[str, ...] = (
        "title",
        "artist",
        "medium",
        "semantic_blob",
    )

    # Whether to print tables
    verbose: bool = True


# -----------------------------
# Helpers: Engine result parsing
# -----------------------------
def unwrap_results(engine_response: Any) -> list[dict[str, Any]]:
    """
    Accept either:
    - {"results": [ ...dicts... ], ...}
    - [ ...dicts... ]
    """
    if engine_response is None:
        return []
    if isinstance(engine_response, dict):
        results = engine_response.get("results", [])
        return results if isinstance(results, list) else []
    if isinstance(engine_response, list):
        return engine_response
    return []


def ensure_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    return "" if s.lower() == "nan" else s


def build_id_to_text(df: pd.DataFrame, fields: Iterable[str]) -> dict[str, str]:
    """
    Build a map: doc_id (string) -> lowercased text blob from df columns.
    Uses df, not GUI result dict, so semantics evaluate correctly.
    """
    fields = list(fields)
    missing = [f for f in fields if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required fields in df for semantic eval: {missing}")

    id_to_text: dict[str, str] = {}
    for _, row in df.iterrows():
        doc_id = ensure_str(row.get("id", ""))
        parts = [ensure_str(row.get(f, "")) for f in fields]
        id_to_text[doc_id] = " ".join(parts).lower()
    return id_to_text


def timed_search(engine: Any, query: str, top_k: int, k_rrf: int) -> tuple[Any, float]:
    """
    Runs engine.hybrid_search and returns (response, elapsed_ms).
    Works even if engine.hybrid_search doesn't accept k_rrf.
    """
    start = time.perf_counter()
    try:
        resp = engine.hybrid_search(query, top_k=top_k, k_rrf=k_rrf)
    except TypeError:
        resp = engine.hybrid_search(query, top_k=top_k)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return resp, elapsed_ms


# -----------------------------
# Metrics
# -----------------------------
def calculate_mrr(ranked_ids: list[str], target_ids: set[str]) -> float:
    for rank, doc_id in enumerate(ranked_ids):
        if doc_id in target_ids:
            return 1.0 / (rank + 1)
    return 0.0


def success_at_k(ranked_ids: list[str], target_ids: set[str], k: int) -> float:
    return 1.0 if any(doc_id in target_ids for doc_id in ranked_ids[:k]) else 0.0


def precision_at_k(rel: list[int], k: int) -> float:
    rel_k = rel[:k]
    return float(sum(rel_k)) / float(k) if k > 0 else 0.0


def average_precision_at_k(rel: list[int], k: int) -> float:
    rel_k = rel[:k]
    hits = 0
    s = 0.0
    for i, r in enumerate(rel_k, start=1):
        if r:
            hits += 1
            s += hits / i
    return s / hits if hits > 0 else 0.0


def ndcg_at_k_binary(rel: list[int], k: int) -> float:
    rel_k = rel[:k]
    dcg = 0.0
    for i, r in enumerate(rel_k):
        if r:
            dcg += 1.0 / np.log2(i + 2)

    ideal_rel_k = sorted(rel_k, reverse=True)
    idcg = 0.0
    for i, r in enumerate(ideal_rel_k):
        if r:
            idcg += 1.0 / np.log2(i + 2)

    return (dcg / idcg) if idcg > 0 else 0.0


# -----------------------------
# Ground truth mapping (Known-item)
# -----------------------------
def resolve_titles_to_ids(df: pd.DataFrame, target_query: str) -> set[str]:
    normalized_query = target_query.lower().replace("-", " ")
    titles = (
        df["title"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace("-", " ", regex=False)
    )
    mask = titles.str.contains(normalized_query, na=False, regex=False)
    return set(ensure_str(i) for i in df.loc[mask, "id"].astype(str).unique().tolist())


# -----------------------------
# Semantic relevance builders
# -----------------------------
def relevance_from_concepts_by_id(
    ranked_ids: list[str],
    id_to_text: dict[str, str],
    concepts: list[str],
    min_hits: int,
) -> list[int]:
    concepts_l = [c.lower() for c in concepts]
    rel: list[int] = []
    for doc_id in ranked_ids:
        blob = id_to_text.get(doc_id, "")
        hits = sum(1 for c in concepts_l if c in blob)
        rel.append(1 if hits >= min_hits else 0)
    return rel


def relevance_from_titles_by_id(
    ranked_ids: list[str], relevant_ids: set[str]
) -> list[int]:
    return [1 if doc_id in relevant_ids else 0 for doc_id in ranked_ids]


# -----------------------------
# Reporting
# -----------------------------
def print_table(rows: list[list[Any]], headers: list[str]) -> None:
    str_rows = [[str(c) for c in r] for r in rows]
    str_headers = [str(h) for h in headers]

    col_widths = [
        max(len(str_headers[i]), max(len(r[i]) for r in str_rows)) + 2
        for i in range(len(headers))
    ]

    def fmt_row(r: list[str]) -> str:
        return "|".join(r[i].ljust(col_widths[i]) for i in range(len(headers)))

    line = "-" * (sum(col_widths) + len(headers) - 1)
    print(line)
    print(fmt_row(str_headers))
    print(line)
    for r in str_rows:
        print(fmt_row(r))
    print(line)


def _print_metric(label: str, value: str, width: int = 34) -> None:
    print(f"{label:<{width}} {value}")


# -----------------------------
# Main evaluation
# -----------------------------
def run_evaluation(config: EvalConfig | None = None) -> None:
    config = config or EvalConfig()

    print("═" * 70)
    print("TATE SEARCH EVALUATION — CONFIGURABLE STRICT HYBRID")
    print("═" * 70)
    print(
        f"[CONFIG] top_k={config.top_k} | k_rrf={config.k_rrf} | "
        f"semantic_mode={config.semantic_mode} | min_hits={config.semantic_min_concept_hits}"
    )

    engine = ArtGallerySearchEngine(config.data_path)
    id_to_text = build_id_to_text(engine.df, config.semantic_text_fields)

    # Warm up (model/index warm-up)
    _ = timed_search(engine, "warmup", top_k=1, k_rrf=config.k_rrf)

    known_items = [
        "a steamer off the coast",
        "river scene",
        "self-portrait",
        "castle on rock",
    ]

    semantic_concepts = {
        "mountain scenery": [
            "mountain",
            "cliff",
            "rock",
            "alps",
            "peak",
            "ridge",
            "summit",
            "crag",
            "hill",
            "landscape",
        ],
        "ocean and boats": [
            "ship",
            "boat",
            "sail",
            "coast",
            "sea",
            "ocean",
            "water",
            "marine",
            "harbour",
            "harbor",
        ],
        "atmospheric studies": [
            "sky",
            "cloud",
            "sunset",
            "storm",
            "mist",
            "fog",
            "dusk",
            "dawn",
            "watercolour",
            "watercolor",
            "light",
            "atmosphere",
        ],
    }

    semantic_title_qrels: dict[str, list[str]] = {}  # optional

    # Aggregates (cold + warm)
    cold_latencies_ms: list[float] = []
    warm_latencies_ms: list[float] = []

    # Known-item aggregates
    mrr_scores: list[float] = []
    s_at_k_known: list[float] = []

    # Semantic aggregates
    ndcg_scores: list[float] = []
    p_at_k_scores: list[float] = []
    map_at_k_scores: list[float] = []
    s_at_k_sem: list[float] = []

    rows_mrr: list[list[Any]] = []
    rows_sem: list[list[Any]] = []

    print("\nPhase 1: Known-Item Retrieval")
    for query in known_items:
        target_ids = resolve_titles_to_ids(engine.df, query)
        user_query = query.replace("-", " ").replace(" on rock", " on a rock")

        # cold + warm runs
        resp_cold, cold_ms = timed_search(
            engine, user_query, config.top_k, config.k_rrf
        )
        resp_warm, warm_ms = timed_search(
            engine, user_query, config.top_k, config.k_rrf
        )

        cold_latencies_ms.append(cold_ms)
        warm_latencies_ms.append(warm_ms)

        results = unwrap_results(resp_warm)
        ranked_ids = [ensure_str(r.get("id", "")) for r in results if "id" in r]

        mrr = calculate_mrr(ranked_ids, target_ids)
        succ = success_at_k(ranked_ids, target_ids, config.top_k)

        mrr_scores.append(mrr)
        s_at_k_known.append(succ)

        status = "PASS" if mrr >= config.pass_mrr_threshold else "FAIL"
        rows_mrr.append(
            [
                user_query,
                query,
                f"{mrr:.4f}",
                f"{succ:.0f}",
                f"{cold_ms:.2f}ms",
                f"{warm_ms:.2f}ms",
                str(ranked_ids[:5]),
                status,
            ]
        )

    print_table(
        rows_mrr,
        [
            "Query",
            "Target",
            "MRR",
            f"S@{config.top_k}",
            "Cold",
            "Warm",
            "Top-5 IDs",
            "Status",
        ],
    )

    print("\nPhase 2: Semantic Discovery")
    for query, concepts in semantic_concepts.items():
        resp_cold, cold_ms = timed_search(engine, query, config.top_k, config.k_rrf)
        resp_warm, warm_ms = timed_search(engine, query, config.top_k, config.k_rrf)

        cold_latencies_ms.append(cold_ms)
        warm_latencies_ms.append(warm_ms)

        results = unwrap_results(resp_warm)
        ranked_ids = [ensure_str(r.get("id", "")) for r in results if "id" in r]

        # allow per-query override of min_hits
        min_hits = config.semantic_min_concept_hits
        if (
            config.semantic_min_hits_by_query
            and query in config.semantic_min_hits_by_query
        ):
            min_hits = config.semantic_min_hits_by_query[query]

        if config.semantic_mode == "concept":
            rel = relevance_from_concepts_by_id(
                ranked_ids,
                id_to_text=id_to_text,
                concepts=concepts,
                min_hits=min_hits,
            )
        else:
            titles = semantic_title_qrels.get(query, [])
            relevant_ids: set[str] = set()
            if titles:
                for t in titles:
                    relevant_ids |= resolve_titles_to_ids(engine.df, t)
            rel = relevance_from_titles_by_id(ranked_ids, relevant_ids)

        ndcg = ndcg_at_k_binary(rel, config.top_k)
        p_at_k = precision_at_k(rel, config.top_k)
        map_at_k = average_precision_at_k(rel, config.top_k)
        succ = 1.0 if any(rel[: config.top_k]) else 0.0

        ndcg_scores.append(ndcg)
        p_at_k_scores.append(p_at_k)
        map_at_k_scores.append(map_at_k)
        s_at_k_sem.append(succ)

        status = "PASS" if ndcg >= config.pass_ndcg_threshold else "FAIL"
        rows_sem.append(
            [
                query,
                f"{min_hits}",
                f"{ndcg:.4f}",
                f"{p_at_k:.4f}",
                f"{map_at_k:.4f}",
                f"{succ:.0f}",
                f"{cold_ms:.2f}ms",
                f"{warm_ms:.2f}ms",
                status,
            ]
        )

    print_table(
        rows_sem,
        [
            "Query",
            "min_hits",
            f"NDCG@{config.top_k}",
            f"P@{config.top_k}",
            f"MAP@{config.top_k}",
            f"S@{config.top_k}",
            "Cold",
            "Warm",
            "Status",
        ],
    )

    print("\nFINAL SYSTEM PERFORMANCE REPORT")
    print("=" * 60)
    if mrr_scores:
        _print_metric(f"MRR@{config.top_k}:", f"{np.mean(mrr_scores):.4f}")
        _print_metric(
            f"Success@{config.top_k} (Known-item):",
            f"{np.mean(s_at_k_known):.4f}",
        )
    if ndcg_scores:
        _print_metric(f"Mean NDCG@{config.top_k}:", f"{np.mean(ndcg_scores):.4f}")
        _print_metric(
            f"Mean Precision@{config.top_k}:", f"{np.mean(p_at_k_scores):.4f}"
        )
        _print_metric(f"Mean MAP@{config.top_k}:", f"{np.mean(map_at_k_scores):.4f}")
        _print_metric(
            f"Success@{config.top_k} (Semantic):", f"{np.mean(s_at_k_sem):.4f}"
        )

    if cold_latencies_ms:
        _print_metric("Mean Cold Latency:", f"{np.mean(cold_latencies_ms):.2f} ms")
    if warm_latencies_ms:
        _print_metric("Mean Warm Latency:", f"{np.mean(warm_latencies_ms):.2f} ms")
    print("=" * 60)


def run_evaluation_to_text(*args, **kwargs) -> str:
    """
    GUI helper: captures run_evaluation() stdout/stderr and returns it as text.

    Usage patterns:
      run_evaluation_to_text()  -> uses defaults
      run_evaluation_to_text(config=EvalConfig(...))
      run_evaluation_to_text(top_k=20, k_rrf=80, ...)  -> convenience overrides
    """
    config = kwargs.pop("config", None)

    # Convenience: allow overrides without requiring caller to build EvalConfig
    if config is None and kwargs:
        config = EvalConfig(**kwargs)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        run_evaluation(config=config)
    return buf.getvalue()


if __name__ == "__main__":
    # CLI usage remains supported
    run_evaluation(
        EvalConfig(
            top_k=10,
            k_rrf=60,
            semantic_mode="concept",
            semantic_min_concept_hits=2,
            semantic_min_hits_by_query={"mountain scenery": 1},
        )
    )
