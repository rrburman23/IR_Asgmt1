# Art Gallery Hybrid Search Engine

This repository contains the codebase for Assignment 1 (Information Retrieval ECS736P/U). It implements a dual-pipeline search engine for a curated corpus of Tate Gallery artwork, following a practical IR architecture with sparse retrieval, dense retrieval, and RRF fusion.

## Project Structure

```text
.
├── ingest_data.py        # ETL pipeline (download, clean, normalize CSV)
├── hybrid_search.py      # Core search engine: BM25 + Dense + RRF
├── main.py               # GUI/CLI entry point, auto-ingest, mode router
├── evaluate_engine.py    # Evaluation (MRR, NDCG@10)
├── test_engine.py        # Unit and latency tests
├── requirements.txt      # Python dependencies (CPU/CUDA compatible)
└── art_gallery_data.csv  # Local search-ready dataset
```

## Architecture Overview

- Sparse indexing: `rank_bm25` over title/artist fields for known-item lookup.
- Dense indexing: `sentence-transformers` (`multi-qa-MiniLM-L6-cos-v1`) for semantic retrieval.
- Fusion: Reciprocal Rank Fusion (RRF).
- Curator boost: optional business weighting to promote finalized masterpieces and demote archival fragments.
- Interfaces: PyQt6 GUI and CLI, with ETL, tests, and evaluation scripts.

## Logical System Architecture

```text
OFFLINE PIPELINE
Remote CSV Source -> ingest_data.py -> BM25 Index + Dense Embeddings -> art_gallery_data.csv

ONLINE PIPELINE
User (GUI/CLI) -> Query Processing -> Sparse Retrieval + Dense Retrieval -> RRF Fusion -> Top-K Results
```

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                            OFFLINE PIPELINE                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
┌───────────────────────┐             ▼             ┌───────────────────────┐
│  Remote Data Source   │──────────────────────────▶│    ingest_data.py     │
│  (GitHub CSV File)    │                           │ (ETL & Normalization) │
└───────────────────────┘                           └───────────┬───────────┘
                                                                │
                                                                ▼
                                                    ┌───────────────────────┐
                                                    │    Document Store     │
                                                    │ (art_gallery_data.csv)│
                                                    └─────┬───────────┬─────┘
                                                          │           │
                               ┌──────────────────────────┘           └──────────────────────────┐
                               ▼                                                                 ▼
                 ┌───────────────────────────┐                                     ┌───────────────────────────┐
                 │   Sparse Index Builder    │                                     │    Dense Index Builder    │
                 │   Model: BM25Okapi        │                                     │   Model: MiniLM-L6-v2     │
                 │   Target: Title, Artist   │                                     │   Target: Description     │
                 └───────────────────────────┘                                     └───────────────────────────┘

===============================================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                             ONLINE PIPELINE                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │    User Interface     │
                          │   (Interactive CLI)   │
                          └───────────┬───────────┘
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │    Query Processor    │
                          │  (Tokenize / Embed)   │
                          └─────┬───────────┬─────┘
                                │           │
              ┌─────────────────┘           └─────────────────┐
              ▼                                               ▼
┌───────────────────────────┐                   ┌───────────────────────────┐
│     Lexical Retrieval     │                   │    Semantic Retrieval     │
│    (Exact Term Match)     │                   │   (Exact k-NN Cosine)     │
└─────────────┬─────────────┘                   └─────────────┬─────────────┘
              │                                               │
              └───────────────────────┬───────────────────────┘
                                      ▼
                          ┌───────────────────────┐
                          │  Fusion Orchestrator  │
                          │ (Reciprocal Rank      │
                          │  Fusion, k=60)        │
                          └───────────┬───────────┘
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │  Presentation Layer   │
                          │   (Formatted Top-K)   │
                          └───────────────────────┘
```

## Setup and Execution

### 1. Install Dependencies

You need Python 3.9+.

Standard install:

```bash
pip install -r requirements.txt
```

- The appropriate PyTorch build is installed automatically.
- SentenceTransformers selects CPU or GPU automatically.

Advanced (force CUDA 12.1 build):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Launch the Search Engine

Start GUI mode:

```bash
python main.py
```

Other modes:

- CLI mode: `python main.py --cli`
- Run tests: `python main.py --test`
- Run evaluation: `python main.py --evaluate`

### 3. Manual Data ETL (Optional)

```bash
python ingest_data.py
```

### 4. Unit Tests

```bash
python test_engine.py
```

### 5. System Evaluation

```bash
python evaluate_engine.py
```

## Evaluation Metrics

- MRR (Mean Reciprocal Rank): known-item retrieval effectiveness.
- NDCG@10 (Normalized Discounted Cumulative Gain): semantic ranking quality in top results.

## Hardware Notes

- Runs on both CPU and GPU.
- No manual CUDA configuration is required for most users.
- If CUDA is available, dense retrieval uses GPU automatically.

## References

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [PyQt6 Documentation](https://doc.qt.io/qtforpython-6/)
- [Rank-BM25 Project](https://github.com/dorianbrown/rank_bm25)
- [RRF Paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)

(c) 2026. ECS736P/U, Queen Mary University of London
