# Art Gallery Hybrid Search Engine

This repository contains the codebase for Assignment 1 (Information Retrieval ECS736P/U). It implements a dual-pipeline search engine designed for a curated corpus of 2,000 art gallery documents.

## Project Structure

```text
.
├── ingest_data.py        # ETL pipeline
├── hybrid_search.py      # Core hybrid search engine (BM25 + dense vectors + RRF)
├── main.py               # Interactive CLI entry point (auto-ETL)
├── evaluate_engine.py    # Evaluation metrics (MRR, NDCG@10)
├── test_engine.py        # Unit tests for engine functionality and latency
├── requirements.txt      # Python dependencies
└── art_gallery_data.csv  # Generated local document store
```

## Architecture Highlights

- **Sparse Indexing:** Uses `Rank-BM25` to index artwork titles and artists for exact known-item matching.
- **Dense Indexing:** Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to embed artwork descriptions, facilitating semantic search via Exact k-NN cosine similarity.
- **Fusion:** Implements Reciprocal Rank Fusion (RRF) to blend lexical and semantic scores.
- **Automated Bootstrap:** A central orchestrator manages data ingestion and system initialization.

## Logical System Architecture

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

## Setup and Execution Instructions

**1. Install Dependencies**  
Ensure Python 3.9+ is installed, then run:

```bash
pip install -r requirements.txt
```

**2. Launch the Search Engine**  
This command initializes the in-memory indexes and starts the interactive CLI. The system will automatically download the dataset if it is not found locally.

```bash
python main.py
```

---

### Optional Manual Setup

**Ingest and Prepare Data**  
If you wish to refresh or modify the document store independently of the main engine:

```bash
python ingest_data.py
```

---

**3. Execute Unit Tests**  
Verify the functional integrity and latency constraints of the core engine components:

```bash
python test_engine.py
```

### Tests include

- Sparse BM25 retrieval
- Dense semantic retrieval
- Hybrid RRF fusion
- Query latency checks  

**4. Run System Evaluation**  
Compute Mean Reciprocal Rank (MRR) and NDCG@10 against the predefined QRELS dataset:

```bash
python evaluate_engine.py
```

## Evaluation Metrics

The system is evaluated using standard Information Retrieval metrics:

- **MRR (Mean Reciprocal Rank)**: Measures known-item retrieval effectiveness.

- **NDCG@10 (Normalized Discounted Cumulative Gain)**: Measures ranking quality for the top 10 results.
