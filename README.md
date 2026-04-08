# Art Gallery Hybrid Search Engine

This repository contains the codebase for Assignment 1 (Information Retrieval ECS736P/U). It implements a **dual-pipeline search engine** for a curated corpus of Tate Gallery artwork, matching real-world IR architectures (fielded sparse, dense encoder, and RRF fusion).

---

## Project Structure

```text
.
├── ingest_data.py        # ETL pipeline (download, clean, normalize CSV)
├── hybrid_search.py      # Core search engine: BM25 (title/artist), Dense (medium/desc), RRF fusion
├── main.py               # GUI/CLI entry point, auto-ingest, mode router
├── evaluate_engine.py    # Evaluation (MRR, NDCG@10)
├── test_engine.py        # Unit/latency tests
├── requirements.txt      # Python dependencies (CPU & CUDA compatible)
└── art_gallery_data.csv  # Local search-ready dataset
```

## Architecture Overview

- **Sparse Indexing:** [`rank_bm25`](https://github.com/dorianbrown/rank_bm25) on artwork titles & artists for exact/known-item lookup.
- **Dense Indexing:** [`sentence-transformers`](https://www.sbert.net/) (default: `"multi-qa-MiniLM-L6-cos-v1"`) for fast semantic search on artwork descriptions/medium.
- **Fusion:** [Reciprocal Rank Fusion (RRF)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf).
- **Full stack:** CLI and PyQt6 GUI, with automated ETL, tests, and benchmarking.

---

## Logical System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                            OFFLINE PIPELINE                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
┌───────────────────────┐             ▼             ┌───────────────────────┐
│  Remote Data Source   │──────────────────────────▶│    ingest_data.py     │
│  (GitHub CSV File)    │                           │    (ETL/clean)        │
└───────────────────────┘                           └───────┬─────┬─────────┘
                                                            │     │
                                  ┌─────────────────────────┘     └─────────────────────────┐
                                  ▼                                                 ▼
                    ┌─────────────────────────┐                        ┌─────────────────────────┐
                    │  BM25 Index (title/artist)│                      │ Dense Index (desc/medium)│
                    └─────────────────────────┘                        └─────────────────────────┘
                                  └─────────────────────────┬───────────┬──────────────────┘
                                                            ▼           ▼
                                                    ┌────────────────────────┐
                                                    │  art_gallery_data.csv  │
                                                    └────────────────────────┘

===============================================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                             ONLINE PIPELINE                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                         ┌────────────────────────────┐
                         │      User Interface        │
                         │ (PyQt6 GUI / CLI / Tests)  │
                         └─────────────┬──────────────┘
                                       │
                                       ▼
                         ┌────────────────────────────┐
                         │    Query Processor         │
                         │  (Tokenize / Embed)        │
                         └─────┬───────────┬──────────┘
                               │           │
           ┌───────────────────┘           └───────────────────┐
           ▼                                                   ▼
┌───────────────────────────┐                 ┌───────────────────────────┐
│      Lexical Retrieval    │                 │      Semantic Retrieval   │
│    (BM25: title/artist)   │                 │    (Dense: medium/desc)  │
└────────────┬──────────────┘                 └────────────┬──────────────┘
             │                                              │
             └───────────────────┬──────────────────────────┘
                                 ▼
                   ┌────────────────────────────┐
                   │   Fusion (RRF, k=60)       │
                   └───────┬────────────┬───────┘
                           ▼            ▼
                   Presentation Layer  (Formatted Top-K)
```

---

## Setup and Execution Instructions

### 1. Install Dependencies

You need **Python 3.9+**.  
This project runs out-of-the-box on CPU or CUDA GPU.

**Standard install:**

```bash
pip install -r requirements.txt
```

- The correct PyTorch build (CPU or CUDA) will be installed automatically for your machine.
- [SentenceTransformers](https://www.sbert.net/) will auto-select your device.

### Advanced: Forcing a specific PyTorch/CUDA version

To force a specific CUDA build (e.g., CUDA 12.1), use:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

(Then, install other packages.) For most users, **this is not necessary**.

---

### 2. Launch the Search Engine

This command initializes indexes and starts the interactive PyQt6 GUI.  
**The engine will automatically download and process the dataset if not found locally.**

```bash
python main.py
```

- **CLI mode**: `python main.py --cli`
- **Run tests**: `python main.py --test`
- **Run system evaluation**: `python main.py --evaluate`

---

### 3. Manual Data ETL (Optional)

To refresh or clean the document store:

```bash
python ingest_data.py
```

---

### 4. Unit Tests

To validate core search/latency:

```bash
python test_engine.py
```

---

### 5. System Evaluation

Run all IR metrics on the current index:

```bash
python evaluate_engine.py
```

---

## Evaluation Metrics

- **MRR (Mean Reciprocal Rank)**: Known-item/exact search effectiveness.
- **NDCG@10 (Normalized Discounted Cumulative Gain):** Semantic ranking quality of top 10.

---

## Hardware Notes

- **Runs on both CPU and GPU** out of the box.
- No CUDA/PyTorch config is required for most users.
- If you have a GPU and CUDA drivers, the engine will use it automatically for dense retrieval.

---

## More

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [PyQt6 Documentation](https://doc.qt.io/qtforpython-6/)
- [Rank-BM25 Project](https://github.com/dorianbrown/rank_bm25)
- [RRF Paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)

---

(c) 2026. ECS736P/U, Queen Mary University of London
