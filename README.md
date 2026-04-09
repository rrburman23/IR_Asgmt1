# Art Gallery Hybrid Search Engine

This repository contains the codebase for Assignment 2 (Information Retrieval ECS736P/U). It implements a dual-pipeline search engine for a curated corpus of Tate Gallery artwork, following a practical IR architecture with sparse retrieval, dense retrieval, and intent-based re-ranking.

## Project Structure

```text
.
├── gui.py                # Main Entry Point: PyQt6 Interface with QThread logic
├── hybrid_search.py      # Core Search Engine: BM25 + Dense + RRF + Intent Shields
├── ingest_data.py        # ETL pipeline (download, clean, normalize CSV)
├── main.py               # Central Dispatcher: Handles routing for GUI, CLI, and Testing
├── evaluate_engine.py    # Evaluation suite (MRR, NDCG@10)
├── test_engine.py        # Unit and latency tests
├── requirements.txt      # Python dependencies (Optimized for CPU-only builds)
└── Tate Search.spec      # PyInstaller build specification
```

## Architecture Overview

- **Sparse Indexing:** rank_bm25 over artist and title fields. The artist surname is heavily weighted (x50) to act as a primary anchor for known-item lookups.
- **Dense Indexing:** Sentence-transformers (all-MiniLM-L6-v2) used for semantic retrieval over the semantic_blob field.
- **Asynchronous Execution:** Background workers (QThread) handle model loading and vector math to prevent GUI thread blockage and ensure high responsiveness.
- **Intent-Based Shields (Soft):** Multipliers applied to finished oil paintings and canonical artists while penalizing archival fragments.
- **Intent-Based Shields (Hard):** A canonical override that enforces dominant artist results (e.g., J.M.W. Turner) to lead single-word surname searches.
- **System Stability:** Native OS stream redirection ensures stability in windowed executable environments.

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

The project uses a CPU-optimized version of PyTorch to minimize environment footprint and ensure compatibility across standard laptop hardware.

```bash
pip install -r requirements.txt
```

### 2. Launching and Commands

The system can be controlled via terminal flags or internal slash commands within the search interfaces.

**Start the Graphical Interface (Default):**

```bash
python main.py
```

**Launch the Command Line Interface (CLI):**

```bash
python main.py --cli
```

**Unified Slash Commands:**
The following commands can be typed directly into the search box in either the GUI or CLI:

- `/help`     : Displays the system manual and available commands.
- `/test`     : Triggers the automated unit tests and latency benchmarks.
- `/evaluate` : Runs the evaluation suite to calculate MRR and NDCG metrics.
- `/exit`     : Safely terminates the application and closes data streams.

### 3. Testing and Evaluation via Terminal

You can bypass the interfaces to run system checks directly from your terminal:

**Run System Tests:**

```bash
python main.py --test
```

**Run Evaluation Suite:**

```bash
python main.py --evaluate
```

### 4. Manual Data ETL (Optional)

The ingestion pipeline runs automatically if data is missing, but can be triggered manually:

```bash
python ingest_data.py
```

## Distribution and Executable

A standalone Windows executable is available for this project.

**Note on Repository Hosting:**
The `dist/` and `build/` folders are not tracked in this repository to prevent binary bloat. The large pre-computed embedding matrix (`embeddings.npy`) is excluded from the main branch due to GitHub's file size limitations.

### Releases

A compiled, portable version of the application is hosted under the **Releases** section of this repository.

### Running the Executable

1. Download and extract the `Tate_Search_v1.0.0.zip` from the Releases section.
2. Launch `Tate Gallery Search Engine.exe`.
3. **Note:** The pre-computed semantic index (`embeddings.npy`) is included in the package. The application will load this into memory immediately, providing sub-100ms hybrid search results on launch.

## Hardware Performance

- The engine supports both CPU and GPU execution.
- On standard laptops without dedicated GPUs, the initial search may incur a "cold start" delay of 5-10 seconds as the 106MB embedding matrix is mapped to system RAM.
- Subsequent searches utilize cached memory and typically execute in sub-100ms.

## References

- Sentence Transformers (all-MiniLM-L6-v2)
- Reciprocal Rank Fusion (Cormack et al.)
- rank_bm25 (Okapi implementation)
- PyQt6 Event Loop Architecture

(c) 2026. ECS736P/U, Queen Mary University of London
