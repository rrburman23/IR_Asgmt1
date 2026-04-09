# Art Gallery Hybrid Search Engine

This repository contains the codebase for **Assignment 2 (Information Retrieval ECS736P/U)**.  
It implements a **hybrid search engine** for a curated corpus of Tate Gallery artworks, using:

- Fielded **BM25** over normalized metadata (title, artist, medium, artist surname),
- **Chunked dense retrieval** using Sentence Transformers (all-MiniLM-L6-v2),
- **Reciprocal Rank Fusion (RRF)** plus **intent-aware re-ranking** (canonical artists, medium boosts, fragment filtering),
- A PyQt6 **GUI**, a **CLI**, and a configurable **evaluation suite** (MRR, NDCG@10, MAP@10, Success@10, latency).

---

## Project Structure

```text
.
├── gui.py                # PyQt6 GUI: main user interface, slash-commands, colored eval/test output
├── hybrid_search.py      # Core hybrid engine: BM25 + chunked dense + RRF + intent logic
# Art Gallery Hybrid Search Engine

This repository contains the codebase for Assignment 2 (Information Retrieval ECS736P/U). It implements a hybrid search engine for a curated corpus of Tate Gallery artworks, using:

- Fielded BM25 over normalized metadata such as title, artist, medium, and artist surname
- Chunked dense retrieval using Sentence Transformers (`all-MiniLM-L6-v2`)
- Reciprocal Rank Fusion (RRF) plus intent-aware re-ranking for canonical artists, medium boosts, and fragment filtering
- A PyQt6 GUI, a CLI, and a configurable evaluation suite covering MRR, NDCG@10, MAP@10, Success@10, and latency

---

## Project Structure

```text
.
├── gui.py                # PyQt6 GUI: main user interface, slash-commands, colored eval/test output
├── hybrid_search.py      # Core hybrid engine: BM25 + chunked dense + RRF + intent logic
├── ingest_data.py        # ETL pipeline: download, clean, normalize CSV, build semantic_blob + description_chunks
├── main.py               # Central dispatcher: routes to GUI, CLI, tests, or evaluation
├── evaluate_engine.py    # Evaluation suite: MRR@K, NDCG@K, MAP@K, Success@K, latency (cold/warm)
├── test_engine.py        # Unit and latency tests for the engine and indexes
├── requirements.txt      # Python dependencies (CPU and optional GPU via torch)
└── Tate Search.spec      # PyInstaller build specification for Windows executable
```

Key generated files at runtime:

- `art_gallery_data.csv`: cleaned document store derived from the Tate CSV
- `chunk_embeddings.npy`: dense embeddings for description chunks
- `chunk_to_doc.npy`: mapping from each chunk to its parent document row

## Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                            OFFLINE PIPELINE                                │
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
│                             ONLINE PIPELINE                                │
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

## Offline Pipeline

### Ingestion (`ingest_data.py`)

- Downloads the Tate artwork CSV from GitHub
- Cleans and filters to the fields used in the GUI:
  - `id`
  - `title`
  - `artist`
  - `medium`
  - `year`
  - `datetext`
  - `dimensions`
  - `creditline`
  - `thumbnailurl`
- Creates hidden normalized columns for sparse search:
  - `search_title`
  - `search_artist`
  - `search_medium`
- Normalization includes lowercasing, punctuation removal, and accent folding via `unicodedata`
- Creates a semantic blob per document in the form `Title: ... Artist: ... Medium: ... Year: ...`
- Builds `description_chunks` as a JSON list of overlapping word windows from `semantic_blob`, for example 40-word chunks with 10-word overlap
- Writes `art_gallery_data.csv`

### Sparse Indexing (`hybrid_search.py`)

- Uses `rank_bm25.BM25Okapi` over a fielded text concatenation:
  - `search_artist_surname x 50`
  - `search_artist x 8`
  - `search_title x 3`
  - `search_medium x 1`
- This heavily anchors artist surname for known-item retrieval and surname-only queries

### Dense Indexing

- Uses Sentence Transformers `all-MiniLM-L6-v2`
- Encodes all description chunks into a single matrix
- Stores embeddings in `chunk_embeddings.npy`
- Stores chunk-to-document mapping in `chunk_to_doc.npy`
- L2-normalizes vectors so dot product is proportional to cosine similarity

## Online Pipeline

### User Interface

- GUI and CLI share the same engine
- GUI supports search, slash commands, pagination, and thumbnail display
- CLI supports search and slash commands in a terminal

### Query Processing

- Spelling suggestion via Levenshtein distance over a vocabulary built from `search_title` and `search_artist`
- Query normalization for BM25 through lowercase conversion, punctuation stripping, and whitespace normalization

### Dual Retrieval

#### Lexical Retrieval (`search_sparse`)

- BM25 scores over the fielded text
- Returns top-k document indices with ranks

#### Dense Retrieval (`search_dense`)

- Encodes the query
- Scores all description chunks via dot product (`chunk_embeddings @ q_vec`)
- Max-pools chunk scores back to document-level scores
- Keeps documents whose dense score exceeds a small threshold to reduce noise

### Fusion And Intent-Aware Re-Ranking (`hybrid_search`)

#### Reciprocal Rank Fusion (RRF)

- Base score per document:

```text
1 / (k_rrf + lexical_rank) + 1 / (k_rrf + dense_rank)
```

#### Intent Multipliers

- Boost when:
  - The document matches the canonical artist for a surname
  - The artist name appears as a whole word in the query
  - The medium suggests primary artworks such as oil, canvas, sculpture, or watercolour
- Penalize when:
  - The title indicates a depiction such as `portrait of` or `bust of` but the user did not request depictions
  - The title contains the query but the artist does not match, producing subject-only matches

#### Junk Filtering

- Filters out uninformative or archival records such as `inscription`, `[recto]`, or blank medium

#### Canonical Hard Shield

- For single-word surname queries such as `Turner`, forces the dominant artist associated with that surname to the top of the ranking

### Presentation

- Results include:
  - Rank
  - ID
  - Title
  - Artist
  - Medium
  - Year
  - Dimensions
  - CreditLine
  - Synthetic Description
  - Thumbnail
  - Score
  - Reasons
  - Suggestion
- The GUI formats results with a thumbnail, bold title and artist, medium, and a short explanation of why the result matched
- Supports server-side pagination through `page`, `per_page`, and `total_pages`

## Evaluation

The evaluation suite in `evaluate_engine.py` is configurable via `EvalConfig` and reports:

- Known-item retrieval:
  - `MRR@K` (Mean Reciprocal Rank)
  - `Success@K` (hit in top-k)
- Semantic discovery:
  - `NDCG@K`
  - `P@K` (Precision@K)
  - `MAP@K`
  - `Success@K`
- Latency:
  - Mean cold latency (first run per query)
  - Mean warm latency (second run per query)

The evaluator:

- Uses the engine document store and IDs, not just the synthetic description shown in the GUI
- Builds a corpus map keyed by `id` using raw fields such as title, artist, medium, and `semantic_blob` to compute concept hits
- Supports:
  - `semantic_min_concept_hits` as a global minimum hit threshold
  - Optional per-query overrides via `semantic_min_hits_by_query`

### Example Current Performance

With the current configuration (`top_k=10`, `k_rrf=60`, chunked dense):

- Known-item retrieval:
  - `MRR@10 ~= 0.78`
  - `Success@10 = 1.00`
  - `3/4` known-item queries have the correct item at rank 1, and the remaining query appears at rank 8 within the top 10
- Semantic discovery:
  - Mean `NDCG@10 ~= 0.97`
  - Mean `Precision@10 ~= 0.63`
  - Mean `MAP@10 ~= 0.94`
  - `Success@10 = 1.00`
- Latency:
  - Mean cold latency is approximately `35-45 ms` per query
  - Mean warm latency is approximately `35-45 ms` per query after embeddings are cached in memory

These numbers indicate strong known-item and semantic retrieval on the Tate subset with fast response times.

## Setup And Execution

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

The environment supports CPU-only by default and uses GPU (CUDA) automatically if available.

### 2. Launching And Commands

#### Graphical Interface (Default)

```bash
python main.py
```

- Loads or ingests data if `art_gallery_data.csv` is missing
- Builds or loads BM25 and dense chunk indexes
- Starts the PyQt6 GUI

#### Command Line Interface (CLI)

```bash
python main.py --cli
```

- Starts a text-based interface using the same engine

#### Global Slash Commands (GUI And CLI)

Type these directly into the search box or CLI prompt:

- `/help`: show help and available slash commands
- `/test`: run the automated unit tests in `test_engine.py`
- `/evaluate`: run the evaluation suite and display metrics such as MRR, NDCG, MAP, Success@K, and latency
- `/exit`: quit the application

### 3. Testing And Evaluation From Terminal

Run tests without starting the GUI:

```bash
python main.py --test
```

Run the evaluation suite:

```bash
python main.py --evaluate
```

Or run it directly:

```bash
python evaluate_engine.py
```

### 4. Manual ETL (Optional)

To force a re-download and rebuild of the corpus:

```bash
python ingest_data.py
```

This regenerates `art_gallery_data.csv` and the `description_chunks` column.

## Distribution And Executable

A PyInstaller spec, `Tate Search.spec`, is included to build a standalone Windows executable.

Runtime-generated artifacts such as the following are not tracked in git to avoid large binary files:

- `art_gallery_data.csv`
- `chunk_embeddings.npy`
- `chunk_to_doc.npy`

A compiled version can be produced via PyInstaller and distributed as a `.zip` containing the `.exe` and required runtime files. On first launch, the app will:

- Ingest the Tate data if missing
- Build or load BM25 and chunk embeddings
- Serve interactive search with sub-100 ms query latency on cached runs

## Hardware Notes

- Supports both CPU and GPU execution via PyTorch
- On CPU-only laptops:
  - First-time embedding computation for approximately 70k records takes a few seconds
  - After embeddings and caches are built, typical hybrid queries complete in tens of milliseconds
- On CUDA-enabled GPUs:
  - Embedding and query latency are reduced further, while the engine logic remains unchanged

## References

- Sentence Transformers: `all-MiniLM-L6-v2`
- Reciprocal Rank Fusion: Cormack, Clarke, and Buettcher, `Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods`
- `rank_bm25`: Okapi BM25 implementation in Python
- PyQt6: GUI framework and event loop

(c) 2026. ECS736P/U, Queen Mary University of London
