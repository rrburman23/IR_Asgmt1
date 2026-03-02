# Art Gallery Hybrid Search Engine

This repository contains the codebase for Assignment 1 (Information Retrieval ECS736P/U). It implements a dual-pipeline search engine designed for a curated corpus of ~2,000 art gallery documents.

## Architecture Highlights

- **Sparse Indexing:** Uses `Rank-BM25` to index artwork titles and artists for exact known-item matching.
- **Dense Indexing:** Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to embed artwork descriptions, facilitating semantic search via Exact k-NN cosine similarity.
- **Fusion:** Implements Reciprocal Rank Fusion (RRF) to blend lexical and semantic scores.

## Setup Instructions

1. **Install Dependencies:**
   Ensure you have Python 3.9+ installed. Run the following command:

   ```bash
   pip install -r requirements.txt
   ```
