"""
File Name: test_engine.py
Description: Automated unit tests for the Tate Search Engine v3.2.
- Updated to resolve Pylance type-safety and attribute access warnings.
- Verifies fielded indexing and RRF fusion schema.
"""

import unittest
import time
import numpy as np
from hybrid_search import ArtGallerySearchEngine


class TestArtGallerySearchEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n[TEST SETUP] Loading hybrid engine for tests...")
        cls.engine = ArtGallerySearchEngine("art_gallery_data.csv")

    def test_engine_initialization(self):
        """Verifies field normalization and store instantiation."""
        self.assertIsNotNone(self.engine.df, "Document DataFrame is None.")
        self.assertGreater(len(self.engine.df), 0, "Document store is empty.")

        # Checking schema presence
        for col in ("title", "artist", "semantic_blob"):
            self.assertIn(
                col, self.engine.df.columns, f"Missing field '{col}' in DataFrame."
            )

    def test_indexes_built(self):
        """Ensures both BM25 and BERT indexes are initialized and matching size."""
        self.assertIsNotNone(self.engine.bm25)
        # Check for None before using len()
        embeddings = self.engine.document_embeddings
        self.assertIsNotNone(embeddings, "Document embeddings were not initialized.")
        if embeddings is not None:
            self.assertEqual(len(embeddings), len(self.engine.df))

    def test_sparse_retrieval(self):
        """BM25 search schema and type check."""
        results = self.engine.search_sparse("river landscape", top_k=10)
        self.assertIsInstance(results, dict)
        if results:
            idx, score = next(iter(results.items()))
            self.assertIsInstance(idx, int)
            self.assertTrue(isinstance(score, (int, float, np.number)))

    def test_dense_retrieval(self):
        """Dense search schema and type check."""
        results = self.engine.search_dense("cloudy mountain", top_k=10)
        self.assertIsInstance(results, dict)
        if results:
            idx, score = next(iter(results.items()))
            self.assertIsInstance(idx, int)
            # Use np.number to cover all numpy float types without generic args
            self.assertTrue(isinstance(score, (int, float, np.number)))

    def test_hybrid_fusion(self):
        """Validates output schema for GUI compatibility."""
        query = "portrait of a woman"
        results = self.engine.hybrid_search(query, top_k=5)
        self.assertEqual(len(results), 5)

        # Verify Name -> Artist -> Medium -> Desc hierarchy keys exist
        required = ["Title", "Artist", "Medium", "Description", "Reasons"]
        for k in required:
            self.assertIn(k, results[0])

    def test_latency_constraint(self):
        """Ensures search latency is under 200ms for typical queries."""
        t0 = time.perf_counter()
        _ = self.engine.hybrid_search("stormy sea", top_k=10)
        latency = time.perf_counter() - t0
        self.assertLess(
            latency, 0.200, f"Latency {latency:.4f}s exceeded 200ms threshold."
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
