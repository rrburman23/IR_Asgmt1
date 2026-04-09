"""
File Name: test_engine.py
Description: Verifies the functional integrity of the search engine components,
             ensuring indexes build correctly, retrieval methods return
             the expected data structures, and latency constraints are met.
"""

import unittest
import time
from hybrid_search import ArtGallerySearchEngine


class TestArtGallerySearchEngine(unittest.TestCase):
    """Automated unit test suite for the Art Gallery Search Engine."""

    @classmethod
    def setUpClass(cls):
        print("\n[TEST SETUP] Initializing engine for automated testing...")
        cls.engine = ArtGallerySearchEngine("art_gallery_data.csv")

    def test_engine_initialization(self):
        self.assertIsNotNone(
            self.engine.df, "Document store DataFrame must not be None."
        )
        self.assertGreater(
            len(self.engine.df), 0, "Document store must contain records."
        )

    def test_indexes_built(self):
        embeddings = self.engine.document_embeddings

        self.assertIsNotNone(
            self.engine.bm25, "BM25 sparse index instantiation failed."
        )
        self.assertIsNotNone(
            embeddings,
            "Dense document embedding generation failed.",
        )
        if embeddings is None:
            self.fail("Dense document embedding generation failed.")

        self.assertGreaterEqual(
            len(embeddings),
            len(self.engine.df),
            "Embedding count should be >= document count due to chunking.",
        )

    def test_sparse_retrieval(self):
        query = "oil painting"
        results = self.engine.search_sparse(query, top_k=10)
        self.assertIsInstance(
            results, dict, "Sparse search output must be a dictionary."
        )
        self.assertLessEqual(len(results), 10, "Sparse search exceeded top_k limit.")

    def test_dense_retrieval(self):
        query = "a gloomy landscape"
        results = self.engine.search_dense(query, top_k=10)
        self.assertIsInstance(
            results, dict, "Dense search output must be a dictionary."
        )
        self.assertLessEqual(len(results), 10, "Dense search exceeded top_k limit.")

    def test_hybrid_fusion(self):
        query = "portrait of a woman"
        top_k = 5
        results = self.engine.hybrid_search(query, top_k=top_k, per_page=top_k)
        # MODIFIED: Now check the returned dict!
        self.assertIsInstance(
            results, dict, "Hybrid search output must be a dictionary."
        )
        self.assertIn("results", results, "Hybrid search output missing 'results' key.")
        self.assertEqual(
            len(results["results"]),
            top_k,
            f"Hybrid search result length must equal {top_k}.",
        )
        first_result = results["results"][0]
        expected_keys = [
            "Rank",
            "Title",
            "Artist",
            "Description",
            "Year",
            "Score",
            "Reasons",
        ]
        for key in expected_keys:
            self.assertIn(key, first_result, f"Result dictionary missing key: {key}")

    def test_latency_constraint(self):
        query = "landscape with clouds"
        start_time = time.perf_counter()
        _ = self.engine.hybrid_search(query, top_k=10)
        latency = time.perf_counter() - start_time
        self.assertLess(
            latency,
            1.0,
            f"Query latency ({latency:.4f}s) exceeded 1s threshold.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
