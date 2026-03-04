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
    """
    Automated unit test suite for the Art Gallery Search Engine.
    Validates data ingestion, index construction, query retrieval,
    and system latency thresholds.
    """

    @classmethod
    def setUpClass(cls):
        """
        Executes preliminary setup. Initializes the engine and builds the
        indexes in memory to prevent reloading the BERT model per test.
        """
        print("\n[TEST SETUP] Initializing engine for automated testing...")
        cls.engine = ArtGallerySearchEngine("art_gallery_data.csv")

    def test_engine_initialization(self):
        """
        Verifies the document store instantiation and corpus generation.
        """
        self.assertIsNotNone(
            self.engine.df, "Document store DataFrame must not be None."
        )
        self.assertGreater(
            len(self.engine.df), 0, "Document store must contain records."
        )
        self.assertEqual(
            len(self.engine.sparse_corpus),
            len(self.engine.df),
            "Sparse corpus size mismatch.",
        )
        self.assertEqual(
            len(self.engine.dense_corpus),
            len(self.engine.df),
            "Dense corpus size mismatch.",
        )

    def test_indexes_built(self):
        """
        Verifies successful instantiation of both BM25 and BERT indexes.
        """
        self.assertIsNotNone(
            self.engine.bm25, "BM25 sparse index instantiation failed."
        )
        self.assertIsNotNone(
            self.engine.document_embeddings,
            "Dense document embedding generation failed.",
        )
        self.assertEqual(
            len(self.engine.document_embeddings),
            len(self.engine.df),
            "Embedding count mismatch.",
        )

    def test_sparse_retrieval(self):
        """
        Verifies BM25 sparse search execution and output format.
        """
        query = "oil painting"
        results = self.engine.search_sparse(query, top_k=10)
        self.assertIsInstance(
            results, dict, "Sparse search output must be a dictionary."
        )
        self.assertLessEqual(len(results), 10, "Sparse search exceeded top_k limit.")

    def test_dense_retrieval(self):
        """
        Verifies Exact k-NN dense search execution and output format.
        """
        query = "a gloomy landscape"
        results = self.engine.search_dense(query, top_k=10)
        self.assertIsInstance(
            results, dict, "Dense search output must be a dictionary."
        )
        self.assertLessEqual(len(results), 10, "Dense search exceeded top_k limit.")

    def test_hybrid_fusion(self):
        """
        Verifies Reciprocal Rank Fusion (RRF) pipeline integration and output schema.
        """
        query = "portrait of a woman"
        top_k = 5
        results = self.engine.hybrid_search(query, top_k=top_k)

        self.assertIsInstance(results, list, "Hybrid search output must be a list.")
        self.assertEqual(
            len(results), top_k, f"Hybrid search output length must equal {top_k}."
        )

        first_result = results[0]
        expected_keys = ["Rank", "Title", "Artist", "Description", "Score"]
        for key in expected_keys:
            self.assertIn(key, first_result, f"Result dictionary missing key: {key}")

    def test_latency_constraint(self):
        """
        Verifies hybrid search latency operates within the 200ms threshold.
        """
        query = "landscape with clouds"

        start_time = time.perf_counter()
        _ = self.engine.hybrid_search(query, top_k=10)
        latency = time.perf_counter() - start_time

        self.assertLess(
            latency, 0.200, f"Query latency ({latency:.4f}s) exceeded 200ms threshold."
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
