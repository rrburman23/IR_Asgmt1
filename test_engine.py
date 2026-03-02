"""
File Name: test_engine.py
Project: Hybrid Art Gallery Search Engine (ECS736P/U)
Description: Verifies the functional integrity of the search engine components,
             ensuring indexes build correctly and retrieval methods return 
             the expected data structures.
"""

import unittest
from hybrid_search import ArtGallerySearchEngine

class TestArtGallerySearchEngine(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """
        Runs once before any tests start. 
        Initializes the engine and builds the indexes so we don't have to 
        reload the BERT model for every single test.
        """
        print("\n[TEST SETUP] Initializing engine for automated testing...")
        # Make sure 'art_gallery_data.csv' is in the same directory
        cls.engine = ArtGallerySearchEngine("art_gallery_data.csv")

    def test_engine_initialization(self):
        """
        Verifies that the document store loaded and the corpora were created.
        """
        self.assertIsNotNone(self.engine.df, "Document store DataFrame should not be None.")
        self.assertGreater(len(self.engine.df), 0, "Document store should contain records.")
        self.assertEqual(len(self.engine.sparse_corpus), len(self.engine.df), "Sparse corpus size mismatch.")
        self.assertEqual(len(self.engine.dense_corpus), len(self.engine.df), "Dense corpus size mismatch.")

    def test_indexes_built(self):
        """
        Verifies that both the BM25 and BERT indexes were successfully instantiated.
        """
        self.assertIsNotNone(self.engine.bm25, "BM25 sparse index was not built.")
        self.assertIsNotNone(self.engine.document_embeddings, "Dense document embeddings were not generated.")
        # Check that embeddings match the number of documents
        self.assertEqual(len(self.engine.document_embeddings), len(self.engine.df), "Embedding count mismatch.")

    def test_sparse_retrieval(self):
        """
        Verifies the BM25 sparse search executes and returns a dictionary of ranks.
        """
        query = "oil painting"
        results = self.engine.search_sparse(query, top_k=10)
        self.assertIsInstance(results, dict, "Sparse search must return a dictionary.")
        self.assertLessEqual(len(results), 10, "Sparse search returned more than top_k results.")

    def test_dense_retrieval(self):
        """
        Verifies the Exact k-NN dense search executes and returns a dictionary of ranks.
        """
        query = "a gloomy landscape"
        results = self.engine.search_dense(query, top_k=10)
        self.assertIsInstance(results, dict, "Dense search must return a dictionary.")
        self.assertLessEqual(len(results), 10, "Dense search returned more than top_k results.")

    def test_hybrid_fusion(self):
        """
        Verifies the Reciprocal Rank Fusion (RRF) successfully combines both
        pipelines and formats the output list correctly.
        """
        query = "portrait of a woman"
        top_k = 5
        results = self.engine.hybrid_search(query, top_k=top_k)
        
        self.assertIsInstance(results, list, "Hybrid search must return a list of dictionaries.")
        self.assertEqual(len(results), top_k, f"Hybrid search should return exactly {top_k} results.")
        
        # Check that the returned dictionaries have the correct keys
        first_result = results[0]
        expected_keys = ["Rank", "Title", "Artist", "Description", "Score"]
        for key in expected_keys:
            self.assertIn(key, first_result, f"Result missing expected key: {key}")

if __name__ == '__main__':
    # Run the tests with moderate verbosity so we can see which ones pass/fail
    unittest.main(verbosity=2)