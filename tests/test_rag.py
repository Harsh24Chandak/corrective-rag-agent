"""
Automated Unit and Integration Test Suite for Corrective Agentic RAG.
Tests hierarchical chunking, BM25 scoring, dense cosine similarity,
Reciprocal Rank Fusion, LangGraph cyclic self-correction, and hallucination guardrails.
"""
import unittest
from fastapi.testclient import TestClient

from app.main import app, vector_store, bm25_retriever, reranker, rag_graph
from app.core.vector_store import generate_dense_embedding, cosine_similarity
from app.core.bm25 import tokenize


class TestCorrectiveRAGAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_check(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertGreaterEqual(data["indexed_parent_docs"], 3)
        self.assertGreaterEqual(data["indexed_chunks"], 3)

    def test_dense_embedding_and_cosine_similarity(self):
        v1 = generate_dense_embedding("distributed systems paxos raft consensus")
        v2 = generate_dense_embedding("distributed database consensus algorithms")
        v3 = generate_dense_embedding("cooking recipe chocolate cake baking")

        sim_related = cosine_similarity(v1, v2)
        sim_unrelated = cosine_similarity(v1, v3)

        self.assertGreater(sim_related, sim_unrelated)

    def test_bm25_tokenization_and_retrieval(self):
        tokens = tokenize("What is the CAP theorem in distributed computing?")
        self.assertIn("cap", tokens)
        self.assertIn("theorem", tokens)
        self.assertIn("distributed", tokens)
        self.assertIn("computing", tokens)
        self.assertNotIn("is", tokens)
        self.assertNotIn("the", tokens)

        results = bm25_retriever.query("CAP theorem consistency availability", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertGreater(results[0][1], 0.0)

    def test_document_ingestion_api(self):
        doc_payload = {
            "title": "Modern Vector Database Architectures",
            "content": (
                "Vector databases index high-dimensional embeddings using Approximate Nearest Neighbor (ANN) algorithms "
                "such as Hierarchical Navigable Small World (HNSW) and Inverted File with Product Quantization (IVF-PQ). "
                "These data structures enable sub-10ms similarity search across billions of dense vectors while preserving "
                "high recall."
            ),
            "metadata": {"category": "vector_search"}
        }
        res = self.client.post("/api/v1/documents/ingest", json=doc_payload)
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["status"], "INGESTED")
        self.assertEqual(data["title"], "Modern Vector Database Architectures")
        self.assertGreaterEqual(data["total_chunks_created"], 1)

    def test_end_to_end_agentic_query_execution(self):
        query_payload = {
            "query": "How do distributed databases handle the CAP theorem and consensus?",
            "top_k": 3,
            "enable_self_correction": True
        }
        res = self.client.post("/api/v1/query", json=query_payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertIn("Distributed Systems Principles", data["answer"])
        self.assertGreaterEqual(data["faithfulness_score"], 0.70)
        self.assertIn(data["confidence_level"], ["HIGH", "MEDIUM"])
        self.assertGreaterEqual(len(data["citations"]), 1)
        self.assertGreaterEqual(len(data["execution_trace"]), 5)

    def test_self_correcting_query_rewrite_on_ambiguous_prompt(self):
        state, latency_ms, steps = rag_graph.run(
            query="can you tell me about what is microservices observability and resiliency",
            top_k=2,
            max_retries=2
        )
        self.assertGreaterEqual(state.iteration_count, 1)
        self.assertGreater(len(state.trace), 4)
        self.assertTrue(any("QueryAnalyzer" in t for t in state.trace))
        self.assertTrue(any("HybridRetriever" in t for t in state.trace))
        self.assertTrue(any("HallucinationChecker" in t for t in state.trace))
        self.assertGreater(latency_ms, 0.0)
        self.assertGreaterEqual(len(steps), 4)


if __name__ == "__main__":
    unittest.main()
