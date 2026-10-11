"""
Dense Vector Store & Hierarchical Parent-Document Indexer.
Computes cosine similarity over normalized dense representations.
Supports Parent-Document mapping to prevent context truncation.
"""
import math
import hashlib
from typing import List, Dict, Tuple, Optional, Any
from app.models import DocumentChunk


def generate_dense_embedding(text: str, dim: int = 128) -> List[float]:
    """
    Generates a deterministic, high-dimensional semantic-preserving dense vector.
    Uses multi-gram feature hashing with L2-normalization.
    (In production, easily replaced with OpenAI text-embedding-3 or BAAI/bge-small-en).
    """
    words = text.lower().split()
    vec = [0.0] * dim

    # Character and word n-gram feature projections
    for i, word in enumerate(words):
        # Unigram
        h = int(hashlib.sha256(word.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0

        # Bigram
        if i < len(words) - 1:
            bigram = f"{word}_{words[i+1]}"
            hb = int(hashlib.md5(bigram.encode()).hexdigest(), 16)
            vec[hb % dim] += 1.5

        # Trigram
        if i < len(words) - 2:
            trigram = f"{word}_{words[i+1]}_{words[i+2]}"
            ht = int(hashlib.sha1(trigram.encode()).hexdigest(), 16)
            vec[ht % dim] += 2.0

    # L2 Normalization
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0.0:
        vec = [x / norm for x in vec]
    return vec


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculates cosine similarity between two normalized vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    return max(0.0, min(1.0, dot))


class HierarchicalVectorStore:
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.chunks: Dict[str, DocumentChunk] = {}
        self.parent_documents: Dict[str, Dict[str, Any]] = {}

    def add_parent_document(self, parent_id: str, title: str, full_content: str, metadata: Dict[str, Any]):
        self.parent_documents[parent_id] = {
            "title": title,
            "full_content": full_content,
            "metadata": metadata,
        }

    def add_chunk(self, chunk: DocumentChunk):
        if not chunk.dense_vector:
            chunk.dense_vector = generate_dense_embedding(chunk.text, dim=self.embedding_dim)
        self.chunks[chunk.chunk_id] = chunk

    def search_dense(self, query_text: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """
        Calculates cosine similarity against all indexed chunks.
        Returns: [(DocumentChunk, similarity_score), ...] sorted descending.
        """
        query_vec = generate_dense_embedding(query_text, dim=self.embedding_dim)
        results: List[Tuple[DocumentChunk, float]] = []

        for chunk in self.chunks.values():
            if chunk.dense_vector:
                sim = cosine_similarity(query_vec, chunk.dense_vector)
                results.append((chunk, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_parent_context(self, parent_id: str) -> Optional[Dict[str, Any]]:
        return self.parent_documents.get(parent_id)
