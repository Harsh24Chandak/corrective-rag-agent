"""
Reciprocal Rank Fusion (RRF) & Cross-Encoder Reranker.
Combines dense semantic vectors and sparse BM25 ranks to defeat naive RAG failure modes.
"""
from typing import List, Dict, Tuple, Any
from app.models import DocumentChunk


class HybridReranker:
    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k

    def reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[DocumentChunk, float]],
        sparse_results: List[Tuple[str, float]],
        chunk_map: Dict[str, DocumentChunk],
        top_k: int = 5,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Combines two ranked lists using standard Reciprocal Rank Fusion:
        RRF(d) = 1/(k + rank_dense) + 1/(k + rank_sparse)
        """
        rrf_scores: Dict[str, float] = {}

        # 1. Score dense ranks
        for rank, (chunk, _) in enumerate(dense_results, 1):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))

        # 2. Score sparse BM25 ranks
        for rank, (chunk_id, _) in enumerate(sparse_results, 1):
            if chunk_id in chunk_map:
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))

        # 3. Sort by RRF score
        sorted_chunk_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        ranked_output: List[Tuple[DocumentChunk, float]] = []
        for cid, score in sorted_chunk_ids[:top_k]:
            if cid in chunk_map:
                ranked_output.append((chunk_map[cid], score))

        return ranked_output

    def cross_encoder_rerank(
        self,
        query: str,
        candidates: List[Tuple[DocumentChunk, float]],
        top_k: int = 4,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Simulates cross-encoder scoring (e.g. BGE-Reranker / Cohere Rerank).
        Cross-encoders evaluate query-document pairs jointly, measuring bidirectional token attention.
        """
        query_words = set(query.lower().split())
        reranked: List[Tuple[DocumentChunk, float]] = []

        for chunk, initial_score in candidates:
            chunk_words = chunk.text.lower().split()
            # Lexical overlap
            overlap = sum(1 for w in chunk_words if w in query_words)
            overlap_ratio = overlap / max(1, len(query_words))

            # Cross-encoder joint confidence score
            joint_score = (0.5 * initial_score * 100.0) + (0.5 * overlap_ratio)
            reranked.append((chunk, min(1.0, joint_score)))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]
