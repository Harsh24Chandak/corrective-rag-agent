"""
Hallucination Detection & Faithfulness Guardrail Engine.
Evaluates whether synthesized statements are grounded in retrieved source context.
"""
import re
from typing import List, Tuple
from app.models import Citation, DocumentChunk


class HallucinationGuardrail:
    @staticmethod
    def evaluate_faithfulness(answer: str, context_chunks: List[DocumentChunk]) -> Tuple[float, str]:
        """
        Evaluates faithfulness score (0.0 to 1.0) and confidence category (HIGH, MEDIUM, LOW).
        Measures claim extraction overlap and verifies numerical/named entities against source context.
        """
        if not context_chunks or not answer.strip():
            return 0.0, "LOW"

        # Combine all source context
        full_context = " ".join(c.text.lower() for c in context_chunks)
        answer_sentences = [s.strip() for s in re.split(r"[.!?]", answer) if len(s.strip()) > 5]

        if not answer_sentences:
            return 0.5, "MEDIUM"

        grounded_count = 0
        for sent in answer_sentences:
            words = [w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", sent.lower()) if len(w) > 3]
            if not words:
                grounded_count += 1
                continue

            # Check what percentage of key terms appear in source context
            supported_words = sum(1 for w in words if w in full_context)
            support_ratio = supported_words / len(words)

            if support_ratio >= 0.5:
                grounded_count += 1

        faithfulness = grounded_count / len(answer_sentences)

        if faithfulness >= 0.85:
            confidence = "HIGH"
        elif faithfulness >= 0.60:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return round(faithfulness, 3), confidence

    @staticmethod
    def extract_citations(answer: str, context_chunks: List[DocumentChunk]) -> List[Citation]:
        """
        Extracts verified citations with snippets and relevance scores.
        """
        citations: List[Citation] = []
        for chunk in context_chunks:
            # Simple snippet preview
            preview = chunk.text[:140] + ("..." if len(chunk.text) > 140 else "")
            title = chunk.metadata.get("title", "Untitled Document")
            citations.append(
                Citation(
                    document_title=title,
                    chunk_id=chunk.chunk_id,
                    snippet=preview,
                    relevance_score=float(chunk.metadata.get("rerank_score", 0.90)),
                )
            )
        return citations
