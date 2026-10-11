"""
Data Models and State Schemas for Corrective Agentic RAG.
Compatible across Pydantic v1 and Pydantic v2.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import pydantic
from pydantic import BaseModel, Field

if hasattr(pydantic, "field_validator"):
    from pydantic import field_validator
    def validate_fields(*fields):
        return field_validator(*fields)
else:
    from pydantic import validator
    def validate_fields(*fields):
        return validator(*fields)


class DocumentIngestRequest(BaseModel):
    title: str = Field(..., min_length=2, max_length=128)
    content: str = Field(..., min_length=10, description="Full raw document text to be hierarchically chunked")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    chunk_id: str
    parent_id: str
    text: str
    metadata: Dict[str, Any]
    dense_vector: Optional[List[float]] = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="User question or research prompt")
    top_k: int = Field(default=4, ge=1, le=20)
    enable_self_correction: bool = Field(default=True, description="Enable cyclic query rewriting if retrieval confidence is low")


class Citation(BaseModel):
    document_title: str
    chunk_id: str
    snippet: str
    relevance_score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    faithfulness_score: float
    confidence_level: str  # HIGH, MEDIUM, LOW
    iterations_run: int
    latency_ms: float = 0.0
    pipeline_steps: List[Dict[str, str]] = Field(default_factory=list)
    execution_trace: List[str]
    citations: List[Citation]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentState(BaseModel):
    """
    State dictionary passing through the multi-agent cyclic graph.
    """
    original_query: str
    current_query: str
    rewritten_queries: List[str] = Field(default_factory=list)
    retrieved_chunks: List[DocumentChunk] = Field(default_factory=list)
    retrieval_score: float = 0.0
    needs_rewrite: bool = False
    synthesized_answer: str = ""
    faithfulness_score: float = 0.0
    citations: List[Citation] = Field(default_factory=list)
    iteration_count: int = 0
    trace: List[str] = Field(default_factory=list)
