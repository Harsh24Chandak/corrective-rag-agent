# Corrective-RAG-Agent — Production Self-Correcting Agentic Retrieval Engine

A stateful, multi-agent Retrieval-Augmented Generation (RAG) system engineered to solve the real-world failure modes of naive RAG: context truncation, false-positive vector similarity, and "lost-in-the-middle" degradation.

Built on the **LangGraph cyclic state machine pattern**, this engine combines **Hierarchical Parent-Document Chunking**, **Hybrid Search** (dense cosine similarity + sparse BM25 keyword matching via Reciprocal Rank Fusion), **Cross-Encoder Reranking**, and **Self-Correcting Query Rewriting** with automated hallucination guardrails.

## 🏗️ System Architecture

```
graph TD
    UserQuery[User Question / Research Prompt] --> QA[Query Analyzer Node]
    QA --> HR[Hybrid Retriever Node]
    
    subgraph Hybrid Search Engine
        HR -->|Dense Semantic| DenseStore[Vector Store L2 Normalized]
        HR -->|Sparse Keyword| BM25Store[Okapi BM25 Term Indexer]
        DenseStore --> RRF[Reciprocal Rank Fusion k=60]
        BM25Store --> RRF
        RRF --> CE[Cross-Encoder Joint Reranker]
    end
    
    CE --> RE[Relevance Evaluator Node]
    
    RE -->|Score < 0.35 Low Confidence| QT[Query Transformer Node Self-Correction Loop]
    QT -->|Rewritten Expanded Query| HR
    
    RE -->|Score >= 0.35 Grounded Context| RS[Response Synthesizer Node]
    RS --> HC[Hallucination & Faithfulness Checker]
    HC --> FinalResponse[Grounded Answer + Verified Citations]

```

## ⚡ Why This Architecture Beats Naive RAG

| Naive RAG Bottleneck | Why It Fails in Production | How Corrective-RAG-Agent Solves It | 
| ----- | ----- | ----- | 
| **Fixed-Length Chunking** | Arbitrary token splitting chops sentences and table rows in half. | **Hierarchical Parent-Document Retrieval:** Queries search fine-grained child chunks but inject rich parent sections for generation. | 
| **Vector Similarity != Relevance** | High cosine similarity frequently pulls in tangential background noise. | **Dense + BM25 Hybrid Search with RRF:** Merges semantic meaning with exact keyword matches, then cross-encoder reranks top candidates. | 
| **"Lost in the Middle"** | Long contexts cause models to ignore documents in the middle. | **Strict Cross-Encoder Filtering:** Prunes candidates to only the top high-confidence chunks. | 
| **Ambiguous / Multi-Hop Queries** | User prompts lack exact entity names needed for search. | **Cyclic Query Transformation:** Evaluator evaluates relevance score; if low, agent rewrites and decomposes query automatically. | 
| **Hallucination on Empty Context** | Naive systems guess when facts are missing. | **Automated Faithfulness Guardrails:** Audits claims against source context, computing a numerical faithfulness score and confidence tier. | 

## 🚀 Quickstart & Setup

### 1. Local Python Setup

```
# Clone the repository
git clone https://github.com/Harsh24Chandak/corrective-rag-agent.git
cd corrective-rag-agent

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows PowerShell: .\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run the test suite
python -m unittest tests/test_rag.py

# Start local server
uvicorn app.main:app --reload --port 8000

```

### 2. Run with Docker Compose

```
docker-compose up --build

```

Interactive Swagger API documentation: `http://localhost:8000/docs`

## 📡 API Endpoints

| Method | Endpoint | Description | 
| ----- | ----- | ----- | 
| `GET` | `/health` | Service status, indexed parent docs count, and chunk stats | 
| `POST` | `/api/v1/documents/ingest` | Ingest and hierarchically chunk raw documents | 
| `POST` | `/api/v1/query` | Execute full multi-agent RAG reasoning with execution trace & citations | 

## 🧪 Testing & Verification

Comprehensive test suite verifying 100% pass rate:

* `test_health_check`: Service startup and seed document verification.

* `test_dense_embedding_and_cosine_similarity`: Dense vector semantic clustering accuracy.

* `test_bm25_tokenization_and_retrieval`: Okapi BM25 term frequency saturation and length normalization.

* `test_document_ingestion_api`: Hierarchical chunking and dual-store indexing.

* `test_end_to_end_agentic_query_execution`: Complete graph traversal, citation attribution, and faithfulness score.

* `test_self_correcting_query_rewrite_on_ambiguous_prompt`: Validates cyclic retry loop on ambiguous inputs.

## 👤 Author

**Harsh Chandak**

* GitHub: [@Harsh24Chandak](https://github.com/Harsh24Chandak)

* LinkedIn: [Harsh Chandak](https://www.linkedin.com/in/harsh-chandak-170b60288)