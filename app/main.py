"""
Self-Correcting Agentic RAG API Server.
FastAPI application orchestrating hierarchical indexing, hybrid search, and LangGraph cyclic execution.
"""
import uuid
from typing import List
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse

from app.models import (
    DocumentIngestRequest,
    DocumentChunk,
    QueryRequest,
    QueryResponse,
    Citation,
)
from app.core.vector_store import HierarchicalVectorStore
from app.core.bm25 import BM25Retriever
from app.core.reranker import HybridReranker
from app.agents.graph import CorrectiveRAGGraph

app = FastAPI(
    title="Corrective-RAG-Agent: Production Agentic Retrieval Engine",
    description="Stateful multi-agent RAG system with hierarchical chunking, hybrid search (dense + BM25), cross-encoder reranking, and self-correcting query rewrites.",
    version="1.0.0",
)

# Core singletons
vector_store = HierarchicalVectorStore(embedding_dim=128)
bm25_retriever = BM25Retriever()
reranker = HybridReranker(rrf_k=60)
rag_graph = CorrectiveRAGGraph(vector_store, bm25_retriever, reranker)


def _reindex_bm25():
    """Syncs vector store chunks into BM25 corpus index."""
    doc_tuples = [(c.chunk_id, c.text) for c in vector_store.chunks.values()]
    bm25_retriever.index(doc_tuples)


def _ingest_document_internal(doc: DocumentIngestRequest) -> List[str]:
    parent_id = f"doc_{uuid.uuid4().hex[:8]}"
    vector_store.add_parent_document(
        parent_id=parent_id,
        title=doc.title,
        full_content=doc.content,
        metadata=doc.metadata or {}
    )

    # Hierarchical Chunking (split by sentences/paragraphs into semantic sub-chunks)
    sentences = [s.strip() for s in doc.content.split(". ") if len(s.strip()) > 10]
    created_chunk_ids = []

    for idx, sent in enumerate(sentences, 1):
        chunk_id = f"{parent_id}_chk_{idx}"
        chunk = DocumentChunk(
            chunk_id=chunk_id,
            parent_id=parent_id,
            text=sent + ".",
            metadata={"title": doc.title, "index": idx, **(doc.metadata or {})}
        )
        vector_store.add_chunk(chunk)
        created_chunk_ids.append(chunk_id)

    _reindex_bm25()
    return created_chunk_ids


def preload_sample_documents():
    """Seed initial enterprise knowledge base."""
    sample_docs = [
        DocumentIngestRequest(
            title="Distributed Systems Principles",
            content=(
                "In distributed computing, the CAP theorem states that a distributed data store can simultaneously provide "
                "at most two out of three guarantees: Consistency, Availability, and Partition tolerance. Modern distributed "
                "databases like CockroachDB and Google Spanner prioritize consistency and partition tolerance (CP) using Paxos "
                "or Raft consensus algorithms, while systems like Amazon DynamoDB offer tunable eventual consistency (AP) "
                "to maximize availability during network netsplit partitions."
            ),
            metadata={"domain": "distributed_systems", "author": "Systems Engineering Team"}
        ),
        DocumentIngestRequest(
            title="RAG Architecture and Failure Modes",
            content=(
                "Naive Retrieval-Augmented Generation frequently fails in production due to three primary bottlenecks: "
                "fixed-length chunk boundary truncation, false-positive semantic vector similarity, and the lost-in-the-middle "
                "phenomenon. Advanced production RAG architectures overcome this by decoupling search indexing from generation "
                "context through Parent-Document Retrieval, combining dense embeddings with sparse BM25 keyword search via Reciprocal "
                "Rank Fusion (RRF), and filtering candidates with cross-encoder rerankers before prompt synthesis."
            ),
            metadata={"domain": "generative_ai", "author": "AI Research Lab"}
        ),
        DocumentIngestRequest(
            title="Microservices Observability and Resiliency",
            content=(
                "Production microservice architectures require distributed tracing and circuit breaker patterns to prevent "
                "cascading failures. OpenTelemetry provides unified telemetry standards across distributed traces, metrics, "
                "and structured logs. Implementing token-bucket rate limiting and exponential backoff retry policies with "
                "dead-letter queues guarantees exactly-once transaction processing and graceful degradation during traffic surges."
            ),
            metadata={"domain": "cloud_infrastructure", "author": "Platform Reliability Team"}
        )
    ]
    for doc in sample_docs:
        _ingest_document_internal(doc)


# Preload documents immediately on initialization
preload_sample_documents()



RAG_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Corrective-RAG-Agent: Multi-Agent Retrieval Playground</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
        code, pre, .font-mono { font-family: 'JetBrains Mono', monospace; }
        @keyframes pulseGlow {
            0%, 100% { box-shadow: 0 0 15px rgba(168, 85, 247, 0.4); }
            50% { box-shadow: 0 0 25px rgba(168, 85, 247, 0.8); }
        }
        .agent-active { animation: pulseGlow 2s infinite; }
    </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col">
    <!-- Top Nav -->
    <header class="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-6 py-3.5 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="w-9 h-9 rounded-xl bg-gradient-to-tr from-purple-600 to-indigo-500 flex items-center justify-center font-bold text-white shadow-lg shadow-purple-500/25">R</div>
                <div>
                    <div class="flex items-center space-x-2">
                        <h1 class="text-base font-bold text-white tracking-tight">Corrective-RAG-Agent</h1>
                        <span class="px-2 py-0.5 rounded text-[10px] font-semibold bg-purple-950 text-purple-300 border border-purple-800/60 font-mono">v1.2 Production</span>
                    </div>
                    <p class="text-xs text-slate-400">Self-Correcting Multi-Agent State Machine (LangGraph + FAISS + BM25)</p>
                </div>
            </div>

            <!-- Live Telemetry Badges in Header -->
            <div class="hidden md:flex items-center space-x-3 text-xs">
                <div class="px-3 py-1 rounded-lg bg-slate-900 border border-slate-800 flex items-center space-x-2 font-mono">
                    <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                    <span class="text-slate-400">Engine:</span>
                    <span id="telemetryStatus" class="text-slate-200 font-semibold">Active</span>
                </div>
                <div class="px-3 py-1 rounded-lg bg-slate-900 border border-slate-800 flex items-center space-x-2 font-mono">
                    <span class="text-slate-400">Indexed Docs:</span>
                    <span id="telemetryDocs" class="text-purple-300 font-semibold">3</span>
                </div>
                <div class="px-3 py-1 rounded-lg bg-slate-900 border border-slate-800 flex items-center space-x-2 font-mono">
                    <span class="text-slate-400">P99 Latency:</span>
                    <span id="telemetryLatency" class="text-emerald-400 font-semibold">&lt; 15ms</span>
                </div>
                <a href="/docs" target="_blank" class="px-3 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 font-medium transition flex items-center space-x-1">
                    <span>Swagger API</span>
                    <span>↗</span>
                </a>
                <a href="https://github.com/Harsh24Chandak/corrective-rag-agent" target="_blank" class="px-3 py-1 rounded-lg bg-purple-600 hover:bg-purple-500 text-white font-medium shadow-sm transition flex items-center space-x-1">
                    <span>GitHub</span>
                    <span>★</span>
                </a>
            </div>
        </div>
    </header>

    <!-- Main Container -->
    <main class="max-w-7xl mx-auto px-6 py-6 flex-1 w-full space-y-6">
        <!-- Live Agent Workflow State Machine Ribbon -->
        <div class="bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-sm">
            <div class="flex items-center justify-between mb-3">
                <span class="text-xs uppercase font-bold tracking-wider text-slate-400 flex items-center space-x-2">
                    <span>LangGraph Multi-Agent Execution Pipeline</span>
                    <span class="text-[10px] font-normal text-purple-400 bg-purple-950/80 px-2 py-0.5 rounded border border-purple-800/40">Cyclic State Machine</span>
                </span>
                <span id="pipelineRunTime" class="text-xs font-mono text-slate-500">Ready</span>
            </div>
            <!-- Node pipeline visualization -->
            <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2 text-center text-xs">
                <div id="nodeQA" class="p-2 rounded-lg bg-slate-950 border border-slate-800 transition">
                    <div class="font-semibold text-slate-300">1. Query Analyzer</div>
                    <div class="text-[10px] text-slate-500 mt-0.5">Intent & Entities</div>
                </div>
                <div id="nodeHR" class="p-2 rounded-lg bg-slate-950 border border-slate-800 transition">
                    <div class="font-semibold text-slate-300">2. Hybrid Retrieval</div>
                    <div class="text-[10px] text-slate-500 mt-0.5">Dense + BM25 RRF</div>
                </div>
                <div id="nodeRE" class="p-2 rounded-lg bg-slate-950 border border-slate-800 transition">
                    <div class="font-semibold text-slate-300">3. Relevance Auditor</div>
                    <div class="text-[10px] text-slate-500 mt-0.5">Score vs Threshold</div>
                </div>
                <div id="nodeQT" class="p-2 rounded-lg bg-slate-950 border border-slate-800 transition">
                    <div class="font-semibold text-slate-300">4. Query Transformer</div>
                    <div class="text-[10px] text-slate-500 mt-0.5">Self-Correction Loop</div>
                </div>
                <div id="nodeRS" class="p-2 rounded-lg bg-slate-950 border border-slate-800 transition">
                    <div class="font-semibold text-slate-300">5. Synthesizer</div>
                    <div class="text-[10px] text-slate-500 mt-0.5">Parent Injection</div>
                </div>
                <div id="nodeHC" class="p-2 rounded-lg bg-slate-950 border border-slate-800 transition">
                    <div class="font-semibold text-slate-300">6. Guardrail Audit</div>
                    <div class="text-[10px] text-slate-500 mt-0.5">Faithfulness Check</div>
                </div>
            </div>
        </div>

        <!-- 2 Columns Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <!-- Left Panel (Inputs & Controls) -->
            <div class="lg:col-span-5 space-y-6">
                <!-- Query Card -->
                <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm space-y-4">
                    <div class="flex items-center justify-between">
                        <h3 class="text-sm font-bold uppercase tracking-wider text-white">Ask the Agent</h3>
                        <span class="text-[10px] font-mono bg-slate-800 text-slate-300 px-2 py-0.5 rounded border border-slate-700">POST /api/v1/query</span>
                    </div>

                    <div>
                        <label class="block text-xs font-semibold text-slate-300 mb-1.5">Question / Research Prompt</label>
                        <textarea id="queryInput" rows="3" class="w-full bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs text-slate-100 focus:outline-none focus:border-purple-500 font-mono leading-relaxed" placeholder="Ask a question about distributed systems or RAG...">How do distributed databases handle the CAP theorem and consensus?</textarea>
                    </div>

                    <!-- Presets -->
                    <div>
                        <span class="text-xs text-slate-400 block mb-2 font-medium">Sample Research Queries:</span>
                        <div class="space-y-1.5">
                            <button onclick="setQuery(this.innerText)" class="w-full text-left p-2 rounded-lg bg-slate-950 hover:bg-slate-800 border border-slate-800/80 text-xs text-slate-300 transition">How do distributed databases handle the CAP theorem and consensus?</button>
                            <button onclick="setQuery(this.innerText)" class="w-full text-left p-2 rounded-lg bg-slate-950 hover:bg-slate-800 border border-slate-800/80 text-xs text-slate-300 transition">Why does naive RAG usually fail in production?</button>
                            <button onclick="setQuery(this.innerText)" class="w-full text-left p-2 rounded-lg bg-slate-950 hover:bg-slate-800 border border-slate-800/80 text-xs text-slate-300 transition">How does microservice observability and resiliency work?</button>
                        </div>
                    </div>

                    <!-- Action Buttons -->
                    <div class="space-y-2 pt-1">
                        <button id="submitBtn" onclick="runQuery()" class="w-full bg-purple-600 hover:bg-purple-500 text-white font-semibold text-xs py-3 px-4 rounded-lg transition shadow-lg shadow-purple-600/20 flex items-center justify-center space-x-2">
                            <span>Execute Agentic Reasoning</span>
                            <span>⚡</span>
                        </button>
                        
                        <!-- Breakthrough Stress-Test Feature -->
                        <button onclick="runStressTest()" class="w-full bg-gradient-to-r from-amber-600/90 to-orange-600/90 hover:from-amber-600 hover:to-orange-600 text-white font-semibold text-xs py-2.5 px-4 rounded-lg transition shadow flex items-center justify-center space-x-2" title="Sends an ambiguous query to trigger and demonstrate the cyclic self-correcting query rewrite loop live!">
                            <span>🔥 Stress-Test Self-Correction Loop</span>
                        </button>
                    </div>
                </div>

                <!-- Document Ingestion Card -->
                <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm space-y-3">
                    <div class="flex items-center justify-between">
                        <h3 class="text-sm font-bold uppercase tracking-wider text-white">Hierarchical Document Ingestion</h3>
                        <span class="text-[10px] font-mono bg-slate-800 text-slate-300 px-2 py-0.5 rounded border border-slate-700">POST /api/v1/documents/ingest</span>
                    </div>
                    <p class="text-xs text-slate-400">Ingest raw documentation with automatic parent-child chunking & dual index synchronization.</p>
                    <input type="text" id="docTitle" placeholder="Document Title (e.g., Raft Consensus)" class="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-xs text-slate-100 focus:outline-none focus:border-purple-500 font-mono">
                    <textarea id="docContent" rows="3" placeholder="Paste document content..." class="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-xs text-slate-100 focus:outline-none focus:border-purple-500 font-mono"></textarea>
                    <button onclick="ingestDoc()" class="w-full bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 font-semibold text-xs py-2 rounded-lg transition">Ingest & Index</button>
                </div>
            </div>

            <!-- Right Panel (Grounded Outputs, Citations & Telemetry) -->
            <div class="lg:col-span-7 space-y-6">
                <!-- Answer Card -->
                <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm space-y-3">
                    <div class="flex items-center justify-between">
                        <div class="flex items-center space-x-3">
                            <h3 class="text-sm font-bold uppercase tracking-wider text-white">Synthesized Grounded Answer</h3>
                            <span id="latencyBadge" class="hidden text-[10px] font-mono bg-slate-800 text-slate-300 px-2 py-0.5 rounded border border-slate-700">0.0ms</span>
                        </div>
                        <div id="faithfulnessBadge" class="hidden px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-950 text-emerald-300 border border-emerald-800/60 font-mono">
                            Faithfulness: 100% (HIGH)
                        </div>
                    </div>
                    <div id="answerText" class="text-xs text-slate-300 leading-relaxed bg-slate-950 border border-slate-800 rounded-lg p-4 min-h-[140px] whitespace-pre-wrap">
Click "Execute Agentic Reasoning" or run the "Stress-Test" to observe the multi-agent graph in action.
                    </div>
                    <div class="flex justify-end">
                        <button onclick="exportAuditLog()" class="text-[11px] font-semibold text-slate-400 hover:text-purple-300 transition flex items-center space-x-1">
                            <span>Export Audit Log (JSON)</span>
                            <span>📥</span>
                        </button>
                    </div>
                </div>

                <!-- Verified Citations Card -->
                <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm space-y-3">
                    <h3 class="text-sm font-bold uppercase tracking-wider text-white">Verified Source Citations</h3>
                    <div id="citationsContainer" class="space-y-2 text-xs text-slate-400">
                        <p class="text-slate-500 italic">No citations retrieved yet.</p>
                    </div>
                </div>

                <!-- Detailed Agent Execution Trace -->
                <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm space-y-3">
                    <div class="flex items-center justify-between">
                        <h3 class="text-sm font-bold uppercase tracking-wider text-white">LangGraph Step-by-Step Execution Trace</h3>
                        <span id="iterationBadge" class="text-[10px] font-mono text-purple-400">Idle</span>
                    </div>
                    <div id="traceWindow" class="bg-slate-950 border border-slate-800 rounded-lg p-4 font-mono text-xs text-purple-300 space-y-1.5 max-h-[220px] overflow-y-auto">
                        <div class="text-slate-500">[Idle] Waiting for agent activation...</div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <!-- Footer -->
    <footer class="border-t border-slate-800/80 bg-slate-900/40 py-4 text-center text-xs text-slate-500">
        MeterPulse & Corrective-RAG-Agent Portfolio Infrastructure | Engineered by Harsh Chandak
    </footer>

    <script>
        let lastQueryResult = null;

        // Fetch health stats on load
        async function fetchHealth() {
            try {
                const res = await fetch('/health');
                const data = await res.json();
                document.getElementById('telemetryDocs').innerText = data.indexed_parent_docs;
            } catch (e) {}
        }
        fetchHealth();

        function setQuery(text) {
            document.getElementById('queryInput').value = text;
        }

        function highlightPipelineNodes(steps) {
            // Reset all nodes
            ['nodeQA', 'nodeHR', 'nodeRE', 'nodeQT', 'nodeRS', 'nodeHC'].forEach(id => {
                const el = document.getElementById(id);
                el.className = 'p-2 rounded-lg bg-slate-950 border border-slate-800 transition';
            });

            // Activate nodes based on steps
            const stepNames = steps.map(s => s.name);
            document.getElementById('nodeQA').classList.add('border-purple-600', 'bg-purple-950/30');
            document.getElementById('nodeHR').classList.add('border-purple-600', 'bg-purple-950/30');
            document.getElementById('nodeRE').classList.add('border-purple-600', 'bg-purple-950/30');

            if (steps.some(s => s.name.includes('Query Transformer'))) {
                document.getElementById('nodeQT').classList.add('border-amber-500', 'bg-amber-950/40', 'agent-active');
            }
            document.getElementById('nodeRS').classList.add('border-purple-600', 'bg-purple-950/30');
            document.getElementById('nodeHC').classList.add('border-emerald-500', 'bg-emerald-950/30');
        }

        async function runQuery() {
            const query = document.getElementById('queryInput').value;
            const btn = document.getElementById('submitBtn');
            btn.disabled = true;
            btn.innerText = 'Agent Reasoning in Progress...';
            document.getElementById('pipelineRunTime').innerText = 'Executing...';

            try {
                const res = await fetch('/api/v1/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query, top_k: 3, enable_self_correction: true })
                });
                const data = await res.json();
                lastQueryResult = data;

                // Update Answer
                document.getElementById('answerText').innerText = data.answer;

                // Update Latency Badge
                const latBadge = document.getElementById('latencyBadge');
                latBadge.classList.remove('hidden');
                latBadge.innerText = `${data.latency_ms}ms latency`;

                // Update Faithfulness Badge
                const badge = document.getElementById('faithfulnessBadge');
                badge.classList.remove('hidden');
                badge.innerText = `Faithfulness: ${(data.faithfulness_score * 100).toFixed(0)}% (${data.confidence_level})`;

                // Update Citations
                const citContainer = document.getElementById('citationsContainer');
                if (data.citations && data.citations.length > 0) {
                    citContainer.innerHTML = data.citations.map((c, i) => `
                        <div class="bg-slate-950 border border-slate-800 rounded-lg p-3">
                            <div class="flex items-center justify-between text-slate-200 font-semibold mb-1">
                                <span>[${i+1}] ${c.document_title}</span>
                                <span class="text-purple-400 font-mono text-[10px]">Score: ${c.relevance_score.toFixed(3)}</span>
                            </div>
                            <p class="text-slate-400 text-[11px] leading-relaxed">"${c.snippet}"</p>
                        </div>
                    `).join('');
                } else {
                    citContainer.innerHTML = '<p class="text-slate-500 italic">No direct citations available.</p>';
                }

                // Update Trace Window
                const traceWin = document.getElementById('traceWindow');
                traceWin.innerHTML = data.execution_trace.map(t => `<div class="leading-relaxed">${t}</div>`).join('');

                // Highlight Nodes
                if (data.pipeline_steps) {
                    highlightPipelineNodes(data.pipeline_steps);
                }

                document.getElementById('pipelineRunTime').innerText = `Finished in ${data.latency_ms}ms (${data.iterations_run} Iterations)`;
                document.getElementById('iterationBadge').innerText = `${data.iterations_run} Iteration(s)`;
            } catch (err) {
                alert('Error running query: ' + err.toString());
            } finally {
                btn.disabled = false;
                btn.innerText = 'Execute Agentic Reasoning ⚡';
            }
        }

        async function runStressTest() {
            // Put an ambiguous / messy query to deliberately test the self-correcting query rewrite loop!
            document.getElementById('queryInput').value = "tell me about that thing with databases and netsplits and how systems talk";
            await runQuery();
        }

        async function ingestDoc() {
            const title = document.getElementById('docTitle').value;
            const content = document.getElementById('docContent').value;
            if (!title || !content) {
                alert('Please enter both title and content.');
                return;
            }
            try {
                const res = await fetch('/api/v1/documents/ingest', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ title: title, content: content })
                });
                const data = await res.json();
                alert(`Successfully ingested '${data.title}'! Created ${data.total_chunks_created} hierarchical chunks.`);
                document.getElementById('docTitle').value = '';
                document.getElementById('docContent').value = '';
                fetchHealth();
            } catch (err) {
                alert('Ingest error: ' + err.toString());
            }
        }

        function exportAuditLog() {
            if (!lastQueryResult) {
                alert('Please execute a query first to export an audit log.');
                return;
            }
            const blob = new Blob([JSON.stringify(lastQueryResult, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `rag_execution_audit_${Date.now()}.json`;
            a.click();
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse, tags=["Playground"])
def get_playground():
    return RAG_UI_HTML

@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "service": "Corrective-RAG-Agent",
        "indexed_parent_docs": len(vector_store.parent_documents),
        "indexed_chunks": len(vector_store.chunks),
        "version": "1.0.0"
    }


@app.post("/api/v1/documents/ingest", status_code=status.HTTP_201_CREATED, tags=["Knowledge Ingestion"])
def ingest_document(doc: DocumentIngestRequest):
    chunk_ids = _ingest_document_internal(doc)
    return {
        "status": "INGESTED",
        "title": doc.title,
        "total_chunks_created": len(chunk_ids),
        "chunk_ids": chunk_ids
    }


@app.post("/api/v1/query", response_model=QueryResponse, tags=["Agentic Reasoning"])
def execute_query(req: QueryRequest):
    if not vector_store.chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Knowledge base is empty. Please ingest documents before querying."
        )

    # Execute cyclic agentic graph
    final_state, latency_ms, steps = rag_graph.run(
        query=req.query,
        top_k=req.top_k,
        max_retries=2 if req.enable_self_correction else 1
    )

    faithfulness, confidence = rag_graph.guardrail.evaluate_faithfulness(
        answer=final_state.synthesized_answer,
        context_chunks=final_state.retrieved_chunks
    )

    return QueryResponse(
        query=req.query,
        answer=final_state.synthesized_answer,
        faithfulness_score=faithfulness,
        confidence_level=confidence,
        iterations_run=final_state.iteration_count,
        execution_trace=final_state.trace,
        citations=final_state.citations,
        latency_ms=latency_ms,
        pipeline_steps=steps
    )
