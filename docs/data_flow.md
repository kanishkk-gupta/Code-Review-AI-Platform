# CodeGuardian AI — Data Flow

> **Version:** 1.0.0 | **Status:** Canonical Reference

---

## 1. End-to-End Request Lifecycle

```
Client              FastAPI           Job Store       LangGraph Agent
  │                    │                  │                  │
  │  POST /review      │                  │                  │
  ├──────────────────>│                  │                  │
  │                    │ validate(body)   │                  │
  │                    │ generate(job_id) │                  │
  │                    ├────────────────>│                  │
  │  202 {job_id}      │                  │                  │
  │<───────────────────┤                  │                  │
  │                    │ BackgroundTask ─────────────────────>│
  │  GET /status/{id}  │                  │                  │
  ├──────────────────>│                  │                  │
  │  {RUNNING, 45%}    │                  │                  │
  │<───────────────────┤                  │                  │
  │                    │                  │<── [complete] ───┤
  │  GET /status/{id}  │                  │                  │
  │  {COMPLETED}       │                  │                  │
  │  GET /report/{id}  │                  │                  │
  │  ReviewResult      │                  │                  │
```

---

## 2. LangGraph Node Data Flow

### `ingest_node`
```
Raw Source (ZIP / GitHub URL)
  → decompress / clone
  → filter allowed extensions
  → extract RepositoryMetadata
  → split into List[CodeChunk]
  → embed via SentenceTransformer → assign CodeChunk.embedding
  → build FAISS index (per-job)
  → ReviewState: metadata, chunks, faiss_index, progress=20
```

### `analyze_node`
```
chunks → asyncio.gather([
  BugAnalyzer       → List[BugFinding],
  SolidAnalyzer     → List[SolidFinding],
  ArchitectureAnalyzer → List[ArchitectureFinding],
  SecurityAnalyzer  → List[SecurityFinding],
  ComplexityAnalyzer → List[ComplexityFinding]
])
→ ReviewState: all findings, progress=60
```

Each analyzer uses a LangChain chain:
`PromptTemplate → ChatModel → PydanticOutputParser[FindingType]`

### `enrich_node`
```
For each finding:
  query = finding.description
  nearest = faiss_index.similarity_search(query, k=3)
  finding.related_chunks = [chunk.chunk_id for chunk in nearest]
→ ReviewState: enriched findings, progress=80
```

### `compile_node`
```
Aggregate all findings
Compute overall_score (0–100):
  base = 100
  CRITICAL → -15 each | HIGH → -8 | MEDIUM → -3 | LOW → -1
  clamp [0, 100]
Build ReviewResult
Write JobStatus: status=COMPLETED, result=ReviewResult, progress=100
Destroy FAISS index and chunk data
```

---

## 3. Embedding & FAISS Flow

```
CodeChunk.content
  → SentenceTransformer.encode()
  → numpy.ndarray(384,) float32
  → FAISS IndexFlatL2.add(vector)
  → chunk_registry[index] = chunk_id

[Query time]
query_vector → IndexFlatL2.search(k)
  → distances[], indices[]
  → chunk_registry[i] → CodeChunk
  → return List[CodeChunk] ranked by similarity
```

---

## 4. Schema Transformation Map

| Input | Internal | Output |
|-------|----------|--------|
| `ReviewRequest` | `ReviewState` | `ReviewResponse (202)` |
| `JobStatus` | (direct) | `JobStatusResponse` |
| `ReviewResult` | (direct) | `ReviewResult` |

---

## 5. Error Propagation

```
Any node exception
  → LangGraph conditional edge catches
  → ReviewState.error = str(exception)
  → Transition to "error" terminal node
  → JobStatus: status=FAILED, error=..., completed_at=utcnow()
  → Client: GET /status/{id} → { "status": "FAILED", "error": "..." }
```

---

## 6. Data Retention

| Data | Retention | Store |
|------|-----------|-------|
| JobStatus (active) | 24h TTL | Memory / Redis |
| ReviewResult | 24h TTL | Memory / Redis |
| FAISS index | Destroyed on compile_node exit | In-process |
| Code chunks | Destroyed on compile_node exit | In-process |
| Raw upload | Never persisted | Request memory |
