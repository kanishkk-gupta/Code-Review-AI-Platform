# CodeGuardian AI — Folder Responsibilities

> **Version:** 1.0.0 | **Status:** Canonical Reference  
> Every file in the project must belong to exactly one of the directories below.  
> Placing logic in the wrong directory is a structural violation.

---

## Project Root Layout

```
codeguardian/
├── api/                        # FastAPI application layer
│   ├── __init__.py
│   ├── main.py                 # App factory, middleware, lifespan
│   ├── dependencies.py         # FastAPI Depends() providers
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── review.py           # POST /review
│   │   ├── status.py           # GET /status/{job_id}
│   │   ├── report.py           # GET /report/{job_id}
│   │   └── health.py           # GET /health
│   └── middleware/
│       ├── __init__.py
│       ├── auth.py             # X-API-Key validation
│       ├── cors.py             # CORS policy
│       └── logging.py          # Structured request logging
│
├── agent/                      # LangGraph orchestration layer
│   ├── __init__.py
│   ├── graph.py                # Graph definition, edges, compile()
│   ├── state.py                # Re-exports ReviewState from schemas.py
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── ingest_node.py      # Source parsing, chunking, embedding
│   │   ├── analyze_node.py     # Parallel analyzer dispatch
│   │   ├── enrich_node.py      # FAISS retrieval, finding enrichment
│   │   └── compile_node.py     # Score computation, result assembly
│   └── analyzers/
│       ├── __init__.py
│       ├── base.py             # Abstract BaseAnalyzer
│       ├── bug_analyzer.py     # → List[BugFinding]
│       ├── solid_analyzer.py   # → List[SolidFinding]
│       ├── architecture_analyzer.py  # → List[ArchitectureFinding]
│       ├── security_analyzer.py      # → List[SecurityFinding]
│       └── complexity_analyzer.py    # → List[ComplexityFinding]
│
├── services/                   # Shared infrastructure services
│   ├── __init__.py
│   ├── job_store.py            # Async job store (memory / Redis)
│   ├── vector_store.py         # FAISS wrapper + SentenceTransformer
│   ├── chunker.py              # Source file → List[CodeChunk]
│   └── llm_client.py           # LangChain ChatModel factory
│
├── codeguardian_ui/            # Streamlit frontend
│   ├── __init__.py
│   ├── app.py                  # Streamlit entrypoint
│   ├── pages/
│   │   ├── 01_submit.py        # Upload / URL submission form
│   │   ├── 02_status.py        # Live job progress page
│   │   └── 03_report.py        # Rendered review report
│   ├── components/
│   │   ├── finding_card.py     # Reusable finding display widget
│   │   ├── severity_badge.py   # Color-coded severity pill
│   │   └── score_gauge.py      # Overall score radial chart
│   └── utils/
│       ├── api_client.py       # HTTP client wrapping all 4 endpoints
│       └── formatters.py       # Markdown rendering helpers
│
├── schemas.py                  # ★ CANONICAL SCHEMA MODULE (this file is root-level)
│
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── conftest.py             # Fixtures, test client, mock job store
│   ├── unit/
│   │   ├── test_schemas.py     # Pydantic model validation tests
│   │   ├── test_chunker.py
│   │   ├── test_analyzers.py   # Mocked LLM responses
│   │   └── test_job_store.py
│   └── integration/
│       ├── test_review_endpoint.py
│       ├── test_status_endpoint.py
│       ├── test_report_endpoint.py
│       └── test_full_pipeline.py   # End-to-end graph execution
│
├── prompts/                    # LangChain prompt templates
│   ├── bug_analysis.jinja2
│   ├── solid_analysis.jinja2
│   ├── architecture_analysis.jinja2
│   ├── security_analysis.jinja2
│   ├── complexity_analysis.jinja2
│   └── summary_generation.jinja2
│
├── config/
│   ├── __init__.py
│   └── settings.py             # Pydantic BaseSettings — env var loading
│
├── architecture.md             # ★ System architecture (this file)
├── data_flow.md                # ★ Data flow documentation
├── api_contracts.md            # ★ API contracts
├── folder_responsibilities.md  # ★ This file
├── pyproject.toml              # Project metadata and dependencies
├── .env.example                # Environment variable template
├── Dockerfile                  # Production container image
└── docker-compose.yml          # Local multi-service orchestration
```

---

## Directory Contracts

### `api/` — FastAPI Application Layer

**Single Responsibility:** HTTP boundary. Validate input, return output. No business logic.

**Owns:**
- Route handlers for all four endpoints
- FastAPI `Depends()` factories
- Middleware (auth, CORS, logging, rate limiting)
- HTTP status code decisions
- Dispatching `BackgroundTasks`

**Must NOT contain:**
- LangGraph graph traversal
- LLM calls
- FAISS operations
- Job computation logic

**Imports allowed from:** `schemas.py`, `services/job_store.py`, `agent/graph.py`

---

### `agent/` — LangGraph Orchestration Layer

**Single Responsibility:** Define and execute the review state machine.

**Owns:**
- Graph topology (`graph.py`)
- All node implementations (`nodes/`)
- All analyzer chains (`analyzers/`)
- State transitions and conditional edges

**`agent/nodes/` Sub-contracts:**

| File | Reads from ReviewState | Writes to ReviewState |
|------|------------------------|----------------------|
| `ingest_node.py` | `source_url`, `source_zip_b64`, `config` | `metadata`, `chunks`, `faiss_index`, `progress=20` |
| `analyze_node.py` | `chunks`, `config` | `bug_findings`, `solid_findings`, `architecture_findings`, `security_findings`, `complexity_findings`, `progress=60` |
| `enrich_node.py` | `*_findings`, `faiss_index`, `config` | updated `*_findings` with `related_chunk_ids`, `progress=80` |
| `compile_node.py` | All findings, `metadata`, `job_id` | `result`, `progress=100` |

**`agent/analyzers/` Sub-contracts:**
- Each analyzer is a class inheriting `BaseAnalyzer`
- Exposes a single `async def run(chunks: list[CodeChunk]) -> list[FindingType]` method
- Uses `services/llm_client.py` for model access
- Uses Jinja2 templates from `prompts/`
- Returns strongly-typed Pydantic V2 finding objects

**Must NOT contain:**
- HTTP request/response handling
- Job store reads/writes (job store is updated by the API layer, not the agent)
- Raw file I/O outside of `ingest_node`

---

### `services/` — Shared Infrastructure Services

**Single Responsibility:** Provide reusable, stateful infrastructure to both `api/` and `agent/`.

**Owns:**
- `job_store.py`: Async key-value store for `JobStatus`. Interface contract:
  ```python
  async def create(job: JobStatus) -> None
  async def get(job_id: str) -> JobStatus | None
  async def update(job_id: str, **fields) -> None
  async def delete(job_id: str) -> None
  async def list_active() -> list[JobStatus]
  ```
- `vector_store.py`: FAISS index factory. Interface contract:
  ```python
  def build_index(chunks: list[CodeChunk]) -> FAISSIndex
  def similarity_search(index: FAISSIndex, query: str, k: int) -> list[CodeChunk]
  ```
- `chunker.py`: File-to-chunk splitter. Interface contract:
  ```python
  def chunk_file(file_path: str, content: str, language: SupportedLanguage,
                 max_lines: int) -> list[CodeChunk]
  ```
- `llm_client.py`: Returns a configured `BaseChatModel` from LangChain.

**Must NOT contain:**
- HTTP routing
- LangGraph state definitions
- Analyzer prompt logic

---

### `codeguardian_ui/` — Streamlit Frontend

**Single Responsibility:** Present data from the API; collect user input. No data transformation.

**Owns:**
- Streamlit page scripts
- Reusable UI components
- API client (`utils/api_client.py`) — thin HTTP wrapper, returns `schemas.py` objects
- Display formatters

**Must NOT contain:**
- Direct LangGraph calls
- Business logic (score calculation, finding classification)
- Database or job store access

---

### `schemas.py` — Canonical Schema Module (Root Level)

**Single Responsibility:** Be the one and only source of truth for all data contracts.

**Rules:**
1. All Pydantic models for the entire system live here.
2. No other file may define a parallel model for a concept defined here.
3. Import path: `from schemas import ReviewState, BugFinding, ...`
4. This file has **zero internal imports** from other project modules — only stdlib and third-party.
5. Any schema change requires updating `api_contracts.md` simultaneously.

---

### `tests/` — Test Suite

**Structure:**
- `unit/`: Tests that mock all I/O (LLM, FAISS, network). Fast.
- `integration/`: Tests that use `TestClient` and real (mocked-LLM) pipeline. Medium.
- `conftest.py`: All shared fixtures — do NOT duplicate fixtures across test files.

**Coverage Targets:**

| Module | Min Coverage |
|--------|-------------|
| `schemas.py` | 100% |
| `services/` | 90% |
| `agent/nodes/` | 85% |
| `agent/analyzers/` | 80% |
| `api/routes/` | 90% |

---

### `prompts/` — LangChain Prompt Templates

**Single Responsibility:** Store prompt text separately from Python code.

**Rules:**
- One `.jinja2` file per analyzer plus one for summary generation.
- Templates receive a `chunks` variable (list of `CodeChunk` dicts) and `config`.
- Templates must request JSON output compatible with the corresponding Pydantic finding model.
- No Python logic inside templates — use Jinja2 `{%- if -%}` sparingly for optional sections only.

---

### `config/settings.py` — Environment Configuration

**Owns all environment variable loading via Pydantic `BaseSettings`:**

```python
class Settings(BaseSettings):
    # API
    api_key: str
    cors_origins: list[str]
    max_upload_size_mb: int = 50

    # LLM
    openai_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    llm_model_name: str = "gpt-4o-mini"

    # Vector Store
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Job Store
    job_ttl_seconds: int = 86400
    redis_url: str | None = None   # None = in-memory

    class Config:
        env_file = ".env"
```

**All other modules read settings exclusively through `config.settings.get_settings()`.**

---

## Cross-Cutting Rules

1. **Schema imports:** All modules import from `schemas` (root-level). Never re-define a model.
2. **No circular imports:** `api → services`, `agent → services`, `api → agent`. Never reverse.
3. **Async consistency:** All I/O operations must be `async def`. Sync wrappers only for FAISS (CPU-bound, use `asyncio.run_in_executor`).
4. **No global state** outside of `services/` singletons initialized at lifespan startup.
5. **Logging:** Use Python `structlog` exclusively. No `print()` statements in non-test code.
6. **Environment variables:** Only `config/settings.py` reads env vars. No `os.environ` calls elsewhere.
