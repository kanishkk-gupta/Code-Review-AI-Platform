"""
graph/workflow.py
==================
CodeGuardian AI — Full LangGraph review pipeline.

Graph topology
--------------
::

    START
      │
      ▼
    repository_processor          ← clone/extract repo, build RepositoryMetadata
      │
      ▼
    chunk_generator               ← chunk files, embed, build FAISS index
      │
      ├──[error]──► error_node ──► END
      │
      ▼
    ┌─────────────────────────────────────────────────────┐
    │              PARALLEL ANALYSIS FANOUT               │
    │  complexity_agent   security_agent   bug_agent      │
    │  solid_agent        architecture_agent              │
    └─────────────────────────────────────────────────────┘
      │  (asyncio.gather — all 5 run concurrently)
      ▼
    analysis_merger               ← collect & merge all 5 finding lists
      │
      ▼
    report_agent_node             ← compute score, generate Markdown + PDF
      │
      ▼
    END

Each non-terminal node writes to ReviewState and routes through a conditional
edge: if ``state["error"]`` is set the graph short-circuits to ``error_node``.

Usage
-----
::

    from graph.workflow import get_workflow, run_review

    # Option 1: direct pipeline run
    result = await run_review(source_url="https://github.com/acme/repo")

    # Option 2: compiled graph (for LangGraph server / checkpointing)
    graph  = get_workflow()
    output = await graph.ainvoke({"source_url": "https://github.com/acme/repo"})
    result = ReviewState(**output).result

Compatibility with graph/graph.py
-----------------------------------
``get_review_graph()`` in the existing ``graph/graph.py`` will be updated to
delegate to this module.  Both entry points return the same compiled graph.
"""
from __future__ import annotations

import asyncio
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional

import structlog

from schemas import (
    ArchitectureFinding,
    BugFinding,
    CodeChunk,
    ComplexityFinding,
    RepositoryMetadata,
    ReviewConfig,
    ReviewResult,
    ReviewState,
    SecurityFinding,
    SolidFinding,
    SupportedLanguage,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Default report output directory
# ---------------------------------------------------------------------------
_REPORT_OUTPUT_DIR = Path(__file__).parent.parent / "reports" / "output"


# ---------------------------------------------------------------------------
# Routing helper
# ---------------------------------------------------------------------------

def _route(state: dict) -> Literal["next", "error"]:
    """Conditional edge: route to 'error' terminal if state['error'] is set."""
    if state.get("error"):
        logger.warning("workflow_routing_to_error", error=state["error"][:200])
        return "error"
    return "next"


# ===========================================================================
# NODE 1: repository_processor
# ===========================================================================

async def repository_processor(state: dict) -> dict:
    """
    Ingest the repository, collect source files, and build RepositoryMetadata.

    For URL sources  : shallow-clones via github_tools.fetch_repo() +
                       list_source_files() in a thread-pool executor.
    For ZIP sources  : decodes base64 ZIP and scans members.

    Reads  : state["source_url"] | state["source_zip_b64"]
    Writes : state["metadata"]   (RepositoryMetadata — always has real counts)
             state["_file_tuples"] (list[(path, content, lang)] — consumed by
                                    chunk_generator so the repo is not
                                    cloned twice)
             state["progress"]  = 10

    Errors: sets state["error"] on any failure; downstream error_node fires.
    """
    job_id = state.get("job_id", "unknown")
    logger.info("node_repository_processor_start", job_id=job_id)

    try:
        source_url     = state.get("source_url")
        source_zip_b64 = state.get("source_zip_b64")

        if not source_url and not source_zip_b64:
            raise ValueError("Either source_url or source_zip_b64 must be provided.")

        from tools.file_parser import parse_source

        file_tuples: list[tuple[str, str, SupportedLanguage]] = await parse_source(
            source_url, source_zip_b64
        )

        if not file_tuples:
            raise ValueError(
                "No analysable source files found in the repository. "
                "Supported extensions: .py .js .ts .java .cpp .c .go .rs .cs .rb .php"
            )

        # Build real metadata from the collected files
        from collections import Counter as _Counter
        lang_counts: _Counter[str] = _Counter(lang.value for _, _, lang in file_tuples)
        primary_raw = lang_counts.most_common(1)[0][0] if lang_counts else "unknown"
        try:
            primary_lang = SupportedLanguage(primary_raw)
        except ValueError:
            primary_lang = SupportedLanguage.UNKNOWN

        total_files = len({fp for fp, _, _ in file_tuples})
        total_lines = sum(
            content.count("\n") + 1
            for _, content, _ in file_tuples
        )
        lang_total  = sum(lang_counts.values()) or 1
        lang_breakdown: dict[str, float] = {
            k: round(v / lang_total * 100, 1)
            for k, v in lang_counts.items()
        }

        if source_url:
            import re as _re
            repo_name = source_url.rstrip("/").split("/")[-1] or "unknown-repo"
            repo_name = _re.sub(r"\.git$", "", repo_name)
        else:
            repo_name = "uploaded-repository"

        metadata = RepositoryMetadata(
            repository_name=repo_name,
            source_url=source_url,
            primary_language=primary_lang,
            language_breakdown=lang_breakdown,
            total_files=total_files,
            total_lines=total_lines,
        )

        logger.info(
            "node_repository_processor_complete",
            job_id=job_id,
            repo=metadata.repository_name,
            files=total_files,
            lines=total_lines,
            primary=str(primary_lang),
        )

        return {
            **state,
            "metadata":      metadata.model_dump(),
            "_file_tuples":  file_tuples,   # handed off to chunk_generator
            "progress":      10,
            "error":         None,
        }

    except Exception as exc:  # noqa: BLE001
        logger.exception("node_repository_processor_error", job_id=job_id, error=str(exc))
        return {
            **state,
            "error":    f"repository_processor: {exc}",
            "progress": state.get("progress", 0),
        }


# ===========================================================================
# NODE 2: chunk_generator
# ===========================================================================

async def chunk_generator(state: dict) -> dict:
    """
    Chunk pre-ingested source files, embed them, and build a FAISS index.

    Reads  : state["_file_tuples"] (list[(path, content, lang)] from
             repository_processor — avoids re-cloning the repository)
             state["config"],
             state["metadata"] (already populated by repository_processor)
    Writes : state["chunks"] (list[CodeChunk dicts], embedding excluded),
             state["faiss_index"] (FAISSIndex),
             state["progress"] = 25
    """
    job_id = state.get("job_id", "unknown")
    logger.info("node_chunk_generator_start", job_id=job_id)

    try:
        from rag.embeddings import load_embedding_model
        from rag.vector_store import build_index
        import numpy as np

        # FIX (BUG 4): Guard against config being a ReviewConfig object vs dict
        raw_config = state.get("config") or {}
        config = raw_config if isinstance(raw_config, ReviewConfig) else ReviewConfig(**raw_config)

        # Consume file_tuples set by repository_processor (no second clone)
        file_tuples: list[tuple[str, str, SupportedLanguage]] = state.get("_file_tuples") or []

        if not file_tuples:
            logger.warning("node_chunk_generator_no_files", job_id=job_id)
            return {**state, "chunks": [], "faiss_index": None, "progress": 25, "error": None}

        logger.info("node_chunk_generator_files", job_id=job_id, files=len(file_tuples))

        # Build CodeChunk objects via tools.chunker.chunk_file (fixed-size line windows)
        from tools.chunker import chunk_file

        raw_chunks: list[CodeChunk] = []
        max_lines = config.max_chunk_lines
        for file_path, content, language in file_tuples:
            raw_chunks.extend(chunk_file(file_path, content, language, max_lines))

        logger.info("node_chunk_generator_chunked", job_id=job_id, chunks=len(raw_chunks))

        if not raw_chunks:
            logger.warning("node_chunk_generator_no_chunks", job_id=job_id)
            return {**state, "chunks": [], "faiss_index": None, "progress": 25, "error": None}

        # Embed all chunks in one batch
        embedding_model = await asyncio.get_event_loop().run_in_executor(
            None, load_embedding_model
        )
        texts   = [c.content for c in raw_chunks]
        vectors = await asyncio.get_event_loop().run_in_executor(
            None, lambda: embedding_model.embed_batch_numpy(texts)
        )

        # Attach float32 embeddings
        embedded_chunks: list[CodeChunk] = [
            chunk.model_copy(update={"embedding": vec.astype(np.float32)})
            for chunk, vec in zip(raw_chunks, vectors)
        ]

        # Build FAISS index
        faiss_index = await asyncio.get_event_loop().run_in_executor(
            None, lambda: build_index(embedded_chunks)
        )

        # Serialize chunks — exclude 'embedding' (numpy, not JSON-safe) AND all
        # @computed_fields (e.g. 'line_count'). CodeChunk has extra='forbid', so
        # model_validate() rejects any key that isn't in model_fields. Before this
        # fix, 'line_count' was included by model_dump() and caused ALL 172 chunks
        # to be silently dropped by the bare except in parallel_analysis.
        _chunk_exclude = set(CodeChunk.model_computed_fields) | {"embedding"}
        chunk_dicts = [c.model_dump(exclude=_chunk_exclude) for c in embedded_chunks]

        logger.info(
            "node_chunk_generator_complete",
            job_id=job_id,
            chunks=len(embedded_chunks),
            faiss_size=faiss_index.size,
            serialized_keys=sorted(chunk_dicts[0].keys()) if chunk_dicts else [],
        )

        return {
            **state,
            "chunks":        chunk_dicts,
            "faiss_index":   faiss_index,
            "_file_tuples":  None,   # free memory — no longer needed
            "progress":      25,
            "error":         None,
        }

    except Exception as exc:  # noqa: BLE001
        logger.exception("node_chunk_generator_error", job_id=job_id, error=str(exc))
        return {**state, "error": f"chunk_generator: {exc}", "progress": state.get("progress", 10)}


# ===========================================================================
# NODE 3: parallel_analysis
# ===========================================================================

async def parallel_analysis(state: dict) -> dict:
    """
    Run all 5 analysis agents concurrently via asyncio.gather.

    Reads  : state["chunks"], state["metadata"], state["config"]
    Writes : state["bug_findings"], state["solid_findings"],
             state["architecture_findings"], state["security_findings"],
             state["complexity_findings"], state["progress"] = 80
    """
    job_id = state.get("job_id", "unknown")
    logger.info("node_parallel_analysis_start", job_id=job_id)

    try:
        from agents.bug_agent        import BugAgent
        from agents.solid_agent      import SolidAgent
        from agents.security_agent   import SecurityAgent
        from agents.complexity_agent import ComplexityAgent
        from agents.architecture_agent import ArchitectureAgent

        # FIX (BUG 4): Guard against config being ReviewConfig object vs dict
        raw_config = state.get("config") or {}
        config = raw_config if isinstance(raw_config, ReviewConfig) else ReviewConfig(**raw_config)

        # Reconstruct CodeChunk objects from serialized dicts
        chunks: list[CodeChunk] = []
        _chunk_errors = 0
        for cd in state.get("chunks", []):
            try:
                chunks.append(CodeChunk.model_validate(cd))
            except Exception as exc:  # noqa: BLE001
                _chunk_errors += 1
                if _chunk_errors <= 3:  # log first 3 failures, not all 172
                    logger.warning(
                        "parallel_analysis_chunk_validation_failed",
                        job_id=job_id,
                        error=str(exc)[:200],
                        keys=list(cd.keys()) if isinstance(cd, dict) else type(cd).__name__,
                    )

        if _chunk_errors:
            logger.error(
                "parallel_analysis_chunks_lost",
                job_id=job_id,
                total_in_state=len(state.get("chunks", [])),
                valid=len(chunks),
                invalid=_chunk_errors,
                hint="Check CodeChunk.model_computed_fields are excluded from chunk_generator serialization",
            )

        logger.info(
            "node_parallel_analysis_chunks_received",
            job_id=job_id,
            raw_in_state=len(state.get("chunks", [])),
            valid_chunks=len(chunks),
        )

        # Reconstruct RepositoryMetadata
        metadata: Optional[RepositoryMetadata] = None
        if state.get("metadata"):
            try:
                metadata = RepositoryMetadata.model_validate(state["metadata"])
            except Exception:  # noqa: BLE001
                pass

        logger.info(
            "node_parallel_analysis_dispatch",
            job_id=job_id,
            chunks=len(chunks),
            enable_bugs=config.enable_bug_analysis,
            enable_solid=config.enable_solid_analysis,
            enable_arch=config.enable_architecture_analysis,
            enable_sec=config.enable_security_analysis,
            enable_complexity=config.enable_complexity_analysis,
        )

        # Build coroutines for enabled agents only
        async def _run_bugs() -> list[BugFinding]:
            if not config.enable_bug_analysis or not chunks:
                return []
            return await BugAgent().run(chunks)

        async def _run_solid() -> list[SolidFinding]:
            if not config.enable_solid_analysis or not chunks:
                return []
            return await SolidAgent().run(chunks)

        async def _run_arch() -> list[ArchitectureFinding]:
            if not config.enable_architecture_analysis or not chunks:
                return []
            return await ArchitectureAgent().analyse_repository(chunks, metadata)

        async def _run_security() -> list[SecurityFinding]:
            if not config.enable_security_analysis or not chunks:
                return []
            return await SecurityAgent().run(chunks)

        async def _run_complexity() -> list[ComplexityFinding]:
            if not config.enable_complexity_analysis or not chunks:
                return []
            return await ComplexityAgent().run(chunks)

        # Run all concurrently — return_exceptions=True so one failure
        # doesn't abort the rest
        results = await asyncio.gather(
            _run_bugs(),
            _run_solid(),
            _run_arch(),
            _run_security(),
            _run_complexity(),
            return_exceptions=True,
        )

        def _safe(r: Any, label: str) -> list:
            if isinstance(r, Exception):
                logger.warning(f"parallel_analysis_{label}_failed", error=str(r))
                return []
            return r or []

        bug_findings          = _safe(results[0], "bugs")
        solid_findings        = _safe(results[1], "solid")
        architecture_findings = _safe(results[2], "architecture")
        security_findings     = _safe(results[3], "security")
        complexity_findings   = _safe(results[4], "complexity")

        logger.info(
            "node_parallel_analysis_complete",
            job_id=job_id,
            bugs=len(bug_findings),
            solid=len(solid_findings),
            architecture=len(architecture_findings),
            security=len(security_findings),
            complexity=len(complexity_findings),
        )

        return {
            **state,
            "bug_findings":          [f.model_dump() for f in bug_findings],
            "solid_findings":        [f.model_dump() for f in solid_findings],
            "architecture_findings": [f.model_dump() for f in architecture_findings],
            "security_findings":     [f.model_dump() for f in security_findings],
            "complexity_findings":   [f.model_dump() for f in complexity_findings],
            "progress": 80,
            "error":    None,
        }

    except Exception as exc:  # noqa: BLE001
        logger.exception("node_parallel_analysis_error", job_id=job_id, error=str(exc))
        return {**state, "error": f"parallel_analysis: {exc}", "progress": state.get("progress", 25)}


# ===========================================================================
# NODE 4: report_agent_node
# ===========================================================================

async def report_agent_node(state: dict) -> dict:
    """
    Compute quality score, generate Markdown + PDF report, build ReviewResult.

    Reads  : state["*_findings"], state["metadata"], state["job_id"]
    Writes : state["result"] (ReviewResult dict), state["chunks"] = [],
             state["faiss_index"] = None (free memory), state["progress"] = 100
    """
    job_id = state.get("job_id", "unknown")
    logger.info("node_report_agent_start", job_id=job_id)

    try:
        from agents.report_agent import ReportAgent
        from schemas import (
            BugFinding, SolidFinding, ArchitectureFinding,
            SecurityFinding, ComplexityFinding,
        )

        # Reconstruct typed finding lists
        def _load(items: list, Model) -> list:
            out = []
            for d in items:
                try:
                    out.append(Model.model_validate(d))
                except Exception:  # noqa: BLE001
                    pass
            return out

        bug_findings          = _load(state.get("bug_findings", []),          BugFinding)
        solid_findings        = _load(state.get("solid_findings", []),        SolidFinding)
        architecture_findings = _load(state.get("architecture_findings", []), ArchitectureFinding)
        security_findings     = _load(state.get("security_findings", []),     SecurityFinding)
        complexity_findings   = _load(state.get("complexity_findings", []),   ComplexityFinding)

        metadata: Optional[RepositoryMetadata] = None
        if state.get("metadata"):
            try:
                metadata = RepositoryMetadata.model_validate(state["metadata"])
            except Exception:  # noqa: BLE001
                pass

        if metadata is None:
            metadata = RepositoryMetadata(
                repository_name="unknown",
                primary_language=SupportedLanguage.UNKNOWN,
                total_files=0,
                total_lines=0,
            )

        agent = ReportAgent(output_dir=_REPORT_OUTPUT_DIR)
        paths = await agent.run(
            job_id=job_id,
            metadata=metadata,
            bug_findings=bug_findings,
            solid_findings=solid_findings,
            architecture_findings=architecture_findings,
            security_findings=security_findings,
            complexity_findings=complexity_findings,
            llm_summary=True,
        )

        # FIX (BUG 8): Use the ReviewResult built inside ReportAgent.run()
        # Do NOT re-read the Markdown file as summary_markdown — that's the full
        # rendered report, not the executive summary. Use paths.summary_markdown.
        from schemas import ReviewResult

        result = ReviewResult(
            job_id=job_id,
            metadata=metadata,
            bug_findings=bug_findings,
            solid_findings=solid_findings,
            architecture_findings=architecture_findings,
            security_findings=security_findings,
            complexity_findings=complexity_findings,
            overall_score=paths.overall_score,
            summary_markdown=paths.summary_markdown or "Report generated.",
        )

        logger.info(
            "node_report_agent_complete",
            job_id=job_id,
            score=paths.overall_score,
            total_findings=result.total_findings,
            markdown=str(paths.markdown_path),
        )

        return {
            **state,
            # Exclude computed_fields (total_findings, critical_count) so that
            # ReviewResult.model_validate(raw_result) succeeds with extra='forbid'
            "result":      result.model_dump(
                               exclude=set(ReviewResult.model_computed_fields)
                           ),
            "chunks":      [],         # free memory
            "faiss_index": None,       # free memory
            "progress":    100,
            "error":       None,
        }

    except Exception as exc:  # noqa: BLE001
        logger.exception("node_report_agent_error", job_id=job_id, error=str(exc))
        return {**state, "error": f"report_agent: {exc}", "progress": state.get("progress", 80)}


# ===========================================================================
# ERROR TERMINAL
# ===========================================================================

async def error_node(state: dict) -> dict:
    """
    Terminal node reached when any preceding node sets state['error'].
    Logs the error and returns state unchanged for the job store to pick up.
    """
    logger.error(
        "workflow_error_terminal",
        job_id=state.get("job_id", "unknown"),
        error=state.get("error", "unknown error"),
    )
    return state


# ===========================================================================
# Graph construction
# ===========================================================================

@lru_cache(maxsize=1)
def get_workflow():
    """
    Build and compile the LangGraph StateGraph (singleton).

    Topology::

        START → repository_processor
            ──[ok]──► chunk_generator
                ──[ok]──► parallel_analysis
                    ──[ok]──► report_agent_node ──► END
            (any node) ──[err]──► error_node ──► END

    Returns:
        Compiled LangGraph graph ready for ``.ainvoke()``.

    Raises:
        ImportError: langgraph is not installed.
    """
    try:
        from langgraph.graph import StateGraph, END
    except ImportError as exc:
        raise ImportError(
            "langgraph is required: pip install langgraph"
        ) from exc

    builder = StateGraph(dict)   # state is a plain dict (ReviewState fields)

    # Register nodes
    builder.add_node("repository_processor", repository_processor)
    builder.add_node("chunk_generator",      chunk_generator)
    builder.add_node("parallel_analysis",    parallel_analysis)
    builder.add_node("report_agent_node",    report_agent_node)
    builder.add_node("error_node",           error_node)

    # Entry point
    builder.set_entry_point("repository_processor")

    # Conditional edges: each node routes to next or error
    builder.add_conditional_edges(
        "repository_processor",
        _route,
        {"next": "chunk_generator", "error": "error_node"},
    )
    builder.add_conditional_edges(
        "chunk_generator",
        _route,
        {"next": "parallel_analysis", "error": "error_node"},
    )
    builder.add_conditional_edges(
        "parallel_analysis",
        _route,
        {"next": "report_agent_node", "error": "error_node"},
    )

    # Terminal edges
    builder.add_edge("report_agent_node", END)
    builder.add_edge("error_node",        END)

    compiled = builder.compile()
    logger.info("workflow_compiled")
    return compiled


# ===========================================================================
# Convenience entry point
# ===========================================================================

async def run_review(
    source_url:      Optional[str] = None,
    source_zip_b64:  Optional[str] = None,
    config:          Optional[ReviewConfig] = None,
    job_id:          Optional[str] = None,
) -> ReviewResult:
    """
    High-level convenience wrapper: run the full review pipeline.

    Args:
        source_url      : GitHub/GitLab repository URL.
        source_zip_b64  : Base64-encoded ZIP archive (mutually exclusive with source_url).
        config          : Optional ReviewConfig (defaults used if None).
        job_id          : Optional UUID string; generated if not provided.

    Returns:
        Completed ReviewResult.

    Raises:
        ValueError    : Neither source was provided.
        RuntimeError  : Pipeline failed (check result.error field for details).
    """
    if not source_url and not source_zip_b64:
        raise ValueError("Provide either source_url or source_zip_b64.")

    _job_id = job_id or str(uuid.uuid4())
    _config = config or ReviewConfig()

    initial_state: dict = {
        "job_id":          _job_id,
        "source_url":      source_url,
        "source_zip_b64":  source_zip_b64,
        "config":          _config.model_dump(),
        "chunks":          [],
        "metadata":        None,
        "bug_findings":          [],
        "solid_findings":        [],
        "architecture_findings": [],
        "security_findings":     [],
        "complexity_findings":   [],
        "result":          None,
        "progress":        0,
        "error":           None,
    }

    logger.info("run_review_start", job_id=_job_id, source=source_url or "zip")

    graph  = get_workflow()
    output = await graph.ainvoke(initial_state)

    if output.get("error"):
        raise RuntimeError(
            f"Review pipeline failed: {output['error']}"
        )

    raw_result = output.get("result")
    if raw_result is None:
        raise RuntimeError("Pipeline completed but produced no result.")

    result = ReviewResult.model_validate(raw_result)
    logger.info(
        "run_review_complete",
        job_id=_job_id,
        score=result.overall_score,
        total_findings=result.total_findings,
    )
    return result
