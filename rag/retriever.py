"""
rag/retriever.py
================
Natural-language → CodeChunk retrieval pipeline.

Public API
----------
    build_chunk_map(chunks)                         -> dict[str, CodeChunk]
    retrieve_chunks(query, handle, chunk_map, ...)  -> list[CodeChunk]
    ChunkRetriever                                  (stateful class API)

Architecture
------------
This module is the **semantic bridge** between the raw FAISS vector store and
the rest of the CodeGuardian pipeline that operates on typed ``CodeChunk``
objects (from ``schemas.py``).

::

    Natural-language query
         │
         ▼
    faiss_store.similarity_search()     ← rag/faiss_store.py
         │ returns list[SearchResult]
         ▼
    _result_to_chunk()                  ← bridge function (this module)
         │ attempts chunk_map lookup first, then reconstructs from metadata
         ▼
    list[CodeChunk]                     ← schemas.py — typed, Pydantic-validated

Chunk Resolution Strategy
--------------------------
When converting a ``SearchResult`` → ``CodeChunk``, two strategies are tried
in order:

  1. **chunk_map lookup** (preferred): If a ``dict[chunk_id → CodeChunk]`` is
     provided (built during ``ingest_node`` and kept in ``ReviewState``), the
     original object is returned.  This preserves the ``.embedding`` ndarray
     and is O(1).

  2. **Reconstruction** (fallback): If the chunk_id is not in the map (or no
     map was provided), a new ``CodeChunk`` is reconstructed from the
     ``SearchResult.metadata`` dict and ``SearchResult.page_content``.
     The reconstructed object has ``embedding=None`` but is otherwise complete.

Usage — Functional API
-----------------------
::

    from rag.retriever import retrieve_chunks, build_chunk_map

    chunk_map = build_chunk_map(state.chunks)
    neighbors = retrieve_chunks(
        query="SQL injection via string concatenation",
        handle=handle,
        chunk_map=chunk_map,
        k=5,
    )

Usage — Class API
-----------------
::

    retriever = ChunkRetriever(handle=handle, chunk_map=chunk_map)
    neighbors = retriever.retrieve("authentication bypass", k=5)
    neighbors = await retriever.aretrieve("authentication bypass", k=5)

Enrich Node Integration
-----------------------
The ``enrich_node`` in ``graph/nodes/enrich_node.py`` should:

  1. Build a ``ChunkRetriever`` once per job (store in ``ReviewState``).
  2. For each finding, call ``retriever.retrieve(finding.description, k=k)``.
  3. Set ``finding.related_chunk_ids = [c.chunk_id for c in neighbors]``.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional

import structlog

from schemas import CodeChunk, SupportedLanguage
from rag.faiss_store import (
    DEFAULT_K,
    DEFAULT_SCORE_THRESHOLD,
    SearchResult,
    VectorStoreHandle,
    similarity_search,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TOP_K: int = 5
_LANGUAGE_FALLBACK: SupportedLanguage = SupportedLanguage.UNKNOWN


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class RetrieverError(RuntimeError):
    """Base exception for retrieval pipeline errors."""


class EmptyQueryError(RetrieverError):
    """Raised when an empty or whitespace-only query is passed."""


# ---------------------------------------------------------------------------
# RetrieverConfig
# ---------------------------------------------------------------------------


@dataclass
class RetrieverConfig:
    """
    Configuration for a ``ChunkRetriever`` instance.

    Attributes:
        top_k           : Maximum number of chunks to return per query.
        score_threshold : Minimum cosine similarity [0.0–1.0] to include a result.
        language_filter : If set, restrict results to this language value
                          (e.g. ``"python"``).  ``None`` = no filter.
        dedup_by_file   : If ``True``, return at most one chunk per file path.
    """
    top_k:           int            = DEFAULT_TOP_K
    score_threshold: float          = DEFAULT_SCORE_THRESHOLD
    language_filter: Optional[str]  = None
    dedup_by_file:   bool           = False


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def build_chunk_map(chunks: list[CodeChunk]) -> dict[str, CodeChunk]:
    """
    Build an O(1) lookup table from chunk_id to CodeChunk.

    Created once per job during ``ingest_node`` and stored in ``ReviewState``.
    Resolving results via the map preserves the original embedding arrays and
    avoids reconstructing objects from metadata.

    Args:
        chunks: List of ``CodeChunk`` objects from the ingestion stage.

    Returns:
        ``dict[chunk_id → CodeChunk]``.  Duplicate chunk_ids are silently
        overwritten (last one wins).
    """
    result: dict[str, CodeChunk] = {chunk.chunk_id: chunk for chunk in chunks}
    logger.debug("chunk_map_built", size=len(result))
    return result


def retrieve_chunks(
    query: str,
    handle: VectorStoreHandle,
    *,
    chunk_map: Optional[dict[str, CodeChunk]] = None,
    k: int = DEFAULT_TOP_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    filters: Optional[dict[str, Any]] = None,
    custom_filter: Optional[Callable[[dict[str, Any]], bool]] = None,
    dedup_by_file: bool = False,
) -> list[CodeChunk]:
    """
    Retrieve the top-k semantically similar ``CodeChunk`` objects for *query*.

    This is the primary functional entry point for the retrieval pipeline.
    It orchestrates ``faiss_store.similarity_search`` and converts each
    ``SearchResult`` into a fully typed ``CodeChunk``.

    Args:
        query           : Natural-language or code-snippet query string.
        handle          : FAISS vector store (from ``build_vector_store`` or
                          ``load_vector_store``).
        chunk_map       : Optional ``dict[chunk_id → CodeChunk]`` built by
                          ``build_chunk_map()``.  When provided, results are
                          resolved from the map (preserves embedding arrays).
                          When absent, objects are reconstructed from metadata.
        k               : Maximum number of chunks to return.
        score_threshold : Discard results below this relevance score.
        filters         : Key-value metadata equality filters forwarded to
                          ``faiss_store.similarity_search``.
        custom_filter   : Custom callable ``(metadata_dict) -> bool``.
        dedup_by_file   : If ``True``, return at most one chunk per unique
                          ``file_path`` (highest-scoring chunk wins).

    Returns:
        List of ``CodeChunk`` objects, sorted by descending relevance.
        Length ≤ *k*.

    Raises:
        EmptyQueryError : *query* is empty or whitespace-only.
        RetrieverError  : FAISS search encountered an unrecoverable error.
    """
    _validate_query(query)

    logger.debug(
        "retriever_search_start",
        query_preview=query[:80],
        k=k,
        has_chunk_map=chunk_map is not None,
        filters=filters,
        score_threshold=score_threshold,
    )

    try:
        search_results: list[SearchResult] = similarity_search(
            handle,
            query,
            k=k * 3 if dedup_by_file else k,
            filters=filters,
            score_threshold=score_threshold,
            custom_filter=custom_filter,
        )
    except Exception as exc:  # noqa: BLE001
        raise RetrieverError(f"Retrieval failed: {exc}") from exc

    chunks: list[CodeChunk] = []
    seen_files: set[str] = set()

    for result in search_results:
        if len(chunks) >= k:
            break

        if dedup_by_file:
            fp = result.file_path or ""
            if fp and fp in seen_files:
                continue
            if fp:
                seen_files.add(fp)

        chunk = _result_to_chunk(result, chunk_map)
        chunks.append(chunk)

    logger.info(
        "retriever_search_complete",
        query_preview=query[:80],
        returned=len(chunks),
        k=k,
    )
    return chunks


# ---------------------------------------------------------------------------
# ChunkRetriever — stateful class API
# ---------------------------------------------------------------------------


class ChunkRetriever:
    """
    Stateful retrieval pipeline wrapping a ``VectorStoreHandle`` and an
    optional ``chunk_map``.

    This is the object created once per review job and stored in
    ``ReviewState`` (or passed into node functions directly).

    Example::

        retriever = ChunkRetriever(
            handle=handle,
            chunk_map=chunk_map,
            config=RetrieverConfig(top_k=5, score_threshold=0.4),
        )
        chunks = retriever.retrieve("null pointer dereference in loop")
        ids = [c.chunk_id for c in chunks]
    """

    def __init__(
        self,
        handle: VectorStoreHandle,
        *,
        chunk_map: Optional[dict[str, CodeChunk]] = None,
        config: Optional[RetrieverConfig] = None,
    ) -> None:
        self.handle: VectorStoreHandle = handle
        self.chunk_map: Optional[dict[str, CodeChunk]] = chunk_map
        self.config: RetrieverConfig = config or RetrieverConfig()

        logger.info(
            "chunk_retriever_created",
            store=handle.store_name,
            doc_count=handle.doc_count,
            chunk_map_size=len(chunk_map) if chunk_map else 0,
            top_k=self.config.top_k,
        )

    # ── Synchronous API ───────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        *,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filters: Optional[dict[str, Any]] = None,
        custom_filter: Optional[Callable[[dict[str, Any]], bool]] = None,
        language: Optional[str] = None,
    ) -> list[CodeChunk]:
        """
        Retrieve the top-k semantically similar ``CodeChunk`` objects.

        All parameters default to the values in ``self.config`` when not
        explicitly supplied.

        Args:
            query           : Natural-language or code-snippet query.
            k               : Override ``config.top_k`` for this call.
            score_threshold : Override ``config.score_threshold`` for this call.
            filters         : Additional metadata equality filters.
            custom_filter   : Custom callable filter for complex predicates.
            language        : Convenience shortcut for ``filters={"language": ...}``.

        Returns:
            List of ``CodeChunk`` objects, sorted by descending relevance.
        """
        effective_k         = k               if k               is not None else self.config.top_k
        effective_threshold = score_threshold if score_threshold is not None else self.config.score_threshold
        effective_filters   = _merge_filters(filters, self.config.language_filter, language)

        return retrieve_chunks(
            query,
            self.handle,
            chunk_map=self.chunk_map,
            k=effective_k,
            score_threshold=effective_threshold,
            filters=effective_filters if effective_filters else None,
            custom_filter=custom_filter,
            dedup_by_file=self.config.dedup_by_file,
        )

    # ── Asynchronous API ──────────────────────────────────────────────────

    async def aretrieve(
        self,
        query: str,
        *,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filters: Optional[dict[str, Any]] = None,
        custom_filter: Optional[Callable[[dict[str, Any]], bool]] = None,
        language: Optional[str] = None,
    ) -> list[CodeChunk]:
        """
        Async wrapper around ``retrieve()``.

        Runs the CPU-bound FAISS search in a thread-pool executor so it does
        not block the ``asyncio`` event loop used by FastAPI and LangGraph.

        Example::

            async def enrich_node(state: ReviewState) -> dict:
                retriever = state["retriever"]
                for finding in state["bug_findings"]:
                    neighbors = await retriever.aretrieve(finding.description, k=5)
                    finding.related_chunk_ids = [c.chunk_id for c in neighbors]
                return {}
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.retrieve(
                query,
                k=k,
                score_threshold=score_threshold,
                filters=filters,
                custom_filter=custom_filter,
                language=language,
            ),
        )

    def retrieve_for_finding(
        self,
        finding_description: str,
        finding_file_path: Optional[str] = None,
        *,
        k: Optional[int] = None,
        prefer_same_file: bool = True,
        score_threshold: float = 0.3,
    ) -> list[CodeChunk]:
        """
        Retrieve context chunks relevant to a specific code finding.

        Higher-level helper for ``enrich_node``.  When *prefer_same_file* is
        ``True`` and *finding_file_path* is known, attempts same-file results
        first, then fills remaining slots from the global index.

        Returns:
            list[CodeChunk], length ≤ k, same-file results first.
        """
        effective_k = k if k is not None else self.config.top_k
        results: list[CodeChunk] = []
        seen_ids: set[str] = set()

        # Step 1: same-file results
        if prefer_same_file and finding_file_path:
            for c in self.retrieve(
                finding_description,
                k=effective_k,
                score_threshold=score_threshold,
                filters={"file_path": finding_file_path},
            ):
                if c.chunk_id not in seen_ids:
                    results.append(c)
                    seen_ids.add(c.chunk_id)

        # Step 2: global fill
        for c in self.retrieve(
            finding_description,
            k=effective_k,
            score_threshold=score_threshold,
        ):
            if c.chunk_id not in seen_ids:
                results.append(c)
                seen_ids.add(c.chunk_id)
            if len(results) >= effective_k:
                break

        logger.debug(
            "retriever_for_finding_complete",
            preview=finding_description[:60],
            file_path=finding_file_path,
            returned=len(results),
        )
        return results[:effective_k]

    # ── Inspection helpers ────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """True if the vector store handle contains at least one document."""
        return self.handle.doc_count > 0

    def chunk_map_size(self) -> int:
        """Return the number of entries in the chunk map (0 if no map)."""
        return len(self.chunk_map) if self.chunk_map else 0

    def __repr__(self) -> str:
        return (
            f"ChunkRetriever("
            f"store={self.handle.store_name!r}, "
            f"docs={self.handle.doc_count}, "
            f"map_size={self.chunk_map_size()}, "
            f"top_k={self.config.top_k})"
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_query(query: str) -> None:
    """Raise EmptyQueryError if *query* is empty or whitespace-only."""
    if not query or not query.strip():
        raise EmptyQueryError(
            "Retrieval query must be a non-empty string. "
            "Pass the finding description or a descriptive code pattern."
        )


def _result_to_chunk(
    result: SearchResult,
    chunk_map: Optional[dict[str, CodeChunk]],
) -> CodeChunk:
    """
    Convert a ``SearchResult`` into a ``CodeChunk``.

    Resolution order:
      1. chunk_map[chunk_id] → returns original object (has embedding).
      2. Reconstruct from result.metadata + result.page_content.
    """
    if chunk_map and result.chunk_id in chunk_map:
        return chunk_map[result.chunk_id]
    return _reconstruct_chunk(result)


def _reconstruct_chunk(result: SearchResult) -> CodeChunk:
    """
    Reconstruct a ``CodeChunk`` from ``SearchResult`` metadata and content.

    Used when the chunk_map is unavailable or the chunk_id is not found.
    The reconstructed chunk has ``embedding=None``.

    Raises:
        RetrieverError : Metadata is too malformed for valid reconstruction.
    """
    meta = result.metadata

    raw_lang = meta.get("language", "unknown")
    try:
        language = SupportedLanguage(raw_lang)
    except ValueError:
        language = _LANGUAGE_FALLBACK

    start_line: int = int(meta.get("start_line") or 1)
    end_line:   int = int(meta.get("end_line")   or start_line)
    if end_line < start_line:
        end_line = start_line

    file_path: str = meta.get("file_path") or "unknown"
    chunk_id:  str = result.chunk_id or _fallback_id(result)

    try:
        return CodeChunk(
            chunk_id=chunk_id,
            file_path=file_path,
            language=language,
            content=result.page_content,
            start_line=start_line,
            end_line=end_line,
            embedding=None,
            related_chunk_ids=[],
        )
    except Exception as exc:  # noqa: BLE001
        raise RetrieverError(
            f"Failed to reconstruct CodeChunk "
            f"(chunk_id={result.chunk_id!r}): {exc}"
        ) from exc


def _fallback_id(result: SearchResult) -> str:
    """Generate a deterministic fallback chunk_id when the result has none."""
    seed = f"{result.file_path}:{result.start_line}:{result.page_content[:40]}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))


def _merge_filters(
    explicit: Optional[dict[str, Any]],
    config_language: Optional[str],
    call_language: Optional[str],
) -> dict[str, Any]:
    """
    Merge filter sources in priority order: call_language > explicit > config.

    Returns an empty dict if no filters apply (caller passes None to avoid
    filtering overhead).
    """
    merged: dict[str, Any] = {}
    if config_language:
        merged["language"] = config_language
    if explicit:
        merged.update(explicit)
    if call_language:
        merged["language"] = call_language
    return merged
