"""
rag/faiss_store.py
==================
Production-grade FAISS vector store wrapper built on LangChain.

Public API
----------
    build_vector_store(documents, embeddings, config) -> VectorStoreHandle
    save_vector_store(handle, persist_dir, store_name) -> Path
    load_vector_store(persist_dir, embeddings, store_name) -> VectorStoreHandle
    similarity_search(handle, query, k, filters, score_threshold) -> list[SearchResult]

Architecture
------------
This module wraps LangChain's ``langchain_community.vectorstores.FAISS`` to
provide:

  1. **Persistence** — ``save_local()`` / ``load_local()`` backed by the
     filesystem.  Two files are written: ``<store_name>.faiss`` and
     ``<store_name>.pkl``.

  2. **Metadata filtering** — ``similarity_search`` accepts an optional
     ``filters`` dict.  Each key-value pair must be present (and equal) in
     the document's metadata for the result to be returned.  Filtering is
     applied as a post-retrieval step because LangChain's FAISS does not
     support pre-retrieval metadata predicates; we over-fetch (``k * _FILTER_OVERSAMPLE``)
     then prune.

  3. **Score thresholding** — Results below ``score_threshold`` (relevance
     score, 0-1) are discarded before returning to the caller.

  4. **Incremental merging** — ``merge_vector_stores`` combines two existing
     ``VectorStoreHandle`` objects into one, useful when processing large repos
     in shards.

Persistence Layout
------------------
::

    <persist_dir>/
    └── <store_name>/
        ├── index.faiss   # Raw FAISS index (binary)
        └── index.pkl     # Docstore + chunk_id registry (pickle)

Search Return Type
------------------
Every result is a ``SearchResult`` dataclass::

    SearchResult(
        chunk_id      = "3fa85f64-...",   # from Document.metadata["chunk_id"]
        page_content  = "def foo(): ...", # the raw text chunk
        metadata      = {...},             # full Document metadata dict
        relevance_score = 0.87,           # cosine similarity [0, 1]
    )

Metadata Filtering Contract
---------------------------
Filters are key-value equality checks applied to Document metadata:

    filters = {
        "language":   "python",
        "class_name": "AuthService",
    }

    # Returned only if doc.metadata["language"] == "python"
    # AND doc.metadata["class_name"] == "AuthService"

``None`` values in the filter dict are treated as "match any value for this key"
(i.e., the key may or may not be present).

Design Decisions
----------------
* ``allow_dangerous_deserialization=True`` is required for ``load_local()`` in
  newer LangChain versions because the pickle file may contain arbitrary Python
  objects.  This is safe here because we control both sides of the pickle.
* FAISS ``IndexFlatL2`` is used internally by LangChain; scores returned by
  ``similarity_search_with_relevance_scores`` are transformed to [0, 1] cosine
  similarity by LangChain's normalization layer.
* Over-sampling factor ``_FILTER_OVERSAMPLE = 5`` means we fetch ``k * 5``
  candidates before filtering.  Increase if metadata filters are very selective.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_STORE_NAME: str = "codeguardian"
DEFAULT_K: int = 5
DEFAULT_SCORE_THRESHOLD: float = 0.0   # accept all results by default
_FILTER_OVERSAMPLE: int = 5            # fetch k×5 candidates before metadata filtering

# ---------------------------------------------------------------------------
# Data Transfer Objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchResult:
    """
    A single result returned by ``similarity_search()``.

    Attributes:
        chunk_id        : Unique chunk identifier (UUID5, from Document metadata).
        page_content    : The raw text content of this chunk.
        metadata        : Complete Document metadata dict.
        relevance_score : Cosine similarity in [0.0, 1.0]; higher = more similar.
    """
    chunk_id:        str
    page_content:    str
    metadata:        dict[str, Any]
    relevance_score: float

    @property
    def file_path(self) -> Optional[str]:
        return self.metadata.get("file_path")

    @property
    def language(self) -> Optional[str]:
        return self.metadata.get("language")

    @property
    def class_name(self) -> Optional[str]:
        return self.metadata.get("class_name")

    @property
    def function_name(self) -> Optional[str]:
        return self.metadata.get("function_name")

    @property
    def start_line(self) -> Optional[int]:
        return self.metadata.get("start_line")

    @property
    def end_line(self) -> Optional[int]:
        return self.metadata.get("end_line")

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (safe for JSON / Pydantic)."""
        return {
            "chunk_id":        self.chunk_id,
            "page_content":    self.page_content,
            "metadata":        self.metadata,
            "relevance_score": self.relevance_score,
        }


@dataclass
class VectorStoreHandle:
    """
    Container holding the active FAISS vector store and its configuration.

    Attributes:
        store      : The LangChain ``FAISS`` instance.
        store_name : Logical name used for persistence filenames.
        doc_count  : Number of documents currently indexed.
        embeddings : The ``CodeGuardianEmbeddings`` used to build the store.
    """
    store:      Any                  # langchain_community.vectorstores.FAISS
    store_name: str
    doc_count:  int
    embeddings: Any                  # rag.embeddings.CodeGuardianEmbeddings

    def __repr__(self) -> str:
        return (
            f"VectorStoreHandle("
            f"name={self.store_name!r}, "
            f"docs={self.doc_count}, "
            f"model={getattr(self.embeddings, 'model_name', '?')})"
        )


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class VectorStoreError(RuntimeError):
    """Base exception for all rag/faiss_store errors."""


class VectorStoreNotFoundError(VectorStoreError):
    """Raised when a persisted store directory or required files do not exist."""


class EmptyDocumentListError(VectorStoreError):
    """Raised when ``build_vector_store`` receives an empty document list."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_vector_store(
    documents: list,
    embeddings: Optional[Any] = None,
    *,
    store_name: str = DEFAULT_STORE_NAME,
    batch_size: int = 64,
) -> VectorStoreHandle:
    """
    Build an in-memory FAISS vector store from a list of LangChain Documents.

    Documents are embedded in batches using ``embeddings``.  If ``embeddings``
    is ``None``, the default ``CodeGuardianEmbeddings`` singleton is loaded.

    Args:
        documents  : List of ``langchain.schema.Document`` objects from
                     ``rag.chunker.chunk_repository()``.
        embeddings : ``CodeGuardianEmbeddings`` instance.  If ``None``, one is
                     created via ``load_embedding_model(warmup=True)``.
        store_name : Logical name for the store (used in persistence filenames).
        batch_size : Documents embedded per batch (tunes memory vs. speed).

    Returns:
        ``VectorStoreHandle`` ready for search or persistence.

    Raises:
        EmptyDocumentListError : ``documents`` is empty.
        ImportError            : ``faiss-cpu`` or ``langchain-community``
                                 is not installed.

    Example::

        from rag.chunker import chunk_repository
        from rag.faiss_store import build_vector_store, save_vector_store

        docs = chunk_repository(parsed_repo)
        handle = build_vector_store(docs)
        save_vector_store(handle, "/tmp/stores")
    """
    if not documents:
        raise EmptyDocumentListError(
            "Cannot build vector store: document list is empty. "
            "Ensure chunk_repository() returned at least one Document."
        )

    embeddings = _resolve_embeddings(embeddings)

    logger.info(
        "faiss_store_build_start",
        store=store_name,
        documents=len(documents),
        model=getattr(embeddings, "model_name", "?"),
    )

    try:
        from langchain_community.vectorstores import FAISS
    except ImportError as exc:
        raise ImportError(
            "langchain-community is required: pip install langchain-community faiss-cpu"
        ) from exc

    # LangChain FAISS.from_documents embeds all texts and builds the index
    store = FAISS.from_documents(documents, embeddings)

    handle = VectorStoreHandle(
        store=store,
        store_name=store_name,
        doc_count=len(documents),
        embeddings=embeddings,
    )

    logger.info(
        "faiss_store_build_complete",
        store=store_name,
        documents=len(documents),
        index_size=store.index.ntotal if hasattr(store, "index") else "?",
    )
    return handle


def save_vector_store(
    handle: VectorStoreHandle,
    persist_dir: str | Path,
    *,
    store_name: Optional[str] = None,
    overwrite: bool = True,
) -> Path:
    """
    Persist *handle* to the filesystem using LangChain's ``save_local()``.

    The store is written to::

        <persist_dir>/<store_name>/
            index.faiss
            index.pkl

    A companion ``metadata.json`` file is also written alongside the index to
    capture provenance information (doc_count, model_name, timestamp).

    Args:
        handle      : ``VectorStoreHandle`` returned by ``build_vector_store()``.
        persist_dir : Parent directory.  Created if it does not exist.
        store_name  : Override the store name.  Defaults to ``handle.store_name``.
        overwrite   : If ``False`` and the target directory already exists,
                      raise ``VectorStoreError``.

    Returns:
        ``Path`` to the store directory (i.e. ``<persist_dir>/<store_name>``).

    Raises:
        VectorStoreError : ``overwrite=False`` and the directory already exists.

    Example::

        path = save_vector_store(handle, "/data/stores", store_name="job-123")
        print(path)  # /data/stores/job-123
    """
    name = store_name or handle.store_name
    store_dir = Path(persist_dir) / name

    if store_dir.exists() and not overwrite:
        raise VectorStoreError(
            f"Store directory already exists: {store_dir}. "
            "Pass overwrite=True to replace it."
        )

    # Clear existing data if overwriting
    if store_dir.exists() and overwrite:
        shutil.rmtree(store_dir)

    store_dir.mkdir(parents=True, exist_ok=True)

    logger.info("faiss_store_save_start", path=str(store_dir), name=name)

    handle.store.save_local(str(store_dir))

    # Write provenance metadata alongside the FAISS files
    _write_store_metadata(store_dir, handle, name)

    logger.info(
        "faiss_store_save_complete",
        path=str(store_dir),
        files=list(store_dir.iterdir()),
    )
    return store_dir


def load_vector_store(
    persist_dir: str | Path,
    embeddings: Optional[Any] = None,
    *,
    store_name: str = DEFAULT_STORE_NAME,
) -> VectorStoreHandle:
    """
    Load a previously persisted FAISS vector store from the filesystem.

    Requires the embedding model used during ``build_vector_store`` (or one
    with an identical output dimension) so that query vectors are compatible
    with the stored index.

    Args:
        persist_dir : Parent directory containing ``<store_name>/``.
        embeddings  : ``CodeGuardianEmbeddings`` instance.  If ``None``, the
                      default singleton is loaded.
        store_name  : Sub-directory name used when saving.

    Returns:
        ``VectorStoreHandle`` ready for similarity search.

    Raises:
        VectorStoreNotFoundError : The directory or required FAISS files are missing.
        ImportError              : ``faiss-cpu`` or ``langchain-community`` is missing.

    Example::

        handle = load_vector_store("/data/stores", store_name="job-123")
        results = similarity_search(handle, "authentication bypass", k=5)
    """
    store_dir = Path(persist_dir) / store_name

    if not store_dir.is_dir():
        raise VectorStoreNotFoundError(
            f"Vector store directory not found: {store_dir}. "
            "Ensure save_vector_store() was called with the same store_name."
        )

    # Check for required FAISS files
    required = ["index.faiss", "index.pkl"]
    for fname in required:
        if not (store_dir / fname).exists():
            raise VectorStoreNotFoundError(
                f"Required FAISS file missing: {store_dir / fname}"
            )

    embeddings = _resolve_embeddings(embeddings)

    logger.info("faiss_store_load_start", path=str(store_dir), name=store_name)

    try:
        from langchain_community.vectorstores import FAISS
    except ImportError as exc:
        raise ImportError(
            "langchain-community is required: pip install langchain-community faiss-cpu"
        ) from exc

    # allow_dangerous_deserialization is required for newer LangChain versions
    # because the .pkl file uses pickle under the hood.
    store = FAISS.load_local(
        str(store_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # Load provenance metadata if available
    meta = _read_store_metadata(store_dir)
    doc_count = meta.get("doc_count", 0)

    handle = VectorStoreHandle(
        store=store,
        store_name=store_name,
        doc_count=doc_count,
        embeddings=embeddings,
    )

    logger.info(
        "faiss_store_load_complete",
        path=str(store_dir),
        name=store_name,
        doc_count=doc_count,
        index_size=store.index.ntotal if hasattr(store, "index") else "?",
    )
    return handle


def similarity_search(
    handle: VectorStoreHandle,
    query: str,
    *,
    k: int = DEFAULT_K,
    filters: Optional[dict[str, Any]] = None,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    custom_filter: Optional[Callable[[dict[str, Any]], bool]] = None,
) -> list[SearchResult]:
    """
    Perform semantic similarity search against the vector store.

    Supports metadata filtering (equality checks) and relevance score
    thresholding.  Results are returned in descending relevance order.

    Args:
        handle          : ``VectorStoreHandle`` returned by ``build_vector_store``
                          or ``load_vector_store``.
        query           : Natural-language or code-snippet query string.
        k               : Maximum number of results to return after filtering.
        filters         : Optional dict of ``{metadata_key: expected_value}``.
                          All pairs must match for a result to be included.
                          Set a value to ``None`` to match any value for that key.
        score_threshold : Minimum relevance score [0.0–1.0] to include a result.
                          Default is 0.0 (include all).
        custom_filter   : Optional callable ``(metadata_dict) -> bool`` for
                          complex filtering logic not expressible as a dict.
                          Applied after the ``filters`` dict check.

    Returns:
        List of ``SearchResult`` objects, sorted by ``relevance_score`` descending.
        May be shorter than *k* if filters or threshold exclude results.

    Raises:
        ValueError : ``query`` is empty.
        VectorStoreError : The store handle is in an invalid state.

    Example::

        results = similarity_search(
            handle,
            query="SQL injection vulnerability",
            k=5,
            filters={"language": "python"},
            score_threshold=0.5,
        )
        for r in results:
            print(r.chunk_id, r.relevance_score, r.file_path)
    """
    if not query.strip():
        raise ValueError("Query must be a non-empty string.")

    # Over-fetch to compensate for metadata filtering reducing result count
    fetch_k = k * _FILTER_OVERSAMPLE if (filters or custom_filter) else k
    fetch_k = max(fetch_k, k)

    logger.debug(
        "faiss_search_start",
        query_preview=query[:80],
        k=k,
        fetch_k=fetch_k,
        filters=filters,
        score_threshold=score_threshold,
    )

    try:
        raw_results: list[tuple] = handle.store.similarity_search_with_relevance_scores(
            query,
            k=fetch_k,
        )
    except Exception as exc:  # noqa: BLE001
        raise VectorStoreError(f"FAISS search failed: {exc}") from exc

    results: list[SearchResult] = []

    for doc, score in raw_results:
        meta = doc.metadata or {}

        # Score threshold gate
        if score < score_threshold:
            logger.debug("faiss_search_skip_score", score=score, threshold=score_threshold)
            continue

        # Metadata equality filter
        if filters and not _matches_filters(meta, filters):
            logger.debug("faiss_search_skip_filter", meta_keys=list(meta.keys()))
            continue

        # Custom callable filter
        if custom_filter and not custom_filter(meta):
            continue

        results.append(
            SearchResult(
                chunk_id=meta.get("chunk_id", ""),
                page_content=doc.page_content,
                metadata=meta,
                relevance_score=float(score),
            )
        )

        if len(results) >= k:
            break

    # Sort descending by score (already sorted by FAISS, but filters may reorder)
    results.sort(key=lambda r: r.relevance_score, reverse=True)

    logger.info(
        "faiss_search_complete",
        query_preview=query[:80],
        requested=k,
        returned=len(results),
        score_threshold=score_threshold,
    )
    return results


# ---------------------------------------------------------------------------
# Additional helpers
# ---------------------------------------------------------------------------


def merge_vector_stores(
    primary: VectorStoreHandle,
    secondary: VectorStoreHandle,
) -> VectorStoreHandle:
    """
    Merge *secondary* into *primary* in-place and return the combined handle.

    Useful when a large repository is chunked and embedded in shards that need
    to be consolidated into a single searchable index.

    Args:
        primary   : Base ``VectorStoreHandle`` (mutated in-place).
        secondary : Source ``VectorStoreHandle`` whose vectors are added.

    Returns:
        Updated *primary* handle with combined document count.

    Raises:
        VectorStoreError : The two stores use incompatible embedding dimensions.
    """
    logger.info(
        "faiss_store_merge_start",
        primary_docs=primary.doc_count,
        secondary_docs=secondary.doc_count,
    )

    try:
        primary.store.merge_from(secondary.store)
    except Exception as exc:  # noqa: BLE001
        raise VectorStoreError(
            f"Failed to merge vector stores: {exc}. "
            "Ensure both stores use the same embedding model and dimension."
        ) from exc

    primary.doc_count += secondary.doc_count

    logger.info(
        "faiss_store_merge_complete",
        total_docs=primary.doc_count,
        index_size=primary.store.index.ntotal if hasattr(primary.store, "index") else "?",
    )
    return primary


def get_store_info(handle: VectorStoreHandle) -> dict[str, Any]:
    """
    Return a summary dict about *handle* suitable for logging or health checks.

    Example::

        {
            "store_name": "codeguardian",
            "doc_count":  342,
            "index_size": 342,
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        }
    """
    index_size = handle.store.index.ntotal if hasattr(handle.store, "index") else None
    return {
        "store_name": handle.store_name,
        "doc_count":  handle.doc_count,
        "index_size": index_size,
        "model_name": getattr(handle.embeddings, "model_name", None),
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_embeddings(embeddings: Optional[Any]) -> Any:
    """Return *embeddings* if provided, or load the default singleton."""
    if embeddings is not None:
        return embeddings
    from rag.embeddings import load_embedding_model
    return load_embedding_model(warmup=True)


def _matches_filters(
    metadata: dict[str, Any],
    filters: dict[str, Any],
) -> bool:
    """
    Return True if *metadata* satisfies all equality conditions in *filters*.

    A filter value of ``None`` means "accept any value for this key" (the key
    may be absent or have any value).

    Args:
        metadata : Document metadata dict.
        filters  : ``{key: required_value}`` pairs.

    Returns:
        True if every non-None filter value matches; False otherwise.
    """
    for key, expected in filters.items():
        if expected is None:
            continue  # wildcard — skip this key
        actual = metadata.get(key)
        if actual != expected:
            return False
    return True


def _write_store_metadata(
    store_dir: Path,
    handle: VectorStoreHandle,
    store_name: str,
) -> None:
    """Write a JSON provenance file alongside the FAISS index files."""
    from datetime import datetime, timezone
    meta = {
        "store_name":  store_name,
        "doc_count":   handle.doc_count,
        "model_name":  getattr(handle.embeddings, "model_name", None),
        "device":      getattr(handle.embeddings, "device", None),
        "saved_at":    datetime.now(timezone.utc).isoformat(),
    }
    meta_path = store_dir / "store_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.debug("faiss_store_metadata_written", path=str(meta_path))


def _read_store_metadata(store_dir: Path) -> dict[str, Any]:
    """Read the JSON provenance file, returning {} if it does not exist."""
    meta_path = store_dir / "store_metadata.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("faiss_store_metadata_read_error", path=str(meta_path), error=str(exc))
        return {}
