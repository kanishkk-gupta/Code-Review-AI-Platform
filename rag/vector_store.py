"""
rag/vector_store.py
====================
FAISS index lifecycle management.

Responsibilities:
  - Build a per-job FAISS IndexFlatL2 from a list of CodeChunks
  - Store the chunk_id → array_index mapping
  - Expose similarity_search() for enrich_node

Per-job lifecycle:
  - Created by ingest_node
  - Queried by enrich_node
  - Destroyed (set to None) by compile_node to free memory

Usage:
    from rag.vector_store import build_index, similarity_search
    index, registry = build_index(chunks)
    neighbors = similarity_search(index, registry, query_vector, k=3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import structlog

from schemas import CodeChunk

logger = structlog.get_logger(__name__)

EMBEDDING_DIM = 384  # sentence-transformers/all-MiniLM-L6-v2 output dimension


@dataclass
class FAISSIndex:
    """
    Container holding a FAISS index and its chunk ID registry.

    Attributes:
        index    : faiss.IndexFlatL2 — the actual FAISS index
        registry : list of chunk_id strings, parallel to FAISS internal array indices
    """
    index: any  # faiss.IndexFlatL2 — untyped to avoid hard import at module load
    registry: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.registry)


def build_index(chunks: list[CodeChunk]) -> FAISSIndex:
    """
    Build a FAISS IndexFlatL2 from the embedded chunks.

    Precondition: Each chunk must have a non-None .embedding (float32 ndarray).

    Args:
        chunks: List of CodeChunk objects with embeddings set.

    Returns:
        FAISSIndex containing the built index and chunk_id registry.

    Raises:
        ValueError: If any chunk is missing an embedding.
        RuntimeError: If FAISS build fails.
    """
    import faiss  # imported here to avoid hard dep at module load time

    embedded = [c for c in chunks if c.embedding is not None]
    if len(embedded) != len(chunks):
        missing = len(chunks) - len(embedded)
        logger.warning("faiss_build_missing_embeddings", missing=missing)

    if not embedded:
        raise ValueError("Cannot build FAISS index: no embedded chunks provided.")

    vectors = np.stack([c.embedding for c in embedded]).astype(np.float32)
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(vectors)

    registry = [c.chunk_id for c in embedded]

    logger.info("faiss_index_built", vector_count=index.ntotal, dim=EMBEDDING_DIM)
    return FAISSIndex(index=index, registry=registry)


def similarity_search(
    faiss_index: FAISSIndex,
    query_vector: np.ndarray,
    k: int = 3,
    chunk_map: Optional[dict[str, CodeChunk]] = None,
) -> list[str]:
    """
    Find the k nearest chunk_ids to the query_vector.

    Args:
        faiss_index  : FAISSIndex built by build_index().
        query_vector : float32 ndarray of shape (384,).
        k            : Number of nearest neighbors to return.
        chunk_map    : Optional dict[chunk_id → CodeChunk] for returning full objects.

    Returns:
        List of chunk_id strings (or CodeChunk objects if chunk_map provided),
        ordered by ascending L2 distance.
    """
    query = query_vector.reshape(1, -1).astype(np.float32)
    effective_k = min(k, faiss_index.size)

    distances, indices = faiss_index.index.search(query, effective_k)

    results: list[str] = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < 0 or idx >= len(faiss_index.registry):
            continue
        chunk_id = faiss_index.registry[idx]
        results.append(chunk_id)
        logger.debug("faiss_neighbor", chunk_id=chunk_id, distance=float(dist))

    return results
