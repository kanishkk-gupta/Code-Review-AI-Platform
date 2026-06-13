"""
rag/embedder.py
================
SentenceTransformer wrapper for producing code embeddings.

Responsibilities:
  - Load and cache the embedding model at startup
  - Expose embed_text(text) and embed_chunks(chunks) for use in ingest_node
  - Run embedding in a thread executor (CPU-bound) to avoid blocking the event loop

Usage:
    from rag.embedder import embed_chunks
    chunks = await embed_chunks(chunks)  # assigns chunk.embedding
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Optional

import numpy as np
import structlog

from schemas import CodeChunk

logger = structlog.get_logger(__name__)


@lru_cache(maxsize=1)
def _get_model() -> "SentenceTransformer":  # type: ignore[name-defined]
    """
    Load and cache the SentenceTransformer model.
    Called once at first embed request; cached for process lifetime.
    """
    from sentence_transformers import SentenceTransformer
    from config.settings import get_settings

    settings = get_settings()
    logger.info("embedder_loading_model", model=settings.embedding_model, device=settings.embedding_device)

    model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)
    logger.info("embedder_model_loaded", model=settings.embedding_model)
    return model


def is_embedder_ready() -> bool:
    """Return True if the model is already loaded (warm)."""
    return _get_model.cache_info().currsize > 0


def embed_text(text: str) -> np.ndarray:
    """
    Synchronously embed a single text string.
    Returns numpy float32 ndarray of shape (384,).
    """
    model = _get_model()
    vector: np.ndarray = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return vector.astype(np.float32)


async def embed_chunks(chunks: list[CodeChunk]) -> list[CodeChunk]:
    """
    Embed all chunks asynchronously (runs model in thread executor).
    Assigns CodeChunk.embedding for each chunk in-place.

    Args:
        chunks: List of CodeChunk objects with content set.

    Returns:
        Same list with .embedding assigned on each chunk.
    """
    if not chunks:
        return chunks

    loop = asyncio.get_running_loop()
    texts = [chunk.content for chunk in chunks]

    # Run CPU-bound model.encode() in a thread pool
    def _batch_encode() -> list[np.ndarray]:
        model = _get_model()
        return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

    embeddings = await loop.run_in_executor(None, _batch_encode)

    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding.astype(np.float32)

    logger.debug("embed_chunks_complete", count=len(chunks))
    return chunks
