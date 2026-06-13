"""
rag/embeddings.py
=================
Thread-safe singleton embedding model wrapper.

Public API
----------
    load_embedding_model(model_name, device) -> CodeGuardianEmbeddings
    generate_embeddings(texts, model)        -> list[list[float]]

Architecture
------------
``CodeGuardianEmbeddings`` is a full LangChain ``Embeddings`` subclass.
This means it can be passed **directly** to LangChain's FAISS integration:

    from langchain_community.vectorstores import FAISS
    from rag.embeddings import load_embedding_model

    model = load_embedding_model()
    vectorstore = FAISS.from_documents(docs, model)
    results = vectorstore.similarity_search(query, k=3)

Singleton Pattern
-----------------
The underlying ``SentenceTransformer`` instance is held in a module-level
registry (``_MODEL_REGISTRY``) keyed by ``(model_name, device)``.

A ``threading.Lock`` guarantees that only one thread ever initialises a
given model, even when multiple workers call ``load_embedding_model()``
concurrently at startup. Subsequent callers return the cached instance
immediately without acquiring the lock.

::

    T1: lock acquired → model loading...
    T2: waiting on lock
    T3: waiting on lock
    T1: model loaded, lock released
    T2: lock acquired → already in registry → returns cached instance
    T3: lock acquired → already in registry → returns cached instance

Model
-----
Default: ``sentence-transformers/all-MiniLM-L6-v2``
  - Output dimension : 384
  - Context window   : 256 word-pieces
  - Normalised output: L2-normalised by default (cosine similarity = dot product)

Logging
-------
Uses structlog (project-wide standard).
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import structlog
from langchain_core.embeddings import Embeddings

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DEVICE: str = "cpu"
EMBEDDING_DIM: int = 384           # all-MiniLM-L6-v2 output dimension
DEFAULT_BATCH_SIZE: int = 64       # sentences per encode() call
DEFAULT_NORMALIZE: bool = True     # L2-normalize for cosine similarity


# ---------------------------------------------------------------------------
# Thread-safe singleton registry
# ---------------------------------------------------------------------------

# Key  : (model_name, device)
# Value: SentenceTransformer instance
_MODEL_REGISTRY: dict[tuple[str, str], "SentenceTransformer"] = {}  # type: ignore[type-arg]
_REGISTRY_LOCK: threading.Lock = threading.Lock()


# ---------------------------------------------------------------------------
# LangChain-compatible Embeddings class
# ---------------------------------------------------------------------------


class CodeGuardianEmbeddings(Embeddings):
    """
    LangChain ``Embeddings`` implementation backed by SentenceTransformer.

    Compatible with both langchain_core 0.x (where Embeddings was a Pydantic
    BaseModel) and 1.x (where Embeddings is a plain ABC).  Uses an explicit
    ``__init__`` instead of Pydantic field declarations so construction works
    in both versions.

        vectorstore = FAISS.from_documents(docs, CodeGuardianEmbeddings())
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        normalize: bool = DEFAULT_NORMALIZE,
    ) -> None:
        self.model_name  = model_name
        self.device      = device
        self.batch_size  = batch_size
        self.normalize   = normalize
        self._model: Optional["SentenceTransformer"] = None  # type: ignore[type-arg]

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def warmup(self) -> "CodeGuardianEmbeddings":
        """
        Eagerly load the model into the singleton registry.

        Call this at application startup (e.g. in FastAPI ``lifespan``) so that
        the first embedding request does not bear the model-load latency.

        Returns:
            ``self`` for method chaining.
        """
        self._get_or_load_model()
        return self

    def is_ready(self) -> bool:
        """Return True if the model is loaded and cached."""
        return (self.model_name, self.device) in _MODEL_REGISTRY

    # ── LangChain Embeddings interface ────────────────────────────────────

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of documents (LangChain interface).

        Called by LangChain's FAISS integration when adding documents to the
        vector store.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of float vectors, one per input text.
            Shape: ``(len(texts), EMBEDDING_DIM)``
        """
        if not texts:
            return []
        return generate_embeddings(texts, model=self)

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string (LangChain interface).

        Called by LangChain when performing similarity search.

        Args:
            text: The query string to embed.

        Returns:
            Float vector of length ``EMBEDDING_DIM``.
        """
        return generate_embeddings([text], model=self)[0]

    # ── Extended API (numpy, for rag/vector_store.py) ─────────────────────

    def embed_text_numpy(self, text: str) -> np.ndarray:
        """
        Embed a single text string and return a ``float32`` numpy array.

        This is a convenience wrapper used by ``rag/vector_store.py`` which
        expects raw ``ndarray`` inputs for FAISS.

        Args:
            text: Source text to embed.

        Returns:
            ``numpy.ndarray`` of shape ``(384,)``, dtype ``float32``.
        """
        model = self._get_or_load_model()
        vector: np.ndarray = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return vector.astype(np.float32)

    def embed_batch_numpy(self, texts: list[str]) -> np.ndarray:
        """
        Embed a batch of texts and return a 2-D ``float32`` numpy array.

        Args:
            texts: List of source texts.

        Returns:
            ``numpy.ndarray`` of shape ``(len(texts), 384)``, dtype ``float32``.
        """
        if not texts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        model = self._get_or_load_model()
        vectors: np.ndarray = model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    # ── Private helpers ───────────────────────────────────────────────────

    def _get_or_load_model(self) -> "SentenceTransformer":  # type: ignore[type-arg]
        """
        Return the cached ``SentenceTransformer`` instance for this
        ``(model_name, device)`` pair, loading it if necessary.

        Thread-safety: double-checked locking — the common path (cache hit)
        never acquires the lock, so there is no contention in steady state.
        """
        key = (self.model_name, self.device)

        # Fast path: already in registry (no lock needed for read)
        if key in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[key]

        # Slow path: first call (or eviction) — acquire the lock
        with _REGISTRY_LOCK:
            # Re-check inside the lock (another thread may have loaded it)
            if key in _MODEL_REGISTRY:
                return _MODEL_REGISTRY[key]

            logger.info(
                "embeddings_loading_model",
                model=self.model_name,
                device=self.device,
            )

            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required: "
                    "pip install sentence-transformers"
                ) from exc

            model = SentenceTransformer(self.model_name, device=self.device)

            # Verify output dimension matches expectation
            try:
                test_vec = model.encode(["test"], convert_to_numpy=True)
                actual_dim = test_vec.shape[-1]
                if actual_dim != EMBEDDING_DIM:
                    logger.warning(
                        "embeddings_dim_mismatch",
                        expected=EMBEDDING_DIM,
                        actual=actual_dim,
                        model=self.model_name,
                    )
            except Exception:  # noqa: BLE001
                pass  # dimension check is best-effort

            _MODEL_REGISTRY[key] = model

            logger.info(
                "embeddings_model_loaded",
                model=self.model_name,
                device=self.device,
                registry_size=len(_MODEL_REGISTRY),
            )

        return _MODEL_REGISTRY[key]


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------


def load_embedding_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = DEFAULT_DEVICE,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    normalize: bool = DEFAULT_NORMALIZE,
    warmup: bool = True,
) -> CodeGuardianEmbeddings:
    """
    Build and return a thread-safe, singleton-backed ``CodeGuardianEmbeddings``
    instance.

    The underlying ``SentenceTransformer`` is loaded **once** per
    ``(model_name, device)`` pair across the entire process lifetime.
    Subsequent calls with the same parameters return the cached wrapper
    immediately (O(1), no I/O).

    Args:
        model_name  : HuggingFace model identifier.
                      Default: ``"sentence-transformers/all-MiniLM-L6-v2"``
        device      : Torch device (``"cpu"``, ``"cuda"``, ``"mps"``).
        batch_size  : Texts encoded per ``model.encode()`` call.
        normalize   : L2-normalise output (recommended for cosine similarity).
        warmup      : If ``True``, eagerly loads the model before returning
                      so the first ``embed_documents`` call has no latency.

    Returns:
        A ready-to-use ``CodeGuardianEmbeddings`` instance compatible with
        ``langchain_community.vectorstores.FAISS``.

    Raises:
        ImportError : ``sentence-transformers`` is not installed.

    Example::

        # At app startup
        model = load_embedding_model()

        # Use with LangChain FAISS directly
        from langchain_community.vectorstores import FAISS
        vectorstore = FAISS.from_documents(documents, model)

        # Use for raw numpy access
        vec = model.embed_text_numpy("def foo(): pass")  # shape (384,)
    """
    # Resolve device from Settings if caller uses the default
    resolved_device = device
    if device == DEFAULT_DEVICE:
        try:
            from config.settings import get_settings
            resolved_device = get_settings().embedding_device
        except Exception:  # noqa: BLE001
            resolved_device = DEFAULT_DEVICE

    resolved_model = model_name
    if model_name == DEFAULT_MODEL_NAME:
        try:
            from config.settings import get_settings
            resolved_model = get_settings().embedding_model
        except Exception:  # noqa: BLE001
            resolved_model = DEFAULT_MODEL_NAME

    embeddings = CodeGuardianEmbeddings(
        model_name=resolved_model,
        device=resolved_device,
        batch_size=batch_size,
        normalize=normalize,
    )

    if warmup:
        embeddings.warmup()

    logger.debug(
        "load_embedding_model_called",
        model=resolved_model,
        device=resolved_device,
        warmup=warmup,
    )
    return embeddings


def generate_embeddings(
    texts: list[str],
    *,
    model: Optional[CodeGuardianEmbeddings] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = DEFAULT_DEVICE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    normalize: bool = DEFAULT_NORMALIZE,
    return_numpy: bool = False,
) -> list[list[float]]:
    """
    Generate embeddings for a list of text strings.

    This function is the primary entry point for one-shot embedding calls.
    It accepts an optional pre-loaded *model* to avoid redundant model-load
    overhead in long-running processes.

    Args:
        texts       : Input strings to embed. Empty strings are allowed but
                      produce a zero vector.
        model       : Optional pre-loaded ``CodeGuardianEmbeddings`` instance.
                      If ``None``, one is created via ``load_embedding_model()``.
        model_name  : Model identifier (used only when *model* is ``None``).
        device      : Torch device (used only when *model* is ``None``).
        batch_size  : Texts per encode call (used only when *model* is ``None``).
        normalize   : L2-normalise output (used only when *model* is ``None``).
        return_numpy: If ``True``, returns a 2-D ``float32`` ndarray instead of
                      ``list[list[float]]``. Useful for feeding FAISS directly.

    Returns:
        ``list[list[float]]`` — one float vector per input text.
        (or ``numpy.ndarray`` of shape ``(N, 384)`` when ``return_numpy=True``)

    Raises:
        ImportError : ``sentence-transformers`` is not installed.
        ValueError  : ``texts`` contains non-string elements.

    Example::

        from rag.embeddings import generate_embeddings

        vecs = generate_embeddings(["def foo(): pass", "class Bar: ..."])
        print(len(vecs))        # 2
        print(len(vecs[0]))     # 384
    """
    if not texts:
        return []

    # Resolve or create the model
    if model is None:
        model = load_embedding_model(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            normalize=normalize,
            warmup=False,  # model will be loaded lazily on first encode
        )

    # Handle empty strings gracefully: replace with a space so encode() works
    sanitized = [t if t.strip() else " " for t in texts]

    logger.debug("generate_embeddings_start", count=len(sanitized))

    np_vecs: np.ndarray = model.embed_batch_numpy(sanitized)

    logger.debug(
        "generate_embeddings_complete",
        count=len(sanitized),
        dim=np_vecs.shape[-1] if np_vecs.ndim > 1 else "N/A",
    )

    if return_numpy:
        return np_vecs  # type: ignore[return-value]

    return np_vecs.tolist()


# ---------------------------------------------------------------------------
# Registry inspection helpers (useful for health-check endpoints)
# ---------------------------------------------------------------------------


def get_loaded_models() -> list[dict[str, str]]:
    """
    Return a list of all currently loaded models in the singleton registry.

    Each entry is a dict with ``model_name`` and ``device`` keys.
    Intended for use in ``GET /health`` responses.

    Example::

        [{"model_name": "sentence-transformers/all-MiniLM-L6-v2", "device": "cpu"}]
    """
    with _REGISTRY_LOCK:
        return [
            {"model_name": name, "device": dev}
            for name, dev in _MODEL_REGISTRY.keys()
        ]


def evict_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = DEFAULT_DEVICE,
) -> bool:
    """
    Remove a model from the singleton registry and release its memory.

    This is mainly useful in tests that need to reset state between runs,
    or in long-running processes that need to hot-swap models.

    Args:
        model_name : Model identifier to evict.
        device     : Device string used when the model was loaded.

    Returns:
        ``True`` if the model was found and evicted; ``False`` otherwise.
    """
    key = (model_name, device)
    with _REGISTRY_LOCK:
        if key in _MODEL_REGISTRY:
            del _MODEL_REGISTRY[key]
            logger.info("embeddings_model_evicted", model=model_name, device=device)
            return True
    return False
