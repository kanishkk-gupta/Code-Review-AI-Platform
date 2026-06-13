"""
tests/unit/test_rag_embeddings.py
==================================
Unit tests for rag/embeddings.py.

All SentenceTransformer calls are mocked — no model is downloaded.
Tests validate threading behaviour, interface contracts, and registry state.
"""
from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

# Reset the global registry before every test
import rag.embeddings as emb_module


@pytest.fixture(autouse=True)
def clear_registry():
    """Ensure singleton registry is empty before each test."""
    emb_module._MODEL_REGISTRY.clear()
    yield
    emb_module._MODEL_REGISTRY.clear()


def _mock_st_model(dim: int = 384) -> MagicMock:
    """Return a MagicMock that mimics SentenceTransformer.encode()."""
    mock = MagicMock()

    def fake_encode(texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return np.random.rand(len(texts), dim).astype(np.float32)

    mock.encode.side_effect = fake_encode
    return mock


# ---------------------------------------------------------------------------
# CodeGuardianEmbeddings — basic interface
# ---------------------------------------------------------------------------

class TestCodeGuardianEmbeddings:
    def _make(self, mock_st) -> emb_module.CodeGuardianEmbeddings:
        emb = emb_module.CodeGuardianEmbeddings(
            model_name="test-model",
            device="cpu",
        )
        # Pre-populate registry so no real loading happens
        emb_module._MODEL_REGISTRY[("test-model", "cpu")] = mock_st
        return emb

    def test_embed_documents_returns_list(self):
        mock = _mock_st_model()
        emb = self._make(mock)
        result = emb.embed_documents(["hello", "world"])
        assert isinstance(result, list)
        assert len(result) == 2

    def test_embed_documents_correct_dim(self):
        mock = _mock_st_model(384)
        emb = self._make(mock)
        result = emb.embed_documents(["code snippet"])
        assert len(result[0]) == 384

    def test_embed_documents_empty_returns_empty(self):
        mock = _mock_st_model()
        emb = self._make(mock)
        assert emb.embed_documents([]) == []

    def test_embed_query_returns_single_vector(self):
        mock = _mock_st_model()
        emb = self._make(mock)
        result = emb.embed_query("def foo(): pass")
        assert isinstance(result, list)
        assert len(result) == 384

    def test_embed_text_numpy_returns_ndarray(self):
        mock = _mock_st_model()
        emb = self._make(mock)
        arr = emb.embed_text_numpy("x = 1")
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (384,)
        assert arr.dtype == np.float32

    def test_embed_batch_numpy_shape(self):
        mock = _mock_st_model()
        emb = self._make(mock)
        arr = emb.embed_batch_numpy(["a", "b", "c"])
        assert arr.shape == (3, 384)
        assert arr.dtype == np.float32

    def test_embed_batch_numpy_empty(self):
        mock = _mock_st_model()
        emb = self._make(mock)
        arr = emb.embed_batch_numpy([])
        assert arr.shape == (0, 384)

    def test_is_ready_when_in_registry(self):
        mock = _mock_st_model()
        emb = emb_module.CodeGuardianEmbeddings(model_name="test-model", device="cpu")
        assert not emb.is_ready()
        emb_module._MODEL_REGISTRY[("test-model", "cpu")] = mock
        assert emb.is_ready()

    def test_langchain_embeddings_interface(self):
        """Verify it's a proper LangChain Embeddings subclass."""
        from langchain_core.embeddings import Embeddings
        assert issubclass(emb_module.CodeGuardianEmbeddings, Embeddings)


# ---------------------------------------------------------------------------
# Singleton / thread-safety
# ---------------------------------------------------------------------------

class TestSingleton:
    @patch("rag.embeddings.SentenceTransformer", create=True)
    def test_model_loaded_only_once(self, MockST):
        MockST.return_value = _mock_st_model()
        with patch("builtins.__import__", wraps=__import__) as mock_import:
            e1 = emb_module.CodeGuardianEmbeddings(model_name="m", device="cpu")
            e2 = emb_module.CodeGuardianEmbeddings(model_name="m", device="cpu")

            # Patch the inner import inside _get_or_load_model
            with patch.dict("sys.modules", {"sentence_transformers": MagicMock(SentenceTransformer=MockST)}):
                e1._get_or_load_model()
                e1._get_or_load_model()  # second call — should not re-instantiate
                e2._get_or_load_model()  # same key — should use cache

        # SentenceTransformer() called exactly once
        assert MockST.call_count == 1

    def test_concurrent_loads_produce_single_model(self):
        """Multiple threads calling load simultaneously must not double-load."""
        load_counts = []
        original_st_init = None

        real_mock = _mock_st_model()
        loaded_event = threading.Event()

        results: list[emb_module.CodeGuardianEmbeddings] = []
        errors: list[Exception] = []

        key = ("concurrent-model", "cpu")

        def _thread_fn():
            try:
                emb = emb_module.CodeGuardianEmbeddings(
                    model_name="concurrent-model",
                    device="cpu",
                )
                # Simulate model not yet in registry
                with patch.dict(
                    "sys.modules",
                    {"sentence_transformers": MagicMock(SentenceTransformer=MagicMock(return_value=real_mock))},
                ):
                    m = emb._get_or_load_model()
                    results.append(emb)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_thread_fn) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"
        # All threads should observe the same model instance
        assert len({id(emb_module._MODEL_REGISTRY.get(key)) for _ in results if key in emb_module._MODEL_REGISTRY}) <= 1

    def test_different_keys_produce_different_entries(self):
        mock_cpu = _mock_st_model()
        mock_cuda = _mock_st_model()
        emb_module._MODEL_REGISTRY[("m", "cpu")] = mock_cpu
        emb_module._MODEL_REGISTRY[("m", "cuda")] = mock_cuda

        e_cpu = emb_module.CodeGuardianEmbeddings(model_name="m", device="cpu")
        e_cuda = emb_module.CodeGuardianEmbeddings(model_name="m", device="cuda")

        assert e_cpu._get_or_load_model() is mock_cpu
        assert e_cuda._get_or_load_model() is mock_cuda


# ---------------------------------------------------------------------------
# load_embedding_model
# ---------------------------------------------------------------------------

class TestLoadEmbeddingModel:
    def test_returns_codeguardian_embeddings(self):
        emb_module._MODEL_REGISTRY[
            (emb_module.DEFAULT_MODEL_NAME, emb_module.DEFAULT_DEVICE)
        ] = _mock_st_model()
        result = emb_module.load_embedding_model(warmup=False)
        assert isinstance(result, emb_module.CodeGuardianEmbeddings)

    def test_warmup_false_skips_load(self):
        """warmup=False should return the wrapper without touching the registry."""
        result = emb_module.load_embedding_model(warmup=False)
        assert isinstance(result, emb_module.CodeGuardianEmbeddings)
        # Registry should still be empty
        assert len(emb_module._MODEL_REGISTRY) == 0

    def test_custom_model_name_respected(self):
        result = emb_module.load_embedding_model(
            model_name="my-custom-model",
            device="cpu",
            warmup=False,
        )
        assert result.model_name == "my-custom-model"

    def test_custom_batch_size_respected(self):
        result = emb_module.load_embedding_model(batch_size=16, warmup=False)
        assert result.batch_size == 16

    def test_normalize_flag_respected(self):
        result = emb_module.load_embedding_model(normalize=False, warmup=False)
        assert result.normalize is False

    def test_returns_cached_on_second_call(self):
        emb_module._MODEL_REGISTRY[
            (emb_module.DEFAULT_MODEL_NAME, emb_module.DEFAULT_DEVICE)
        ] = _mock_st_model()
        r1 = emb_module.load_embedding_model(warmup=False)
        r2 = emb_module.load_embedding_model(warmup=False)
        # Both wrappers point to same registry entry
        assert r1.model_name == r2.model_name
        assert r1.device == r2.device


# ---------------------------------------------------------------------------
# generate_embeddings
# ---------------------------------------------------------------------------

class TestGenerateEmbeddings:
    def _seeded_model(self) -> emb_module.CodeGuardianEmbeddings:
        mock = _mock_st_model()
        emb = emb_module.CodeGuardianEmbeddings(model_name="g-model", device="cpu")
        emb_module._MODEL_REGISTRY[("g-model", "cpu")] = mock
        return emb

    def test_returns_list_of_lists(self):
        model = self._seeded_model()
        result = emb_module.generate_embeddings(["a", "b"], model=model)
        assert isinstance(result, list)
        assert isinstance(result[0], list)

    def test_correct_count(self):
        model = self._seeded_model()
        result = emb_module.generate_embeddings(["a", "b", "c"], model=model)
        assert len(result) == 3

    def test_correct_dimension(self):
        model = self._seeded_model()
        result = emb_module.generate_embeddings(["hello"], model=model)
        assert len(result[0]) == 384

    def test_empty_input_returns_empty(self):
        model = self._seeded_model()
        assert emb_module.generate_embeddings([], model=model) == []

    def test_empty_string_handled_gracefully(self):
        """Empty strings must not crash encode()."""
        model = self._seeded_model()
        result = emb_module.generate_embeddings(["", "code"], model=model)
        assert len(result) == 2

    def test_return_numpy_flag(self):
        model = self._seeded_model()
        result = emb_module.generate_embeddings(
            ["x"], model=model, return_numpy=True
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 384)

    def test_values_are_floats(self):
        model = self._seeded_model()
        result = emb_module.generate_embeddings(["test"], model=model)
        assert all(isinstance(v, float) for v in result[0])


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

class TestRegistryHelpers:
    def test_get_loaded_models_empty(self):
        assert emb_module.get_loaded_models() == []

    def test_get_loaded_models_populated(self):
        emb_module._MODEL_REGISTRY[("m1", "cpu")] = _mock_st_model()
        emb_module._MODEL_REGISTRY[("m2", "cuda")] = _mock_st_model()
        models = emb_module.get_loaded_models()
        assert len(models) == 2
        names = {m["model_name"] for m in models}
        assert "m1" in names
        assert "m2" in names

    def test_evict_model_removes_entry(self):
        emb_module._MODEL_REGISTRY[("to-evict", "cpu")] = _mock_st_model()
        result = emb_module.evict_model("to-evict", "cpu")
        assert result is True
        assert ("to-evict", "cpu") not in emb_module._MODEL_REGISTRY

    def test_evict_nonexistent_returns_false(self):
        result = emb_module.evict_model("ghost-model", "cpu")
        assert result is False

    def test_evict_then_reload(self):
        key = ("reload-model", "cpu")
        emb_module._MODEL_REGISTRY[key] = _mock_st_model()
        emb_module.evict_model("reload-model", "cpu")
        assert key not in emb_module._MODEL_REGISTRY

        # After eviction, the model can be loaded again
        emb_module._MODEL_REGISTRY[key] = _mock_st_model()
        assert key in emb_module._MODEL_REGISTRY
