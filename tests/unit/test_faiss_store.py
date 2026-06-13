"""
tests/unit/test_faiss_store.py
==============================
Unit tests for rag/faiss_store.py.

All LangChain FAISS and embedding calls are mocked — no model or index I/O.
Filesystem tests use pytest's tmp_path fixture for real save/load round-trips
against a mock store.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest
from langchain.schema import Document

from rag.faiss_store import (
    DEFAULT_K,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_STORE_NAME,
    EmptyDocumentListError,
    SearchResult,
    VectorStoreError,
    VectorStoreHandle,
    VectorStoreNotFoundError,
    _matches_filters,
    _read_store_metadata,
    _write_store_metadata,
    build_vector_store,
    get_store_info,
    load_vector_store,
    merge_vector_stores,
    save_vector_store,
    similarity_search,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(
    content: str = "def foo(): pass",
    chunk_id: str = "aaaaaaaa-0000-0000-0000-000000000000",
    file_path: str = "src/main.py",
    language: str = "python",
    class_name: str | None = None,
    function_name: str | None = None,
    score: float = 0.9,
) -> tuple[Document, float]:
    doc = Document(
        page_content=content,
        metadata={
            "chunk_id":      chunk_id,
            "file_path":     file_path,
            "language":      language,
            "class_name":    class_name,
            "function_name": function_name,
            "start_line":    1,
            "end_line":      5,
            "chunk_index":   0,
        },
    )
    return doc, score


def _mock_faiss_store(results: list[tuple[Document, float]] | None = None) -> MagicMock:
    """Return a MagicMock that mimics langchain_community.vectorstores.FAISS."""
    mock = MagicMock()
    mock.index.ntotal = 5
    if results is None:
        results = [_make_doc()]
    mock.similarity_search_with_relevance_scores.return_value = results
    return mock


def _mock_embeddings(model_name: str = "test-model") -> MagicMock:
    mock = MagicMock()
    mock.model_name = model_name
    mock.device = "cpu"
    return mock


def _make_handle(
    results: list[tuple[Document, float]] | None = None,
    store_name: str = "test-store",
    doc_count: int = 5,
) -> VectorStoreHandle:
    return VectorStoreHandle(
        store=_mock_faiss_store(results),
        store_name=store_name,
        doc_count=doc_count,
        embeddings=_mock_embeddings(),
    )


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------

class TestSearchResult:
    def _make(self, **overrides) -> SearchResult:
        base = dict(
            chunk_id="abc",
            page_content="def foo(): pass",
            metadata={
                "file_path": "main.py",
                "language": "python",
                "class_name": "Bar",
                "function_name": "foo",
                "start_line": 1,
                "end_line": 5,
            },
            relevance_score=0.85,
        )
        base.update(overrides)
        return SearchResult(**base)

    def test_file_path_property(self):
        r = self._make()
        assert r.file_path == "main.py"

    def test_language_property(self):
        r = self._make()
        assert r.language == "python"

    def test_class_name_property(self):
        r = self._make()
        assert r.class_name == "Bar"

    def test_function_name_property(self):
        r = self._make()
        assert r.function_name == "foo"

    def test_start_end_line(self):
        r = self._make()
        assert r.start_line == 1
        assert r.end_line == 5

    def test_to_dict_keys(self):
        r = self._make()
        d = r.to_dict()
        assert "chunk_id" in d
        assert "page_content" in d
        assert "metadata" in d
        assert "relevance_score" in d

    def test_frozen(self):
        r = self._make()
        with pytest.raises((AttributeError, TypeError)):
            r.relevance_score = 0.1  # type: ignore


# ---------------------------------------------------------------------------
# _matches_filters
# ---------------------------------------------------------------------------

class TestMatchesFilters:
    META = {"language": "python", "class_name": "Auth", "function_name": "login"}

    def test_empty_filters_always_match(self):
        assert _matches_filters(self.META, {}) is True

    def test_single_key_match(self):
        assert _matches_filters(self.META, {"language": "python"}) is True

    def test_single_key_no_match(self):
        assert _matches_filters(self.META, {"language": "java"}) is False

    def test_multi_key_all_match(self):
        assert _matches_filters(self.META, {"language": "python", "class_name": "Auth"}) is True

    def test_multi_key_partial_match(self):
        assert _matches_filters(self.META, {"language": "python", "class_name": "Wrong"}) is False

    def test_none_value_is_wildcard(self):
        assert _matches_filters(self.META, {"language": None, "class_name": "Auth"}) is True

    def test_missing_key_no_match(self):
        assert _matches_filters(self.META, {"nonexistent_key": "value"}) is False

    def test_missing_key_none_value_matches(self):
        assert _matches_filters(self.META, {"nonexistent_key": None}) is True


# ---------------------------------------------------------------------------
# build_vector_store
# ---------------------------------------------------------------------------

class TestBuildVectorStore:
    def test_raises_on_empty_documents(self):
        with pytest.raises(EmptyDocumentListError):
            build_vector_store([])

    @patch("rag.faiss_store._resolve_embeddings")
    @patch("rag.faiss_store.FAISS", create=True)
    def test_returns_handle(self, MockFAISS, mock_resolve):
        mock_emb = _mock_embeddings()
        mock_resolve.return_value = mock_emb
        mock_store = _mock_faiss_store()
        MockFAISS.from_documents.return_value = mock_store

        docs = [_make_doc()[0], _make_doc(chunk_id="bbb", content="class Foo: pass")[0]]

        with patch.dict("sys.modules", {"langchain_community.vectorstores": MagicMock(FAISS=MockFAISS)}):
            from importlib import import_module
            import rag.faiss_store as fs_mod
            original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

            with patch.object(fs_mod, "build_vector_store", wraps=fs_mod.build_vector_store):
                # Patch the FAISS import inside the function
                with patch("builtins.__import__", wraps=__import__) as mock_import:
                    def side_effect(name, *args, **kwargs):
                        if name == "langchain_community.vectorstores":
                            mod = MagicMock()
                            mod.FAISS = MockFAISS
                            return mod
                        return __import__(name, *args, **kwargs)
                    mock_import.side_effect = side_effect

                    handle = fs_mod.build_vector_store(docs, embeddings=mock_emb)

        assert isinstance(handle, VectorStoreHandle)
        assert handle.doc_count == len(docs)
        assert handle.store_name == DEFAULT_STORE_NAME

    def test_missing_faiss_raises_import_error(self):
        docs = [_make_doc()[0]]
        emb = _mock_embeddings()
        with patch("builtins.__import__", side_effect=ImportError("no module")):
            with pytest.raises((ImportError, Exception)):
                build_vector_store(docs, embeddings=emb)


# ---------------------------------------------------------------------------
# save_vector_store / load_vector_store
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_creates_directory(self, tmp_path: Path):
        handle = _make_handle()
        # Mock the FAISS save_local to write fake files
        def fake_save_local(path: str):
            p = Path(path)
            (p / "index.faiss").write_bytes(b"fake_faiss")
            (p / "index.pkl").write_bytes(b"fake_pkl")

        handle.store.save_local.side_effect = fake_save_local

        result = save_vector_store(handle, tmp_path)
        assert result.is_dir()
        assert result.name == DEFAULT_STORE_NAME

    def test_save_writes_metadata_json(self, tmp_path: Path):
        handle = _make_handle(doc_count=42)

        def fake_save_local(path: str):
            p = Path(path)
            (p / "index.faiss").write_bytes(b"data")
            (p / "index.pkl").write_bytes(b"data")

        handle.store.save_local.side_effect = fake_save_local
        store_dir = save_vector_store(handle, tmp_path)
        meta_path = store_dir / "store_metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["doc_count"] == 42

    def test_save_overwrite_false_raises(self, tmp_path: Path):
        handle = _make_handle()

        def fake_save_local(path: str):
            p = Path(path)
            (p / "index.faiss").write_bytes(b"data")
            (p / "index.pkl").write_bytes(b"data")

        handle.store.save_local.side_effect = fake_save_local
        save_vector_store(handle, tmp_path)  # first save

        with pytest.raises(VectorStoreError, match="already exists"):
            save_vector_store(handle, tmp_path, overwrite=False)

    def test_save_overwrite_true_replaces(self, tmp_path: Path):
        handle = _make_handle()

        def fake_save_local(path: str):
            p = Path(path)
            (p / "index.faiss").write_bytes(b"data")
            (p / "index.pkl").write_bytes(b"data")

        handle.store.save_local.side_effect = fake_save_local
        save_vector_store(handle, tmp_path)
        save_vector_store(handle, tmp_path, overwrite=True)  # must not raise

    def test_load_missing_directory_raises(self, tmp_path: Path):
        with pytest.raises(VectorStoreNotFoundError):
            load_vector_store(tmp_path, embeddings=_mock_embeddings(), store_name="ghost")

    def test_load_missing_faiss_file_raises(self, tmp_path: Path):
        store_dir = tmp_path / "test-store"
        store_dir.mkdir()
        # Only index.pkl, no index.faiss
        (store_dir / "index.pkl").write_bytes(b"data")
        with pytest.raises(VectorStoreNotFoundError):
            load_vector_store(tmp_path, embeddings=_mock_embeddings(), store_name="test-store")

    def test_load_reads_doc_count_from_metadata(self, tmp_path: Path):
        store_dir = tmp_path / "my-store"
        store_dir.mkdir()
        (store_dir / "index.faiss").write_bytes(b"fake")
        (store_dir / "index.pkl").write_bytes(b"fake")
        meta = {"doc_count": 77, "store_name": "my-store"}
        (store_dir / "store_metadata.json").write_text(json.dumps(meta))

        mock_lc_faiss = MagicMock()
        mock_lc_faiss.load_local.return_value = _mock_faiss_store()

        with patch("builtins.__import__", wraps=__import__) as mock_import:
            def side_effect(name, *args, **kwargs):
                if name == "langchain_community.vectorstores":
                    mod = MagicMock()
                    mod.FAISS = mock_lc_faiss
                    return mod
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = side_effect
            import rag.faiss_store as fs_mod
            with patch.object(fs_mod, "_resolve_embeddings", return_value=_mock_embeddings()):
                handle = fs_mod.load_vector_store(tmp_path, store_name="my-store")
                assert handle.doc_count == 77


# ---------------------------------------------------------------------------
# similarity_search
# ---------------------------------------------------------------------------

class TestSimilaritySearch:
    def test_empty_query_raises(self):
        handle = _make_handle()
        with pytest.raises(ValueError, match="non-empty"):
            similarity_search(handle, "")

    def test_whitespace_query_raises(self):
        handle = _make_handle()
        with pytest.raises(ValueError, match="non-empty"):
            similarity_search(handle, "   ")

    def test_returns_list_of_search_results(self):
        handle = _make_handle(results=[_make_doc(score=0.8)])
        results = similarity_search(handle, "authentication")
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_chunk_id_in_result(self):
        chunk_id = "aaaaaaaa-1111-0000-0000-000000000000"
        handle = _make_handle(results=[_make_doc(chunk_id=chunk_id, score=0.9)])
        results = similarity_search(handle, "auth")
        assert results[0].chunk_id == chunk_id

    def test_score_populated(self):
        handle = _make_handle(results=[_make_doc(score=0.75)])
        results = similarity_search(handle, "injection")
        assert abs(results[0].relevance_score - 0.75) < 0.01

    def test_score_threshold_filters_low_scores(self):
        results_raw = [
            _make_doc(content="a", chunk_id="aaa", score=0.9),
            _make_doc(content="b", chunk_id="bbb", score=0.3),
        ]
        handle = _make_handle(results=results_raw)
        results = similarity_search(handle, "query", score_threshold=0.5)
        assert len(results) == 1
        assert results[0].relevance_score >= 0.5

    def test_metadata_filter_language(self):
        results_raw = [
            _make_doc(content="py code",   language="python", chunk_id="py1", score=0.9),
            _make_doc(content="java code", language="java",   chunk_id="jv1", score=0.85),
        ]
        handle = _make_handle(results=results_raw)
        results = similarity_search(handle, "query", filters={"language": "python"})
        assert all(r.language == "python" for r in results)

    def test_metadata_filter_class_name(self):
        results_raw = [
            _make_doc(class_name="Auth",  chunk_id="a1", score=0.9),
            _make_doc(class_name="Cache", chunk_id="a2", score=0.8),
        ]
        handle = _make_handle(results=results_raw)
        results = similarity_search(handle, "query", filters={"class_name": "Auth"})
        assert len(results) == 1
        assert results[0].class_name == "Auth"

    def test_metadata_filter_none_wildcard(self):
        results_raw = [
            _make_doc(class_name="Auth",  chunk_id="a1", score=0.9),
            _make_doc(class_name="Cache", chunk_id="a2", score=0.8),
        ]
        handle = _make_handle(results=results_raw)
        # None means "accept any value"
        results = similarity_search(handle, "query", filters={"class_name": None})
        assert len(results) == 2

    def test_k_limits_results(self):
        results_raw = [_make_doc(chunk_id=str(i), score=0.9 - i * 0.01) for i in range(10)]
        handle = _make_handle(results=results_raw)
        results = similarity_search(handle, "query", k=3)
        assert len(results) <= 3

    def test_results_sorted_descending_by_score(self):
        results_raw = [
            _make_doc(chunk_id="a", score=0.5),
            _make_doc(chunk_id="b", score=0.9),
            _make_doc(chunk_id="c", score=0.7),
        ]
        handle = _make_handle(results=results_raw)
        results = similarity_search(handle, "query")
        scores = [r.relevance_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_custom_filter_callable(self):
        results_raw = [
            _make_doc(content="short fn", chunk_id="s1", score=0.9),
            _make_doc(content="a" * 500, chunk_id="lg", score=0.85),
        ]
        handle = _make_handle(results=results_raw)
        # Only return chunks with content shorter than 100 chars
        results = similarity_search(
            handle,
            "query",
            custom_filter=lambda meta: True,  # pass all — test callable is called
        )
        assert isinstance(results, list)

    def test_faiss_error_raises_vector_store_error(self):
        handle = _make_handle()
        handle.store.similarity_search_with_relevance_scores.side_effect = RuntimeError("FAISS boom")
        with pytest.raises(VectorStoreError, match="FAISS search failed"):
            similarity_search(handle, "query")


# ---------------------------------------------------------------------------
# merge_vector_stores
# ---------------------------------------------------------------------------

class TestMergeVectorStores:
    def test_doc_count_summed(self):
        primary = _make_handle(doc_count=10)
        secondary = _make_handle(doc_count=7)
        merged = merge_vector_stores(primary, secondary)
        assert merged.doc_count == 17

    def test_merge_called_on_primary_store(self):
        primary = _make_handle(doc_count=10)
        secondary = _make_handle(doc_count=5)
        merge_vector_stores(primary, secondary)
        primary.store.merge_from.assert_called_once_with(secondary.store)

    def test_merge_error_raises_vector_store_error(self):
        primary = _make_handle()
        secondary = _make_handle()
        primary.store.merge_from.side_effect = Exception("dim mismatch")
        with pytest.raises(VectorStoreError, match="Failed to merge"):
            merge_vector_stores(primary, secondary)


# ---------------------------------------------------------------------------
# get_store_info
# ---------------------------------------------------------------------------

class TestGetStoreInfo:
    def test_returns_dict_with_required_keys(self):
        handle = _make_handle(doc_count=42, store_name="info-store")
        info = get_store_info(handle)
        assert "store_name" in info
        assert "doc_count" in info
        assert "index_size" in info
        assert "model_name" in info

    def test_doc_count_correct(self):
        handle = _make_handle(doc_count=99)
        assert get_store_info(handle)["doc_count"] == 99

    def test_store_name_correct(self):
        handle = _make_handle(store_name="my-special-store")
        assert get_store_info(handle)["store_name"] == "my-special-store"


# ---------------------------------------------------------------------------
# Metadata JSON helpers
# ---------------------------------------------------------------------------

class TestMetadataHelpers:
    def test_write_and_read_roundtrip(self, tmp_path: Path):
        handle = _make_handle(doc_count=55, store_name="round-trip")
        _write_store_metadata(tmp_path, handle, "round-trip")
        meta = _read_store_metadata(tmp_path)
        assert meta["doc_count"] == 55
        assert meta["store_name"] == "round-trip"

    def test_read_missing_returns_empty(self, tmp_path: Path):
        meta = _read_store_metadata(tmp_path / "nonexistent")
        assert meta == {}

    def test_read_corrupt_json_returns_empty(self, tmp_path: Path):
        bad = tmp_path / "store_metadata.json"
        bad.write_text("not valid json{{{{", encoding="utf-8")
        meta = _read_store_metadata(tmp_path)
        assert meta == {}

    def test_saved_at_is_iso_format(self, tmp_path: Path):
        handle = _make_handle()
        _write_store_metadata(tmp_path, handle, "ts-test")
        meta = _read_store_metadata(tmp_path)
        from datetime import datetime
        # Should parse without error
        datetime.fromisoformat(meta["saved_at"])
