"""
tests/unit/test_retriever.py
=============================
Unit tests for rag/retriever.py — fully offline, no FAISS or model I/O.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from schemas import CodeChunk, SupportedLanguage
from rag.faiss_store import SearchResult, VectorStoreHandle
from rag.retriever import (
    ChunkRetriever,
    EmptyQueryError,
    RetrieverConfig,
    RetrieverError,
    _fallback_id,
    _merge_filters,
    _reconstruct_chunk,
    _result_to_chunk,
    _validate_query,
    build_chunk_map,
    retrieve_chunks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    chunk_id: str = "aaaa-0000",
    content: str = "def foo(): pass",
    file_path: str = "src/main.py",
    language: str = "python",
    start_line: int = 1,
    end_line: int = 5,
    score: float = 0.9,
    class_name: str | None = None,
    function_name: str | None = None,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        page_content=content,
        metadata={
            "chunk_id":      chunk_id,
            "file_path":     file_path,
            "language":      language,
            "class_name":    class_name,
            "function_name": function_name,
            "start_line":    start_line,
            "end_line":      end_line,
            "chunk_index":   0,
        },
        relevance_score=score,
    )


def _make_chunk(
    chunk_id: str = "aaaa-0000",
    file_path: str = "src/main.py",
    language: SupportedLanguage = SupportedLanguage.PYTHON,
    content: str = "def foo(): pass",
    start_line: int = 1,
    end_line: int = 5,
) -> CodeChunk:
    return CodeChunk(
        chunk_id=chunk_id,
        file_path=file_path,
        language=language,
        content=content,
        start_line=start_line,
        end_line=end_line,
    )


def _mock_handle(results: list[SearchResult] | None = None) -> VectorStoreHandle:
    handle = MagicMock(spec=VectorStoreHandle)
    handle.store_name = "test"
    handle.doc_count = 10
    handle.embeddings = MagicMock()
    handle.embeddings.model_name = "test-model"
    return handle


# ---------------------------------------------------------------------------
# _validate_query
# ---------------------------------------------------------------------------

class TestValidateQuery:
    def test_valid_query_passes(self):
        _validate_query("SQL injection in login form")

    def test_empty_string_raises(self):
        with pytest.raises(EmptyQueryError):
            _validate_query("")

    def test_whitespace_raises(self):
        with pytest.raises(EmptyQueryError):
            _validate_query("   \t\n  ")

    def test_none_raises(self):
        with pytest.raises((EmptyQueryError, TypeError)):
            _validate_query(None)  # type: ignore


# ---------------------------------------------------------------------------
# _reconstruct_chunk
# ---------------------------------------------------------------------------

class TestReconstructChunk:
    def test_returns_code_chunk(self):
        result = _make_result()
        chunk = _reconstruct_chunk(result)
        assert isinstance(chunk, CodeChunk)

    def test_content_preserved(self):
        result = _make_result(content="class Foo: pass")
        chunk = _reconstruct_chunk(result)
        assert chunk.content == "class Foo: pass"

    def test_chunk_id_preserved(self):
        result = _make_result(chunk_id="my-id-123")
        chunk = _reconstruct_chunk(result)
        assert chunk.chunk_id == "my-id-123"

    def test_file_path_preserved(self):
        result = _make_result(file_path="api/auth.py")
        chunk = _reconstruct_chunk(result)
        assert chunk.file_path == "api/auth.py"

    def test_language_resolved(self):
        result = _make_result(language="python")
        chunk = _reconstruct_chunk(result)
        assert chunk.language == SupportedLanguage.PYTHON

    def test_unknown_language_fallback(self):
        result = _make_result(language="cobol")
        chunk = _reconstruct_chunk(result)
        assert chunk.language == SupportedLanguage.UNKNOWN

    def test_start_end_lines_preserved(self):
        result = _make_result(start_line=10, end_line=25)
        chunk = _reconstruct_chunk(result)
        assert chunk.start_line == 10
        assert chunk.end_line == 25

    def test_end_line_clamped_when_less_than_start(self):
        result = _make_result(start_line=15, end_line=5)  # invalid range
        chunk = _reconstruct_chunk(result)
        assert chunk.end_line >= chunk.start_line

    def test_embedding_is_none(self):
        result = _make_result()
        chunk = _reconstruct_chunk(result)
        assert chunk.embedding is None

    def test_related_chunk_ids_empty(self):
        result = _make_result()
        chunk = _reconstruct_chunk(result)
        assert chunk.related_chunk_ids == []

    def test_missing_file_path_uses_unknown(self):
        result = SearchResult(
            chunk_id="x",
            page_content="code",
            metadata={"language": "python", "start_line": 1, "end_line": 1},
            relevance_score=0.5,
        )
        chunk = _reconstruct_chunk(result)
        assert chunk.file_path == "unknown"


# ---------------------------------------------------------------------------
# _result_to_chunk
# ---------------------------------------------------------------------------

class TestResultToChunk:
    def test_uses_chunk_map_when_available(self):
        original = _make_chunk(chunk_id="known-id", content="original content")
        result = _make_result(chunk_id="known-id", content="different content")
        chunk = _result_to_chunk(result, {"known-id": original})
        assert chunk is original  # exact same object

    def test_reconstructs_when_id_not_in_map(self):
        result = _make_result(chunk_id="unknown-id")
        chunk = _result_to_chunk(result, {"other-id": _make_chunk()})
        assert isinstance(chunk, CodeChunk)
        assert chunk.chunk_id == "unknown-id"

    def test_reconstructs_when_no_map(self):
        result = _make_result(chunk_id="some-id")
        chunk = _result_to_chunk(result, None)
        assert isinstance(chunk, CodeChunk)

    def test_empty_map_falls_through_to_reconstruct(self):
        result = _make_result(chunk_id="abc")
        chunk = _result_to_chunk(result, {})
        assert chunk.chunk_id == "abc"


# ---------------------------------------------------------------------------
# _merge_filters
# ---------------------------------------------------------------------------

class TestMergeFilters:
    def test_empty_inputs_return_empty(self):
        assert _merge_filters(None, None, None) == {}

    def test_config_language_included(self):
        result = _merge_filters(None, "python", None)
        assert result["language"] == "python"

    def test_explicit_overrides_config(self):
        result = _merge_filters({"language": "java"}, "python", None)
        assert result["language"] == "java"

    def test_call_language_overrides_explicit(self):
        result = _merge_filters({"language": "java"}, "python", "typescript")
        assert result["language"] == "typescript"

    def test_explicit_non_language_keys_preserved(self):
        result = _merge_filters({"class_name": "Auth"}, None, None)
        assert result["class_name"] == "Auth"
        assert "language" not in result

    def test_all_three_sources_merged(self):
        result = _merge_filters({"file_path": "main.py"}, "python", "typescript")
        assert result["file_path"] == "main.py"
        assert result["language"] == "typescript"


# ---------------------------------------------------------------------------
# build_chunk_map
# ---------------------------------------------------------------------------

class TestBuildChunkMap:
    def test_returns_dict(self):
        chunks = [_make_chunk(chunk_id="a"), _make_chunk(chunk_id="b")]
        m = build_chunk_map(chunks)
        assert isinstance(m, dict)

    def test_keys_are_chunk_ids(self):
        chunks = [_make_chunk(chunk_id="x"), _make_chunk(chunk_id="y")]
        m = build_chunk_map(chunks)
        assert set(m.keys()) == {"x", "y"}

    def test_values_are_code_chunks(self):
        chunk = _make_chunk(chunk_id="z")
        m = build_chunk_map([chunk])
        assert m["z"] is chunk

    def test_empty_list_returns_empty_dict(self):
        assert build_chunk_map([]) == {}

    def test_duplicate_ids_last_wins(self):
        c1 = _make_chunk(chunk_id="dup", content="first")
        c2 = _make_chunk(chunk_id="dup", content="second")
        m = build_chunk_map([c1, c2])
        assert m["dup"] is c2


# ---------------------------------------------------------------------------
# retrieve_chunks
# ---------------------------------------------------------------------------

class TestRetrieveChunks:
    def _run(
        self,
        query: str = "authentication",
        results: list[SearchResult] | None = None,
        k: int = 5,
        chunk_map=None,
    ) -> list[CodeChunk]:
        if results is None:
            results = [_make_result()]
        handle = _mock_handle()

        with patch("rag.retriever.similarity_search", return_value=results):
            return retrieve_chunks(query, handle, chunk_map=chunk_map, k=k)

    def test_returns_list_of_code_chunks(self):
        chunks = self._run()
        assert isinstance(chunks, list)
        assert all(isinstance(c, CodeChunk) for c in chunks)

    def test_empty_query_raises(self):
        handle = _mock_handle()
        with pytest.raises(EmptyQueryError):
            retrieve_chunks("", handle)

    def test_k_limits_results(self):
        results = [_make_result(chunk_id=str(i)) for i in range(10)]
        chunks = self._run(results=results, k=3)
        assert len(chunks) <= 3

    def test_uses_chunk_map(self):
        original = _make_chunk(chunk_id="mapped-id", content="original")
        results = [_make_result(chunk_id="mapped-id")]
        chunks = self._run(results=results, chunk_map={"mapped-id": original})
        assert chunks[0] is original

    def test_dedup_by_file(self):
        results = [
            _make_result(chunk_id="a1", file_path="main.py"),
            _make_result(chunk_id="a2", file_path="main.py"),
            _make_result(chunk_id="b1", file_path="utils.py"),
        ]
        handle = _mock_handle()
        with patch("rag.retriever.similarity_search", return_value=results):
            chunks = retrieve_chunks("query", handle, k=5, dedup_by_file=True)

        file_paths = [c.file_path for c in chunks]
        assert file_paths.count("main.py") == 1
        assert "utils.py" in file_paths

    def test_faiss_error_raises_retriever_error(self):
        handle = _mock_handle()
        with patch("rag.retriever.similarity_search", side_effect=RuntimeError("boom")):
            with pytest.raises(RetrieverError, match="Retrieval failed"):
                retrieve_chunks("auth", handle)


# ---------------------------------------------------------------------------
# ChunkRetriever
# ---------------------------------------------------------------------------

class TestChunkRetriever:
    def _make_retriever(
        self,
        results: list[SearchResult] | None = None,
        chunk_map=None,
        config: RetrieverConfig | None = None,
    ) -> ChunkRetriever:
        handle = _mock_handle()
        return ChunkRetriever(handle=handle, chunk_map=chunk_map, config=config)

    # ── Constructor ───────────────────────────────────────────────────────

    def test_default_config_assigned(self):
        r = self._make_retriever()
        assert isinstance(r.config, RetrieverConfig)
        assert r.config.top_k == 5

    def test_custom_config_respected(self):
        cfg = RetrieverConfig(top_k=10, score_threshold=0.5)
        r = self._make_retriever(config=cfg)
        assert r.config.top_k == 10

    def test_is_ready_when_doc_count_positive(self):
        r = self._make_retriever()
        r.handle.doc_count = 5
        assert r.is_ready is True

    def test_not_ready_when_doc_count_zero(self):
        r = self._make_retriever()
        r.handle.doc_count = 0
        assert r.is_ready is False

    def test_chunk_map_size(self):
        chunk_map = {f"id{i}": _make_chunk(chunk_id=f"id{i}") for i in range(7)}
        r = self._make_retriever(chunk_map=chunk_map)
        assert r.chunk_map_size() == 7

    def test_chunk_map_size_none(self):
        r = self._make_retriever()
        assert r.chunk_map_size() == 0

    def test_repr_contains_store_name(self):
        r = self._make_retriever()
        assert "test" in repr(r)

    # ── retrieve() ────────────────────────────────────────────────────────

    def test_retrieve_returns_code_chunks(self):
        r = self._make_retriever()
        with patch("rag.retriever.retrieve_chunks", return_value=[_make_chunk()]) as mock:
            result = r.retrieve("sql injection")
        assert isinstance(result, list)
        assert all(isinstance(c, CodeChunk) for c in result)

    def test_retrieve_k_override(self):
        r = self._make_retriever(config=RetrieverConfig(top_k=5))
        with patch("rag.retriever.retrieve_chunks", return_value=[]) as mock:
            r.retrieve("query", k=3)
            _, kwargs = mock.call_args
            assert kwargs["k"] == 3

    def test_retrieve_language_filter_merged(self):
        r = self._make_retriever()
        with patch("rag.retriever.retrieve_chunks", return_value=[]) as mock:
            r.retrieve("query", language="typescript")
            _, kwargs = mock.call_args
            assert kwargs["filters"]["language"] == "typescript"

    def test_retrieve_config_language_applied(self):
        r = self._make_retriever(config=RetrieverConfig(language_filter="java"))
        with patch("rag.retriever.retrieve_chunks", return_value=[]) as mock:
            r.retrieve("query")
            _, kwargs = mock.call_args
            assert kwargs["filters"]["language"] == "java"

    def test_retrieve_call_language_overrides_config(self):
        r = self._make_retriever(config=RetrieverConfig(language_filter="java"))
        with patch("rag.retriever.retrieve_chunks", return_value=[]) as mock:
            r.retrieve("query", language="python")
            _, kwargs = mock.call_args
            assert kwargs["filters"]["language"] == "python"

    # ── aretrieve() ───────────────────────────────────────────────────────

    def test_aretrieve_returns_coroutine(self):
        r = self._make_retriever()
        with patch("rag.retriever.retrieve_chunks", return_value=[_make_chunk()]):
            coro = r.aretrieve("query")
            assert asyncio.iscoroutine(coro)
            # Clean up by running it
            asyncio.get_event_loop().run_until_complete(coro)

    def test_aretrieve_result_matches_retrieve(self):
        expected = [_make_chunk(chunk_id="async-result")]
        r = self._make_retriever()
        with patch("rag.retriever.retrieve_chunks", return_value=expected):
            result = asyncio.get_event_loop().run_until_complete(
                r.aretrieve("query")
            )
        assert result == expected

    # ── retrieve_for_finding() ────────────────────────────────────────────

    def test_retrieve_for_finding_returns_code_chunks(self):
        r = self._make_retriever()
        with patch("rag.retriever.retrieve_chunks", return_value=[_make_chunk()]):
            result = r.retrieve_for_finding("SQL injection via fmt.Sprintf")
        assert isinstance(result, list)

    def test_retrieve_for_finding_deduplicates(self):
        """Same chunk returned from both same-file and global should appear once."""
        chunk = _make_chunk(chunk_id="shared")
        r = self._make_retriever()
        call_count = [0]

        def fake_retrieve_chunks(query, handle, **kwargs):
            call_count[0] += 1
            return [chunk]

        with patch("rag.retriever.retrieve_chunks", side_effect=fake_retrieve_chunks):
            result = r.retrieve_for_finding("desc", "main.py", k=5)

        ids = [c.chunk_id for c in result]
        assert ids.count("shared") == 1

    def test_retrieve_for_finding_respects_k(self):
        chunks = [_make_chunk(chunk_id=f"c{i}") for i in range(10)]
        r = self._make_retriever()
        with patch("rag.retriever.retrieve_chunks", return_value=chunks):
            result = r.retrieve_for_finding("query", k=3)
        assert len(result) <= 3


# ---------------------------------------------------------------------------
# _fallback_id
# ---------------------------------------------------------------------------

class TestFallbackId:
    def test_returns_string(self):
        result = _make_result(chunk_id="")
        fid = _fallback_id(result)
        assert isinstance(fid, str)

    def test_deterministic(self):
        result = _make_result(chunk_id="")
        assert _fallback_id(result) == _fallback_id(result)

    def test_different_inputs_produce_different_ids(self):
        r1 = _make_result(chunk_id="", file_path="a.py", start_line=1)
        r2 = _make_result(chunk_id="", file_path="b.py", start_line=2)
        assert _fallback_id(r1) != _fallback_id(r2)
