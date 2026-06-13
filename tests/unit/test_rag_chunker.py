"""
tests/unit/test_rag_chunker.py
==============================
Unit tests for rag/chunker.py — fully offline, no LLM or network calls.
Uses pytest tmp_path for real file I/O; all LangChain calls execute locally.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest
from langchain.schema import Document

from schemas import RepositoryMetadata, SupportedLanguage
from tools.parser import (
    ParsedClass,
    ParsedFile,
    ParsedFunction,
    ParsedRepository,
)
from rag.chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    _extract_lines,
    _get_splitter,
    _merge_ranges,
    _split_text,
    _uncovered_lines,
    chunk_file,
    chunk_repository,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_parsed_file(
    rel_path: str = "src/app.py",
    language: SupportedLanguage = SupportedLanguage.PYTHON,
    total_lines: int = 10,
    classes: Optional[list[ParsedClass]] = None,
    functions: Optional[list[ParsedFunction]] = None,
) -> ParsedFile:
    return ParsedFile(
        rel_path=rel_path,
        language=language,
        total_lines=total_lines,
        classes=classes or [],
        functions=functions or [],
    )


PYTHON_SAMPLE = """\
import os
import sys

GLOBAL_CONSTANT = 42


class Animal:
    \"\"\"Base animal.\"\"\"
    def __init__(self, name: str):
        self.name = name

    def speak(self):
        return "..."


class Dog(Animal):
    def speak(self):
        return "Woof"


def top_level_util(x: int) -> int:
    return x * 2
"""


# ---------------------------------------------------------------------------
# _extract_lines
# ---------------------------------------------------------------------------

class TestExtractLines:
    LINES = ["line1", "line2", "line3", "line4", "line5"]

    def test_basic_extraction(self):
        result = _extract_lines(self.LINES, 2, 4)
        assert result == "line2\nline3\nline4"

    def test_single_line(self):
        assert _extract_lines(self.LINES, 3, 3) == "line3"

    def test_full_file(self):
        result = _extract_lines(self.LINES, 1, 5)
        assert result == "\n".join(self.LINES)

    def test_clamps_beyond_end(self):
        result = _extract_lines(self.LINES, 4, 100)
        assert result == "line4\nline5"

    def test_clamps_before_start(self):
        result = _extract_lines(self.LINES, 0, 2)
        assert result == "line1\nline2"


# ---------------------------------------------------------------------------
# _merge_ranges
# ---------------------------------------------------------------------------

class TestMergeRanges:
    def test_empty(self):
        assert _merge_ranges([]) == []

    def test_single(self):
        assert _merge_ranges([(3, 7)]) == [(3, 7)]

    def test_non_overlapping(self):
        assert _merge_ranges([(1, 3), (5, 8)]) == [(1, 3), (5, 8)]

    def test_overlapping(self):
        assert _merge_ranges([(1, 5), (3, 8)]) == [(1, 8)]

    def test_adjacent(self):
        assert _merge_ranges([(1, 4), (5, 8)]) == [(1, 8)]

    def test_unsorted_input(self):
        assert _merge_ranges([(10, 15), (1, 5)]) == [(1, 5), (10, 15)]

    def test_multiple_overlapping(self):
        result = _merge_ranges([(1, 3), (2, 5), (4, 8), (10, 12)])
        assert result == [(1, 8), (10, 12)]


# ---------------------------------------------------------------------------
# _uncovered_lines
# ---------------------------------------------------------------------------

class TestUncoveredLines:
    LINES = ["import os", "x = 1", "def foo():", "    return 1", "y = 2"]

    def test_no_covered_returns_all(self):
        result = _uncovered_lines(self.LINES, [])
        assert result == "\n".join(self.LINES)

    def test_covered_range_excluded(self):
        result = _uncovered_lines(self.LINES, [(3, 4)])
        assert "def foo():" not in result
        assert "    return 1" not in result
        assert "import os" in result
        assert "y = 2" in result

    def test_all_covered_returns_empty(self):
        result = _uncovered_lines(self.LINES, [(1, 5)])
        assert result.strip() == ""


# ---------------------------------------------------------------------------
# _get_splitter
# ---------------------------------------------------------------------------

class TestGetSplitter:
    @pytest.mark.parametrize("lang", [
        SupportedLanguage.PYTHON,
        SupportedLanguage.JAVA,
        SupportedLanguage.CPP,
        SupportedLanguage.JAVASCRIPT,
        SupportedLanguage.TYPESCRIPT,
    ])
    def test_returns_splitter_for_supported_lang(self, lang):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = _get_splitter(lang, 800, 100)
        assert isinstance(splitter, RecursiveCharacterTextSplitter)

    def test_fallback_for_unknown_lang(self):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = _get_splitter(SupportedLanguage.UNKNOWN, 800, 100)
        assert isinstance(splitter, RecursiveCharacterTextSplitter)

    def test_chunk_size_respected(self):
        splitter = _get_splitter(SupportedLanguage.PYTHON, 50, 0)
        text = "x = 1\n" * 20
        chunks = splitter.split_text(text)
        assert all(len(c) <= 60 for c in chunks)  # some slack for splitter strategy


# ---------------------------------------------------------------------------
# _split_text
# ---------------------------------------------------------------------------

class TestSplitText:
    def _splitter(self):
        return _get_splitter(SupportedLanguage.PYTHON, 200, 20)

    def test_returns_documents(self):
        docs = _split_text("x = 1\n" * 5, {"file_path": "a.py", "start_line": 1}, self._splitter())
        assert all(isinstance(d, Document) for d in docs)

    def test_metadata_chunk_id_present(self):
        docs = _split_text("def foo(): pass\n", {"file_path": "a.py", "start_line": 1}, self._splitter())
        assert all("chunk_id" in d.metadata for d in docs)

    def test_chunk_ids_are_uuids(self):
        import re
        UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
        docs = _split_text("def foo(): pass\n" * 3, {"file_path": "b.py", "start_line": 5}, self._splitter())
        for d in docs:
            assert UUID_RE.match(d.metadata["chunk_id"]), f"Bad UUID: {d.metadata['chunk_id']}"

    def test_chunk_ids_are_deterministic(self):
        splitter = self._splitter()
        text = "def foo(): pass\n" * 3
        meta = {"file_path": "c.py", "start_line": 1}
        docs1 = _split_text(text, meta, splitter)
        docs2 = _split_text(text, meta, splitter)
        assert [d.metadata["chunk_id"] for d in docs1] == [d.metadata["chunk_id"] for d in docs2]

    def test_chunk_index_sequential(self):
        big_text = "x = 1\n" * 100
        splitter = _get_splitter(SupportedLanguage.PYTHON, 50, 0)
        docs = _split_text(big_text, {"file_path": "d.py", "start_line": 1}, splitter)
        indices = [d.metadata["chunk_index"] for d in docs]
        assert indices == list(range(len(docs)))

    def test_empty_text_returns_empty(self):
        docs = _split_text("   \n  ", {"file_path": "e.py", "start_line": 1}, self._splitter())
        assert docs == []

    def test_metadata_passthrough(self):
        meta = {
            "file_path": "f.py",
            "language": "python",
            "class_name": "Foo",
            "function_name": "bar",
            "start_line": 10,
            "end_line": 20,
        }
        docs = _split_text("def bar():\n    pass\n", meta, self._splitter())
        for d in docs:
            assert d.metadata["class_name"] == "Foo"
            assert d.metadata["function_name"] == "bar"
            assert d.metadata["file_path"] == "f.py"


# ---------------------------------------------------------------------------
# chunk_file
# ---------------------------------------------------------------------------

class TestChunkFile:
    def _parsed_file_from_content(self, content: str, rel_path: str = "app.py") -> ParsedFile:
        from tools.parser import parse_file
        return parse_file(rel_path, content, SupportedLanguage.PYTHON)

    def test_returns_list_of_documents(self):
        pf = self._parsed_file_from_content(PYTHON_SAMPLE)
        docs = chunk_file(pf, PYTHON_SAMPLE)
        assert isinstance(docs, list)
        assert all(isinstance(d, Document) for d in docs)

    def test_all_docs_have_chunk_id(self):
        pf = self._parsed_file_from_content(PYTHON_SAMPLE)
        docs = chunk_file(pf, PYTHON_SAMPLE)
        assert all("chunk_id" in d.metadata for d in docs)

    def test_all_docs_have_file_path(self):
        pf = self._parsed_file_from_content(PYTHON_SAMPLE, "src/app.py")
        docs = chunk_file(pf, PYTHON_SAMPLE)
        assert all(d.metadata["file_path"] == "src/app.py" for d in docs)

    def test_all_docs_have_language(self):
        pf = self._parsed_file_from_content(PYTHON_SAMPLE)
        docs = chunk_file(pf, PYTHON_SAMPLE)
        assert all(d.metadata["language"] == "python" for d in docs)

    def test_class_level_docs_have_class_name(self):
        pf = self._parsed_file_from_content(PYTHON_SAMPLE)
        docs = chunk_file(pf, PYTHON_SAMPLE)
        class_docs = [d for d in docs if d.metadata.get("class_name") is not None]
        assert len(class_docs) > 0
        class_names = {d.metadata["class_name"] for d in class_docs}
        assert "Animal" in class_names
        assert "Dog" in class_names

    def test_method_docs_have_function_name(self):
        pf = self._parsed_file_from_content(PYTHON_SAMPLE)
        docs = chunk_file(pf, PYTHON_SAMPLE)
        method_docs = [
            d for d in docs
            if d.metadata.get("class_name") is not None
            and d.metadata.get("function_name") is not None
        ]
        fn_names = {d.metadata["function_name"] for d in method_docs}
        assert "speak" in fn_names or "__init__" in fn_names

    def test_module_function_docs_have_no_class_name(self):
        pf = self._parsed_file_from_content(PYTHON_SAMPLE)
        docs = chunk_file(pf, PYTHON_SAMPLE)
        fn_docs = [
            d for d in docs
            if d.metadata.get("function_name") == "top_level_util"
        ]
        assert len(fn_docs) > 0
        assert all(d.metadata["class_name"] is None for d in fn_docs)

    def test_empty_file_returns_empty(self):
        pf = ParsedFile(rel_path="empty.py", language=SupportedLanguage.PYTHON, total_lines=0)
        docs = chunk_file(pf, "")
        assert docs == []

    def test_whitespace_only_returns_empty(self):
        pf = ParsedFile(rel_path="blank.py", language=SupportedLanguage.PYTHON, total_lines=1)
        docs = chunk_file(pf, "   \n\n  ")
        assert docs == []

    def test_repo_name_in_metadata(self):
        pf = self._parsed_file_from_content(PYTHON_SAMPLE)
        docs = chunk_file(pf, PYTHON_SAMPLE, repo_name="my-repo")
        assert all(d.metadata["repo_name"] == "my-repo" for d in docs)

    def test_source_url_in_metadata(self):
        pf = self._parsed_file_from_content(PYTHON_SAMPLE)
        docs = chunk_file(pf, PYTHON_SAMPLE, source_url="https://github.com/x/y")
        assert all(d.metadata["source_url"] == "https://github.com/x/y" for d in docs)

    def test_large_file_splits_into_multiple(self):
        big_content = "x = 1\n" * 300  # well over 800 chars
        pf = ParsedFile(rel_path="big.py", language=SupportedLanguage.PYTHON, total_lines=300)
        docs = chunk_file(pf, big_content, chunk_size=200, chunk_overlap=20)
        assert len(docs) > 1

    def test_page_content_not_empty(self):
        pf = self._parsed_file_from_content(PYTHON_SAMPLE)
        docs = chunk_file(pf, PYTHON_SAMPLE)
        assert all(d.page_content.strip() != "" for d in docs)

    def test_java_file_chunks(self):
        java_src = (
            "import java.util.List;\n\n"
            "public class Foo {\n"
            "    public void bar() {\n"
            "        System.out.println(\"hello\");\n"
            "    }\n"
            "}\n"
        )
        from tools.parser import parse_file
        pf = parse_file("Foo.java", java_src, SupportedLanguage.JAVA)
        docs = chunk_file(pf, java_src)
        assert len(docs) > 0
        assert all("chunk_id" in d.metadata for d in docs)


# ---------------------------------------------------------------------------
# chunk_repository
# ---------------------------------------------------------------------------

class TestChunkRepository:
    def _make_repo(self, tmp_path: Path) -> ParsedRepository:
        """Create a real on-disk repo with two Python files."""
        (tmp_path / "main.py").write_text(PYTHON_SAMPLE, encoding="utf-8")
        (tmp_path / "utils.py").write_text(
            "def helper(x):\n    return x + 1\n\nHELPER_CONST = 10\n",
            encoding="utf-8",
        )
        from tools.parser import parse_repository
        return parse_repository(tmp_path, name="test-repo")

    def test_returns_list_of_documents(self, tmp_path: Path):
        repo = self._make_repo(tmp_path)
        docs = chunk_repository(repo)
        assert isinstance(docs, list)
        assert all(isinstance(d, Document) for d in docs)

    def test_documents_produced(self, tmp_path: Path):
        repo = self._make_repo(tmp_path)
        docs = chunk_repository(repo)
        assert len(docs) > 0

    def test_all_chunk_ids_unique(self, tmp_path: Path):
        repo = self._make_repo(tmp_path)
        docs = chunk_repository(repo)
        ids = [d.metadata["chunk_id"] for d in docs]
        # UUIDs should be unique or at least close to unique
        assert len(ids) == len(set(ids))

    def test_metadata_populated_from_repository_metadata(self, tmp_path: Path):
        repo = self._make_repo(tmp_path)
        md = RepositoryMetadata(
            repository_name="my-service",
            source_url="https://github.com/acme/my-service",
            primary_language=SupportedLanguage.PYTHON,
            language_breakdown={"python": 100.0},
            total_files=2,
            total_lines=20,
        )
        docs = chunk_repository(repo, metadata=md)
        assert all(d.metadata["repo_name"] == "my-service" for d in docs)
        assert all(d.metadata["source_url"] == "https://github.com/acme/my-service" for d in docs)

    def test_skips_missing_file_gracefully(self, tmp_path: Path):
        repo = self._make_repo(tmp_path)
        # Corrupt one file's path so it can't be read
        bad_pf = ParsedFile(
            rel_path="nonexistent/ghost.py",
            language=SupportedLanguage.PYTHON,
            total_lines=0,
        )
        repo.files.append(bad_pf)
        docs = chunk_repository(repo)  # must not raise
        assert isinstance(docs, list)

    def test_custom_chunk_size(self, tmp_path: Path):
        repo = self._make_repo(tmp_path)
        docs_small = chunk_repository(repo, chunk_size=100, chunk_overlap=10)
        docs_large = chunk_repository(repo, chunk_size=5000, chunk_overlap=100)
        # Smaller chunk_size should produce more Documents
        assert len(docs_small) >= len(docs_large)

    def test_empty_repo_returns_empty(self, tmp_path: Path):
        repo = ParsedRepository(root_path=tmp_path, name="empty", source_url=None)
        docs = chunk_repository(repo)
        assert docs == []
