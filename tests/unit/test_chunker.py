"""
tests/unit/test_chunker.py
===========================
Unit tests for tools/chunker.py
"""

from __future__ import annotations

import pytest

from schemas import SupportedLanguage
from tools.chunker import chunk_file, detect_language, is_allowed_file


class TestDetectLanguage:
    def test_python(self):
        assert detect_language("src/main.py") == SupportedLanguage.PYTHON

    def test_typescript(self):
        assert detect_language("app/index.tsx") == SupportedLanguage.TYPESCRIPT

    def test_unknown(self):
        assert detect_language("README.md") == SupportedLanguage.UNKNOWN

    def test_case_insensitive(self):
        assert detect_language("Main.PY") == SupportedLanguage.PYTHON


class TestIsAllowedFile:
    def test_python_allowed(self):
        assert is_allowed_file("main.py") is True

    def test_markdown_not_allowed(self):
        assert is_allowed_file("README.md") is False

    def test_yaml_not_allowed(self):
        assert is_allowed_file("config.yaml") is False


class TestChunkFile:
    def test_empty_content_returns_empty(self):
        result = chunk_file("main.py", "   \n  \n", SupportedLanguage.PYTHON, max_lines=80)
        assert result == []

    def test_single_chunk_for_small_file(self):
        content = "\n".join([f"line {i}" for i in range(10)])
        chunks = chunk_file("main.py", content, SupportedLanguage.PYTHON, max_lines=80)
        assert len(chunks) == 1
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 10

    def test_multiple_chunks_for_large_file(self):
        content = "\n".join([f"line {i}" for i in range(200)])
        chunks = chunk_file("main.py", content, SupportedLanguage.PYTHON, max_lines=80)
        assert len(chunks) == 3  # ceil(200/80) = 3

    def test_chunk_line_numbers_are_correct(self):
        content = "\n".join([f"line {i}" for i in range(10)])
        chunks = chunk_file("main.py", content, SupportedLanguage.PYTHON, max_lines=5)
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 5
        assert chunks[1].start_line == 6
        assert chunks[1].end_line == 10

    def test_chunk_file_path_preserved(self):
        content = "x = 1\ny = 2\n"
        chunks = chunk_file("src/utils.py", content, SupportedLanguage.PYTHON)
        assert all(c.file_path == "src/utils.py" for c in chunks)

    def test_chunk_language_preserved(self):
        content = "const x = 1;\n"
        chunks = chunk_file("app.ts", content, SupportedLanguage.TYPESCRIPT)
        assert chunks[0].language == SupportedLanguage.TYPESCRIPT
