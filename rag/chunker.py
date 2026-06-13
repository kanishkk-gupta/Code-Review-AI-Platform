"""
rag/chunker.py
==============
Hierarchical source-code → LangChain Document pipeline.

Public API
----------
    chunk_repository(parsed_repo, metadata, chunk_size, chunk_overlap)
        -> list[Document]

    chunk_file(parsed_file, content, chunk_size, chunk_overlap)
        -> list[Document]

Chunking Hierarchy
------------------
    Repository
    └── File
        ├── Class
        │   └── Method  (function nested inside a class)
        └── Function    (module-level; no parent class)

Processing strategy per file:
  1. For every **class** → extract its full source span, split with the
     language-aware RecursiveCharacterTextSplitter, tag with class_name.
  2. For every **method** inside each class → extract the method's span,
     split, tag with both class_name and function_name.
  3. For every **module-level function** → extract its span, split, tag
     with function_name only.
  4. For lines NOT covered by any class or function → split as file-level
     chunks (covers imports, module docstrings, global constants).

Document Metadata Schema
------------------------
Every returned Document carries the following metadata dict:

    {
        "chunk_id":      str,       # UUID4 — stable across identical content
        "file_path":     str,       # POSIX path relative to repository root
        "language":      str,       # SupportedLanguage.value ("python", etc.)
        "class_name":    str | None,
        "function_name": str | None,
        "start_line":    int,       # 1-indexed
        "end_line":      int,       # 1-indexed (inclusive)
        "chunk_index":   int,       # 0-based index within the same scope
        "repo_name":     str | None,
        "source_url":    str | None,
    }

Splitter Configuration
----------------------
    chunk_size    : 800 characters (default)
    chunk_overlap : 100 characters (default)

Language-aware separators via ``RecursiveCharacterTextSplitter.from_language``
are used when LangChain supports the language; generic separators used as
fallback.

Design Decisions
----------------
* Content is extracted from disk per file (``ParsedFile.rel_path`` + repo root).
  ``ParsedFile`` intentionally omits content to keep the parser lean.
* Files that cannot be read are skipped with a WARNING log (no crash).
* When a scope (class / function) spans > chunk_size chars it is split into
  multiple Documents; each inherits the same metadata plus a ``chunk_index``.
* Line-range tracking uses a sorted merge of covered ranges so that
  "file-level" chunks are exactly the complement of class/function spans.
* ``chunk_id`` values are deterministic within a run (UUID5 of file_path +
  start_line + chunk_index) to enable deduplication across incremental runs.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

import structlog
from langchain.schema import Document
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

from schemas import RepositoryMetadata, SupportedLanguage
from tools.parser import ParsedClass, ParsedFile, ParsedFunction, ParsedRepository

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE: int = 800
DEFAULT_CHUNK_OVERLAP: int = 100

# Namespace UUID for deterministic chunk_id generation (UUID5)
_CHUNK_NS = uuid.UUID("12345678-1234-5678-1234-567812345678")

# SupportedLanguage → LangChain Language enum
_LANG_TO_LC: dict[str, Language] = {
    SupportedLanguage.PYTHON:     Language.PYTHON,
    SupportedLanguage.JAVA:       Language.JAVA,
    SupportedLanguage.CPP:        Language.CPP,
    SupportedLanguage.C:          Language.CPP,   # LangChain has no separate C enum
    SupportedLanguage.JAVASCRIPT: Language.JS,
    SupportedLanguage.TYPESCRIPT: Language.TS,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_repository(
    parsed_repo: ParsedRepository,
    metadata: Optional[RepositoryMetadata] = None,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Walk every file in *parsed_repo*, apply hierarchical chunking, and return
    all resulting LangChain Documents.

    Args:
        parsed_repo   : Output of ``tools.parser.parse_repository()``.
        metadata      : Optional ``RepositoryMetadata`` used to populate
                        ``repo_name`` and ``source_url`` in each Document.
        chunk_size    : Maximum character length per Document page_content.
        chunk_overlap : Number of characters shared between adjacent Documents
                        produced from the same scope.

    Returns:
        Flat list of ``langchain.schema.Document`` objects, ordered by
        file → class → function, with file-level residual chunks appended last
        per file.

    Example::

        parsed = parse_repository("/path/to/repo")
        docs = chunk_repository(parsed)
        print(len(docs))        # e.g. 342
        print(docs[0].metadata) # {"chunk_id": "...", "file_path": "src/main.py", ...}
    """
    all_docs: list[Document] = []
    repo_name = metadata.repository_name if metadata else parsed_repo.name
    source_url = metadata.source_url if metadata else parsed_repo.source_url

    logger.info(
        "chunk_repository_start",
        repo=repo_name,
        files=parsed_repo.total_files,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    for parsed_file in parsed_repo.files:
        abs_path = parsed_repo.root_path / parsed_file.rel_path
        try:
            content = abs_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning(
                "chunk_skip_read_error",
                path=str(abs_path),
                error=str(exc),
            )
            continue

        docs = chunk_file(
            parsed_file,
            content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            repo_name=repo_name,
            source_url=source_url,
        )
        all_docs.extend(docs)

    logger.info(
        "chunk_repository_complete",
        repo=repo_name,
        total_documents=len(all_docs),
    )
    return all_docs


def chunk_file(
    parsed_file: ParsedFile,
    content: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    repo_name: Optional[str] = None,
    source_url: Optional[str] = None,
) -> list[Document]:
    """
    Chunk a single parsed source file into hierarchical LangChain Documents.

    This is the primary chunking function. The caller supplies *content*
    separately because ``ParsedFile`` stores structure (line numbers) but
    not the raw text (to keep memory usage lean in the parser).

    Args:
        parsed_file   : Structural metadata from ``tools.parser.parse_file()``.
        content       : Full decoded source text of the file.
        chunk_size    : Max chars per Document.
        chunk_overlap : Overlap chars between adjacent chunks.
        repo_name     : Passed into every Document's metadata.
        source_url    : Passed into every Document's metadata.

    Returns:
        List of Documents in hierarchy order:
        class docs → method docs → module-function docs → file-residual docs.
    """
    if not content.strip():
        return []

    lines = content.splitlines()
    splitter = _get_splitter(parsed_file.language, chunk_size, chunk_overlap)

    base_meta: dict = {
        "file_path": parsed_file.rel_path,
        "language": str(parsed_file.language),
        "repo_name": repo_name,
        "source_url": source_url,
        "class_name": None,
        "function_name": None,
    }

    all_docs: list[Document] = []
    covered_ranges: list[tuple[int, int]] = []  # (start_line, end_line), 1-indexed

    # ── 1. Classes + their methods ─────────────────────────────────────────
    for cls in parsed_file.classes:
        # (a) Class-level chunk (the full class body)
        class_content = _extract_lines(lines, cls.line_start, cls.line_end)
        class_meta = {
            **base_meta,
            "class_name": cls.name,
            "function_name": None,
            "start_line": cls.line_start,
            "end_line": cls.line_end,
        }
        docs = _split_text(class_content, class_meta, splitter)
        all_docs.extend(docs)
        covered_ranges.append((cls.line_start, cls.line_end))

        # (b) Method-level chunks within the class
        for method in cls.methods:
            method_content = _extract_lines(lines, method.line_start, method.line_end)
            method_meta = {
                **base_meta,
                "class_name": cls.name,
                "function_name": method.name,
                "start_line": method.line_start,
                "end_line": method.line_end,
            }
            docs = _split_text(method_content, method_meta, splitter)
            all_docs.extend(docs)
            # Methods are nested; don't add to covered_ranges separately
            # (their range is already inside the class range)

    # ── 2. Module-level functions ──────────────────────────────────────────
    for fn in parsed_file.functions:
        fn_content = _extract_lines(lines, fn.line_start, fn.line_end)
        fn_meta = {
            **base_meta,
            "class_name": None,
            "function_name": fn.name,
            "start_line": fn.line_start,
            "end_line": fn.line_end,
        }
        docs = _split_text(fn_content, fn_meta, splitter)
        all_docs.extend(docs)
        covered_ranges.append((fn.line_start, fn.line_end))

    # ── 3. File-level residual (imports, globals, module docstrings) ───────
    residual_lines = _uncovered_lines(lines, covered_ranges)
    if residual_lines.strip():
        residual_start, residual_end = _first_last_nonempty(lines, covered_ranges)
        file_meta = {
            **base_meta,
            "class_name": None,
            "function_name": None,
            "start_line": residual_start,
            "end_line": residual_end,
        }
        docs = _split_text(residual_lines, file_meta, splitter)
        all_docs.extend(docs)

    logger.debug(
        "chunk_file_done",
        file_path=parsed_file.rel_path,
        documents=len(all_docs),
        classes=len(parsed_file.classes),
        functions=len(parsed_file.functions),
    )
    return all_docs


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_splitter(
    language: SupportedLanguage,
    chunk_size: int,
    chunk_overlap: int,
) -> RecursiveCharacterTextSplitter:
    """
    Return a ``RecursiveCharacterTextSplitter`` tuned for *language*.

    Uses ``from_language()`` when LangChain supports it; falls back to a
    generic splitter with sensible code separators otherwise.
    """
    lc_lang = _LANG_TO_LC.get(language)
    if lc_lang is not None:
        try:
            return RecursiveCharacterTextSplitter.from_language(
                language=lc_lang,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        except Exception:  # noqa: BLE001
            pass  # fall through to generic splitter

    # Generic fallback — works for any language
    return RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )


def _extract_lines(lines: list[str], start: int, end: int) -> str:
    """
    Extract the slice ``lines[start-1 : end]`` (1-indexed, inclusive) and
    join with newlines.

    Clamps the range to valid indices to avoid IndexError on malformed data.
    """
    s = max(0, start - 1)
    e = min(len(lines), end)
    return "\n".join(lines[s:e])


def _split_text(
    text: str,
    base_metadata: dict,
    splitter: RecursiveCharacterTextSplitter,
) -> list[Document]:
    """
    Split *text* with *splitter* and wrap each piece in a ``Document``.

    Each Document receives:
      - A copy of *base_metadata*
      - ``chunk_id``: UUID5 derived from (file_path + start_line + chunk_index)
      - ``chunk_index``: 0-based index within this scope's chunk sequence
    """
    if not text.strip():
        return []

    raw_chunks = splitter.split_text(text)
    docs: list[Document] = []

    file_path = base_metadata.get("file_path", "")
    start_line = base_metadata.get("start_line", 0)

    for i, chunk_text in enumerate(raw_chunks):
        if not chunk_text.strip():
            continue

        # Deterministic chunk_id: UUID5(namespace, "file_path:start_line:index")
        uid_seed = f"{file_path}:{start_line}:{i}"
        chunk_id = str(uuid.uuid5(_CHUNK_NS, uid_seed))

        meta = {
            **base_metadata,
            "chunk_id": chunk_id,
            "chunk_index": i,
        }
        docs.append(Document(page_content=chunk_text, metadata=meta))

    return docs


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Merge overlapping or adjacent 1-indexed inclusive ranges into a sorted,
    non-overlapping list.

    Example::
        _merge_ranges([(1, 5), (3, 8), (10, 15)]) -> [(1, 8), (10, 15)]
    """
    if not ranges:
        return []
    sorted_ranges = sorted(ranges)
    merged: list[tuple[int, int]] = [sorted_ranges[0]]
    for start, end in sorted_ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _uncovered_lines(
    lines: list[str],
    covered: list[tuple[int, int]],
) -> str:
    """
    Return a single string containing the lines NOT covered by any range in
    *covered* (1-indexed, inclusive). Used to produce file-level residual chunks.
    """
    merged = _merge_ranges(covered)
    covered_set: set[int] = set()
    for start, end in merged:
        covered_set.update(range(start, end + 1))

    residual: list[str] = []
    for i, line in enumerate(lines, start=1):
        if i not in covered_set:
            residual.append(line)

    return "\n".join(residual)


def _first_last_nonempty(
    lines: list[str],
    covered: list[tuple[int, int]],
) -> tuple[int, int]:
    """
    Identify the first and last 1-indexed line numbers that are NOT in any
    covered range. Used to populate ``start_line`` / ``end_line`` for
    residual (file-level) documents.

    Returns (1, 1) if all lines are covered or there are no lines.
    """
    merged = _merge_ranges(covered)
    covered_set: set[int] = set()
    for start, end in merged:
        covered_set.update(range(start, end + 1))

    uncovered = [i for i in range(1, len(lines) + 1) if i not in covered_set]
    if not uncovered:
        return 1, 1
    return uncovered[0], uncovered[-1]
