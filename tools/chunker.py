"""
tools/chunker.py
=================
Source file → List[CodeChunk] splitter.

Interface contract:
    chunk_file(file_path, content, language, max_lines) → List[CodeChunk]

Strategy:
  - Split on function/class boundaries where possible (language-aware)
  - Fall back to fixed-size windows (max_lines) with no overlap
  - Each chunk gets a stable chunk_id (UUID4)

Language-aware splitting is Phase 2.
Phase 1: Fixed-size line windows.
"""

from __future__ import annotations

import textwrap

import structlog

from schemas import CodeChunk, SupportedLanguage

logger = structlog.get_logger(__name__)

# File extensions to programming language mapping
EXTENSION_TO_LANGUAGE: dict[str, SupportedLanguage] = {
    ".py":   SupportedLanguage.PYTHON,
    ".js":   SupportedLanguage.JAVASCRIPT,
    ".mjs":  SupportedLanguage.JAVASCRIPT,
    ".ts":   SupportedLanguage.TYPESCRIPT,
    ".tsx":  SupportedLanguage.TYPESCRIPT,
    ".go":   SupportedLanguage.GO,
    ".java": SupportedLanguage.JAVA,
    ".rs":   SupportedLanguage.RUST,
    ".cpp":  SupportedLanguage.CPP,
    ".cc":   SupportedLanguage.CPP,
    ".cxx":  SupportedLanguage.CPP,
    ".c":    SupportedLanguage.C,
    ".h":    SupportedLanguage.C,
    ".cs":   SupportedLanguage.CSHARP,
    ".rb":   SupportedLanguage.RUBY,
    ".php":  SupportedLanguage.PHP,
}

# File extensions that are allowed to be analyzed
ALLOWED_EXTENSIONS: frozenset[str] = frozenset(EXTENSION_TO_LANGUAGE.keys())


def detect_language(file_path: str) -> SupportedLanguage:
    """Infer SupportedLanguage from file extension. Returns UNKNOWN if not recognized."""
    from pathlib import Path
    ext = Path(file_path).suffix.lower()
    return EXTENSION_TO_LANGUAGE.get(ext, SupportedLanguage.UNKNOWN)


def is_allowed_file(file_path: str) -> bool:
    """Return True if the file extension is in the allowed analysis set."""
    from pathlib import Path
    return Path(file_path).suffix.lower() in ALLOWED_EXTENSIONS


def chunk_file(
    file_path: str,
    content: str,
    language: SupportedLanguage,
    max_lines: int = 80,
) -> list[CodeChunk]:
    """
    Split a source file into non-overlapping CodeChunk objects.

    Phase 1: Fixed-size line windows.
    Phase 2: Language-aware splitting on function/class boundaries.

    Args:
        file_path : Relative path within the repository.
        content   : Full source file content as a string.
        language  : SupportedLanguage enum value.
        max_lines : Maximum number of lines per chunk.

    Returns:
        List[CodeChunk] — empty list if content is blank.
    """
    if not content.strip():
        return []

    lines = content.splitlines()
    chunks: list[CodeChunk] = []

    for window_start in range(0, len(lines), max_lines):
        window_end = min(window_start + max_lines, len(lines))
        chunk_content = "\n".join(lines[window_start:window_end])

        if not chunk_content.strip():
            continue

        chunk = CodeChunk(
            file_path=file_path,
            language=language,
            content=chunk_content,
            start_line=window_start + 1,  # 1-indexed
            end_line=window_end,           # 1-indexed inclusive
        )
        chunks.append(chunk)

    logger.debug(
        "file_chunked",
        file_path=file_path,
        total_lines=len(lines),
        chunk_count=len(chunks),
        max_lines=max_lines,
    )
    return chunks
