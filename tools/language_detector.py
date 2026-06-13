"""
tools/language_detector.py
===========================
Language statistics extraction from a file tree.

Used by ingest_node to populate RepositoryMetadata.language_breakdown
and determine RepositoryMetadata.primary_language.
"""

from __future__ import annotations

from collections import Counter

from schemas import SupportedLanguage
from tools.chunker import detect_language, is_allowed_file


def compute_language_breakdown(
    file_entries: list[tuple[str, str, SupportedLanguage]],
) -> tuple[SupportedLanguage, dict[str, float]]:
    """
    Compute language statistics from a list of (file_path, content, language) tuples.

    Args:
        file_entries: Output from tools.file_parser.parse_source().

    Returns:
        Tuple of:
          - primary_language: SupportedLanguage with the most lines
          - language_breakdown: dict mapping language value → percentage of total lines
    """
    line_counts: Counter[str] = Counter()

    for file_path, content, language in file_entries:
        line_count = content.count("\n") + 1
        line_counts[language] += line_count

    total = sum(line_counts.values()) or 1  # avoid division by zero

    breakdown = {
        lang: round((count / total) * 100, 1)
        for lang, count in line_counts.most_common()
    }

    if not line_counts:
        return SupportedLanguage.UNKNOWN, {}

    primary_lang_str = line_counts.most_common(1)[0][0]
    try:
        primary_language = SupportedLanguage(primary_lang_str)
    except ValueError:
        primary_language = SupportedLanguage.UNKNOWN

    return primary_language, breakdown
