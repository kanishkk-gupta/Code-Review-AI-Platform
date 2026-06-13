"""tools package"""
from tools.chunker import chunk_file, detect_language, is_allowed_file, ALLOWED_EXTENSIONS
from tools.file_parser import parse_source
from tools.language_detector import compute_language_breakdown
from tools.parser import (
    parse_repository,
    parse_file,
    build_metadata_from_parsed,
    ParsedRepository,
    ParsedFile,
    ParsedClass,
    ParsedFunction,
    ParsedImport,
    ParserError,
    UnsupportedLanguageError,
)
from tools.github_tools import (
    fetch_repo,
    list_source_files,
    detect_languages,
    cleanup_repo,
    build_repository_metadata,
    ClonedRepo,
    SourceFile,
    LanguageStats,
    GitHubToolError,
    InvalidRepositoryURLError,
    CloneFailedError,
    EmptyRepositoryError,
    GITHUB_TOOL_LANGUAGES,
    MAX_FILE_SIZE_BYTES,
    MAX_FILES_PER_REPO,
)

__all__ = [
    # chunker
    "chunk_file",
    "detect_language",
    "is_allowed_file",
    "ALLOWED_EXTENSIONS",
    # file_parser
    "parse_source",
    # language_detector
    "compute_language_breakdown",
    # parser
    "parse_repository",
    "parse_file",
    "build_metadata_from_parsed",
    "ParsedRepository",
    "ParsedFile",
    "ParsedClass",
    "ParsedFunction",
    "ParsedImport",
    "ParserError",
    "UnsupportedLanguageError",
    # github_tools
    "fetch_repo",
    "list_source_files",
    "detect_languages",
    "cleanup_repo",
    "build_repository_metadata",
    "ClonedRepo",
    "SourceFile",
    "LanguageStats",
    "GitHubToolError",
    "InvalidRepositoryURLError",
    "CloneFailedError",
    "EmptyRepositoryError",
    "GITHUB_TOOL_LANGUAGES",
    "MAX_FILE_SIZE_BYTES",
    "MAX_FILES_PER_REPO",
]
