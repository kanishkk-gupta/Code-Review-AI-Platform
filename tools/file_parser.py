"""
tools/file_parser.py
=====================
Repository source ingestion — handles both GitHub URL cloning and ZIP extraction.

Interface
---------
    parse_source(source_url, source_zip_b64)
        -> list[tuple[rel_path: str, content: str, language: SupportedLanguage]]

URL path  : delegates to tools.github_tools.fetch_repo + list_source_files
            (GitPython shallow clone → temp dir → file walk → cleanup)

ZIP path  : decodes base64 ZIP → extracts to temp dir → applies same
            extension/size/binary filters as the URL path → cleanup

Both paths return the same tuple format so chunk_generator is agnostic to
the ingestion method.

Security
--------
* ZIP extraction: path-traversal guard (zip-slip) — members whose resolved
  path escapes the temp root are silently skipped with a WARNING.
* File size ceiling: 512 KB per file (same constant as github_tools).
* Binary detection: files containing null bytes are skipped.
* Max total files: 5 000 (same constant as github_tools).
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import structlog

from schemas import SupportedLanguage
from tools.chunker import ALLOWED_EXTENSIONS, EXTENSION_TO_LANGUAGE, is_allowed_file

logger = structlog.get_logger(__name__)

# Mirror the constants from github_tools for consistency
_MAX_FILE_SIZE_BYTES: int = 512 * 1024   # 512 KB
_MAX_FILES_PER_REPO:  int = 5_000

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def parse_source(
    source_url:     Optional[str],
    source_zip_b64: Optional[str],
) -> list[tuple[str, str, SupportedLanguage]]:
    """
    Parse a repository from either a GitHub/GitLab URL or a base64 ZIP.

    Exactly one of the two arguments must be non-None/non-empty.

    Args:
        source_url     : GitHub/GitLab HTTPS URL (e.g. "https://github.com/psf/requests").
        source_zip_b64 : Base64-encoded ZIP archive of the repository root.

    Returns:
        List of ``(relative_file_path, file_content, SupportedLanguage)`` tuples.
        The list is sorted by relative path for deterministic downstream processing.

    Raises:
        ValueError   : Both or neither argument supplied.
        RuntimeError : Clone / extraction failed.
    """
    if source_url and source_zip_b64:
        raise ValueError("Provide either source_url or source_zip_b64, not both.")
    if not source_url and not source_zip_b64:
        raise ValueError("At least one of source_url or source_zip_b64 must be provided.")

    if source_url:
        return await asyncio.get_event_loop().run_in_executor(
            None, _parse_git_url, source_url
        )
    else:
        return await asyncio.get_event_loop().run_in_executor(
            None, _parse_zip, source_zip_b64  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# URL path — GitPython shallow clone
# ---------------------------------------------------------------------------


def _parse_git_url(url: str) -> list[tuple[str, str, SupportedLanguage]]:
    """
    Clone the repository at *url* to a temp directory, walk all source files,
    and return (rel_path, content, language) tuples.

    Uses tools.github_tools.fetch_repo + list_source_files so that all
    filtering rules (excluded dirs, size limits, binary detection) are
    applied consistently.  The temp directory is always deleted on exit.

    Args:
        url: Validated GitHub/GitLab HTTPS URL.

    Returns:
        List of (rel_path, content, SupportedLanguage) sorted by rel_path.

    Raises:
        RuntimeError: Clone or file-listing failed.
    """
    from tools.github_tools import (
        CloneFailedError,
        EmptyRepositoryError,
        InvalidRepositoryURLError,
        fetch_repo,
        list_source_files,
    )

    logger.info("parse_git_url_start", url=url)

    try:
        repo = fetch_repo(url)
    except (InvalidRepositoryURLError, CloneFailedError, TimeoutError) as exc:
        raise RuntimeError(f"Failed to clone repository {url!r}: {exc}") from exc

    try:
        try:
            source_files = list_source_files(repo)
        except EmptyRepositoryError:
            logger.warning(
                "parse_git_url_empty_repo",
                url=url,
                note="No analysable source files found — returning empty list",
            )
            return []

        results: list[tuple[str, str, SupportedLanguage]] = [
            (sf.rel_path, sf.content, sf.language)
            for sf in source_files
        ]

        logger.info("parse_git_url_complete", url=url, files=len(results))
        return results

    finally:
        # Always clean up, even on exception
        from tools.github_tools import cleanup_repo
        try:
            cleanup_repo(repo)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# ZIP path — base64 → extract → filter → return
# ---------------------------------------------------------------------------


def _parse_zip(zip_b64: str) -> list[tuple[str, str, SupportedLanguage]]:
    """
    Decode a base64 ZIP archive, extract source files to a temp directory,
    apply the same filtering rules as the URL path, and return tuples.

    Security:
        * Zip-slip guard: members whose resolved absolute path escapes the
          temp root are skipped with a WARNING.
        * Per-file size ceiling: _MAX_FILE_SIZE_BYTES.
        * Binary detection: files containing null bytes are skipped.
        * Member count ceiling: _MAX_FILES_PER_REPO.

    Args:
        zip_b64: Base64-encoded ZIP archive (no data: URI prefix needed).

    Returns:
        List of (rel_path, content, SupportedLanguage) sorted by rel_path.

    Raises:
        ValueError  : Payload is not valid base64 or not a ZIP file.
        RuntimeError: Extraction failed.
    """
    logger.info("parse_zip_start", b64_len=len(zip_b64))

    try:
        raw = base64.b64decode(zip_b64)
    except Exception as exc:
        raise ValueError(f"Invalid base64 payload: {exc}") from exc

    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Payload is not a valid ZIP archive: {exc}") from exc

    tmpdir = Path(tempfile.mkdtemp(prefix="codeguardian_zip_"))
    results: list[tuple[str, str, SupportedLanguage]] = []

    try:
        members = zf.infolist()
        processed = 0

        for member in members:
            if processed >= _MAX_FILES_PER_REPO:
                logger.warning("parse_zip_file_limit_reached", limit=_MAX_FILES_PER_REPO)
                break

            # Skip directories
            if member.filename.endswith("/") or member.is_dir():
                continue

            # Extension filter — skip non-source files early
            if not is_allowed_file(member.filename):
                continue

            # Zip-slip guard
            member_path = (tmpdir / member.filename).resolve()
            if not str(member_path).startswith(str(tmpdir.resolve())):
                logger.warning("parse_zip_path_traversal_blocked", member=member.filename)
                continue

            # Per-file size ceiling
            if member.file_size > _MAX_FILE_SIZE_BYTES:
                logger.warning(
                    "parse_zip_file_too_large",
                    member=member.filename,
                    size=member.file_size,
                    limit=_MAX_FILE_SIZE_BYTES,
                )
                continue

            # Read and binary detection
            try:
                raw_bytes = zf.read(member.filename)
            except Exception as exc:  # noqa: BLE001
                logger.warning("parse_zip_read_error", member=member.filename, error=str(exc))
                continue

            if b"\x00" in raw_bytes:
                logger.debug("parse_zip_skip_binary", member=member.filename)
                continue

            try:
                content = raw_bytes.decode("utf-8", errors="replace")
            except Exception as exc:  # noqa: BLE001
                logger.warning("parse_zip_decode_error", member=member.filename, error=str(exc))
                continue

            # Compute a clean relative path (strip any leading top-level dir
            # that ZIP archives typically add, e.g. "repo-main/src/foo.py" → "src/foo.py")
            rel_path = _strip_zip_prefix(member.filename, members)
            language = EXTENSION_TO_LANGUAGE.get(
                Path(member.filename).suffix.lower(), SupportedLanguage.UNKNOWN
            )

            results.append((rel_path, content, language))
            processed += 1

            logger.debug("parse_zip_file_accepted", rel_path=rel_path, language=language)

    finally:
        zf.close()
        _force_remove_dir(tmpdir)

    results.sort(key=lambda t: t[0])
    logger.info("parse_zip_complete", files=len(results))
    return results


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _strip_zip_prefix(
    filename: str,
    all_members: list[zipfile.ZipInfo],
) -> str:
    """
    Remove a common top-level directory prefix that GitHub adds when
    downloading a ZIP (e.g. "requests-main/src/foo.py" → "src/foo.py").

    If all members share exactly one top-level directory component,
    that component is stripped.  Otherwise the filename is returned as-is.
    """
    parts = filename.replace("\\", "/").split("/")
    if len(parts) <= 1:
        return filename

    # Collect the first path component of every non-directory member
    roots: set[str] = set()
    for m in all_members:
        p = m.filename.replace("\\", "/").split("/")
        if p:
            roots.add(p[0])

    # Strip only if there is exactly one common root (GitHub archive style)
    if len(roots) == 1:
        stripped = "/".join(parts[1:])
        return stripped if stripped else filename

    return filename


def _force_remove_dir(path: Path) -> None:
    """
    Remove a directory tree, resetting read-only permissions on Windows
    (where git objects have read-only bits set).
    """
    def _on_error(func: object, path_str: str, exc_info: object) -> None:
        try:
            os.chmod(path_str, stat.S_IWRITE | stat.S_IREAD)
            if callable(func):
                func(path_str)
        except Exception:  # noqa: BLE001
            pass

    try:
        shutil.rmtree(str(path), onerror=_on_error)
    except Exception:  # noqa: BLE001
        pass
