"""
tools/github_tools.py
======================
GitHub repository ingestion via GitPython.

Public API
----------
    fetch_repo(url, dest_dir, *, depth, timeout) -> ClonedRepo
    list_source_files(repo: ClonedRepo)          -> list[SourceFile]
    detect_languages(files: list[SourceFile])    -> LanguageStats
    cleanup_repo(repo: ClonedRepo)               -> None
    build_repository_metadata(url, name)         -> RepositoryMetadata

Design Decisions
----------------
* Shallow clone (depth=1) by default — avoids pulling full history for analysis.
* All I/O is synchronous; callers in async context must use run_in_executor().
* ClonedRepo is a dataclass context manager — cleanup is always guaranteed via
  __exit__() even on exception.
* Files > MAX_FILE_SIZE_BYTES are skipped and logged as warnings (not errors).
* Binary files are detected and silently skipped.
* Symlinks pointing outside the repo root are rejected (path traversal guard).

Supported Languages (as required)
----------------------------------
    Python · Java · C++ · JavaScript · TypeScript

All other extensions recognised by tools/chunker.py are also passed through
so this module stays forward-compatible with schema additions.

Logging
-------
Uses structlog (project-wide standard). Every public function emits structured
log events at the appropriate level:
    DEBUG  — per-file decisions
    INFO   — function entry/exit with key metrics
    WARNING — skipped files (too large, binary, symlink escape)
    ERROR  — clone failures, permission errors
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import stat
import tempfile
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import urlparse

import structlog

from schemas import RepositoryMetadata, SupportedLanguage
from tools.chunker import EXTENSION_TO_LANGUAGE, ALLOWED_EXTENSIONS

# ---------------------------------------------------------------------------
# Module-level logger (structlog, project standard)
# ---------------------------------------------------------------------------

logger = structlog.get_logger(__name__)

# Also configure stdlib logging so GitPython's internal messages are visible
_git_log = logging.getLogger("git")
_git_log.setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Subset required by this task (Python · Java · C++ · JavaScript · TypeScript)
GITHUB_TOOL_LANGUAGES: frozenset[SupportedLanguage] = frozenset({
    SupportedLanguage.PYTHON,
    SupportedLanguage.JAVA,
    SupportedLanguage.CPP,
    SupportedLanguage.JAVASCRIPT,
    SupportedLanguage.TYPESCRIPT,
})

# Extensions that map to each required language
_REQUIRED_EXTENSIONS: frozenset[str] = frozenset(
    ext
    for ext, lang in EXTENSION_TO_LANGUAGE.items()
    if lang in GITHUB_TOOL_LANGUAGES
)

# Files larger than this are skipped (guards against minified bundles / generated code)
MAX_FILE_SIZE_BYTES: int = 512 * 1024  # 512 KB

# Maximum total files to analyse in a single repo (safety limit)
MAX_FILES_PER_REPO: int = 5_000

# Default shallow clone depth (1 = tip of default branch only)
DEFAULT_CLONE_DEPTH: int = 1

# Clone timeout in seconds
DEFAULT_CLONE_TIMEOUT: int = 120

# Directories always excluded from analysis
_EXCLUDED_DIRS: frozenset[str] = frozenset({
    ".git", ".github", ".gitlab", "node_modules", "__pycache__",
    ".venv", "venv", "env", ".env", "vendor", "third_party",
    "build", "dist", "out", "target", ".gradle", ".idea",
    ".vscode", "coverage", ".nyc_output", ".pytest_cache",
})

# ---------------------------------------------------------------------------
# Data Transfer Objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceFile:
    """
    Represents a single analysable source file inside a cloned repository.

    Attributes:
        abs_path    : Absolute filesystem path to the file.
        rel_path    : POSIX-style path relative to the repository root.
        language    : Detected SupportedLanguage.
        size_bytes  : File size in bytes.
        line_count  : Number of lines (newline-separated).
        content     : Full decoded file content.
    """
    abs_path:   Path
    rel_path:   str          # POSIX relative path, e.g. "src/main/App.java"
    language:   SupportedLanguage
    size_bytes: int
    line_count: int
    content:    str


@dataclass
class LanguageStats:
    """
    Aggregated language statistics computed by detect_languages().

    Attributes:
        primary_language    : Language with the most lines of code.
        language_breakdown  : Map of language enum value → percentage of total lines.
        line_counts         : Raw per-language line counts (before normalisation).
        total_files         : Total number of source files analysed.
        total_lines         : Sum of all source lines across all files.
    """
    primary_language:   SupportedLanguage
    language_breakdown: dict[str, float]   # SupportedLanguage.value → percentage
    line_counts:        dict[str, int]     # SupportedLanguage.value → raw line count
    total_files:        int
    total_lines:        int


@dataclass
class ClonedRepo:
    """
    Handle for a locally cloned repository.

    Implements the context manager protocol for guaranteed cleanup:

        with fetch_repo(url) as repo:
            files = list_source_files(repo)
            ...
        # repo directory is deleted here

    Attributes:
        url         : Original GitHub/GitLab URL.
        local_path  : Absolute path to the cloned directory on disk.
        name        : Repository name derived from the URL (e.g. "my-repo").
        default_branch: Name of the checked-out branch.
        commit_sha  : Full SHA of the HEAD commit.
        _owned      : Whether this object is responsible for deleting local_path.
    """
    url:             str
    local_path:      Path
    name:            str
    default_branch:  str
    commit_sha:      str
    _owned:          bool = field(default=True, repr=False, compare=False)

    # ── Context Manager ───────────────────────────────────────────────────

    def __enter__(self) -> "ClonedRepo":
        return self

    def __exit__(self, *_: object) -> None:
        if self._owned:
            cleanup_repo(self)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class GitHubToolError(RuntimeError):
    """Base exception for all tools/github_tools errors."""


class InvalidRepositoryURLError(GitHubToolError):
    """Raised when the supplied URL does not look like a valid git remote."""


class CloneFailedError(GitHubToolError):
    """Raised when git clone fails (network, auth, timeout, etc.)."""


class EmptyRepositoryError(GitHubToolError):
    """Raised when no analysable source files are found after cloning."""


# ---------------------------------------------------------------------------
# URL Validation
# ---------------------------------------------------------------------------

# Accepts:
#   https://github.com/owner/repo
#   https://github.com/owner/repo.git
#   https://gitlab.com/owner/repo
#   git@github.com:owner/repo.git
_GIT_URL_RE = re.compile(
    r"^(?:"
    r"https?://(?:github|gitlab|bitbucket)\.(?:com|org)/[\w.\-]+/[\w.\-]+"
    r"|git@[\w.\-]+:[\w.\-]+/[\w.\-]+"
    r")(?:\.git)?/?$",
    re.IGNORECASE,
)


def _validate_url(url: str) -> str:
    """
    Validate and normalise a GitHub/GitLab/Bitbucket URL.

    Args:
        url: Raw URL string from the caller.

    Returns:
        Stripped, normalised URL string.

    Raises:
        InvalidRepositoryURLError: If the URL does not match known git patterns.
    """
    url = url.strip().rstrip("/")
    if not _GIT_URL_RE.match(url):
        raise InvalidRepositoryURLError(
            f"URL does not appear to be a valid GitHub/GitLab repository: {url!r}. "
            "Expected format: https://github.com/owner/repo"
        )
    return url


def _extract_repo_name(url: str) -> str:
    """
    Derive a human-readable repository name from the clone URL.

    Examples:
        "https://github.com/acme/my-service.git" -> "my-service"
        "git@github.com:acme/my-service"         -> "my-service"
    """
    # Strip .git suffix and take the last path segment
    path = urlparse(url).path or url.split(":")[-1]
    name = Path(path).stem  # removes .git extension
    return name or "unknown-repo"


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------


def fetch_repo(
    url: str,
    dest_dir: Optional[str | Path] = None,
    *,
    depth: int = DEFAULT_CLONE_DEPTH,
    timeout: int = DEFAULT_CLONE_TIMEOUT,
    branch: Optional[str] = None,
) -> ClonedRepo:
    """
    Clone a remote git repository to a local directory.

    Performs a shallow clone (depth=1) by default to minimise network
    transfer when full history is not required for code analysis.

    Args:
        url      : Public GitHub/GitLab/Bitbucket HTTPS or SSH URL.
        dest_dir : Parent directory to clone into.  If ``None``, a
                   temporary directory is created and owned by the
                   returned ``ClonedRepo`` (deleted on context exit).
        depth    : Shallow clone depth.  Pass ``0`` for a full clone.
        timeout  : Maximum seconds allowed for the clone operation.
        branch   : Specific branch/tag to clone. ``None`` = default branch.

    Returns:
        ``ClonedRepo`` dataclass with repository handle and metadata.

    Raises:
        InvalidRepositoryURLError : URL failed validation.
        CloneFailedError          : git clone subprocess failed.
        TimeoutError              : Clone exceeded ``timeout`` seconds.

    Example:
        with fetch_repo("https://github.com/acme/my-service") as repo:
            files = list_source_files(repo)
    """
    try:
        import git
    except ImportError as exc:
        raise ImportError(
            "GitPython is required: pip install gitpython"
        ) from exc

    url = _validate_url(url)
    repo_name = _extract_repo_name(url)

    # Determine clone destination
    owned = dest_dir is None
    if dest_dir is None:
        dest_dir = Path(tempfile.mkdtemp(prefix=f"codeguardian_{repo_name}_"))
    else:
        dest_dir = Path(dest_dir) / repo_name
        dest_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "fetch_repo_start",
        url=url,
        dest=str(dest_dir),
        depth=depth,
        branch=branch or "default",
        timeout=timeout,
    )

    clone_kwargs: dict[str, object] = {
        "to_path": str(dest_dir),
        "single_branch": True,
    }
    if depth > 0:
        clone_kwargs["depth"] = depth
    if branch:
        clone_kwargs["branch"] = branch

    # GitPython's clone_from does not natively support a timeout,
    # so we run it in a thread and join with a deadline.
    exc_holder: list[Exception] = []
    result_holder: list[git.Repo] = []

    def _do_clone() -> None:
        try:
            result_holder.append(
                git.Repo.clone_from(url, **clone_kwargs)  # type: ignore[arg-type]
            )
        except Exception as e:  # noqa: BLE001
            exc_holder.append(e)

    thread = threading.Thread(target=_do_clone, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        # Thread is still running → timeout exceeded
        logger.error("fetch_repo_timeout", url=url, timeout=timeout)
        # Best-effort cleanup of partial clone
        _force_remove_dir(dest_dir)
        raise TimeoutError(
            f"git clone of {url!r} exceeded {timeout}s timeout. "
            "Consider increasing DEFAULT_CLONE_TIMEOUT or using a shallower depth."
        )

    if exc_holder:
        error = exc_holder[0]
        logger.error("fetch_repo_clone_failed", url=url, error=str(error))
        _force_remove_dir(dest_dir)
        raise CloneFailedError(
            f"Failed to clone {url!r}: {error}"
        ) from error

    git_repo: git.Repo = result_holder[0]
    head = git_repo.head.commit
    active_branch = "HEAD"
    try:
        active_branch = git_repo.active_branch.name
    except TypeError:
        # Detached HEAD state (e.g. when cloning a tag)
        pass

    logger.info(
        "fetch_repo_complete",
        url=url,
        name=repo_name,
        branch=active_branch,
        commit=head.hexsha[:12],
        dest=str(dest_dir),
    )

    return ClonedRepo(
        url=url,
        local_path=dest_dir,
        name=repo_name,
        default_branch=active_branch,
        commit_sha=head.hexsha,
        _owned=owned,
    )


def list_source_files(
    repo: ClonedRepo,
    *,
    max_file_size: int = MAX_FILE_SIZE_BYTES,
    max_files: int = MAX_FILES_PER_REPO,
    languages: Optional[frozenset[SupportedLanguage]] = None,
) -> list[SourceFile]:
    """
    Walk the cloned repository and return analysable source files.

    Filtering rules (applied in order):
      1. Skip files in excluded directories (``_EXCLUDED_DIRS``).
      2. Skip files whose extension is not in ``ALLOWED_EXTENSIONS``
         (or the caller-supplied ``languages`` subset).
      3. Skip symlinks that resolve outside the repository root (path traversal guard).
      4. Skip binary files (null byte detection).
      5. Skip files larger than ``max_file_size`` bytes.
      6. Stop after ``max_files`` files and emit a warning.

    Args:
        repo         : ``ClonedRepo`` returned by ``fetch_repo()``.
        max_file_size: Per-file size ceiling in bytes.
        max_files    : Global file count ceiling.
        languages    : Optional whitelist of ``SupportedLanguage`` values.
                       ``None`` = accept all ``ALLOWED_EXTENSIONS``.

    Returns:
        List of ``SourceFile`` objects, sorted by ``rel_path`` (deterministic order).

    Raises:
        EmptyRepositoryError: No analysable source files found.
    """
    if not repo.local_path.is_dir():
        raise CloneFailedError(
            f"Repository path does not exist: {repo.local_path}"
        )

    # Build the effective extension set
    if languages is not None:
        allowed_exts = frozenset(
            ext for ext, lang in EXTENSION_TO_LANGUAGE.items() if lang in languages
        )
    else:
        allowed_exts = _REQUIRED_EXTENSIONS  # task-scoped default

    repo_root = repo.local_path.resolve()
    source_files: list[SourceFile] = []
    skipped_excluded = 0
    skipped_extension = 0
    skipped_symlink = 0
    skipped_binary = 0
    skipped_size = 0

    for abs_path in _walk_repo(repo_root):
        if len(source_files) >= max_files:
            logger.warning(
                "list_source_files_limit_reached",
                max_files=max_files,
                repo=repo.name,
            )
            break

        # 1. Excluded directories
        if _is_in_excluded_dir(abs_path, repo_root):
            skipped_excluded += 1
            logger.debug("skip_excluded_dir", path=str(abs_path))
            continue

        # 2. Extension filter
        if abs_path.suffix.lower() not in allowed_exts:
            skipped_extension += 1
            continue

        # 3. Symlink safety
        if abs_path.is_symlink():
            real = abs_path.resolve()
            if not str(real).startswith(str(repo_root)):
                logger.warning(
                    "skip_symlink_escape",
                    path=str(abs_path),
                    resolved=str(real),
                )
                skipped_symlink += 1
                continue

        # 4. Size check (before read)
        try:
            size_bytes = abs_path.stat().st_size
        except OSError as exc:
            logger.warning("skip_stat_error", path=str(abs_path), error=str(exc))
            continue

        if size_bytes > max_file_size:
            logger.warning(
                "skip_file_too_large",
                path=str(abs_path),
                size_bytes=size_bytes,
                limit_bytes=max_file_size,
            )
            skipped_size += 1
            continue

        # 5. Read and binary detection
        try:
            raw = abs_path.read_bytes()
        except OSError as exc:
            logger.warning("skip_read_error", path=str(abs_path), error=str(exc))
            continue

        if b"\x00" in raw:
            logger.debug("skip_binary", path=str(abs_path))
            skipped_binary += 1
            continue

        try:
            content = raw.decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            logger.warning("skip_decode_error", path=str(abs_path), error=str(exc))
            continue

        rel_path = abs_path.relative_to(repo_root).as_posix()
        language = EXTENSION_TO_LANGUAGE.get(abs_path.suffix.lower(), SupportedLanguage.UNKNOWN)
        line_count = content.count("\n") + 1

        source_files.append(
            SourceFile(
                abs_path=abs_path,
                rel_path=rel_path,
                language=language,
                size_bytes=size_bytes,
                line_count=line_count,
                content=content,
            )
        )

        logger.debug(
            "source_file_accepted",
            rel_path=rel_path,
            language=language,
            lines=line_count,
            bytes=size_bytes,
        )

    logger.info(
        "list_source_files_complete",
        repo=repo.name,
        accepted=len(source_files),
        skipped_excluded=skipped_excluded,
        skipped_extension=skipped_extension,
        skipped_symlink=skipped_symlink,
        skipped_binary=skipped_binary,
        skipped_size=skipped_size,
    )

    if not source_files:
        raise EmptyRepositoryError(
            f"No analysable source files found in {repo.url!r}. "
            f"Supported extensions: {sorted(allowed_exts)}"
        )

    # Deterministic ordering for reproducible downstream processing
    return sorted(source_files, key=lambda f: f.rel_path)


def detect_languages(files: list[SourceFile]) -> LanguageStats:
    """
    Aggregate per-language line counts from a list of source files.

    The ``language_breakdown`` percentages sum to 100.0 (±0.1 due to rounding).
    The ``primary_language`` is the language with the most lines of code.

    Args:
        files: List of ``SourceFile`` objects from ``list_source_files()``.

    Returns:
        ``LanguageStats`` with breakdown percentages and primary language.

    Raises:
        ValueError: If ``files`` is empty.

    Example:
        stats = detect_languages(source_files)
        print(stats.primary_language)           # SupportedLanguage.PYTHON
        print(stats.language_breakdown)         # {"python": 78.4, "java": 21.6}
    """
    if not files:
        raise ValueError("Cannot detect languages from an empty file list.")

    # Accumulate line counts keyed by SupportedLanguage enum value (str)
    line_counts: Counter[str] = Counter()
    for f in files:
        line_counts[f.language] += f.line_count

    total_lines = sum(line_counts.values())
    total_files = len(files)

    # Sort descending by line count for deterministic ranking
    ranked = line_counts.most_common()

    # Compute percentages with rounding; adjust last bucket to ensure sum == 100.0
    breakdown: dict[str, float] = {}
    running_sum = 0.0
    for i, (lang, count) in enumerate(ranked):
        if i < len(ranked) - 1:
            pct = round((count / total_lines) * 100, 1)
            breakdown[lang] = pct
            running_sum += pct
        else:
            # Last entry absorbs rounding error
            breakdown[lang] = round(100.0 - running_sum, 1)

    primary_lang_str = ranked[0][0]
    try:
        primary_language = SupportedLanguage(primary_lang_str)
    except ValueError:
        primary_language = SupportedLanguage.UNKNOWN

    raw_counts: dict[str, int] = dict(line_counts)

    logger.info(
        "detect_languages_complete",
        primary=primary_language,
        total_files=total_files,
        total_lines=total_lines,
        languages=list(breakdown.keys()),
    )

    return LanguageStats(
        primary_language=primary_language,
        language_breakdown=breakdown,
        line_counts=raw_counts,
        total_files=total_files,
        total_lines=total_lines,
    )


def cleanup_repo(repo: ClonedRepo) -> None:
    """
    Remove the locally cloned repository directory from disk.

    This is always a best-effort operation: if removal fails (e.g. due to
    file locks on Windows), a warning is logged rather than raising.

    On Windows, git objects have read-only bits set. ``_force_remove_dir``
    handles this by resetting permissions before deletion.

    Args:
        repo: ``ClonedRepo`` handle returned by ``fetch_repo()``.
    """
    if not repo.local_path.exists():
        logger.debug("cleanup_repo_already_gone", path=str(repo.local_path))
        return

    logger.info("cleanup_repo_start", path=str(repo.local_path), name=repo.name)
    _force_remove_dir(repo.local_path)
    logger.info("cleanup_repo_complete", path=str(repo.local_path))


# ---------------------------------------------------------------------------
# High-Level Orchestration
# ---------------------------------------------------------------------------


def build_repository_metadata(
    url: str,
    repository_name: Optional[str] = None,
    *,
    depth: int = DEFAULT_CLONE_DEPTH,
    timeout: int = DEFAULT_CLONE_TIMEOUT,
    branch: Optional[str] = None,
    languages: Optional[frozenset[SupportedLanguage]] = None,
) -> RepositoryMetadata:
    """
    Clone a GitHub repository and return a populated ``RepositoryMetadata``.

    This is the primary entry point for the ingest_node. It orchestrates:
      1. ``fetch_repo()``      — clone
      2. ``list_source_files()`` — filter
      3. ``detect_languages()``  — stats
      4. ``cleanup_repo()``    — delete local clone
      5. Return ``RepositoryMetadata`` (matches canonical schema)

    The local clone is cleaned up even if an error occurs during steps 2–3.

    Args:
        url             : GitHub/GitLab repository URL.
        repository_name : Display name for the report. Defaults to repo slug.
        depth           : Shallow clone depth.
        timeout         : Clone timeout in seconds.
        branch          : Specific branch to clone. ``None`` = default branch.
        languages       : Optional language filter. ``None`` = all 5 required langs.

    Returns:
        ``RepositoryMetadata`` populated with real stats from the cloned repo.

    Raises:
        InvalidRepositoryURLError : Invalid URL.
        CloneFailedError          : git clone failed.
        EmptyRepositoryError      : No analysable source files found.
        TimeoutError              : Clone exceeded timeout.

    Example:
        metadata = build_repository_metadata(
            "https://github.com/acme/my-service",
            repository_name="my-service",
        )
        print(metadata.primary_language)   # "python"
        print(metadata.overall_score)      # N/A — not set here
    """
    logger.info(
        "build_repository_metadata_start",
        url=url,
        name=repository_name or "(auto)",
    )

    repo = fetch_repo(url, depth=depth, timeout=timeout, branch=branch)
    try:
        files = list_source_files(repo, languages=languages)
        stats = detect_languages(files)
    finally:
        # Always clean up, even on error
        cleanup_repo(repo)

    name = repository_name or repo.name

    metadata = RepositoryMetadata(
        repository_name=name,
        source_url=url,
        primary_language=stats.primary_language,
        language_breakdown=stats.language_breakdown,
        total_files=stats.total_files,
        total_lines=stats.total_lines,
    )

    logger.info(
        "build_repository_metadata_complete",
        name=name,
        primary_language=metadata.primary_language,
        total_files=metadata.total_files,
        total_lines=metadata.total_lines,
    )

    return metadata


# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------


def _walk_repo(root: Path) -> Iterator[Path]:
    """
    Recursively yield all regular files under ``root``.
    Skips top-level excluded directories early (``os.walk`` prune strategy).
    """
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # Prune excluded directories in-place (modifying dirnames prevents descent)
        dirnames[:] = [
            d for d in dirnames
            if d not in _EXCLUDED_DIRS and not d.startswith(".")
        ]
        for filename in filenames:
            yield Path(dirpath) / filename


def _is_in_excluded_dir(path: Path, root: Path) -> bool:
    """Return True if any component of ``path`` (relative to ``root``) is excluded."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        return True  # path not under root — treat as excluded
    return any(part in _EXCLUDED_DIRS or part.startswith(".") for part in rel.parts[:-1])


def _force_remove_dir(path: Path) -> None:
    """
    Remove a directory tree, forcibly resetting read-only permissions first.

    Git objects on Windows have read-only bits set; standard ``shutil.rmtree``
    fails on these. This handler resets permissions before retry.
    """
    def _on_error(func: object, path_str: str, exc_info: object) -> None:
        """onerror handler: chmod and retry."""
        try:
            os.chmod(path_str, stat.S_IWRITE | stat.S_IREAD)
            if callable(func):
                func(path_str)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "cleanup_permission_error",
                path=path_str,
                error=str(e),
            )

    try:
        shutil.rmtree(str(path), onerror=_on_error)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "cleanup_failed",
            path=str(path),
            error=str(exc),
        )
