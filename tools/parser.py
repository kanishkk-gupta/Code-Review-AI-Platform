"""
tools/parser.py
===============
Source code structure extractor for 5 languages.

Public API
----------
    parse_repository(repo_path, name, source_url) -> ParsedRepository
    parse_file(rel_path, content, language)        -> ParsedFile
    build_metadata_from_parsed(parsed)             -> RepositoryMetadata

Parsers
-------
    Python      : stdlib ast (full fidelity)
    Java        : regex (classes, methods, imports)
    C++         : regex (classes, functions, #includes)
    JavaScript  : regex (classes, functions, imports/require)
    TypeScript  : regex (classes, interfaces, functions, imports)

All output conforms to schemas.py (RepositoryMetadata).
Detailed structure (ParsedFile / ParsedClass / etc.) uses local dataclasses.
"""

from __future__ import annotations

import ast
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import structlog

from schemas import RepositoryMetadata, SupportedLanguage
from tools.chunker import EXTENSION_TO_LANGUAGE, is_allowed_file

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses — detailed structure output
# ---------------------------------------------------------------------------

@dataclass
class ParsedImport:
    module: str
    alias: Optional[str] = None
    symbols: list[str] = field(default_factory=list)

@dataclass
class ParsedFunction:
    name: str
    line_start: int
    line_end: int
    params: list[str] = field(default_factory=list)
    is_async: bool = False
    decorators: list[str] = field(default_factory=list)
    docstring: Optional[str] = None

@dataclass
class ParsedClass:
    name: str
    line_start: int
    line_end: int
    base_classes: list[str] = field(default_factory=list)
    methods: list[ParsedFunction] = field(default_factory=list)
    docstring: Optional[str] = None

@dataclass
class ParsedFile:
    rel_path: str
    language: SupportedLanguage
    total_lines: int
    classes: list[ParsedClass] = field(default_factory=list)
    functions: list[ParsedFunction] = field(default_factory=list)
    imports: list[ParsedImport] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)

    @property
    def class_count(self) -> int:
        return len(self.classes)

    @property
    def function_count(self) -> int:
        return len(self.functions) + sum(len(c.methods) for c in self.classes)

    @property
    def import_count(self) -> int:
        return len(self.imports)

@dataclass
class ParsedRepository:
    root_path: Path
    name: str
    source_url: Optional[str]
    files: list[ParsedFile] = field(default_factory=list)

    @property
    def total_files(self) -> int:
        return len(self.files)

    @property
    def total_lines(self) -> int:
        return sum(f.total_lines for f in self.files)

    @property
    def total_classes(self) -> int:
        return sum(f.class_count for f in self.files)

    @property
    def total_functions(self) -> int:
        return sum(f.function_count for f in self.files)

    @property
    def primary_language(self) -> SupportedLanguage:
        counts: Counter[str] = Counter()
        for f in self.files:
            counts[f.language] += f.total_lines
        if not counts:
            return SupportedLanguage.UNKNOWN
        top = counts.most_common(1)[0][0]
        try:
            return SupportedLanguage(top)
        except ValueError:
            return SupportedLanguage.UNKNOWN

    @property
    def language_breakdown(self) -> dict[str, float]:
        counts: Counter[str] = Counter()
        for f in self.files:
            counts[f.language] += f.total_lines
        total = sum(counts.values()) or 1
        ranked = counts.most_common()
        result: dict[str, float] = {}
        running = 0.0
        for i, (lang, cnt) in enumerate(ranked):
            if i < len(ranked) - 1:
                pct = round((cnt / total) * 100, 1)
                result[lang] = pct
                running += pct
            else:
                result[lang] = round(100.0 - running, 1)
        return result

    def metadata(self) -> RepositoryMetadata:
        return build_metadata_from_parsed(self)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class ParserError(RuntimeError):
    """Raised when a file cannot be parsed and recovery is impossible."""

class UnsupportedLanguageError(ParserError):
    """Raised when parse_file is called with an unsupported language."""


# ---------------------------------------------------------------------------
# Excluded directories (mirrors github_tools)
# ---------------------------------------------------------------------------

_EXCLUDED_DIRS: frozenset[str] = frozenset({
    ".git", ".github", ".gitlab", "node_modules", "__pycache__",
    ".venv", "venv", "env", "vendor", "third_party",
    "build", "dist", "out", "target", ".gradle",
})

_MAX_FILE_BYTES = 512 * 1024   # 512 KB
_MAX_FILES      = 5_000


# ===========================================================================
# Public API
# ===========================================================================

def parse_repository(
    repo_path: str | Path,
    name: Optional[str] = None,
    source_url: Optional[str] = None,
) -> ParsedRepository:
    """
    Walk *repo_path*, parse every supported source file, and return a
    ``ParsedRepository`` containing full structural information.

    Args:
        repo_path  : Absolute or relative path to the repository root.
        name       : Human-readable project name (defaults to directory name).
        source_url : Original remote URL (optional, stored in metadata).

    Returns:
        ``ParsedRepository`` with per-file classes, functions, and imports.

    Raises:
        FileNotFoundError : ``repo_path`` does not exist or is not a directory.
    """
    root = Path(repo_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Repository path is not a directory: {root}")

    project_name = name or root.name
    parsed = ParsedRepository(root_path=root, name=project_name, source_url=source_url)
    file_count = 0

    logger.info("parse_repository_start", path=str(root), name=project_name)

    for abs_path in _walk(root):
        if file_count >= _MAX_FILES:
            logger.warning("parse_repository_file_limit", limit=_MAX_FILES)
            break
        if not is_allowed_file(str(abs_path)):
            continue
        try:
            size = abs_path.stat().st_size
        except OSError:
            continue
        if size > _MAX_FILE_BYTES:
            logger.warning("parse_skip_large", path=str(abs_path), bytes=size)
            continue
        try:
            raw = abs_path.read_bytes()
        except OSError as exc:
            logger.warning("parse_skip_read_error", path=str(abs_path), error=str(exc))
            continue
        if b"\x00" in raw:
            logger.debug("parse_skip_binary", path=str(abs_path))
            continue
        content = raw.decode("utf-8", errors="replace")
        rel_path = abs_path.relative_to(root).as_posix()
        lang = EXTENSION_TO_LANGUAGE.get(abs_path.suffix.lower(), SupportedLanguage.UNKNOWN)

        pf = parse_file(rel_path, content, lang)
        parsed.files.append(pf)
        file_count += 1
        logger.debug(
            "parse_file_done",
            rel=rel_path,
            lang=lang,
            classes=pf.class_count,
            functions=pf.function_count,
            imports=pf.import_count,
        )

    logger.info(
        "parse_repository_complete",
        files=parsed.total_files,
        lines=parsed.total_lines,
        classes=parsed.total_classes,
        functions=parsed.total_functions,
        primary=parsed.primary_language,
    )
    return parsed


def parse_file(
    rel_path: str,
    content: str,
    language: SupportedLanguage,
) -> ParsedFile:
    """
    Parse a single source file and extract structural elements.

    Args:
        rel_path : Repository-relative POSIX path (used in output only).
        content  : Full decoded file content.
        language : ``SupportedLanguage`` enum value.

    Returns:
        ``ParsedFile`` with classes, functions, and imports extracted.

    Raises:
        UnsupportedLanguageError : Language has no registered parser.
    """
    total_lines = content.count("\n") + 1

    _DISPATCH = {
        SupportedLanguage.PYTHON:     _parse_python,
        SupportedLanguage.JAVA:       _parse_java,
        SupportedLanguage.CPP:        _parse_cpp,
        SupportedLanguage.C:          _parse_cpp,
        SupportedLanguage.JAVASCRIPT: _parse_javascript,
        SupportedLanguage.TYPESCRIPT: _parse_typescript,
    }

    parser_fn = _DISPATCH.get(language)
    if parser_fn is None:
        logger.debug("parse_file_unsupported", lang=language, path=rel_path)
        return ParsedFile(rel_path=rel_path, language=language, total_lines=total_lines)

    try:
        classes, functions, imports = parser_fn(content)
    except Exception as exc:   # noqa: BLE001
        logger.warning("parse_file_error", path=rel_path, error=str(exc))
        return ParsedFile(
            rel_path=rel_path,
            language=language,
            total_lines=total_lines,
            parse_errors=[str(exc)],
        )

    return ParsedFile(
        rel_path=rel_path,
        language=language,
        total_lines=total_lines,
        classes=classes,
        functions=functions,
        imports=imports,
    )


def build_metadata_from_parsed(parsed: ParsedRepository) -> RepositoryMetadata:
    """
    Convert a ``ParsedRepository`` into the canonical ``RepositoryMetadata`` schema.

    Args:
        parsed: Output from ``parse_repository()``.

    Returns:
        ``RepositoryMetadata`` validated by Pydantic V2.

    Raises:
        ValueError: If the repository contains no parseable files.
    """
    if not parsed.files:
        raise ValueError(
            f"Cannot build RepositoryMetadata: no parseable files in '{parsed.name}'."
        )

    bd = parsed.language_breakdown
    # Ensure breakdown sums to ~100 (validator tolerance is ±1.0)
    if bd and abs(sum(bd.values()) - 100.0) > 0.5:
        # Re-normalise last bucket
        keys = list(bd.keys())
        bd[keys[-1]] = round(100.0 - sum(list(bd.values())[:-1]), 1)

    return RepositoryMetadata(
        repository_name=parsed.name,
        source_url=parsed.source_url,
        primary_language=parsed.primary_language,
        language_breakdown=bd,
        total_files=parsed.total_files,
        total_lines=max(parsed.total_lines, 1),
    )


# ===========================================================================
# Language Parsers
# ===========================================================================

# ── Python (AST) ─────────────────────────────────────────────────────────────

def _parse_python(
    content: str,
) -> tuple[list[ParsedClass], list[ParsedFunction], list[ParsedImport]]:
    """Full-fidelity Python parser using the stdlib ``ast`` module."""
    tree = ast.parse(content)
    lines = content.splitlines()

    classes: list[ParsedClass] = []
    module_functions: list[ParsedFunction] = []
    imports: list[ParsedImport] = []

    for node in ast.walk(tree):
        # ── Imports ──────────────────────────────────────────────────────
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(ParsedImport(module=alias.name, alias=alias.asname))

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            symbols = [a.name for a in node.names]
            imports.append(ParsedImport(module=module, symbols=symbols))

    # Two-pass: top-level classes and functions first
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append(_py_class(node, lines))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            module_functions.append(_py_function(node, lines))

    return classes, module_functions, imports


def _py_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    lines: list[str],
) -> ParsedFunction:
    params = [arg.arg for arg in node.args.args]
    decorators = [
        ast.unparse(d) if hasattr(ast, "unparse") else getattr(d, "id", "")
        for d in node.decorator_list
    ]
    docstring: Optional[str] = None
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        docstring = node.body[0].value.value[:500]  # cap at 500 chars

    end_line = getattr(node, "end_lineno", node.lineno)
    return ParsedFunction(
        name=node.name,
        line_start=node.lineno,
        line_end=end_line,
        params=params,
        is_async=isinstance(node, ast.AsyncFunctionDef),
        decorators=decorators,
        docstring=docstring,
    )


def _py_class(node: ast.ClassDef, lines: list[str]) -> ParsedClass:
    bases = [
        ast.unparse(b) if hasattr(ast, "unparse") else getattr(b, "id", "")
        for b in node.bases
    ]
    docstring: Optional[str] = None
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        docstring = node.body[0].value.value[:500]

    methods: list[ParsedFunction] = []
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods.append(_py_function(child, lines))

    end_line = getattr(node, "end_lineno", node.lineno)
    return ParsedClass(
        name=node.name,
        line_start=node.lineno,
        line_end=end_line,
        base_classes=bases,
        methods=methods,
        docstring=docstring,
    )


# ── Java (regex) ─────────────────────────────────────────────────────────────

# import [static] com.example.Foo;
_JAVA_IMPORT = re.compile(
    r"^\s*import\s+(static\s+)?([\w.]+)\s*;", re.MULTILINE
)
# class / interface / enum Foo [extends Bar] [implements Baz]
_JAVA_CLASS = re.compile(
    r"^[ \t]*(?:(?:public|private|protected|abstract|final|static)\s+)*"
    r"(class|interface|enum)\s+(\w+)"
    r"(?:\s+extends\s+([\w<>, ]+))?"
    r"(?:\s+implements\s+([\w<>, ]+))?",
    re.MULTILINE,
)
# [modifiers] ReturnType methodName(...)
_JAVA_METHOD = re.compile(
    r"^[ \t]*(?:(?:public|private|protected|static|final|abstract|"
    r"synchronized|native|default|override)\s+)+"
    r"(?:[\w<>\[\]?,\s]+?\s+)?(\w+)\s*\(([^)]*)\)\s*(?:throws\s+[\w,\s]+)?\s*[{;]",
    re.MULTILINE,
)


def _parse_java(
    content: str,
) -> tuple[list[ParsedClass], list[ParsedFunction], list[ParsedImport]]:
    imports: list[ParsedImport] = [
        ParsedImport(
            module=m.group(2),
            symbols=["static"] if m.group(1) else [],
        )
        for m in _JAVA_IMPORT.finditer(content)
    ]

    lines_list = content.splitlines()

    classes: list[ParsedClass] = []
    for m in _JAVA_CLASS.finditer(content):
        line_no = content[: m.start()].count("\n") + 1
        bases: list[str] = []
        if m.group(3):
            bases += [b.strip() for b in m.group(3).split(",")]
        if m.group(4):
            bases += [b.strip() for b in m.group(4).split(",")]
        classes.append(
            ParsedClass(
                name=m.group(2),
                line_start=line_no,
                line_end=_find_block_end(lines_list, line_no - 1),
                base_classes=bases,
            )
        )

    methods: list[ParsedFunction] = []
    for m in _JAVA_METHOD.finditer(content):
        line_no = content[: m.start()].count("\n") + 1
        raw_params = m.group(2).strip()
        params = [p.strip().split()[-1] for p in raw_params.split(",") if p.strip()]
        methods.append(
            ParsedFunction(
                name=m.group(1),
                line_start=line_no,
                line_end=line_no,
                params=params,
            )
        )

    return classes, methods, imports


# ── C++ (regex) ──────────────────────────────────────────────────────────────

_CPP_INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.MULTILINE)
_CPP_CLASS = re.compile(
    r"^[ \t]*(?:template\s*<[^>]*>\s*)?"
    r"(class|struct)\s+(\w+)"
    r"(?:\s*:\s*(?:public|private|protected)\s+\w+(?:\s*,\s*(?:public|private|protected)\s+\w+)*)?",
    re.MULTILINE,
)
_CPP_FUNCTION = re.compile(
    r"^([ \t]*)(?:(?:virtual|static|inline|explicit|constexpr|override|"
    r"[[nodiscard]]\s*)?)"
    r"(?:[\w:*&<>, ]+?\s+)(\w+)\s*\(([^)]*)\)\s*(?:const\s*)?(?:noexcept\s*)?[{;]",
    re.MULTILINE,
)
_CPP_KEYWORDS = frozenset({
    "if", "else", "for", "while", "do", "switch", "case", "return",
    "class", "struct", "namespace", "template", "try", "catch",
})


def _parse_cpp(
    content: str,
) -> tuple[list[ParsedClass], list[ParsedFunction], list[ParsedImport]]:
    imports: list[ParsedImport] = [
        ParsedImport(module=m.group(1))
        for m in _CPP_INCLUDE.finditer(content)
    ]

    lines_list = content.splitlines()

    classes: list[ParsedClass] = []
    for m in _CPP_CLASS.finditer(content):
        line_no = content[: m.start()].count("\n") + 1
        classes.append(
            ParsedClass(
                name=m.group(2),
                line_start=line_no,
                line_end=_find_block_end(lines_list, line_no - 1),
            )
        )

    functions: list[ParsedFunction] = []
    for m in _CPP_FUNCTION.finditer(content):
        name = m.group(2)
        if name in _CPP_KEYWORDS:
            continue
        line_no = content[: m.start()].count("\n") + 1
        raw_params = m.group(3).strip()
        params = [p.strip() for p in raw_params.split(",") if p.strip()]
        functions.append(
            ParsedFunction(name=name, line_start=line_no, line_end=line_no, params=params)
        )

    return classes, functions, imports


# ── JavaScript (regex) ────────────────────────────────────────────────────────

_JS_IMPORT_ES6 = re.compile(
    r"""^\s*import\s+(?:(?:\w+|\{[^}]*\}|\*\s+as\s+\w+)(?:\s*,\s*(?:\w+|\{[^}]*\}|\*\s+as\s+\w+))*\s+from\s+)?['"]([^'"]+)['"]""",
    re.MULTILINE,
)
_JS_REQUIRE = re.compile(
    r"""(?:const|let|var)\s+(?:\w+|\{[^}]*\})\s*=\s*require\s*\(\s*['"]([^'"]+)['"]\s*\)""",
    re.MULTILINE,
)
_JS_CLASS = re.compile(
    r"^[ \t]*(?:export\s+(?:default\s+)?)?class\s+(\w+)(?:\s+extends\s+(\w+))?",
    re.MULTILINE,
)
_JS_FUNCTION = re.compile(
    r"^[ \t]*(?:export\s+(?:default\s+)?)?"
    r"(?:(async)\s+)?"
    r"(?:"
    r"function\s*\*?\s*(\w+)\s*\(([^)]*)\)"
    r"|(?:const|let|var)\s+(\w+)\s*=\s*(?:(async)\s+)?(?:function\s*\*?\s*\(([^)]*)\)|\(([^)]*)\)\s*=>|(\w+)\s*=>)"
    r")",
    re.MULTILINE,
)


def _parse_javascript(
    content: str,
) -> tuple[list[ParsedClass], list[ParsedFunction], list[ParsedImport]]:
    imports: list[ParsedImport] = []
    for m in _JS_IMPORT_ES6.finditer(content):
        imports.append(ParsedImport(module=m.group(1)))
    for m in _JS_REQUIRE.finditer(content):
        imports.append(ParsedImport(module=m.group(1)))

    lines_list = content.splitlines()
    classes: list[ParsedClass] = []
    for m in _JS_CLASS.finditer(content):
        line_no = content[: m.start()].count("\n") + 1
        bases = [m.group(2)] if m.group(2) else []
        classes.append(
            ParsedClass(
                name=m.group(1),
                line_start=line_no,
                line_end=_find_block_end(lines_list, line_no - 1),
                base_classes=bases,
            )
        )

    functions: list[ParsedFunction] = []
    for m in _JS_FUNCTION.finditer(content):
        # group(2) = `function name`, group(4) = `const name =`
        name = m.group(2) or m.group(4)
        if not name:
            continue
        line_no = content[: m.start()].count("\n") + 1
        raw_p = m.group(3) or m.group(6) or m.group(7) or ""
        params = [p.strip() for p in raw_p.split(",") if p.strip()]
        is_async = bool(m.group(1) or m.group(5))
        functions.append(
            ParsedFunction(
                name=name,
                line_start=line_no,
                line_end=line_no,
                params=params,
                is_async=is_async,
            )
        )

    return classes, functions, imports


# ── TypeScript (regex) ────────────────────────────────────────────────────────

_TS_INTERFACE = re.compile(
    r"^[ \t]*(?:export\s+)?interface\s+(\w+)(?:\s+extends\s+([\w, ]+))?",
    re.MULTILINE,
)
_TS_TYPE_ALIAS = re.compile(
    r"^[ \t]*(?:export\s+)?type\s+(\w+)\s*(?:<[^>]*>)?\s*=",
    re.MULTILINE,
)


def _parse_typescript(
    content: str,
) -> tuple[list[ParsedClass], list[ParsedFunction], list[ParsedImport]]:
    # TypeScript is a superset of JavaScript — reuse JS parser
    classes, functions, imports = _parse_javascript(content)

    lines_list = content.splitlines()

    # Add interfaces as classes (structurally identical for our purposes)
    for m in _TS_INTERFACE.finditer(content):
        line_no = content[: m.start()].count("\n") + 1
        bases = [b.strip() for b in m.group(2).split(",")] if m.group(2) else []
        classes.append(
            ParsedClass(
                name=m.group(1),
                line_start=line_no,
                line_end=_find_block_end(lines_list, line_no - 1),
                base_classes=bases,
            )
        )

    return classes, functions, imports


# ===========================================================================
# Private helpers
# ===========================================================================

def _walk(root: Path):
    """Yield all files under *root*, pruning excluded directories."""
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [
            d for d in dirnames
            if d not in _EXCLUDED_DIRS and not d.startswith(".")
        ]
        for name in filenames:
            yield Path(dirpath) / name


def _find_block_end(lines: list[str], start_idx: int) -> int:
    """
    Walk forward from *start_idx* counting braces to find the closing ``}``
    of the first block opened at or after that line.

    Returns the 1-indexed line number of the closing ``}`` or the last line
    if the block is not closed (e.g. truncated file).
    """
    depth = 0
    opened = False
    for i in range(start_idx, len(lines)):
        line = lines[i]
        for ch in line:
            if ch == "{":
                depth += 1
                opened = True
            elif ch == "}":
                depth -= 1
        if opened and depth <= 0:
            return i + 1  # 1-indexed
    return len(lines)
