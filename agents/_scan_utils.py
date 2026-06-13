"""
agents/_scan_utils.py
======================
Shared pre-processing helpers for all regex-based scan agents.

Provides:
  - Comment / string-literal stripping before pattern matching
  - File-role classification (test / docs / example / production)
  - Severity downgrade for non-production files
  - Shannon entropy + placeholder detection for secret rules
  - Proximity-aware deduplication helper
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from schemas import Severity

# ---------------------------------------------------------------------------
# Severity ladder
# ---------------------------------------------------------------------------

_LADDER: list[Severity] = [
    Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO,
]


def downgrade_severity(severity: Severity, steps: int = 2) -> Severity:
    """Move *severity* down the ladder by *steps* (floor = INFO)."""
    try:
        idx = _LADDER.index(severity)
    except ValueError:
        return severity
    return _LADDER[min(idx + steps, len(_LADDER) - 1)]


# ---------------------------------------------------------------------------
# File-role classification
# ---------------------------------------------------------------------------

_TEST_DIR_RE = re.compile(
    r'(?:^|[/\\])(?:tests?|spec|fixtures?|conftest|mock|fake|stub)[/\\]',
    re.IGNORECASE,
)
_TEST_FILE_RE = re.compile(
    r'(?:^test_|_test\.py$|_spec\.py$|test\.py$)',
    re.IGNORECASE,
)
_DOCS_RE = re.compile(
    r'(?:^|[/\\])(?:docs?|documentation|readme|changelog|contributing)[/\\]',
    re.IGNORECASE,
)
_EXAMPLE_RE = re.compile(
    r'(?:^|[/\\])(?:examples?|samples?|sample|demo|tutorials?|tutorial|cookbook)[/\\]',
    re.IGNORECASE,
)
_BENCHMARK_RE = re.compile(
    r'(?:^|[/\\])(?:benchmarks?|bench)[/\\]',
    re.IGNORECASE,
)
_MIGRATIONS_RE = re.compile(
    r'(?:^|[/\\])migrations[/\\]',
    re.IGNORECASE,
)
# Singular test/ directory (tests/ is covered by _TEST_DIR_RE)
_TEST_SINGULAR_RE = re.compile(r'(?:^|[/\\])test[/\\]', re.IGNORECASE)
# Alternate test directory naming conventions (unit_tests/, __tests__/, etc.)
_EXTRA_TEST_DIR_RE = re.compile(
    r'(?:^|[/\\])(?:__tests__|unit_tests|integration_tests|testing)[/\\]',
    re.IGNORECASE,
)
_STATIC_RE = re.compile(
    r'(?:^|[/\\])static[/\\]',
    re.IGNORECASE,
)


def classify_file_role(file_path: str) -> str:
    """Return 'test' | 'docs' | 'example' | 'production'."""
    p = file_path.replace('\\', '/')
    name = Path(p).name
    if (
        _TEST_DIR_RE.search(p)
        or _TEST_SINGULAR_RE.search(p)
        or _EXTRA_TEST_DIR_RE.search(p)
        or _TEST_FILE_RE.search(name)
        or 'conftest' in name
    ):
        return 'test'
    if _DOCS_RE.search(p):
        return 'docs'
    if _EXAMPLE_RE.search(p):
        return 'example'
    return 'production'


def is_non_production_path(file_path: str) -> bool:
    """
    True for paths excluded from production analysis:
    tests, examples, docs, tutorials, samples, benchmarks, migrations, static assets.
    """
    role = classify_file_role(file_path)
    if role in ('test', 'docs', 'example'):
        return True
    p = file_path.replace('\\', '/')
    if _MIGRATIONS_RE.search(p) or _BENCHMARK_RE.search(p) or _STATIC_RE.search(p):
        return True
    return False


# Values commonly used as test-only or fixture secrets (not real credentials)
_TEST_FIXTURE_SECRETS = frozenset({
    'config', 'test', 'test key', 'testing', 'dev', 'development',
    'secret', 'key', 'changeme', 'dummy', 'placeholder', 'not-a-secret',
    'my-secret-key', 'super secret key',
})


def is_test_fixture_secret(value: str) -> bool:
    """True if *value* is a typical test/config placeholder secret."""
    v = value.strip().lower()
    if not v:
        return True
    if v in _TEST_FIXTURE_SECRETS:
        return True
    if v.startswith('test ') or v.endswith(' test'):
        return True
    if 'test' in v and len(v) < 20:
        return True
    return False


# ---------------------------------------------------------------------------
# Line pre-processing
# ---------------------------------------------------------------------------

def is_comment_line(line: str) -> bool:
    """True if the entire logical line is a comment (starts with #)."""
    return line.lstrip().startswith('#')


def strip_inline_comment(line: str) -> str:
    """Remove the inline comment portion of a line (# not inside a string)."""
    in_single = False
    in_double = False
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == '\\' and (in_single or in_double) and i + 1 < len(line):
            i += 2
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == '#' and not in_single and not in_double:
            return line[:i]
        i += 1
    return line


# Matches single-line string literals (no escaped-quote handling needed
# for the FP cases we target — we just need to remove path content).
_DOUBLE_STR_RE = re.compile(r'"(?:[^"\\]|\\.)*"')
_SINGLE_STR_RE = re.compile(r"'(?:[^'\\]|\\.)*'")


def strip_string_contents(line: str) -> str:
    """
    Replace the *content* of string literals with empty strings while keeping
    the surrounding quotes.  This removes URL paths, route strings, etc. that
    would otherwise look like division operators to a naive regex.

    Example:
        '@app.route("/auth/login")'  →  '@app.route("")'
        'result = total / count'     →  unchanged
    """
    line = _DOUBLE_STR_RE.sub('""', line)
    line = _SINGLE_STR_RE.sub("''", line)
    return line


def compute_docstring_lines(content: str) -> frozenset:
    """
    Return a frozenset of 0-based line indices that fall inside triple-quoted
    strings (docstrings, multi-line strings) in *content*.

    This is the single highest-impact pre-filter: Flask and requests have
    extensive docstrings containing code examples, URLs, and None references
    that are NOT executable code.

    Algorithm: track whether we are inside a triple-double or triple-single
    quoted block, line by line.  Handles escaped quotes by checking the
    surrounding context character-by-character.
    """
    lines = content.splitlines()
    in_triple_dq = False  # inside  \"\"\"...\"\"\"\
    in_triple_sq = False  # inside  \'\'\'...\'\'\'
    doc_lines: set = set()

    for idx, line in enumerate(lines):
        # If already inside a triple-quoted block, mark this line immediately
        if in_triple_dq or in_triple_sq:
            doc_lines.add(idx)

        j = 0
        while j < len(line):
            # Skip escaped characters
            if line[j] == '\\' and j + 1 < len(line):
                j += 2
                continue

            triple_dq = line[j:j+3] == '"""'
            triple_sq = line[j:j+3] == "'''"

            if not in_triple_dq and not in_triple_sq:
                if triple_dq:
                    in_triple_dq = True
                    doc_lines.add(idx)
                    j += 3
                    continue
                elif triple_sq:
                    in_triple_sq = True
                    doc_lines.add(idx)
                    j += 3
                    continue
                # Skip single-quoted strings to avoid false triple-quote detection
                elif line[j] == '"':
                    j += 1
                    while j < len(line) and line[j] != '"':
                        if line[j] == '\\':
                            j += 1
                        j += 1
                elif line[j] == "'":
                    j += 1
                    while j < len(line) and line[j] != "'":
                        if line[j] == '\\':
                            j += 1
                        j += 1
            else:
                if in_triple_dq and triple_dq:
                    in_triple_dq = False
                    j += 3
                    continue
                elif in_triple_sq and triple_sq:
                    in_triple_sq = False
                    j += 3
                    continue
            j += 1

    return frozenset(doc_lines)


# ---------------------------------------------------------------------------
# AST-based arithmetic operator detection (Python only)
# ---------------------------------------------------------------------------

@dataclass
class AstHit:
    """Result of an AST-level arithmetic operator scan."""
    line_no: int      # 1-based line number in the chunk
    op_type: str      # 'div' or 'mod'
    divisor: str      # name of the right-hand variable
    is_pathlib: bool  # heuristic: likely a pathlib path-join operation


# Variable names that strongly suggest pathlib Path division (not arithmetic)
_PATHLIB_NAMES: frozenset = frozenset({
    'path', 'filepath', 'file_path', 'dir', 'directory', 'root', 'base',
    'cwd', 'fname', 'filename', 'folder', 'p', 'src', 'dst', 'dest',
    'target', 'source', 'prefix', 'suffix', 'parent', 'child',
    'base_dir', 'base_path', 'work_dir', 'repo_path', 'data_dir',
})

# Patterns on the same line that strongly suggest pathlib context
_PATHLIB_LINE_RE = re.compile(
    r'\b(?:Path|pathlib|os\.path|PurePath|PosixPath|WindowsPath)\b',
    re.IGNORECASE,
)

# Numeric-suggestive variable names where division would be arithmetic
_NUMERIC_NAMES_RE = re.compile(
    r'^(?:count|total|n|num|number|size|length|len|sum|avg|'
    r'denominator|divisor|factor|ratio|rate|pct|percent|'
    r'width|height|cols|rows|value|val|x|y|z|i|j|k)$',
    re.IGNORECASE,
)

# Divisors that are never runtime arithmetic risks (keywords, type vars, etc.)
_ARITHMETIC_SKIP_DIVISORS = frozenset({
    'await', 'async', 'lambda', 'True', 'False', 'None',
    'self', 'cls', 'type', 'object', 'Any', 'Union', 'Optional',
})

# PascalCase names are usually TypeVars / classes, not numeric divisors
_PASCAL_CASE_RE = re.compile(r'^[A-Z][a-zA-Z0-9]*$')


def _is_runtime_arithmetic_divisor(name: str, *, is_pathlib: bool) -> bool:
    """True when *name* plausibly represents a numeric runtime divisor."""
    if is_pathlib:
        return False
    if name in _ARITHMETIC_SKIP_DIVISORS:
        return False
    if _PASCAL_CASE_RE.match(name):
        return False
    if len(name) == 1 and name.upper() == name and name not in ('n', 'x', 'y', 'z'):
        return False
    if _NUMERIC_NAMES_RE.match(name):
        return True
    # Lowercase identifiers that are not pathlib-ish
    low = name.lower()
    if low in _PATHLIB_NAMES or 'path' in low or 'dir' in low or 'file' in low:
        return False
    return name.islower() and len(name) >= 2


def ast_find_arithmetic_hits(
    source: str,
    start_line: int = 1,
    *,
    parents: dict | None = None,
) -> list:
    """
    Parse *source* as Python and return AstHit objects for runtime BinOp(Div/Mod)
    where the right operand is a variable that plausibly represents a numeric divisor.

  Must run on complete file source — never on partial chunks.
    """
    import ast as _ast

    try:
        tree = _ast.parse(source)
    except SyntaxError:
        return []

    if parents is None:
        parents = {}
        for node in _ast.walk(tree):
            for child in _ast.iter_child_nodes(node):
                parents[child] = node

    hits = []
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.BinOp):
            continue
        if not isinstance(node.op, (_ast.Div, _ast.Mod)):
            continue
        if _in_arithmetic_annotation(node, parents):
            continue

        right = node.right
        if not isinstance(right, _ast.Name):
            continue

        op_type = 'div' if isinstance(node.op, _ast.Div) else 'mod'

        if op_type == 'mod' and isinstance(node.left, _ast.Constant):
            if isinstance(node.left.value, str):
                continue

        divisor_name = right.id
        is_path = (
            divisor_name.lower() in _PATHLIB_NAMES
            or 'path' in divisor_name.lower()
            or 'dir' in divisor_name.lower()
        )
        if not _is_runtime_arithmetic_divisor(divisor_name, is_pathlib=is_path):
            continue

        hits.append(AstHit(
            line_no=start_line + node.lineno - 1,
            op_type=op_type,
            divisor=divisor_name,
            is_pathlib=is_path,
        ))

    return hits


def _in_arithmetic_annotation(node: object, parents: dict) -> bool:
    """True if BinOp appears inside a type-annotation position."""
    import ast as _ast
    cur = node
    while cur is not None:
        parent = parents.get(cur)
        if parent is None:
            break
        if isinstance(parent, _ast.arg) and parent.annotation is cur:
            return True
        if isinstance(parent, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            if parent.returns is cur:
                return True
        if isinstance(parent, _ast.AnnAssign) and parent.annotation is cur:
            return True
        cur = parent
    return False


# ---------------------------------------------------------------------------
# HTTP client discriminator (for BUG-002)
# ---------------------------------------------------------------------------

# These names are HTTP clients; .get(url) on them returns a Response, not None
_HTTP_CLIENT_NAMES_RE = re.compile(
    r'\b(?:requests?|session|client|http|resp|response|adapter|'
    r'pool|conn|connection|urllib|httpx|aiohttp)\b',
    re.IGNORECASE,
)


def is_http_client_get(line: str) -> bool:
    """
    Return True if the `.get(...)` on this line is an HTTP client call
    (returns a Response object, not None) rather than a dict lookup.

    HTTP client .get() is safe to chain — it raises on error rather than
    returning None.  Dict .get() returns None when key is absent.

    Heuristic: check if a known HTTP client name precedes .get( on the line.
    """
    # Find the .get( position
    m = re.search(r'\.get\s*\(', line)
    if not m:
        return False
    # Check what precedes .get( on the same line
    prefix = line[:m.start()]
    return bool(_HTTP_CLIENT_NAMES_RE.search(prefix))



# ---------------------------------------------------------------------------
# Entropy + placeholder detection (for secret rules)
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(
    r'^(?:'
    # Generic weak/obvious words
    r'test(?:ing|123)?|password\d*|passwd|secret\d*|example\d*|changeme|'
    r'change.?me|replace.?me|'
    # "your-X" patterns
    r'your.?(?:secret|key|password|token|api.?key|secret.?key)|'
    # Common placeholder phrases (hyphenated or underscored)
    r'your[-_]?key[-_]here|your-secret-key-here|your-secret-key|secret-key-here|'
    r'replace.?me.?with.?key|insert.?(?:key|secret|token).?here|'
    r'put.?(?:key|secret|token).?here|'
    # Single-word weak values
    r'todo|fixme|foo|bar|baz|admin|letmein|welcome|default|'
    r'sample\d*|dummy|fake|xxxx+|aaaa+|1234+|abcd+|qwerty|'
    r'placeholder|enter.?(?:key|secret|value)|'
    # Angle-bracket template vars
    r'<[^>]+>|\.\.\.|s3cr3t|my.?(?:password|secret|key|token)|'
    # ALL_CAPS_WITH_UNDERSCORES that look like template slots
    r'[A-Z][A-Z_]{3,}(?:KEY|SECRET|TOKEN|PASSWORD|PASS)'
    r')$',
    re.IGNORECASE,
)

# Secondary heuristic: obvious placeholder signals not caught by regex
_PLACEHOLDER_WORDS = frozenset({
    'wrongpassword', 'badpassword', 'incorrectpassword',
    'mypassword', 'mypassword1', 'password1', 'pass123',
    'not-a-real-secret', 'not_a_real_secret', 'fake-secret',
    'supersecret', 'topsecret',  # too obvious
})


def shannon_entropy(s: str) -> float:
    """Shannon entropy in bits per character."""
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def is_placeholder_secret(value: str) -> bool:
    """
    Return True if *value* looks like a non-real placeholder rather than
    an actual secret.  Multi-layer detection:
      1. Regex pattern match (common placeholder phrases)
      2. Lowercase word set (wrongpassword, etc.)
      3. All-caps-with-separator token (REPLACE_ME, INSERT_KEY_HERE, …)
      4. Shannon entropy < 2.5 bits/char (low-complexity strings)
    """
    v = value.strip()
    if not v:
        return True

    # Layer 1: regex pattern
    if _PLACEHOLDER_RE.match(v):
        return True

    # Layer 2: known placeholder words (lowercase comparison)
    if v.lower() in _PLACEHOLDER_WORDS:
        return True

    # Layer 3: ALL_CAPS_SEPARATOR token — e.g. REPLACE_ME_WITH_KEY
    if re.match(r'^[A-Z][A-Z0-9_\-]{4,}$', v) and not re.search(r'[a-z]', v):
        return True

    # Layer 4: low entropy — repetitive or dictionary text
    if len(v) > 0 and shannon_entropy(v) < 2.5:
        return True

    return False


_SECRET_VALUE_RE = re.compile(r'["\']([^"\']*)["\']')


def extract_secret_value(match_text: str) -> str:
    """Pull the quoted string value out of a secret-assignment match."""
    m = _SECRET_VALUE_RE.search(match_text)
    return m.group(1) if m else ''


# ---------------------------------------------------------------------------
# Proximity-aware deduplication
# ---------------------------------------------------------------------------

F = TypeVar('F')


def proximity_deduplicate(
    findings: list[F],
    *,
    key_fn,          # finding → (file_path, category_or_pattern)
    line_fn,         # finding → int
    window: int = 5,
) -> list[F]:
    """
    Remove duplicate findings where the same (file, category) fires within
    *window* lines of a previously seen finding.
    """
    seen: dict[tuple, int] = {}   # key → last_line
    result: list[F] = []
    for f in findings:
        k = key_fn(f)
        ln = line_fn(f)
        last = seen.get(k)
        if last is None or abs(ln - last) > window:
            seen[k] = ln
            result.append(f)
    return result
