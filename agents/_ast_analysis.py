"""
agents/_ast_analysis.py
=======================
AST-based static analysis for high-precision rule detection.

Replaces regex for Python chunks where regex produces systematic false positives:
  BUG-001  — runtime None dereference (not type hints / annotations)
  BUG-002  — dict .get() chained attribute access (not HTTP clients)
  BUG-003  — unguarded Optional parameter use
  BUG-020  — while True without exit
  SEC-030  — path traversal with untrusted input flow
  SEC-040  — SSRF with untrusted URL flow
  SEC-070  — production-exposed debug mode
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Iterator

from agents._scan_utils import ast_find_arithmetic_hits, classify_file_role

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AstRuleHit:
    """A single AST-level rule match."""
    rule_id:    str
    line_no:    int      # 1-based line in file
    match_text: str
    evidence:   str
    reasoning:  str
    confidence: float    # 0.0–1.0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# TypeAlias added in Python 3.12
_TYPE_ALIAS_NODE = getattr(ast, "TypeAlias", type("_TypeAliasStub", (), {}))

# Names / dotted paths treated as untrusted input sources
_TAINT_NAME_RE = (
    "request", "req", "g", "form", "args", "files", "values", "cookies",
    "user_input", "user_data", "user_path", "user_id", "query_string",
    "params", "payload", "body", "upload", "filename",
)

_TAINT_ROOTS = frozenset({
    "request", "flask", "werkzeug", "input", "sys", "os", "urllib",
})

_HTTP_CLIENT_ATTRS = frozenset({
    "get", "post", "put", "patch", "delete", "head", "options", "request",
})

# Receivers whose .get() is an HTTP verb, not dict lookup
_HTTP_GET_RECEIVER_ROOTS = frozenset({
    "requests", "httpx", "aiohttp", "urllib", "urllib3", "http", "https",
    "session", "client", "adapter", "pool", "conn", "connection",
    "response", "resp",
})


def _dotted_name(node: ast.AST) -> str:
    """Best-effort dotted name for Attribute/Name nodes."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _is_none_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _in_annotation(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """True if *node* appears inside a type-annotation position."""
    cur: ast.AST | None = node
    while cur is not None:
        parent = parents.get(cur)
        if parent is None:
            break
        if isinstance(parent, ast.arg) and parent.annotation is cur:
            return True
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if parent.returns is cur:
                return True
        if isinstance(parent, ast.AnnAssign) and parent.annotation is cur:
            return True
        if isinstance(parent, _TYPE_ALIAS_NODE) and getattr(parent, "value", None) is cur:
            return True
        if isinstance(parent, ast.Subscript):
            # Subscripts in annotations: Optional[X], list[T | None]
            anc = parent
            while anc is not None:
                p = parents.get(anc)
                if p is None:
                    break
                if isinstance(p, ast.arg) and p.annotation is anc:
                    return True
                if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)) and p.returns is anc:
                    return True
                if isinstance(p, ast.AnnAssign) and p.annotation is anc:
                    return True
                anc = p
        cur = parent
    return False


def _build_parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _is_always_true(test: ast.AST) -> bool:
    if isinstance(test, ast.Constant) and test.value is True:
        return True
    if isinstance(test, ast.Constant) and test.value == 1:
        return True
    return False


def _body_has_exit(stmts: list[ast.stmt]) -> bool:
    """Return True if *stmts* contains break/return/raise (including nested blocks)."""
    for stmt in stmts:
        if isinstance(stmt, (ast.Break, ast.Return, ast.Raise)):
            return True
        for child in ast.walk(stmt):
            if isinstance(child, (ast.Break, ast.Return, ast.Raise)):
                return True
    return False


def _is_tainted_expr(node: ast.AST, *, local_tainted: frozenset[str] = frozenset()) -> bool:
    """
    Lightweight taint check: does *node* reference an untrusted source?

    High-confidence sources:
      request.*, flask.request.*, input(), sys.argv[...]
    Medium-confidence:
      parameters / variables whose names suggest user data when combined with
      subscript of request-like roots.
    """
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            fn = _dotted_name(sub.func)
            if fn in ("input", "raw_input"):
                return True
            if fn.endswith(".get") and any(
                _dotted_name(a).split(".")[0] in _TAINT_ROOTS for a in ast.walk(sub.func)
            ):
                pass  # handled by attribute walk below

        if isinstance(sub, ast.Name):
            low = sub.id.lower()
            if sub.id in local_tainted:
                return True
            if low in ("request", "req", "user_input", "user_data", "form", "args", "files"):
                return True
            # User-supplied URL/path naming conventions
            if low.startswith("user_") or low.endswith("_url") or low.endswith("_uri"):
                return True
            if "supplied" in low and any(t in low for t in ("url", "uri", "path", "host")):
                return True

        if isinstance(sub, ast.Attribute):
            path = _dotted_name(sub).lower()
            if path.startswith("request.") or path.startswith("flask.request."):
                return True
            if any(path.startswith(f"{r}.") for r in ("werkzeug",)):
                if any(p in path for p in ("args", "form", "files", "values", "data", "json")):
                    return True

        if isinstance(sub, ast.Subscript):
            base = _dotted_name(sub.value).lower()
            if base in ("sys.argv", "os.environ") or base.startswith("request."):
                return True

    return False


def _inside_main_guard(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """True if *node* is nested inside `if __name__ == '__main__'`."""
    cur: ast.AST | None = node
    while cur is not None:
        parent = parents.get(cur)
        if isinstance(parent, ast.If) and _is_main_guard_test(parent.test):
            return True
        cur = parent
    return False


def _is_main_guard_test(test: ast.AST) -> bool:
    if not isinstance(test, ast.Compare):
        return False
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    left, right = test.left, test.comparators[0]
    if isinstance(left, ast.Name) and left.id == "__name__":
        return isinstance(right, ast.Constant) and right.value == "__main__"
    if isinstance(right, ast.Name) and right.id == "__name__":
        return isinstance(left, ast.Constant) and left.value == "__main__"
    return False


def _is_optional_annotation(ann: ast.AST | None) -> bool:
    if ann is None:
        return False
    text = ast.unparse(ann) if hasattr(ast, "unparse") else ""
    low = text.lower()
    return (
        "optional" in low
        or "none" in low
        or "| none" in low
        or "union[" in low and "none" in low
    )


def _param_is_optional(arg: ast.arg, defaults_offset: int, defaults: list[ast.expr]) -> bool:
    """True if parameter is typed Optional or defaults to None."""
    if _is_optional_annotation(arg.annotation):
        return True
    # Match default value to parameter (defaults align to tail of posonlyargs+args)
    return False


def _get_defaults_map(
    args: ast.arguments,
) -> dict[str, ast.expr | None]:
    """Map positional parameter name → default value (None if required)."""
    result: dict[str, ast.expr | None] = {}
    pos_args: list[ast.arg] = []
    pos_args.extend(args.posonlyargs)
    pos_args.extend(args.args)
    defaults = list(args.defaults)
    offset = len(pos_args) - len(defaults)
    for i, arg in enumerate(pos_args):
        if i >= offset:
            result[arg.arg] = defaults[i - offset]
        else:
            result[arg.arg] = None
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        result[arg.arg] = default
    return result


def _is_param_is_none_compare(test: ast.AST, param: str) -> bool:
    if not isinstance(test, ast.Compare):
        return False
    for op, comp in zip(test.ops, test.comparators):
        if isinstance(op, ast.Is):
            if isinstance(test.left, ast.Name) and test.left.id == param and _is_none_constant(comp):
                return True
            if isinstance(comp, ast.Name) and comp.id == param and _is_none_constant(test.left):
                return True
    return False


def _expr_tests_param_truthiness(test: ast.AST, param: str) -> bool:
    """True when *test* checks *param* is truthy (if param:, param and …)."""
    if isinstance(test, ast.Name) and test.id == param:
        return True
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        if isinstance(test.operand, ast.Name) and test.operand.id == param:
            return True
    if isinstance(test, ast.Compare):
        for op, comp in zip(test.ops, test.comparators):
            if isinstance(op, ast.IsNot):
                if isinstance(test.left, ast.Name) and test.left.id == param and _is_none_constant(comp):
                    return True
                if isinstance(comp, ast.Name) and comp.id == param and _is_none_constant(test.left):
                    return True
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And) and test.values:
        return _expr_tests_param_truthiness(test.values[0], param)
    return False


def _function_guards_param(func: ast.FunctionDef | ast.AsyncFunctionDef, param: str) -> bool:
    """True when the function prologue guards *param* before use."""
    for stmt in func.body[:8]:
        if isinstance(stmt, ast.If):
            if _is_param_is_none_compare(stmt.test, param) and _body_has_exit(stmt.body):
                return True
            if (
                isinstance(stmt.test, ast.UnaryOp)
                and isinstance(stmt.test.op, ast.Not)
                and isinstance(stmt.test.operand, ast.Name)
                and stmt.test.operand.id == param
                and _body_has_exit(stmt.body)
            ):
                return True
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == param:
                    if isinstance(stmt.value, ast.BoolOp) and isinstance(stmt.value.op, ast.Or):
                        if isinstance(stmt.value.values[0], ast.Name) and stmt.value.values[0].id == param:
                            return True
    return False


def _node_is_guarded_optional_use(
    node: ast.AST,
    param: str,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    """True when *node* is guarded by if/and/conditional patterns on *param*."""
    cur: ast.AST | None = node
    while cur is not None:
        parent = parents.get(cur)
        if parent is None:
            break

        if isinstance(parent, ast.IfExp):
            if cur is parent.body and _expr_tests_param_truthiness(parent.test, param):
                return True

        if isinstance(parent, ast.BoolOp) and isinstance(parent.op, ast.And):
            if cur in parent.values[1:] and _expr_tests_param_truthiness(parent.values[0], param):
                return True

        if isinstance(parent, ast.If):
            if cur in parent.body and _expr_tests_param_truthiness(parent.test, param):
                return True
            if (
                cur in parent.orelse
                and isinstance(parent.test, ast.UnaryOp)
                and isinstance(parent.test.op, ast.Not)
                and isinstance(parent.test.operand, ast.Name)
                and parent.test.operand.id == param
            ):
                return True

        cur = parent
    return False


def _is_safe_none_idiom(node: ast.AST) -> bool:
    """Whitelist None.__class__ and type(None) — not runtime dereference bugs."""
    if isinstance(node, ast.Attribute) and _is_none_constant(node.value):
        return node.attr == "__class__"
    if isinstance(node, ast.Call):
        fn = _dotted_name(node.func)
        if fn == "type" and node.args and _is_none_constant(node.args[0]):
            return True
    return False


def _line_source(source: str, lineno: int) -> str:
    lines = source.splitlines()
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1].strip()
    return ""


def _get_call_has_safe_default(call: ast.Call) -> bool:
    """True when dict.get() supplies a non-None default (absent-key safe)."""
    if len(call.args) >= 2 and not _is_none_constant(call.args[1]):
        return True
    for kw in call.keywords:
        if kw.arg == "default" and not _is_none_constant(kw.value):
            return True
    return False


def _get_call_first_arg_is_route(call: ast.Call) -> bool:
    """HTTP clients commonly use .get('/path') — dict keys rarely look like routes."""
    if not call.args:
        return False
    first = call.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value.startswith("/")
    return False


def _is_http_get_call(call: ast.Call) -> bool:
    """
    True when *call* is an HTTP client .get(), not dict.get().

    Covers requests/session/client test clients and chained forms like
    app.test_client().get(...).
    """
    if not isinstance(call.func, ast.Attribute):
        return False
    if call.func.attr != "get":
        return False

    if _get_call_first_arg_is_route(call):
        return True

    receiver = _dotted_name(call.func.value).lower()
    if not receiver:
        return False

    if "test_client" in receiver:
        return True

    root = receiver.split(".")[0]
    if root in _HTTP_GET_RECEIVER_ROOTS:
        return True

    if receiver == "client" or receiver.endswith(".client"):
        return True
    if receiver.endswith("session") or ".session" in receiver:
        return True

    # Flask app / blueprint test helpers
    if receiver in ("app", "bp", "blueprint") or receiver.endswith(".app"):
        return True

    return False


def _analyze_bug002(
    tree: ast.AST,
    source: str,
    start_line: int,
) -> list[AstRuleHit]:
    """
    Dict .get(key).attr chains where .get() may return None.

    Excludes HTTP client .get() (requests, session, test client, etc.).
    """
    hits: list[AstRuleHit] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if not isinstance(node.value, ast.Call):
            continue

        call = node.value
        if not isinstance(call.func, ast.Attribute):
            continue
        if call.func.attr != "get":
            continue
        if _is_http_get_call(call):
            continue
        if _get_call_has_safe_default(call):
            continue

        receiver = _dotted_name(call.func.value)
        ln = start_line + node.lineno - 1
        src = _line_source(source, node.lineno)
        hits.append(AstRuleHit(
            rule_id="BUG-002",
            line_no=ln,
            match_text=src[:120] or f"{receiver}.get(...).{node.attr}",
            evidence=(
                f"Chained attribute `.{node.attr}` on result of `{receiver}.get()` "
                f"without a non-None default at line {node.lineno}."
            ),
            reasoning=(
                "AST confirms dict-like .get() (not HTTP client) whose result is "
                "immediately dereferenced; absent keys return None."
            ),
            confidence=0.86,
        ))

    return hits


# ---------------------------------------------------------------------------
# Per-rule analyzers
# ---------------------------------------------------------------------------


def _analyze_bug001(
    tree: ast.AST,
    source: str,
    start_line: int,
    parents: dict[ast.AST, ast.AST],
) -> list[AstRuleHit]:
    """Runtime None attribute/subscript access only."""
    hits: list[AstRuleHit] = []

    for node in ast.walk(tree):
        if _is_safe_none_idiom(node):
            continue
        if isinstance(node, ast.Attribute) and _is_none_constant(node.value):
            if _in_annotation(node, parents):
                continue
            ln = start_line + node.lineno - 1
            src = _line_source(source, node.lineno)
            hits.append(AstRuleHit(
                rule_id="BUG-001",
                line_no=ln,
                match_text=src[:120] or f"None.{node.attr}",
                evidence=f"Attribute access `.{node.attr}` on literal `None` at line {node.lineno}",
                reasoning=(
                    "AST confirms the receiver of the attribute access is the runtime "
                    "constant `None`, not a type annotation or comparison expression."
                ),
                confidence=0.92,
            ))
        elif isinstance(node, ast.Subscript) and _is_none_constant(node.value):
            if _in_annotation(node, parents):
                continue
            ln = start_line + node.lineno - 1
            src = _line_source(source, node.lineno)
            hits.append(AstRuleHit(
                rule_id="BUG-001",
                line_no=ln,
                match_text=src[:120] or "None[...]",
                evidence=f"Subscript on literal `None` at line {node.lineno}",
                reasoning="AST confirms subscript operation on runtime `None` constant.",
                confidence=0.90,
            ))

    return hits


def _analyze_bug003(
    tree: ast.AST,
    source: str,
    start_line: int,
    parents: dict[ast.AST, ast.AST],
) -> list[AstRuleHit]:
    """Functions with Optional/default-None params used without guard."""
    hits: list[AstRuleHit] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        defaults_map = _get_defaults_map(node.args)
        optional_params: list[str] = []
        for arg in list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs):
            name = arg.arg
            default = defaults_map.get(name)
            if _is_optional_annotation(arg.annotation) or _is_none_constant(default):
                optional_params.append(name)

        if not optional_params:
            continue

        for param in optional_params:
            if _function_guards_param(node, param):
                continue
            for child in ast.walk(node):
                if child is node:
                    continue
                if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
                    if child.value.id != param:
                        continue
                    if _node_is_guarded_optional_use(child, param, parents):
                        continue
                    ln = start_line + child.lineno - 1
                    src = _line_source(source, child.lineno)
                    hits.append(AstRuleHit(
                        rule_id="BUG-003",
                        line_no=ln,
                        match_text=src[:120] or f"{param}.{child.attr}",
                        evidence=(
                            f"Parameter `{param}` is Optional/default-None but "
                            f"`.{child.attr}` is accessed without a recognized guard."
                        ),
                        reasoning=(
                            "AST shows optional parameter dereferenced outside "
                            "if-param, param-and-attr, or param-if-else guard patterns."
                        ),
                        confidence=0.78,
                    ))
                    break

    return hits


def _analyze_bug010_011(
    tree: ast.AST,
    source: str,
    start_line: int,
    parents: dict[ast.AST, ast.AST],
) -> list[AstRuleHit]:
    """Runtime division/modulo by variable without zero-check (AST-only)."""
    hits: list[AstRuleHit] = []
    for ah in ast_find_arithmetic_hits(source, start_line, parents=parents):
        rule_id = "BUG-010" if ah.op_type == "div" else "BUG-011"
        src = _line_source(source, ah.line_no - start_line + 1)
        hits.append(AstRuleHit(
            rule_id=rule_id,
            line_no=ah.line_no,
            match_text=src[:120] or f"/ {ah.divisor}",
            evidence=f"Division/modulo by variable `{ah.divisor}` (AST BinOp).",
            reasoning="AST confirms runtime arithmetic with a variable divisor.",
            confidence=0.85,
        ))
    return hits


def _analyze_bug020(
    tree: ast.AST,
    source: str,
    start_line: int,
) -> list[AstRuleHit]:
    """while True / while 1 without break/return/raise in body."""
    hits: list[AstRuleHit] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.While):
            continue
        is_true = _is_always_true(node.test)
        is_one = isinstance(node.test, ast.Constant) and node.test.value == 1
        if not (is_true or is_one):
            continue
        if _body_has_exit(node.body):
            continue

        rule_id = "BUG-020" if is_true else "BUG-021"
        ln = start_line + node.lineno - 1
        src = _line_source(source, node.lineno)
        hits.append(AstRuleHit(
            rule_id=rule_id,
            line_no=ln,
            match_text=src[:120] or ("while True:" if is_true else "while 1:"),
            evidence=f"Infinite loop at line {node.lineno} with no break/return/raise in body.",
            reasoning=(
                "AST walk of the loop body found no `break`, `return`, or `raise` "
                "on any path — the loop may never terminate."
            ),
            confidence=0.82,
        ))

    return hits


def _analyze_sec030(
    tree: ast.AST,
    source: str,
    start_line: int,
    file_path: str,
) -> list[AstRuleHit]:
    """File operations only when path argument has untrusted input flow."""
    if classify_file_role(file_path) in ("test", "docs", "example"):
        return []

    hits: list[AstRuleHit] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        fn = _dotted_name(node.func)
        sink_kind: str | None = None
        if fn in ("open", "io.open"):
            sink_kind = "open"
        elif fn in ("Path", "pathlib.Path"):
            sink_kind = "Path"
        elif fn.endswith("path.join") or fn == "os.path.join":
            sink_kind = "os.path.join"
        else:
            continue

        tainted_args: list[str] = []
        for arg in node.args:
            if _is_tainted_expr(arg):
                tainted_args.append(ast.unparse(arg) if hasattr(ast, "unparse") else "...")
        for kw in node.keywords:
            if kw.arg in ("file", "path", "filename", "filepath", "name", "dest", "src"):
                if _is_tainted_expr(kw.value):
                    tainted_args.append(kw.arg or "...")

        if not tainted_args:
            continue

        ln = start_line + node.lineno - 1
        src = _line_source(source, node.lineno)
        hits.append(AstRuleHit(
            rule_id="SEC-030",
            line_no=ln,
            match_text=src[:120] or fn,
            evidence=f"{sink_kind}() called with tainted path component(s): {', '.join(tainted_args[:3])}",
            reasoning=(
                "AST taint analysis links the file-path argument to an untrusted "
                "source (request.*, input(), sys.argv, etc.)."
            ),
            confidence=0.80,
        ))

    return hits


def _analyze_sec040(
    tree: ast.AST,
    source: str,
    start_line: int,
    file_path: str,
) -> list[AstRuleHit]:
    """HTTP client calls only when URL flows from untrusted source."""
    if classify_file_role(file_path) in ("test", "docs", "example"):
        return []

    hits: list[AstRuleHit] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        fn = _dotted_name(node.func)
        parts = fn.split(".")
        if not parts:
            continue
        method = parts[-1]
        if method not in _HTTP_CLIENT_ATTRS:
            continue

        # Must be a known HTTP client module/object
        root = parts[0].lower()
        if root not in ("requests", "httpx", "urllib", "aiohttp", "session", "client") and "request" not in fn.lower():
            # Also allow self.get patterns on sessions
            if method in _HTTP_CLIENT_ATTRS and len(parts) >= 2:
                pass
            else:
                continue

        url_node: ast.AST | None = None
        if node.args:
            url_node = node.args[0]
        for kw in node.keywords:
            if kw.arg in ("url", "uri", "endpoint", "href"):
                url_node = kw.value
                break

        if url_node is None:
            continue
        if not _is_tainted_expr(url_node):
            continue

        url_text = ast.unparse(url_node) if hasattr(ast, "unparse") else _dotted_name(url_node)
        ln = start_line + node.lineno - 1
        src = _line_source(source, node.lineno)
        hits.append(AstRuleHit(
            rule_id="SEC-040",
            line_no=ln,
            match_text=src[:120] or fn,
            evidence=f"HTTP {method}() with tainted URL expression: {url_text[:80]}",
            reasoning=(
                "AST taint analysis shows the request URL derives from an untrusted "
                "input source, enabling potential SSRF."
            ),
            confidence=0.82,
        ))

    return hits


def _analyze_sec070(
    tree: ast.AST,
    source: str,
    start_line: int,
    file_path: str,
    parents: dict[ast.AST, ast.AST],
) -> list[AstRuleHit]:
    """Production debug mode — not tests, __main__, or docstrings."""
    role = classify_file_role(file_path)
    if role in ("test", "docs", "example"):
        return []

    base = file_path.replace("\\", "/").rsplit("/", 1)[-1]
    if base in ("__main__.py", "conftest.py"):
        return []

    hits: list[AstRuleHit] = []

    for node in ast.walk(tree):
        if _inside_main_guard(node, parents):
            continue

        # DEBUG = True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "DEBUG":
                    if isinstance(node.value, ast.Constant) and node.value.value is True:
                        ln = start_line + node.lineno - 1
                        hits.append(AstRuleHit(
                            rule_id="SEC-070",
                            line_no=ln,
                            match_text=_line_source(source, node.lineno)[:120],
                            evidence="Module-level assignment `DEBUG = True`.",
                            reasoning="AST confirms a literal True assignment to DEBUG at module scope.",
                            confidence=0.88,
                        ))

        # app.debug = True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == "debug":
                    if isinstance(node.value, ast.Constant) and node.value.value is True:
                        ln = start_line + node.lineno - 1
                        hits.append(AstRuleHit(
                            rule_id="SEC-070",
                            line_no=ln,
                            match_text=_line_source(source, node.lineno)[:120],
                            evidence="Assignment `app.debug = True` in production code.",
                            reasoning="AST confirms debug flag enabled via attribute assignment.",
                            confidence=0.85,
                        ))

        # app.run(..., debug=True)
        if isinstance(node, ast.Call):
            fn = _dotted_name(node.func)
            if fn.endswith(".run") or fn == "run":
                for kw in node.keywords:
                    if kw.arg == "debug" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        ln = start_line + node.lineno - 1
                        hits.append(AstRuleHit(
                            rule_id="SEC-070",
                            line_no=ln,
                            match_text=_line_source(source, node.lineno)[:120],
                            evidence="`run(debug=True)` in production code path.",
                            reasoning="AST confirms debug=True passed to a run() call outside __main__ guard.",
                            confidence=0.83,
                        ))

        # os.environ['FLASK_DEBUG'] = '1'
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript):
                    key = target.slice
                    if isinstance(key, ast.Constant) and str(key.value).upper() == "FLASK_DEBUG":
                        ln = start_line + node.lineno - 1
                        hits.append(AstRuleHit(
                            rule_id="SEC-070",
                            line_no=ln,
                            match_text=_line_source(source, node.lineno)[:120],
                            evidence="FLASK_DEBUG environment variable set in source.",
                            reasoning="AST confirms FLASK_DEBUG assignment in production code.",
                            confidence=0.86,
                        ))

    return hits


def _sql_string_has_taint(node: ast.AST) -> bool:
    """True when an SQL-looking string expression includes tainted data."""
    if _is_tainted_expr(node):
        return True
    if isinstance(node, ast.JoinedStr):
        for val in node.values:
            if isinstance(val, ast.FormattedValue) and _is_tainted_expr(val.value):
                return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _sql_string_has_taint(node.left) or _sql_string_has_taint(node.right)
    if isinstance(node, ast.Call):
        fn = _dotted_name(node.func)
        if fn.endswith(".format") or fn == "format":
            for arg in node.args:
                if _is_tainted_expr(arg):
                    return True
            for kw in node.keywords:
                if _is_tainted_expr(kw.value):
                    return True
    return False


def _is_parameterized_execute(call: ast.Call) -> bool:
    """True when execute() passes bind parameters as a second argument."""
    if len(call.args) >= 2:
        second = call.args[1]
        if isinstance(second, (ast.Tuple, ast.List, ast.Dict)):
            return True
    for kw in call.keywords:
        if kw.arg in ("params", "parameters", "vars", "args") and kw.value is not None:
            return True
    return False


def _analyze_sec010(
    tree: ast.AST,
    source: str,
    start_line: int,
    file_path: str,
) -> list[AstRuleHit]:
    """SQL injection only when user input flows into SQL string passed to execute()."""
    if classify_file_role(file_path) in ("test", "docs", "example"):
        return []

    hits: list[AstRuleHit] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = _dotted_name(node.func)
        if not (fn.endswith(".execute") or fn == "execute"):
            continue
        if _is_parameterized_execute(node):
            continue
        if not node.args:
            continue
        sql_arg = node.args[0]
        if not _sql_string_has_taint(sql_arg):
            continue
        ln = start_line + node.lineno - 1
        src = _line_source(source, node.lineno)
        hits.append(AstRuleHit(
            rule_id="SEC-010",
            line_no=ln,
            match_text=src[:120] or fn,
            evidence="execute() called with SQL string built from tainted input.",
            reasoning=(
                "AST taint analysis links SQL string construction to an untrusted "
                "source without parameterized bind arguments."
            ),
            confidence=0.84,
        ))
    return hits


def _analyze_sec050(
    tree: ast.AST,
    source: str,
    start_line: int,
    file_path: str,
) -> list[AstRuleHit]:
    """Unsafe deserialization only when untrusted data reaches the sink."""
    if classify_file_role(file_path) in ("test", "docs", "example"):
        return []

    hits: list[AstRuleHit] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = _dotted_name(node.func)
        is_pickle = fn in ("pickle.load", "pickle.loads")
        is_yaml = fn == "yaml.load"
        if not is_pickle and not is_yaml:
            continue
        if is_yaml:
            uses_safe_loader = False
            for kw in node.keywords:
                if kw.arg == "Loader" and isinstance(kw.value, ast.Attribute):
                    if kw.value.attr == "SafeLoader":
                        uses_safe_loader = True
                        break
            if uses_safe_loader:
                continue

        data_arg = node.args[0] if node.args else None
        if data_arg is None:
            for kw in node.keywords:
                if kw.arg in ("data", "buf", "buffer", "s", "bytes"):
                    data_arg = kw.value
                    break
        if data_arg is None or not _is_tainted_expr(data_arg):
            continue

        ln = start_line + node.lineno - 1
        src = _line_source(source, node.lineno)
        hits.append(AstRuleHit(
            rule_id="SEC-050",
            line_no=ln,
            match_text=src[:120] or fn,
            evidence=f"{fn}() called with tainted/untrusted data argument.",
            reasoning=(
                "AST taint analysis shows deserialization sink receives data "
                "from an untrusted input source."
            ),
            confidence=0.82,
        ))
    return hits


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_AST_RULE_IDS = frozenset({
    "BUG-001", "BUG-002", "BUG-003", "BUG-010", "BUG-011", "BUG-020", "BUG-021",
    "SEC-010", "SEC-030", "SEC-040", "SEC-050", "SEC-070",
})


def ast_analyze_python(
    source: str,
    start_line: int = 1,
    *,
    file_path: str = "",
    rule_ids: frozenset[str] | None = None,
) -> list[AstRuleHit]:
    """
    Run AST analyzers on Python *source*.

    Returns empty list on SyntaxError (caller may fall back to regex for
    non-Python or unparseable chunks).
    """
    active = rule_ids or _AST_RULE_IDS
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    parents = _build_parent_map(tree)
    hits: list[AstRuleHit] = []

    if "BUG-001" in active:
        hits.extend(_analyze_bug001(tree, source, start_line, parents))
    if "BUG-002" in active:
        hits.extend(_analyze_bug002(tree, source, start_line))
    if "BUG-003" in active:
        hits.extend(_analyze_bug003(tree, source, start_line, parents))
    if "BUG-010" in active or "BUG-011" in active:
        hits.extend(_analyze_bug010_011(tree, source, start_line, parents))
    if "BUG-020" in active or "BUG-021" in active:
        hits.extend(_analyze_bug020(tree, source, start_line))
    if "SEC-010" in active:
        hits.extend(_analyze_sec010(tree, source, start_line, file_path))
    if "SEC-030" in active:
        hits.extend(_analyze_sec030(tree, source, start_line, file_path))
    if "SEC-040" in active:
        hits.extend(_analyze_sec040(tree, source, start_line, file_path))
    if "SEC-050" in active:
        hits.extend(_analyze_sec050(tree, source, start_line, file_path))
    if "SEC-070" in active:
        hits.extend(_analyze_sec070(tree, source, start_line, file_path, parents))

    return hits


def confidence_to_level(confidence: float) -> str:
    """Map numeric confidence to High / Medium / Low label."""
    if confidence >= 0.80:
        return "High"
    if confidence >= 0.55:
        return "Medium"
    return "Low"
