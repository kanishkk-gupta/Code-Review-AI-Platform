"""Unit tests for AST-based rule analysis."""
from __future__ import annotations

from agents._ast_analysis import ast_analyze_python


def _ids(hits) -> set[str]:
    return {h.rule_id for h in hits}


def _lines(hits, rule_id: str) -> list[int]:
    return [h.line_no for h in hits if h.rule_id == rule_id]


class TestBug001Ast:
    def test_flags_runtime_none_deref(self):
        src = "def f():\n    return None.items()\n"
        hits = ast_analyze_python(src)
        assert "BUG-001" in _ids(hits)

    def test_ignores_type_annotation_none_union(self):
        src = "def get(self, name: str, default: t.Any | None = None) -> t.Any:\n    pass\n"
        hits = ast_analyze_python(src)
        assert "BUG-001" not in _ids(hits)

    def test_ignores_optional_return_self(self):
        src = (
            "@t.overload\n"
            "def __get__(self, obj: None, owner: None) -> te.Self: ...\n"
        )
        hits = ast_analyze_python(src)
        assert "BUG-001" not in _ids(hits)

    def test_ignores_is_not_none_and_chain(self):
        src = (
            "def f(value):\n"
            "    if value is not None and value.foo():\n"
            "        return value\n"
        )
        hits = ast_analyze_python(src)
        assert "BUG-001" not in _ids(hits)

    def test_ignores_setdefault_none_key(self):
        src = "d.setdefault(None, []).append(1)\n"
        hits = ast_analyze_python(src)
        assert "BUG-001" not in _ids(hits)

    def test_ignores_none_class_idiom(self):
        src = "NoneType = None.__class__\n"
        hits = ast_analyze_python(src)
        assert "BUG-001" not in _ids(hits)

    def test_ignores_type_none_call(self):
        src = "t = type(None)\n"
        hits = ast_analyze_python(src)
        assert "BUG-001" not in _ids(hits)


class TestBug002Ast:
    def test_flags_dict_get_chain(self):
        src = "def f(d):\n    return d.get('key').upper()\n"
        hits = ast_analyze_python(src)
        assert "BUG-002" in _ids(hits)

    def test_ignores_client_get_chain(self):
        src = 'def f(client):\n    return client.get("/route").data\n'
        hits = ast_analyze_python(src)
        assert "BUG-002" not in _ids(hits)

    def test_ignores_requests_get_chain(self):
        src = "def f():\n    return requests.get(url).json()\n"
        hits = ast_analyze_python(src)
        assert "BUG-002" not in _ids(hits)

    def test_ignores_session_get_chain(self):
        src = "def f(session):\n    return session.get(endpoint).text\n"
        hits = ast_analyze_python(src)
        assert "BUG-002" not in _ids(hits)

    def test_ignores_get_with_default(self):
        src = "def f(d):\n    return d.get('key', '').strip()\n"
        hits = ast_analyze_python(src)
        assert "BUG-002" not in _ids(hits)

    def test_ignores_short_client_var_route_get(self):
        src = (
            "def f(app):\n"
            "    c = app.test_client()\n"
            '    return c.get("/").data\n'
        )
        hits = ast_analyze_python(src)
        assert "BUG-002" not in _ids(hits)


class TestBug003Ast:
    def test_flags_unguarded_optional_use(self):
        src = (
            "def process(data: Optional[str] = None):\n"
            "    return data.strip()\n"
        )
        hits = ast_analyze_python(src)
        assert "BUG-003" in _ids(hits)

    def test_ignores_guarded_optional(self):
        src = (
            "def process(data: Optional[str] = None):\n"
            "    if data is None:\n"
            "        return ''\n"
            "    return data.strip()\n"
        )
        hits = ast_analyze_python(src)
        assert "BUG-003" not in _ids(hits)

    def test_ignores_conditional_expression_guard(self):
        src = (
            "def f(p: Optional[Path] = None):\n"
            "    return {'file_size': p.size if p else None}\n"
        )
        hits = ast_analyze_python(src)
        assert "BUG-003" not in _ids(hits)

    def test_ignores_and_guard(self):
        src = (
            "def f(p: Optional[str] = None):\n"
            "    return p and p.strip()\n"
        )
        hits = ast_analyze_python(src)
        assert "BUG-003" not in _ids(hits)

    def test_ignores_if_not_guard(self):
        src = (
            "def f(p: Optional[str] = None):\n"
            "    if not p:\n"
            "        return ''\n"
            "    return p.strip()\n"
        )
        hits = ast_analyze_python(src)
        assert "BUG-003" not in _ids(hits)


class TestBug010Ast:
    def test_flags_numeric_division(self):
        src = "def f(total, count):\n    return total / count\n"
        hits = ast_analyze_python(src)
        assert "BUG-010" in _ids(hits)

    def test_ignores_typevar_divisor(self):
        src = "def f():\n    return Base / Custom\n"
        hits = ast_analyze_python(src)
        assert "BUG-010" not in _ids(hits)

    def test_ignores_await_divisor(self):
        src = "async def f():\n    return x / await coro()\n"
        hits = ast_analyze_python(src)
        assert "BUG-010" not in _ids(hits)


class TestBug020Ast:
    def test_flags_while_true_without_break(self):
        src = "while True:\n    x = 1\n"
        hits = ast_analyze_python(src)
        assert "BUG-020" in _ids(hits)

    def test_ignores_while_true_with_break(self):
        src = "while True:\n    if done:\n        break\n"
        hits = ast_analyze_python(src)
        assert "BUG-020" not in _ids(hits)


class TestSec030Ast:
    def test_flags_request_path_join(self):
        src = 'open(os.path.join("/uploads", request.args["filename"]))\n'
        hits = ast_analyze_python(src, file_path="src/app.py")
        assert "SEC-030" in _ids(hits)

    def test_ignores_safe_config_path(self):
        src = "filename = os.path.join(self.root_path, filename)\n"
        hits = ast_analyze_python(src, file_path="src/config.py")
        assert "SEC-030" not in _ids(hits)


class TestSec040Ast:
    def test_flags_user_supplied_url(self):
        src = "requests.get(url=user_supplied_url)\n"
        hits = ast_analyze_python(src, file_path="src/client.py")
        assert "SEC-040" in _ids(hits)

    def test_ignores_test_file_url_variable(self):
        src = "requests.get(url=url)\n"
        hits = ast_analyze_python(src, file_path="tests/test_api.py")
        assert "SEC-040" not in _ids(hits)


class TestSec010Ast:
    def test_flags_tainted_sql_execute(self):
        src = 'cursor.execute(f"SELECT * FROM users WHERE id={request.args[\'id\']}")\n'
        hits = ast_analyze_python(src, file_path="src/db.py")
        assert "SEC-010" in _ids(hits)

    def test_ignores_parameterized_execute(self):
        src = 'cursor.execute("SELECT * FROM users WHERE id=%s", (user_id,))\n'
        hits = ast_analyze_python(src, file_path="src/db.py")
        assert "SEC-010" not in _ids(hits)


class TestSec050Ast:
    def test_flags_tainted_pickle(self):
        src = "pickle.loads(request.data)\n"
        hits = ast_analyze_python(src, file_path="src/app.py")
        assert "SEC-050" in _ids(hits)

    def test_ignores_static_pickle(self):
        src = "pickle.loads(cached_blob)\n"
        hits = ast_analyze_python(src, file_path="src/cache.py")
        assert "SEC-050" not in _ids(hits)


class TestSec070Ast:
    def test_flags_debug_assignment(self):
        src = "DEBUG = True\n"
        hits = ast_analyze_python(src, file_path="src/settings.py")
        assert "SEC-070" in _ids(hits)

    def test_ignores_test_debug(self):
        src = "app.debug = True\n"
        hits = ast_analyze_python(src, file_path="tests/test_app.py")
        assert "SEC-070" not in _ids(hits)

    def test_ignores_main_guard(self):
        src = (
            "if __name__ == '__main__':\n"
            "    app.run(debug=True)\n"
        )
        hits = ast_analyze_python(src, file_path="src/app.py")
        assert "SEC-070" not in _ids(hits)

    def test_hit_has_evidence_and_reasoning(self):
        src = "DEBUG = True\n"
        hits = ast_analyze_python(src, file_path="src/settings.py")
        assert hits[0].evidence
        assert hits[0].reasoning
        assert hits[0].confidence >= 0.80
