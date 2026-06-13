"""
tests/unit/test_parser.py
==========================
Unit tests for tools/parser.py — all language parsers.
No filesystem fixtures are needed for the core parsing logic;
repository-level tests use pytest's tmp_path.
"""
from __future__ import annotations
from pathlib import Path
import pytest

from schemas import SupportedLanguage, RepositoryMetadata
from tools.parser import (
    ParsedClass,
    ParsedFile,
    ParsedFunction,
    ParsedImport,
    ParsedRepository,
    ParserError,
    UnsupportedLanguageError,
    build_metadata_from_parsed,
    parse_file,
    parse_repository,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pf(content: str, lang: SupportedLanguage, path: str = "test_file") -> ParsedFile:
    return parse_file(path, content, lang)


# ===========================================================================
# Python (AST)
# ===========================================================================

class TestParsePython:
    def test_imports_simple(self):
        pf = _pf("import os\nimport sys\n", SupportedLanguage.PYTHON)
        assert any(i.module == "os" for i in pf.imports)
        assert any(i.module == "sys" for i in pf.imports)

    def test_imports_from(self):
        pf = _pf("from pathlib import Path, PurePath\n", SupportedLanguage.PYTHON)
        imp = next(i for i in pf.imports if i.module == "pathlib")
        assert "Path" in imp.symbols
        assert "PurePath" in imp.symbols

    def test_module_level_function(self):
        src = "def greet(name: str) -> str:\n    return f'hello {name}'\n"
        pf = _pf(src, SupportedLanguage.PYTHON)
        assert len(pf.functions) == 1
        fn = pf.functions[0]
        assert fn.name == "greet"
        assert "name" in fn.params
        assert not fn.is_async

    def test_async_function(self):
        src = "async def fetch(url):\n    pass\n"
        pf = _pf(src, SupportedLanguage.PYTHON)
        assert pf.functions[0].is_async

    def test_class_with_methods(self):
        src = (
            "class Animal:\n"
            "    def speak(self):\n"
            "        pass\n"
            "    def move(self):\n"
            "        pass\n"
        )
        pf = _pf(src, SupportedLanguage.PYTHON)
        assert len(pf.classes) == 1
        cls = pf.classes[0]
        assert cls.name == "Animal"
        assert len(cls.methods) == 2
        assert {m.name for m in cls.methods} == {"speak", "move"}

    def test_class_base_classes(self):
        src = "class Dog(Animal, Runnable):\n    pass\n"
        pf = _pf(src, SupportedLanguage.PYTHON)
        cls = pf.classes[0]
        assert "Animal" in cls.base_classes
        assert "Runnable" in cls.base_classes

    def test_class_docstring(self):
        src = 'class Foo:\n    """This is Foo."""\n    pass\n'
        pf = _pf(src, SupportedLanguage.PYTHON)
        assert pf.classes[0].docstring == "This is Foo."

    def test_function_docstring(self):
        src = 'def bar():\n    """Bar does stuff."""\n    pass\n'
        pf = _pf(src, SupportedLanguage.PYTHON)
        assert pf.functions[0].docstring == "Bar does stuff."

    def test_decorator_captured(self):
        src = "@staticmethod\ndef util():\n    pass\n"
        pf = _pf(src, SupportedLanguage.PYTHON)
        assert any("staticmethod" in d for d in pf.functions[0].decorators)

    def test_syntax_error_returns_parse_error(self):
        pf = _pf("def broken(:\n    pass\n", SupportedLanguage.PYTHON)
        assert len(pf.parse_errors) == 1

    def test_line_numbers_correct(self):
        src = "x = 1\n\ndef foo():\n    return 1\n"
        pf = _pf(src, SupportedLanguage.PYTHON)
        assert pf.functions[0].line_start == 3

    def test_total_lines(self):
        pf = _pf("a = 1\nb = 2\nc = 3\n", SupportedLanguage.PYTHON)
        assert pf.total_lines == 3

    def test_empty_file(self):
        pf = _pf("", SupportedLanguage.PYTHON)
        assert pf.classes == []
        assert pf.functions == []
        assert pf.imports == []


# ===========================================================================
# Java (regex)
# ===========================================================================

class TestParseJava:
    SAMPLE = """\
import java.util.List;
import static org.junit.Assert.assertEquals;

public class Animal {
    private String name;

    public Animal(String name) {
        this.name = name;
    }

    public String speak() {
        return "...";
    }
}

interface Runnable {
    void run();
}
"""

    def test_imports_captured(self):
        pf = _pf(self.SAMPLE, SupportedLanguage.JAVA)
        modules = [i.module for i in pf.imports]
        assert "java.util.List" in modules
        assert "org.junit.Assert.assertEquals" in modules

    def test_class_captured(self):
        pf = _pf(self.SAMPLE, SupportedLanguage.JAVA)
        names = {c.name for c in pf.classes}
        assert "Animal" in names

    def test_interface_captured(self):
        pf = _pf(self.SAMPLE, SupportedLanguage.JAVA)
        names = {c.name for c in pf.classes}
        assert "Runnable" in names

    def test_methods_captured(self):
        pf = _pf(self.SAMPLE, SupportedLanguage.JAVA)
        fn_names = {f.name for f in pf.functions}
        assert "speak" in fn_names

    def test_line_number_class(self):
        pf = _pf(self.SAMPLE, SupportedLanguage.JAVA)
        animal = next(c for c in pf.classes if c.name == "Animal")
        assert animal.line_start == 4


# ===========================================================================
# C++ (regex)
# ===========================================================================

class TestParseCpp:
    SAMPLE = """\
#include <iostream>
#include <vector>

class Shape {
public:
    virtual double area() const = 0;
    void describe();
};

struct Point {
    int x, y;
};

int main(int argc, char** argv) {
    return 0;
}
"""

    def test_includes_captured(self):
        pf = _pf(self.SAMPLE, SupportedLanguage.CPP)
        modules = [i.module for i in pf.imports]
        assert "iostream" in modules
        assert "vector" in modules

    def test_class_captured(self):
        pf = _pf(self.SAMPLE, SupportedLanguage.CPP)
        names = {c.name for c in pf.classes}
        assert "Shape" in names

    def test_struct_captured(self):
        pf = _pf(self.SAMPLE, SupportedLanguage.CPP)
        names = {c.name for c in pf.classes}
        assert "Point" in names

    def test_function_captured(self):
        pf = _pf(self.SAMPLE, SupportedLanguage.CPP)
        fn_names = {f.name for f in pf.functions}
        assert "main" in fn_names

    def test_c_file_uses_cpp_parser(self):
        src = '#include <stdio.h>\nint main() { return 0; }\n'
        pf = _pf(src, SupportedLanguage.C)
        assert any(i.module == "stdio.h" for i in pf.imports)


# ===========================================================================
# JavaScript (regex)
# ===========================================================================

class TestParseJavaScript:
    SAMPLE = """\
import React from 'react';
import { useState, useEffect } from 'react';
const fs = require('fs');

class EventEmitter {
    constructor() {}
    emit(event) {}
}

async function fetchData(url) {
    return await fetch(url);
}

const handler = async (req, res) => {
    res.send('ok');
};

const square = x => x * x;
"""

    def test_es6_imports(self):
        pf = _pf(self.SAMPLE, SupportedLanguage.JAVASCRIPT)
        modules = [i.module for i in pf.imports]
        assert "react" in modules

    def test_require_import(self):
        pf = _pf(self.SAMPLE, SupportedLanguage.JAVASCRIPT)
        modules = [i.module for i in pf.imports]
        assert "fs" in modules

    def test_class_captured(self):
        pf = _pf(self.SAMPLE, SupportedLanguage.JAVASCRIPT)
        assert any(c.name == "EventEmitter" for c in pf.classes)

    def test_async_function_captured(self):
        pf = _pf(self.SAMPLE, SupportedLanguage.JAVASCRIPT)
        fn = next(f for f in pf.functions if f.name == "fetchData")
        assert fn.is_async

    def test_arrow_function_captured(self):
        pf = _pf(self.SAMPLE, SupportedLanguage.JAVASCRIPT)
        fn_names = {f.name for f in pf.functions}
        assert "handler" in fn_names


# ===========================================================================
# TypeScript (regex)
# ===========================================================================

class TestParseTypeScript:
    SAMPLE = """\
import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';

interface UserProfile {
    id: number;
    name: string;
}

interface Serializable extends UserProfile {
    serialize(): string;
}

class UserService implements OnInit {
    constructor(private http: HttpClient) {}
    ngOnInit(): void {}
}

export async function loadUser(id: number): Promise<UserProfile> {
    return {} as UserProfile;
}
"""

    def test_imports_captured(self):
        pf = _pf(self.SAMPLE, SupportedLanguage.TYPESCRIPT)
        modules = [i.module for i in pf.imports]
        assert "@angular/core" in modules

    def test_interface_captured(self):
        pf = _pf(self.SAMPLE, SupportedLanguage.TYPESCRIPT)
        names = {c.name for c in pf.classes}
        assert "UserProfile" in names
        assert "Serializable" in names

    def test_interface_extends(self):
        pf = _pf(self.SAMPLE, SupportedLanguage.TYPESCRIPT)
        iface = next(c for c in pf.classes if c.name == "Serializable")
        assert "UserProfile" in iface.base_classes

    def test_class_captured(self):
        pf = _pf(self.SAMPLE, SupportedLanguage.TYPESCRIPT)
        names = {c.name for c in pf.classes}
        assert "UserService" in names

    def test_function_captured(self):
        pf = _pf(self.SAMPLE, SupportedLanguage.TYPESCRIPT)
        fn_names = {f.name for f in pf.functions}
        assert "loadUser" in fn_names


# ===========================================================================
# ParsedFile computed properties
# ===========================================================================

class TestParsedFileProperties:
    def test_class_count(self):
        pf = _pf("class A: pass\nclass B: pass\n", SupportedLanguage.PYTHON)
        assert pf.class_count == 2

    def test_function_count_includes_methods(self):
        src = "class A:\n    def m1(self): pass\n    def m2(self): pass\ndef top(): pass\n"
        pf = _pf(src, SupportedLanguage.PYTHON)
        assert pf.function_count == 3  # 2 methods + 1 top-level

    def test_import_count(self):
        pf = _pf("import os\nimport sys\nimport re\n", SupportedLanguage.PYTHON)
        assert pf.import_count == 3

    def test_unsupported_language_returns_empty_file(self):
        pf = parse_file("test.rb", "puts 'hello'", SupportedLanguage.RUBY)
        assert pf.classes == []
        assert pf.functions == []
        assert pf.imports == []


# ===========================================================================
# ParsedRepository
# ===========================================================================

class TestParsedRepository:
    def _make(self, tmp_path: Path) -> ParsedRepository:
        (tmp_path / "main.py").write_text("def main(): pass\n", encoding="utf-8")
        (tmp_path / "utils.py").write_text("class Helper:\n    def run(self): pass\n", encoding="utf-8")
        return parse_repository(tmp_path, name="test-repo")

    def test_files_collected(self, tmp_path: Path):
        repo = self._make(tmp_path)
        assert repo.total_files == 2

    def test_primary_language(self, tmp_path: Path):
        repo = self._make(tmp_path)
        assert repo.primary_language == SupportedLanguage.PYTHON

    def test_total_lines(self, tmp_path: Path):
        repo = self._make(tmp_path)
        assert repo.total_lines >= 2

    def test_raises_on_missing_dir(self):
        with pytest.raises(FileNotFoundError):
            parse_repository("/nonexistent/path")

    def test_metadata_returns_repository_metadata(self, tmp_path: Path):
        repo = self._make(tmp_path)
        md = repo.metadata()
        assert isinstance(md, RepositoryMetadata)
        assert md.repository_name == "test-repo"
        assert md.primary_language == "python"

    def test_language_breakdown_sums_to_100(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("x = 1\n" * 50, encoding="utf-8")
        (tmp_path / "b.js").write_text("const x = 1;\n" * 50, encoding="utf-8")
        repo = parse_repository(tmp_path, name="mixed")
        total = sum(repo.language_breakdown.values())
        assert abs(total - 100.0) <= 0.5


# ===========================================================================
# build_metadata_from_parsed
# ===========================================================================

class TestBuildMetadataFromParsed:
    def test_raises_on_empty_files(self, tmp_path: Path):
        repo = ParsedRepository(root_path=tmp_path, name="empty", source_url=None)
        with pytest.raises(ValueError, match="no parseable files"):
            build_metadata_from_parsed(repo)

    def test_source_url_preserved(self, tmp_path: Path):
        (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
        repo = parse_repository(tmp_path, name="repo", source_url="https://github.com/x/y")
        md = build_metadata_from_parsed(repo)
        assert md.source_url == "https://github.com/x/y"

    def test_total_files_correct(self, tmp_path: Path):
        for i in range(3):
            (tmp_path / f"f{i}.py").write_text(f"x = {i}\n", encoding="utf-8")
        repo = parse_repository(tmp_path)
        md = build_metadata_from_parsed(repo)
        assert md.total_files == 3
