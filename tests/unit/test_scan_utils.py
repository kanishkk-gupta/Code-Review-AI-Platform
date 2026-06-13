"""Unit tests for production-path classification in agents/_scan_utils.py."""
from __future__ import annotations

import pytest

from agents._scan_utils import classify_file_role, is_non_production_path


@pytest.mark.parametrize(
    "path",
    [
        "unit_tests/helper.py",
        "integration_tests/utils.py",
        "__tests__/setup.js",
        "pkg/testing/support.py",
        "django/test/runner.py",
    ],
)
def test_extra_test_dirs_classified_as_test(path: str) -> None:
    assert classify_file_role(path) == "test"
    assert is_non_production_path(path) is True


@pytest.mark.parametrize(
    "path",
    [
        "django/contrib/admin/static/admin/js/core.js",
        "static/admin/js/inlines.js",
        "app/static/scripts/main.js",
    ],
)
def test_static_paths_are_non_production(path: str) -> None:
    assert is_non_production_path(path) is True


@pytest.mark.parametrize(
    "path",
    [
        "src/main.py",
        "django/core/handlers/base.py",
        "lib/service.py",
    ],
)
def test_production_paths_still_allowed(path: str) -> None:
    assert classify_file_role(path) == "production"
    assert is_non_production_path(path) is False
