"""
tests/conftest.py
==================
Shared pytest fixtures for all test modules.

Rules:
  - ALL shared fixtures live here. Never duplicate across test files.
  - Use pytest-asyncio with asyncio_mode="auto" (set in pyproject.toml).
  - Use httpx.AsyncClient with FastAPI's ASGITransport for integration tests.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient, ASGITransport

from api.main import create_app
from services.job_store import InMemoryJobStore
from schemas import (
    JobStatus,
    JobStatusEnum,
    ReviewConfig,
    ReviewRequest,
    ReviewResult,
    RepositoryMetadata,
    SupportedLanguage,
    CodeChunk,
)


# ── App Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def app():
    """Return a fresh FastAPI app instance for testing."""
    return create_app()


@pytest.fixture
async def async_client(app):
    """Return an async test client for the FastAPI app."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


# ── Auth Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def api_key():
    """Return a test API key that matches the test environment settings."""
    return "test-api-key"


@pytest.fixture
def auth_headers(api_key):
    """Return HTTP headers with the test API key."""
    return {"X-API-Key": api_key}


# ── Job Store Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def job_store():
    """Return a fresh InMemoryJobStore for testing."""
    return InMemoryJobStore()


@pytest.fixture
async def pending_job(job_store):
    """Create and store a PENDING job, return its JobStatus."""
    job = JobStatus(status=JobStatusEnum.PENDING)
    await job_store.create(job)
    return job


# ── Schema Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def sample_review_request():
    """Return a valid ReviewRequest with a GitHub URL."""
    return ReviewRequest(
        repository_name="test-repo",
        source_url="https://github.com/test/test-repo",
    )


@pytest.fixture
def sample_code_chunk():
    """Return a minimal valid CodeChunk."""
    return CodeChunk(
        file_path="src/main.py",
        language=SupportedLanguage.PYTHON,
        content="def hello():\n    return 'world'\n",
        start_line=1,
        end_line=2,
    )


@pytest.fixture
def sample_repository_metadata():
    """Return a valid RepositoryMetadata."""
    return RepositoryMetadata(
        repository_name="test-repo",
        primary_language=SupportedLanguage.PYTHON,
        language_breakdown={"python": 100.0},
        total_files=5,
        total_lines=250,
    )
