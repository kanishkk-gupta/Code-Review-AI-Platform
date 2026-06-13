"""
tests/integration/test_review_endpoint.py
==========================================
Integration tests for POST /review endpoint.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from schemas import JobStatusEnum


class TestPostReview:
    async def test_submit_with_url_returns_202(self, async_client: AsyncClient, auth_headers):
        response = await async_client.post(
            "/review",
            json={
                "repository_name": "test-repo",
                "source_url": "https://github.com/test/test-repo",
            },
            headers=auth_headers,
        )
        assert response.status_code == 202
        body = response.json()
        assert body["status"] == JobStatusEnum.PENDING
        assert "job_id" in body
        assert "poll_url" in body

    async def test_submit_without_api_key_returns_401(self, async_client: AsyncClient):
        response = await async_client.post(
            "/review",
            json={
                "repository_name": "test-repo",
                "source_url": "https://github.com/test/test-repo",
            },
        )
        assert response.status_code == 401

    async def test_submit_without_source_returns_422(self, async_client: AsyncClient, auth_headers):
        response = await async_client.post(
            "/review",
            json={"repository_name": "test-repo"},
            headers=auth_headers,
        )
        assert response.status_code == 422

    async def test_submit_with_both_sources_returns_422(self, async_client: AsyncClient, auth_headers):
        response = await async_client.post(
            "/review",
            json={
                "repository_name": "test-repo",
                "source_url": "https://github.com/test/test-repo",
                "source_zip_b64": "dGVzdA==",
            },
            headers=auth_headers,
        )
        assert response.status_code == 422
