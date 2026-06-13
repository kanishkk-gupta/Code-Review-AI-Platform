"""
ui/utils/api_client.py
=======================
Thin HTTP client wrapping all 4 CodeGuardian API endpoints.
Returns typed Pydantic schema objects — no raw dicts in UI code.

Usage:
    from ui.utils.api_client import APIClient
    client = APIClient()
    response = client.submit_review(body)
    status = client.get_status(job_id)
"""

from __future__ import annotations

import httpx

from schemas import (
    JobStatusResponse,
    ReviewRequest,
    ReviewResponse,
    ReviewResult,
    HealthResponse,
)


class APIClient:
    """
    Synchronous HTTP client for the CodeGuardian API.
    Streamlit runs synchronously so we use httpx in sync mode.
    """

    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        import os
        self.base_url = base_url or os.environ.get("STREAMLIT_API_BASE_URL", "http://localhost:8000")
        self.api_key = api_key or os.environ.get("API_KEY", "")
        self._headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
        }

    def submit_review(self, body: ReviewRequest) -> ReviewResponse:
        """POST /review — Submit repository for analysis."""
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{self.base_url}/review",
                content=body.model_dump_json(),
                headers=self._headers,
            )
            resp.raise_for_status()
            return ReviewResponse.model_validate(resp.json())

    def get_status(self, job_id: str) -> JobStatusResponse:
        """GET /status/{job_id} — Poll job lifecycle state."""
        with httpx.Client(timeout=10) as client:
            resp = client.get(
                f"{self.base_url}/status/{job_id}",
                headers=self._headers,
            )
            resp.raise_for_status()
            return JobStatusResponse.model_validate(resp.json())

    def get_report(self, job_id: str) -> ReviewResult:
        """GET /report/{job_id} — Retrieve the complete review report."""
        with httpx.Client(timeout=10) as client:
            resp = client.get(
                f"{self.base_url}/report/{job_id}",
                headers=self._headers,
            )
            resp.raise_for_status()
            return ReviewResult.model_validate(resp.json())

    def health_check(self) -> HealthResponse:
        """GET /health — Check API health status."""
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{self.base_url}/health")
            return HealthResponse.model_validate(resp.json())
