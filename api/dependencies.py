"""
api/dependencies.py
===================
FastAPI Depends() providers shared across all route handlers.

Rules:
  - All shared injectable state lives here.
  - Route files import from this module — never from each other.
  - No business logic; only DI wiring.
"""

from __future__ import annotations

from fastapi import Header, HTTPException, status, Depends

from config.settings import Settings, get_settings
from services.job_store import JobStore, get_job_store


# ── Authentication ────────────────────────────────────────────────────────────


async def verify_api_key(
    x_api_key: str = Header(..., alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> str:
    """
    Validate the X-API-Key header against the configured secret.

    Raises:
        HTTPException 401 if key is missing or does not match.

    Returns:
        The validated API key string.
    """
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHORIZED", "message": "Invalid or missing API key.", "details": None},
        )
    return x_api_key


# ── Job Store ─────────────────────────────────────────────────────────────────


async def get_store(store: JobStore = Depends(get_job_store)) -> JobStore:
    """
    Provide the shared JobStore instance to route handlers.
    Swappable between in-memory and Redis without changing route code.
    """
    return store


# ── Settings ──────────────────────────────────────────────────────────────────


async def get_app_settings(settings: Settings = Depends(get_settings)) -> Settings:
    """Provide Settings to routes that need runtime configuration values."""
    return settings
