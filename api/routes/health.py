"""
api/routes/health.py
====================
GET /health — Liveness and readiness probe. No authentication required.

Contract: api_contracts.md § GET /health
Schema  : HealthResponse (200 OK) | HealthResponse (503 Service Unavailable)

Status:
  "ok"       — all subsystems ready (embedding model loaded)
  "degraded" — app running but embedding model not yet warm; requests accepted
               but analysis may be slower on the first job
"""

from __future__ import annotations

import time

import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from config.settings import Settings, get_settings
from schemas import HealthResponse, JobStatusEnum
from services.job_store import JobStore, get_job_store

logger = structlog.get_logger(__name__)
router = APIRouter()

# Module-level start time — overwritten by api/main.py lifespan on startup
_PROCESS_START: float = time.monotonic()


def _is_embedding_model_ready() -> bool:
    """
    Check whether the SentenceTransformer model has been loaded.
    FIX (BUG 7): Uses public get_loaded_models() instead of private _MODEL_REGISTRY.
    Returns True if at least one model is registered and ready.
    """
    try:
        from rag.embeddings import get_loaded_models
        return len(get_loaded_models()) > 0
    except Exception:   # noqa: BLE001
        return False


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description=(
        "Liveness and readiness probe. "
        "Returns 200 when all systems are nominal, 503 when degraded. "
        "No authentication required. "
        "Safe to call from load-balancer health checks every 10 seconds."
    ),
    responses={
        200: {"description": "All systems healthy."},
        503: {"description": "Application running but not fully ready (embedding model loading)."},
    },
)
async def health_check(
    store:    JobStore = Depends(get_job_store),
    settings: Settings = Depends(get_settings),
) -> JSONResponse:
    """
    GET /health

    Checks:
      - Uptime (always available)
      - Active RUNNING job count
      - SentenceTransformer embedding model readiness
    """
    uptime       = time.monotonic() - _PROCESS_START
    active_jobs  = await store.count_by_status(JobStatusEnum.RUNNING)
    model_ready  = _is_embedding_model_ready()

    overall_status = "ok" if model_ready else "degraded"

    payload = HealthResponse(
        status=overall_status,
        version=settings.app_version,
        uptime_seconds=round(uptime, 2),
        active_jobs=active_jobs,
        vector_store_ready=model_ready,
    )

    http_status = 200 if overall_status == "ok" else 503

    logger.debug(
        "health_check",
        status=overall_status,
        uptime=round(uptime, 1),
        active_jobs=active_jobs,
        model_ready=model_ready,
    )

    return JSONResponse(content=payload.model_dump(), status_code=http_status)
