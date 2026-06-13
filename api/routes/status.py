"""
api/routes/status.py
====================
GET /status/{job_id} — Poll for current review job lifecycle state.

Contract: api_contracts.md § GET /status/{job_id}
Schema  : JobStatus → JobStatusResponse (200 OK)
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_store, verify_api_key
from schemas import JobStatusResponse
from services.job_store import JobStore

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get(
    "/status/{job_id}",
    response_model=JobStatusResponse,
    summary="Poll review job status",
    description=(
        "Returns current lifecycle state and progress percentage. "
        "When status=COMPLETED, the full ReviewResult is embedded inline. "
        "See api_contracts.md for recommended polling intervals."
    ),
)
async def get_job_status(
    job_id: str,
    store: JobStore = Depends(get_store),
    _key: str = Depends(verify_api_key),
) -> JobStatusResponse:
    """
    GET /status/{job_id}

    Steps:
      1. Look up job_id in the job store.
      2. Return 404 if not found.
      3. Map JobStatus → JobStatusResponse and return.
    """
    job = await store.get(job_id)

    if job is None:
        logger.warning("status_job_not_found", job_id=job_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "JOB_NOT_FOUND", "message": f"No job with id '{job_id}'.", "details": None},
        )

    logger.debug("status_polled", job_id=job_id, status=job.status, progress=job.progress)

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        result=job.result,
        error=job.error,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )
