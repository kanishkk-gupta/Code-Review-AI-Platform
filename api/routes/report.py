"""
api/routes/report.py
====================
GET /report/{job_id} — Retrieve the complete ReviewResult for a finished job.

Contract: api_contracts.md § GET /report/{job_id}
Schema  : ReviewResult (200 OK) | ErrorDetail (404 / 409)
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_store, verify_api_key
from schemas import JobStatusEnum, ReviewResult
from services.job_store import JobStore

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get(
    "/report/{job_id}",
    response_model=ReviewResult,
    summary="Retrieve the complete code review report",
    description=(
        "Returns the full ReviewResult for a completed job. "
        "Returns 409 if the job is still PENDING or RUNNING. "
        "Returns 409 with code=JOB_FAILED if the job failed."
    ),
)
async def get_report(
    job_id: str,
    store: JobStore = Depends(get_store),
    _key: str = Depends(verify_api_key),
) -> ReviewResult:
    """
    GET /report/{job_id}

    Steps:
      1. Look up job in the store → 404 if not found.
      2. Guard: 409 if PENDING or RUNNING.
      3. Guard: 409 if FAILED (direct user to /status for error details).
      4. Return the frozen ReviewResult.
    """
    job = await store.get(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "JOB_NOT_FOUND", "message": f"No job with id '{job_id}'.", "details": None},
        )

    if job.status in (JobStatusEnum.PENDING, JobStatusEnum.RUNNING):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "REPORT_NOT_READY",
                "message": f"Job '{job_id}' is still {job.status}. Poll GET /status/{job_id}.",
                "details": {"progress": job.progress},
            },
        )

    if job.status == JobStatusEnum.FAILED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "JOB_FAILED",
                "message": f"Job '{job_id}' failed. See GET /status/{job_id} for error details.",
                "details": None,
            },
        )

    # status == COMPLETED; result is guaranteed non-None by JobStatus validator
    logger.info("report_retrieved", job_id=job_id)
    return job.result  # type: ignore[return-value]
