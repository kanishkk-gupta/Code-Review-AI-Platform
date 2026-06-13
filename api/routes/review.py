"""
api/routes/review.py
====================
POST /review — Submit a repository for AI-powered code review.

Contract: api_contracts.md § POST /review
Schema  : ReviewRequest → ReviewResponse (202 Accepted)

Responsibilities:
  - Validate the request body (Pydantic V2 via FastAPI)
  - Create a JobStatus record in the job store
  - Dispatch the LangGraph review pipeline as a BackgroundTask
  - Return 202 Accepted with job_id and poll_url

Must NOT contain:
  - Any analysis logic
  - Direct LangGraph calls (delegated to graph.workflow)
  - Job store write logic beyond creating the initial record
"""

from __future__ import annotations

import asyncio

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, status
from fastapi.responses import JSONResponse

from api.dependencies import get_store, get_app_settings, verify_api_key
from config.settings import Settings
from schemas import (
    JobStatus,
    JobStatusEnum,
    ReviewRequest,
    ReviewResponse,
    ReviewState,
)
from services.job_store import JobStore

logger = structlog.get_logger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Background task — runs the full LangGraph pipeline
# ---------------------------------------------------------------------------

async def _run_review_pipeline(
    job_id: str,
    state:  ReviewState,
    store:  JobStore,
) -> None:
    """
    Background task: invoke the compiled LangGraph workflow.

    Progress updates written to the job store at each node transition
    so the GET /status endpoint can report live progress.

    Error handling:
      - Any unhandled exception marks the job FAILED with an error message.
      - The background task never raises — FastAPI BackgroundTask errors
        are silently swallowed otherwise.
    """
    try:
        logger.info("review_pipeline_start", job_id=job_id)
        await store.update(job_id, status=JobStatusEnum.RUNNING, progress=0)

        # Import here to avoid circular imports at module load time
        from graph.workflow import get_workflow

        graph = get_workflow()

        # Build the initial state dict for the graph
        initial: dict = {
            "job_id":          job_id,
            "source_url":      state.source_url,
            "source_zip_b64":  state.source_zip_b64,
            "config":          state.config.model_dump(),
            "chunks":          [],
            "metadata":        None,
            "bug_findings":          [],
            "solid_findings":        [],
            "architecture_findings": [],
            "security_findings":     [],
            "complexity_findings":   [],
            "result":          None,
            "progress":        0,
            "error":           None,
        }

        # Progress callback — poll the state dict for progress updates
        # and sync them to the job store every N seconds while the graph runs
        _progress_task: asyncio.Task | None = None

        async def _progress_syncer(current_state_ref: list[dict]) -> None:
            """Periodically push progress from graph state into job store."""
            last_progress = 0
            while True:
                await asyncio.sleep(3)
                st = current_state_ref[0] if current_state_ref else {}
                prog = st.get("progress", 0)
                if prog != last_progress:
                    try:
                        await store.update(job_id, progress=prog)
                        last_progress = prog
                    except Exception:  # noqa: BLE001
                        pass

        # Mutable reference to the live state (updated after graph returns)
        state_ref: list[dict] = [initial]

        _progress_task = asyncio.create_task(_progress_syncer(state_ref))

        try:
            output = await graph.ainvoke(initial)
            state_ref[0] = output   # update reference so syncer sees final progress
        finally:
            if _progress_task is not None:  # FIX (BUG 3): guard None before cancel
                _progress_task.cancel()
                try:
                    await _progress_task
                except asyncio.CancelledError:
                    pass

        # Check for pipeline error
        if output.get("error"):
            error_msg = str(output["error"])
            logger.error("review_pipeline_graph_error", job_id=job_id, error=error_msg)
            await store.update(
                job_id,
                status=JobStatusEnum.FAILED,
                error=error_msg,
                progress=output.get("progress", 0),
            )
            return

        # Extract ReviewResult from output
        from schemas import ReviewResult
        raw_result = output.get("result")
        if raw_result is None:
            raise RuntimeError("Graph completed but produced no result.")

        result = ReviewResult.model_validate(raw_result)

        logger.info(
            "review_pipeline_complete",
            job_id=job_id,
            score=result.overall_score,
            total_findings=result.total_findings,
        )

        await store.update(
            job_id,
            status=JobStatusEnum.COMPLETED,
            progress=100,
            result=result,
        )

    except Exception as exc:  # noqa: BLE001
        logger.exception("review_pipeline_error", job_id=job_id, error=str(exc))
        await store.update(
            job_id,
            status=JobStatusEnum.FAILED,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# POST /review
# ---------------------------------------------------------------------------

@router.post(
    "/review",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ReviewResponse,
    summary="Submit a repository for code review",
    description=(
        "Accepts a GitHub URL or base64-encoded ZIP archive. "
        "Returns a job_id immediately; analysis runs asynchronously. "
        "Poll GET /status/{job_id} for progress updates. "
        "Retrieve the full report via GET /report/{job_id} once completed."
    ),
    responses={
        202: {"description": "Review job accepted and queued."},
        400: {"description": "Neither source_url nor source_zip_b64 provided."},
        401: {"description": "Missing or invalid X-API-Key header."},
        422: {"description": "Request body validation failed."},
    },
)
async def submit_review(
    body:             ReviewRequest,
    background_tasks: BackgroundTasks,
    store:            JobStore  = Depends(get_store),
    settings:         Settings  = Depends(get_app_settings),
    _key:             str       = Depends(verify_api_key),
) -> ReviewResponse:
    """
    POST /review

    Steps:
      1. Build initial ReviewState from the validated request body.
      2. Create a PENDING JobStatus in the job store.
      3. Enqueue the LangGraph pipeline as a BackgroundTask.
      4. Return 202 with job_id and poll_url immediately.
    """
    # Validate that at least one source was provided
    if not body.source_url and not body.source_zip_b64:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "code":    "MISSING_SOURCE",
                "message": "Provide either source_url or source_zip_b64.",
                "details": None,
            },
        )

    # 1. Build initial state
    initial_state = ReviewState(
        source_url=body.source_url,
        source_zip_b64=body.source_zip_b64,
        config=body.config,
    )
    job_id = initial_state.job_id

    log = logger.bind(
        job_id=job_id,
        repository=body.repository_name,
        source_type="url" if body.source_url else "zip",
    )
    log.info("review_submitted")

    # 2. Persist initial PENDING status
    job = JobStatus(job_id=job_id, status=JobStatusEnum.PENDING)
    await store.create(job)

    # 3. Dispatch background pipeline (non-blocking)
    background_tasks.add_task(_run_review_pipeline, job_id, initial_state, store)

    log.info("review_pipeline_enqueued")

    # 4. Return 202 immediately
    poll_url = f"{settings.api_base_url}/status/{job_id}"

    return ReviewResponse(
        job_id=job_id,
        status=JobStatusEnum.PENDING,
        poll_url=poll_url,
        estimated_duration_seconds=60,
    )
