"""
Issue Triage Endpoints.
Exposes single-issue and batch-issue triage capabilities.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from app.schemas import (
    BatchItemResult,
    BatchTriageRequest,
    BatchTriageResponse,
    IssueTriageRequest
)
from app.dependencies import get_triage_engine, get_settings
from src.triage_ml.prioritization.schema import TriageResult
from src.triage_ml.prioritization.triage_engine import TriageEngine
from src.triage_ml.utils.logger import get_logger

logger = get_logger("routes_triage")

router = APIRouter(prefix="/api/v1/triage", tags=["Issue Triage"])


@router.post(
    "",
    response_model=TriageResult,
    summary="Triage Single GitHub Issue",
    description="Analyzes an issue's title, body, and metadata to predict category, priority, severity, urgency, and recommendations."
)
def triage_single_issue(
    request: IssueTriageRequest,
    engine: TriageEngine = Depends(get_triage_engine)
) -> TriageResult:
    """Performs end-to-end triage analysis for a single GitHub issue."""
    try:
        combined_metadata = request.metadata.copy() if request.metadata else {}
        if request.repository:
            combined_metadata["repository"] = request.repository
        if request.labels:
            combined_metadata["existing_labels"] = request.labels
        if request.state:
            combined_metadata["state"] = request.state
        if request.author:
            combined_metadata["author"] = request.author
        if request.url:
            combined_metadata["url"] = request.url

        effective_id = request.issue_number or request.issue_id

        result = engine.analyze_issue(
            title=request.title,
            body=request.body or "",
            issue_id=effective_id,
            metadata=combined_metadata
        )
        return result

    except Exception as e:
        logger.error(f"Error triaging issue '{request.title[:40]}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Triage engine encountered an internal error: {str(e)}"
        )


@router.post(
    "/batch",
    response_model=BatchTriageResponse,
    summary="Batch Triage Multiple GitHub Issues",
    description="Analyzes up to 50 GitHub issues concurrently, reporting individual results and summary statistics."
)
def triage_batch_issues(
    request: BatchTriageRequest,
    engine: TriageEngine = Depends(get_triage_engine),
    settings=Depends(get_settings)
) -> BatchTriageResponse:
    """Performs resilient batch triage analysis with isolated per-item exception handling."""
    if len(request.issues) > settings.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size exceeds maximum limit of {settings.MAX_BATCH_SIZE} issues."
        )

    results: List[BatchItemResult] = []
    successful_count = 0
    failed_count = 0

    for idx, item in enumerate(request.issues):
        try:
            effective_id = item.issue_number or item.issue_id or idx + 1
            combined_meta = item.metadata.copy() if item.metadata else {}
            if item.repository:
                combined_meta["repository"] = item.repository

            triage_res = engine.analyze_issue(
                title=item.title,
                body=item.body or "",
                issue_id=effective_id,
                metadata=combined_meta
            )
            results.append(BatchItemResult(
                index=idx,
                success=True,
                result=triage_res,
                error=None
            ))
            successful_count += 1

        except Exception as e:
            logger.warning(f"Error processing batch item {idx}: {e}")
            results.append(BatchItemResult(
                index=idx,
                success=False,
                result=None,
                error=str(e)
            ))
            failed_count += 1

    return BatchTriageResponse(
        total_submitted=len(request.issues),
        total_successful=successful_count,
        total_failed=failed_count,
        results=results
    )
