"""
Health & Diagnostic Endpoints.
Provides service status, API versioning, and documentation links.
"""
from datetime import datetime
from fastapi import APIRouter, Depends
from app.schemas import DetailedHealthResponse, HealthResponse
from app.dependencies import get_triage_engine, get_settings
from src.triage_ml.prioritization.triage_engine import TriageEngine

router = APIRouter(tags=["Health & Diagnostics"])


@router.get(
    "/",
    summary="API Root Information",
    description="Returns service metadata, version, and interactive documentation links."
)
def get_root(settings=Depends(get_settings)):
    """API root landing endpoint."""
    return {
        "title": settings.API_TITLE,
        "version": settings.API_VERSION,
        "description": settings.API_DESCRIPTION,
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "endpoints": {
            "health": "/health",
            "detailed_health": "/api/v1/health",
            "single_triage": "/api/v1/triage",
            "batch_triage": "/api/v1/triage/batch",
            "model_info": "/api/v1/model/info"
        },
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Basic Health Check",
    description="Liveness probe returning basic service operational status."
)
def get_health():
    """Basic health check probe."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat() + "Z"
    )


@router.get(
    "/api/v1/health",
    response_model=DetailedHealthResponse,
    summary="Detailed Diagnostic Health Check",
    description="Deep diagnostic probe reporting model readiness, active classes, and triage engine status."
)
def get_detailed_health(
    engine: TriageEngine = Depends(get_triage_engine),
    settings=Depends(get_settings)
):
    """Deep readiness and health diagnostic check."""
    is_ready = engine.predictor.is_ready()
    status_str = "healthy" if is_ready else "degraded"
    engine_status = "operational" if is_ready else "model_unavailable"

    return DetailedHealthResponse(
        status=status_str,
        api_title=settings.API_TITLE,
        api_version=settings.API_VERSION,
        timestamp=datetime.utcnow().isoformat() + "Z",
        model_loaded=is_ready,
        triage_engine_status=engine_status,
        supported_classes=engine.predictor.classes
    )
