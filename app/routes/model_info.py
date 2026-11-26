"""
Model Information & Metadata Endpoints.
Exposes sanitized model evaluation metrics and classification parameters.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from app.schemas import ModelInfoResponse
from app.dependencies import get_triage_engine
from src.triage_ml.prioritization.triage_engine import TriageEngine

router = APIRouter(prefix="/api/v1/model", tags=["Model Information"])


@router.get(
    "/info",
    response_model=ModelInfoResponse,
    summary="Get Active Model Information",
    description="Returns model architecture, version, target classes, and benchmark performance metrics."
)
def get_model_info(engine: TriageEngine = Depends(get_triage_engine)) -> ModelInfoResponse:
    """Returns sanitized ML model architecture and benchmark evaluation summary."""
    predictor = engine.predictor
    if not predictor.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning model is not currently loaded."
        )

    meta = predictor.metadata
    metrics = meta.get("metrics", {})

    # Extract sanitized metrics summary (excluding raw paths)
    metrics_summary = {
        "accuracy": metrics.get("accuracy", 0.0),
        "macro_f1": metrics.get("macro_f1", 0.0),
        "weighted_f1": metrics.get("weighted_f1", 0.0),
        "macro_precision": metrics.get("macro_precision", 0.0),
        "macro_recall": metrics.get("macro_recall", 0.0),
        "sample_count": metrics.get("sample_count", 0)
    }

    vec_config = meta.get("vectorizer_config", {})
    vectorizer_summary = {
        "ngram_range": vec_config.get("ngram_range", [1, 2]),
        "max_features": vec_config.get("max_features", 5000),
        "vocab_size": vec_config.get("vocab_size", 0)
    }

    return ModelInfoResponse(
        model_name=meta.get("model_name", "Complement Naive Bayes"),
        version=meta.get("version", "2.0.0"),
        task=meta.get("task", "Issue Type Classification"),
        classes=predictor.classes,
        selection_metric=meta.get("selection_metric", "macro_f1"),
        model_loaded=True,
        metrics_summary=metrics_summary,
        vectorizer_summary=vectorizer_summary
    )
