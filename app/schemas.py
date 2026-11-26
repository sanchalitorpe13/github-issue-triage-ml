"""
Pydantic Request and Response Schemas for FastAPI Endpoints.
Provides strict validation, descriptive field limits, OpenAPI documentation, and example payloads.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from src.triage_ml.prioritization.schema import TriageResult


class IssueTriageRequest(BaseModel):
    """Input payload for triaging a single GitHub issue."""
    title: str = Field(
        min_length=1,
        max_length=500,
        description="GitHub issue title",
        examples=["Fatal crash: Segmentation fault in QuantizedConv2D op"]
    )
    body: Optional[str] = Field(
        default="",
        max_length=50000,
        description="GitHub issue body / description / traceback",
        examples=["Calling tf.raw_ops.QuantizedConv2D with empty filter produces core dump in production."]
    )
    issue_number: Optional[int] = Field(
        default=None,
        ge=1,
        description="GitHub issue number (e.g. 1024)",
        examples=[1024]
    )
    issue_id: Optional[Union[int, str]] = Field(
        default=None,
        description="Internal or GitHub issue ID",
        examples=["GH-1024"]
    )
    repository: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Repository name in 'owner/repo' format",
        examples=["tensorflow/tensorflow"]
    )
    labels: Optional[List[str]] = Field(
        default_factory=list,
        description="Existing labels currently assigned to the issue",
        examples=[["type:bug", "comp:core"]]
    )
    state: Optional[str] = Field(
        default="open",
        description="Issue state (e.g. 'open' or 'closed')",
        examples=["open"]
    )
    url: Optional[str] = Field(
        default=None,
        description="Link to the issue on GitHub",
        examples=["https://github.com/tensorflow/tensorflow/issues/1024"]
    )
    author: Optional[str] = Field(
        default=None,
        description="GitHub username of the issue author",
        examples=["octocat"]
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional arbitrary metadata attributes",
        examples=[{"comments_count": 3, "created_at": "2026-11-20T10:00:00Z"}]
    )


class BatchTriageRequest(BaseModel):
    """Input payload for batch issue triage."""
    issues: List[IssueTriageRequest] = Field(
        min_length=1,
        max_length=50,
        description="List of issues to triage (1 to 50 items per batch)"
    )


class BatchItemResult(BaseModel):
    """Result for an individual issue inside a batch request."""
    index: int = Field(description="Zero-based index of the issue in the submitted batch")
    success: bool = Field(description="Whether triage succeeded for this item")
    result: Optional[TriageResult] = Field(default=None, description="Triage output if successful")
    error: Optional[str] = Field(default=None, description="Error explanation if processing failed")


class BatchTriageResponse(BaseModel):
    """Consolidated response for batch issue triage."""
    total_submitted: int = Field(description="Total issues submitted in batch")
    total_successful: int = Field(description="Total issues successfully triaged")
    total_failed: int = Field(description="Total issues that encountered errors")
    results: List[BatchItemResult] = Field(description="Individual issue triage results")
    processed_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class HealthResponse(BaseModel):
    """Basic service health check."""
    status: str = Field(examples=["healthy"])
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class DetailedHealthResponse(BaseModel):
    """Detailed system diagnostic status."""
    status: str = Field(examples=["healthy"])
    api_title: str
    api_version: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    model_loaded: bool
    triage_engine_status: str
    supported_classes: List[str]


class ModelInfoResponse(BaseModel):
    """Sanitized machine learning model metadata."""
    model_name: str
    version: str
    task: str
    classes: List[str]
    selection_metric: str
    model_loaded: bool
    metrics_summary: Dict[str, Any]
    vectorizer_summary: Dict[str, Any]


class ErrorResponse(BaseModel):
    """Standardized API error response."""
    error: str = Field(description="Error type identifier")
    message: str = Field(description="Human-readable error description")
    status_code: int = Field(description="HTTP status code")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    path: Optional[str] = Field(default=None, description="Requested endpoint path")
