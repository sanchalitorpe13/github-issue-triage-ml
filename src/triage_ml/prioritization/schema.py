"""
Unified Triage Result Schemas & Data Contracts.
Provides typed Pydantic models for structured, explainable issue triage results.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class AlternativePrediction(BaseModel):
    """Secondary predicted category and score."""
    category: str
    probability: float
    percentage: str


class PriorityAssessment(BaseModel):
    """Intelligent rule-based priority assessment."""
    level: str = Field(description="Priority tier: P0, P1, P2, P3, or P4")
    name: str = Field(description="Priority name: Critical, High, Medium, Low, or Trivial")
    score: int = Field(ge=0, le=100, description="Heuristic priority score from 0 to 100")
    method: str = Field(default="rule_based", description="Assessment method: rule_based (heuristic)")
    sla_hours: int = Field(description="Recommended operational SLA response window in hours")
    signals_detected: List[str] = Field(default_factory=list, description="Specific triggers detected in issue text")
    reasoning: List[str] = Field(default_factory=list, description="Step-by-step scoring breakdown")
    is_heuristic: bool = Field(default=True, description="Explicit disclaimer that score is heuristic, not ML-trained")


class SeverityAssessment(BaseModel):
    """Technical severity and blast-radius evaluation."""
    level: str = Field(description="Severity tier: Critical, High, Medium, or Low")
    factors: Dict[str, str] = Field(default_factory=dict, description="Dimensions evaluated: functionality, availability, etc.")
    reasoning: List[str] = Field(default_factory=list, description="Reasoning for assigned severity level")


class UrgencyAssessment(BaseModel):
    """Operational urgency and time-sensitivity evaluation."""
    level: str = Field(description="Urgency tier: Immediate, High, Medium, or Low")
    signals: List[str] = Field(default_factory=list, description="Time-sensitivity indicators detected")
    reasoning: str = Field(description="Summary rationale for urgency level")


class ExplanationDetails(BaseModel):
    """Detailed explainability bundle for triage transparency."""
    category_reasoning: str
    top_keywords: List[List[Union[str, float]]] = Field(default_factory=list, description="Top influential TF-IDF n-grams")
    priority_reasoning: str
    severity_reasoning: str
    urgency_reasoning: str
    limitations: List[str] = Field(default_factory=list, description="Known assessment constraints and caveats")


class TriageResult(BaseModel):
    """Complete, unified triage result for a single GitHub issue."""
    issue_id: Optional[Union[int, str]] = None
    title: str
    predicted_category: str
    confidence: float = Field(ge=0.0, le=1.0, description="Calibrated confidence of primary prediction")
    confidence_status: str = Field(description="Reliability tier: High, Medium, or Low")
    alternative_predictions: List[AlternativePrediction] = Field(default_factory=list)
    priority: PriorityAssessment
    severity: SeverityAssessment
    urgency: UrgencyAssessment
    recommendations: List[str] = Field(default_factory=list)
    explanation: ExplanationDetails
    model_name: str
    model_version: str
    analyzed_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    is_fallback: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)
