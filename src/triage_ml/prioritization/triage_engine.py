"""
Unified Intelligent Issue Triage Engine.
Integrates trained ML inference, calibrated confidence estimation,
rule-based priority scoring, severity analysis, urgency detection,
contextual recommendations, and explainability into a structured service.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from config.settings import settings
from src.triage_ml.nlp.text_cleaner import extract_structural_features
from src.triage_ml.models.predictor import IssuePredictor
from src.triage_ml.prioritization.schema import (
    AlternativePrediction,
    ExplanationDetails,
    TriageResult
)
from src.triage_ml.prioritization.priority_scorer import PriorityScorer
from src.triage_ml.prioritization.severity_analyzer import SeverityAnalyzer
from src.triage_ml.prioritization.urgency_detector import UrgencyDetector
from src.triage_ml.prioritization.recommendation_engine import RecommendationEngine
from src.triage_ml.utils.logger import get_logger

logger = get_logger("triage_engine")


class TriageEngine:
    """Production triage service coordinating ML models and rule-based intelligence."""

    def __init__(
        self,
        predictor: Optional[IssuePredictor] = None,
        priority_scorer: Optional[PriorityScorer] = None,
        severity_analyzer: Optional[SeverityAnalyzer] = None,
        urgency_detector: Optional[UrgencyDetector] = None,
        recommendation_engine: Optional[RecommendationEngine] = None
    ):
        # Reusable ML predictor (cached, loaded once)
        self.predictor = predictor or IssuePredictor()
        self.priority_scorer = priority_scorer or PriorityScorer()
        self.severity_analyzer = severity_analyzer or SeverityAnalyzer()
        self.urgency_detector = urgency_detector or UrgencyDetector()
        self.recommendation_engine = recommendation_engine or RecommendationEngine()

    def analyze_issue(
        self,
        title: Optional[str] = None,
        body: Optional[str] = None,
        text: Optional[str] = None,
        issue_id: Optional[Union[int, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TriageResult:
        """
        Analyzes a single GitHub issue end-to-end.

        Args:
            title: Issue title string.
            body: Issue body string.
            text: Combined raw text (if title/body not provided separately).
            issue_id: Optional GitHub issue ID or number.
            metadata: Optional dictionary with repository, author, comments, labels.

        Returns:
            TriageResult: Structured, validated triage output.
        """
        t = str(title) if title is not None else ""
        b = str(body) if body is not None else ""

        if not t and text:
            t = text.split("\n")[0][:100]
            b = text

        # 1. Structural features
        structural = extract_structural_features(t, b)

        # 2. ML Prediction (Category, Confidence, Probabilities, Top Keywords)
        pred = self.predictor.predict(title=t, body=b)
        cat = pred["predicted_label"]
        conf = float(pred["confidence"])
        is_fallback = pred.get("is_fallback", False)

        # Confidence status calculation
        if is_fallback or conf < 0.30:
            conf_status = "Low"
        elif conf < 0.55:
            conf_status = "Medium"
        else:
            conf_status = "High"

        # Alternative predictions sorted by probability
        alt_preds: List[AlternativePrediction] = []
        for c, p in sorted(pred.get("probabilities", {}).items(), key=lambda x: x[1], reverse=True):
            if c != cat and p > 0.05:
                alt_preds.append(AlternativePrediction(
                    category=c,
                    probability=round(p, 4),
                    percentage=f"{p * 100:.1f}%"
                ))

        # 3. Rule-Based Priority Scoring
        priority = self.priority_scorer.calculate_priority(
            title=t,
            body=b,
            predicted_category=cat,
            structural_features=structural,
            metadata=metadata
        )

        # 4. Severity Assessment
        severity = self.severity_analyzer.evaluate_severity(
            title=t,
            body=b,
            priority_score=priority.score,
            predicted_category=cat
        )

        # 5. Urgency Detection
        urgency = self.urgency_detector.detect_urgency(
            title=t,
            body=b,
            priority_level=priority.level
        )

        # 6. Action Recommendations
        recs = self.recommendation_engine.generate_recommendations(
            predicted_category=cat,
            priority_level=priority.level,
            severity_level=severity.level,
            urgency_level=urgency.level,
            structural_features=structural,
            priority_signals=priority.signals_detected
        )

        # 7. Explainability Details
        top_kws = [[kw, score] for kw, score in pred.get("top_keywords", [])]
        kw_str = ", ".join([f"'{k}'" for k, _ in top_kws[:3]]) if top_kws else "generic patterns"
        cat_reasoning = (
            f"Classified as '{cat}' with {conf * 100:.1f}% confidence based on textual alignment "
            f"and key n-grams including {kw_str}."
        )

        explanation = ExplanationDetails(
            category_reasoning=cat_reasoning,
            top_keywords=top_kws,
            priority_reasoning=f"Calculated heuristic priority {priority.level} ({priority.score}/100) based on {len(priority.signals_detected)} signals.",
            severity_reasoning=f"Evaluated blast radius as '{severity.level}': availability={severity.factors.get('availability')}, data={severity.factors.get('data_integrity')}.",
            urgency_reasoning=urgency.reasoning,
            limitations=[
                "Category prediction is generated by an NLP classifier trained on open-source GitHub issues; unusual domain syntax may affect accuracy.",
                "Priority and severity scores are rule-based heuristic assessments intended for maintainer triage guidance, not verified ground truth labels."
            ]
        )

        model_name = self.predictor.metadata.get("model_name", "Complement Naive Bayes")
        model_version = self.predictor.metadata.get("version", settings.VERSION)

        return TriageResult(
            issue_id=issue_id,
            title=t,
            predicted_category=cat,
            confidence=conf,
            confidence_status=conf_status,
            alternative_predictions=alt_preds,
            priority=priority,
            severity=severity,
            urgency=urgency,
            recommendations=recs,
            explanation=explanation,
            model_name=model_name,
            model_version=model_version,
            is_fallback=is_fallback,
            metadata=metadata or {}
        )

    def analyze_batch(self, issues: List[Dict[str, Any]]) -> List[TriageResult]:
        """
        Analyzes a batch of issue dictionaries without reloading model artifacts.

        Args:
            issues: List of dictionaries containing 'title', 'body', optional 'id', 'metadata'.

        Returns:
            List[TriageResult]: List of processed triage results.
        """
        results: List[TriageResult] = []
        for item in issues:
            res = self.analyze_issue(
                title=item.get("title", ""),
                body=item.get("body", ""),
                text=item.get("text"),
                issue_id=item.get("id") or item.get("number"),
                metadata=item.get("metadata")
            )
            results.append(res)
        logger.info(f"Completed batch triage for {len(results)} issues.")
        return results
