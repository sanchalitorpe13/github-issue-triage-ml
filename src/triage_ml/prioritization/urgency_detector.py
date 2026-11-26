"""
Operational Urgency Detector.
Detects time-sensitive factors, production incidents, and regression deadlines.
"""
from typing import Any, Dict, List, Optional
from src.triage_ml.prioritization.schema import UrgencyAssessment
from src.triage_ml.utils.logger import get_logger

logger = get_logger("urgency_detector")


class UrgencyDetector:
    """Detects immediate operational time sensitivity in issue text and metadata."""

    def detect_urgency(
        self,
        title: str,
        body: str,
        priority_level: str
    ) -> UrgencyAssessment:
        """
        Evaluates operational urgency level and explicit deadlines.

        Args:
            title: Issue title string.
            body: Issue body string.
            priority_level: Priority tier ('P0', 'P1', etc.).

        Returns:
            UrgencyAssessment: Detected signals and operational urgency tier.
        """
        combined = f"{title} {body}".lower()
        signals: List[str] = []

        # Urgent phrases
        urgent_triggers = [
            ("production outage", "Active production outage in progress"),
            ("blocking release", "Impending software release blocked"),
            ("urgent", "Explicit urgent flag in description"),
            ("asap", "Explicit ASAP request from user"),
            ("deadline", "Time-sensitive deadline cited"),
            ("regression", "Regression in recent patch release"),
            ("broken in prod", "Production deployment disrupted"),
            ("critical vulnerability", "Critical security vulnerability disclosed"),
            ("cve", "Security vulnerability advisory (CVE)"),
            ("denial of service", "Denial of service hazard"),
            ("in production", "Production environment impacted")
        ]

        for trigger, desc in urgent_triggers:
            if trigger in combined:
                signals.append(desc)

        # Assign urgency level
        if (
            priority_level == "P0"
            or "Active production outage in progress" in signals
            or any("cve" in s.lower() for s in signals)
            or any("denial of service" in s.lower() for s in signals)
        ):
            level = "Immediate"
            reasoning = "Immediate attention required: active production disruption, crash, or critical security hazard."
        elif priority_level == "P1" or len(signals) >= 2:
            level = "High"
            reasoning = "High time sensitivity: major regression or release-blocking functionality defect."
        elif priority_level in ["P2", "P3"] and len(signals) > 0:
            level = "Medium"
            reasoning = "Standard urgency: issue contains operational concerns but is not currently causing total outage."
        else:
            level = "Low"
            reasoning = "Normal queue: routine development item, documentation change, or feature proposal."

        return UrgencyAssessment(
            level=level,
            signals=signals,
            reasoning=reasoning
        )
