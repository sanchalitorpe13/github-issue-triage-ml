"""
Severity Assessment Analyzer.
Evaluates technical blast radius across 5 dimensions:
Functionality, Availability, Data Integrity, Security Risk, and Workarounds.
"""
from typing import Any, Dict, List, Optional
from src.triage_ml.prioritization.schema import SeverityAssessment
from src.triage_ml.utils.logger import get_logger

logger = get_logger("severity_analyzer")


class SeverityAnalyzer:
    """Evaluates the structural impact and blast radius of an issue."""

    def evaluate_severity(
        self,
        title: str,
        body: str,
        priority_score: int,
        predicted_category: str
    ) -> SeverityAssessment:
        """
        Assesses technical severity based on multidimensional impact signals.

        Args:
            title: Issue title string.
            body: Issue body string.
            priority_score: Priority score calculated by PriorityScorer.
            predicted_category: Category from ML classifier.

        Returns:
            SeverityAssessment: Multidimensional severity evaluation.
        """
        combined = f"{title} {body}".lower()
        factors: Dict[str, str] = {}
        reasoning: List[str] = []

        # 1. System Availability
        if any(term in combined for term in ["outage", "service down", "crash", "segfault", "panic"]):
            factors["availability"] = "Down / Crash"
            reasoning.append("System execution halted or crashing.")
        elif any(term in combined for term in ["timeout", "hang", "leak", "slow"]):
            factors["availability"] = "Degraded"
            reasoning.append("Performance degradation or thread lockup.")
        else:
            factors["availability"] = "Operational"

        # 2. Data Integrity
        if any(term in combined for term in ["data loss", "corrupt", "deleted data"]):
            factors["data_integrity"] = "Corrupted / Lost"
            reasoning.append("Irreversible data corruption or loss.")
        elif "inconsistent" in combined:
            factors["data_integrity"] = "Inconsistent State"
        else:
            factors["data_integrity"] = "Preserved"

        # 3. Security
        if any(term in combined for term in ["cve", "exploit", "rce", "auth bypass", "vulnerability"]):
            factors["security"] = "Active Vulnerability / Exploit"
            reasoning.append("Potential security vulnerability identified.")
        else:
            factors["security"] = "Standard / None"

        # 4. Functionality
        if any(term in combined for term in ["cannot start", "fails to build", "unhandled exception", "blocking"]):
            factors["functionality"] = "Core Functionality Blocked"
            reasoning.append("Core capability completely broken.")
        elif predicted_category in ["Documentation", "Question"]:
            factors["functionality"] = "Non-blocking / Informational"
        else:
            factors["functionality"] = "Partial Impairment"

        # 5. Workaround
        if any(term in combined for term in ["workaround:", "can be bypassed", "temporary workaround"]):
            factors["workaround"] = "Available"
            reasoning.append("User documented a functional workaround.")
        else:
            factors["workaround"] = "None Documented"

        # Overall Level Determination
        if (
            factors["availability"] == "Down / Crash"
            or factors["data_integrity"] == "Corrupted / Lost"
            or factors["security"] == "Active Vulnerability / Exploit"
            or priority_score >= 80
        ):
            level = "Critical"
        elif factors["functionality"] == "Core Functionality Blocked" or priority_score >= 60:
            level = "High"
        elif priority_score >= 35 or predicted_category == "Bug":
            level = "Medium"
        else:
            level = "Low"

        reasoning.append(f"Overall severity classified as '{level}' based on blast-radius assessment.")

        return SeverityAssessment(
            level=level,
            factors=factors,
            reasoning=reasoning
        )
