"""
Intelligent Priority Scoring Engine.
Calculates a transparent 0-100 severity score based on configurable domain rules.
Includes robust negation detection and contextual signal dampening.
Clearly discloses rule-based heuristic methodology.
"""
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from config.settings import settings
from src.triage_ml.prioritization.schema import PriorityAssessment
from src.triage_ml.utils.logger import get_logger

logger = get_logger("priority_scorer")


class PriorityScorer:
    """Configurable, rule-based priority scoring engine with negation awareness."""

    def __init__(self, custom_weights: Optional[Dict[str, int]] = None):
        self.weights = custom_weights or getattr(settings, "PRIORITY_WEIGHTS", {
            "critical_failure": 35,
            "security_vulnerability": 35,
            "data_loss": 30,
            "production_outage": 30,
            "regression": 20,
            "major_breakage": 20,
            "performance_leak": 15,
            "has_stack_trace": 10,
            "workaround_available": -15,
            "feature_penalty": -15,
            "doc_typo_penalty": -25
        })

    def _is_negated(self, text: str, keyword: str) -> bool:
        """
        Detects if a keyword is preceded by a negation token within a 4-word window.
        E.g., 'not a crash', 'no data loss', 'without any segmentation fault'.
        """
        text_lower = text.lower()
        # Regex searching for negation within 1-3 words before the target keyword
        pattern = rf"\b(not|no|never|isn't|is not|without|doesn't|does not|avoid|avoiding|prevent|fixed|unrelated to)\b(?:\s+\w+){{0,3}}\s+\b{re.escape(keyword.lower())}\b"
        return bool(re.search(pattern, text_lower))

    def calculate_priority(
        self,
        title: str,
        body: str,
        predicted_category: str,
        structural_features: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PriorityAssessment:
        """
        Calculates transparent priority score (0-100) and maps to P0-P4 tiers.

        Args:
            title: Issue title string.
            body: Issue body string.
            predicted_category: Category from ML classifier ('Bug', 'Feature', etc.).
            structural_features: Metadata dictionary (e.g. has_code_block, has_stack_trace).
            metadata: Optional GitHub issue metadata (labels, comments, author).

        Returns:
            PriorityAssessment: Structured priority result with step-by-step reasoning.
        """
        combined_text = f"{title} {body}".lower()
        score = 25  # Base default score for an unclassified developer issue
        signals_detected: List[str] = []
        reasoning: List[str] = [f"Base starting score: 25 pts (neutral baseline)."]

        # 1. Critical Failures (crash, segfault, core dump, panic)
        crash_words = ["segmentation fault", "segfault", "core dump", "core dumped", "fatal error", "kernel panic", "fatal crash"]
        crash_detected = False
        for cw in crash_words:
            if cw in combined_text:
                if self._is_negated(combined_text, cw):
                    reasoning.append(f"Detected '{cw}', but suppressed due to preceding negation ('not/no').")
                else:
                    crash_detected = True
                    signals_detected.append(f"Critical failure: {cw}")
                    break
        if crash_detected:
            pts = self.weights.get("critical_failure", 35)
            score += pts
            reasoning.append(f"Critical failure detected (+{pts} pts).")

        # 2. Security Vulnerabilities (CVE, exploit, vulnerability, auth bypass)
        sec_words = ["vulnerability", "exploit", "cve", "security advisory", "denial of service", "rce", "remote code execution", "auth bypass"]
        sec_detected = False
        for sw in sec_words:
            if sw in combined_text:
                if self._is_negated(combined_text, sw):
                    reasoning.append(f"Mentioned security keyword '{sw}', but in negated context.")
                else:
                    sec_detected = True
                    signals_detected.append(f"Security concern: {sw}")
                    break
        if sec_detected:
            pts = self.weights.get("security_vulnerability", 35)
            score += pts
            reasoning.append(f"Security vulnerability or exploit signal (+{pts} pts).")

        # 3. Data Loss & Corruption
        dl_words = ["data loss", "corrupt", "corruption", "database corruption", "deleted data", "dropped table"]
        dl_detected = False
        for dw in dl_words:
            if dw in combined_text:
                if self._is_negated(combined_text, dw):
                    reasoning.append(f"Mentioned '{dw}', but in negated context.")
                else:
                    dl_detected = True
                    signals_detected.append(f"Data integrity hazard: {dw}")
                    break
        if dl_detected:
            pts = self.weights.get("data_loss", 30)
            score += pts
            reasoning.append(f"Data loss or storage corruption risk (+{pts} pts).")

        # 4. Production Outage
        prod_words = [
            "production outage", "broken in production", "service down", "outage",
            "production down", "500 internal server error", "in production", "production server"
        ]
        prod_detected = False
        for pw in prod_words:
            if pw in combined_text:
                if not self._is_negated(combined_text, pw):
                    prod_detected = True
                    signals_detected.append(f"Production impact: {pw}")
                    break
        if prod_detected:
            pts = self.weights.get("production_outage", 30)
            score += pts
            reasoning.append(f"Production environment impact reported (+{pts} pts).")

        # 5. Regressions
        reg_words = ["regression", "broke in version", "worked in previous", "broken after upgrade", "introduced in v"]
        reg_detected = any(rw in combined_text and not self._is_negated(combined_text, rw) for rw in reg_words)
        if reg_detected:
            pts = self.weights.get("regression", 20)
            score += pts
            signals_detected.append("Regression in recent release")
            reasoning.append(f"Regression breaking working functionality (+{pts} pts).")

        # 6. Major Breakage / Denial of Service
        breakage_words = ["cannot start", "fails to build", "unhandled exception", "blocking", "deadlock", "denial of service", "lockup", "infinite loop"]
        breakage_detected = any(bw in combined_text and not self._is_negated(combined_text, bw) for bw in breakage_words)
        if breakage_detected:
            pts = self.weights.get("major_breakage", 20)
            score += pts
            signals_detected.append("Major functionality breakage / blocking")
            reasoning.append(f"Major functionality breakage or blocking defect (+{pts} pts).")

        # 6. Stack Trace Confirmation
        has_stack_trace = False
        if structural_features and structural_features.get("has_stack_trace"):
            has_stack_trace = True
        elif "traceback (most recent call last)" in combined_text or "unhandled exception" in combined_text:
            has_stack_trace = True

        if has_stack_trace:
            pts = self.weights.get("has_stack_trace", 10)
            score += pts
            signals_detected.append("Traceback / stack trace verified")
            reasoning.append(f"Verified stack trace / exception details present (+{pts} pts).")

        # 7. Workaround Mitigation (Reduces Priority)
        workaround_patterns = ["workaround:", "temporary workaround", "found a workaround", "can be bypassed by", "mitigation:"]
        workaround_detected = any(wp in combined_text for wp in workaround_patterns)
        if workaround_detected:
            pts = self.weights.get("workaround_available", -15)
            score += pts
            signals_detected.append("Workaround available")
            reasoning.append(f"Workaround documented by user ({pts} pts mitigation).")

        # 8. Category Dampeners
        if predicted_category == "Documentation":
            pts = self.weights.get("doc_typo_penalty", -25)
            score += pts
            reasoning.append(f"Category is Documentation / Typo ({pts} pts).")
        elif predicted_category == "Feature":
            pts = self.weights.get("feature_penalty", -15)
            score += pts
            reasoning.append(f"Category is Feature Request ({pts} pts non-outage adjustment).")

        # Clamp score between 0 and 100
        final_score = int(min(100, max(0, score)))

        # Determine Tier
        if final_score >= 80:
            level = "P0"
            name = "Critical"
            sla_hours = 2
        elif final_score >= 60:
            level = "P1"
            name = "High"
            sla_hours = 8
        elif final_score >= 40:
            level = "P2"
            name = "Medium"
            sla_hours = 24
        elif final_score >= 20:
            level = "P3"
            name = "Low"
            sla_hours = 72
        else:
            level = "P4"
            name = "Trivial"
            sla_hours = 168

        reasoning.append(f"Final calculated score: {final_score}/100 -> Priority Tier: {level} ({name}), Target SLA: {sla_hours}h.")

        return PriorityAssessment(
            level=level,
            name=name,
            score=final_score,
            method="rule_based",
            sla_hours=sla_hours,
            signals_detected=signals_detected,
            reasoning=reasoning,
            is_heuristic=True
        )
