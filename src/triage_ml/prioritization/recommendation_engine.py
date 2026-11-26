"""
Action Recommendation Engine.
Produces actionable, context-aware engineering recommendations for maintainers
based on issue category, severity, urgency, and technical content.
"""
from typing import Any, Dict, List, Optional
from src.triage_ml.utils.logger import get_logger

logger = get_logger("recommendation_engine")


class RecommendationEngine:
    """Generates specific, non-generic maintainer next steps."""

    def generate_recommendations(
        self,
        predicted_category: str,
        priority_level: str,
        severity_level: str,
        urgency_level: str,
        structural_features: Dict[str, Any],
        priority_signals: List[str]
    ) -> List[str]:
        """
        Synthesizes maintainer action recommendations.

        Args:
            predicted_category: Category from ML model ('Bug', 'Feature', etc.).
            priority_level: Assigned priority tier ('P0', 'P1', etc.).
            severity_level: Severity tier ('Critical', 'High', etc.).
            urgency_level: Urgency tier ('Immediate', 'High', etc.).
            structural_features: Metadata (has_code_block, has_stack_trace).
            priority_signals: List of signals detected by PriorityScorer.

        Returns:
            List[str]: Formatted action items.
        """
        recs: List[str] = []

        # 1. P0 / Immediate Incident Handling
        if priority_level == "P0" or urgency_level == "Immediate":
            recs.append("🚨 [IMMEDIATE TRIAGE]: Convene on-call maintainers; assess need for rollback or emergency hotfix.")

        # 2. Security Handling
        if any("security" in s.lower() or "vulnerability" in s.lower() for s in priority_signals):
            recs.append("🔒 [SECURITY PROTOCOL]: Do not discuss reproduction payloads publicly. Route to Security Response Team under private advisory.")

        # 3. Bug-Specific Diagnostic Checks
        if predicted_category == "Bug":
            if not structural_features.get("has_stack_trace") and not structural_features.get("has_code_block"):
                recs.append("📋 [REPRODUCTION NEEDED]: Request minimal reproducible code snippet (MRE) and execution environment details (OS, Python version, dependencies).")
            elif structural_features.get("has_stack_trace"):
                recs.append("🔍 [STACK TRACE]: Cross-reference reported traceback against master branch line numbers to identify invalid state or unhandled exception.")

            if any("regression" in s.lower() for s in priority_signals):
                recs.append("⏪ [REGRESSION CHECK]: Inspect recent pull requests and git commit log; consider git bisect between current and prior release tag.")

        # 4. Feature-Specific Guidance
        elif predicted_category == "Feature":
            recs.append("💡 [ROADMAP REVIEW]: Evaluate proposal against project scope, architectural roadmap, and backward-compatibility guidelines.")
            recs.append("👥 [COMMUNITY RFC]: Solicit community input and design specifications before accepting pull requests.")

        # 5. Documentation-Specific Guidance
        elif predicted_category == "Documentation":
            recs.append("📖 [DOC UPDATE]: Verify reported inaccuracy or broken link against current API specification.")
            recs.append("🏷️ [COMMUNITY CONTRIBUTION]: Mark issue as 'good first issue' / 'documentation' for external contributors.")

        # 6. Question-Specific Guidance
        elif predicted_category == "Question":
            recs.append("💬 [SUPPORT REDIRECT]: Direct user to GitHub Discussions, Stack Overflow, or community Discord channel.")
            recs.append("📚 [DOC GAP CHECK]: If this question recurs frequently, consider adding an explicit FAQ section in documentation.")

        # 7. Workaround Verification
        if any("workaround" in s.lower() for s in priority_signals):
            recs.append("💡 [WORKAROUND PIN]: Pin the reported temporary workaround in issue comments to help other impacted users.")

        # Ensure at least one concrete recommendation exists
        if not recs:
            recs.append("📌 [STANDARD TRIAGE]: Assign to component maintainer and label for review during next sprint planning.")

        return recs
