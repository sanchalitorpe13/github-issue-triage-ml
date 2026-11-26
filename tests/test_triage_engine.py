"""
Phase 4 Intelligent Triage Engine Unit Tests.
Tests priority scoring, negation handling, severity evaluation, urgency detection,
recommendation generation, schema serialization, batch processing, and edge cases.
"""
import unittest
from src.triage_ml.prioritization.schema import TriageResult
from src.triage_ml.prioritization.priority_scorer import PriorityScorer
from src.triage_ml.prioritization.severity_analyzer import SeverityAnalyzer
from src.triage_ml.prioritization.urgency_detector import UrgencyDetector
from src.triage_ml.prioritization.recommendation_engine import RecommendationEngine
from src.triage_ml.prioritization.triage_engine import TriageEngine


class TestPhase4TriageEngine(unittest.TestCase):
    """Test suite for Phase 4 triage components."""

    def setUp(self):
        self.priority_scorer = PriorityScorer()
        self.severity_analyzer = SeverityAnalyzer()
        self.urgency_detector = UrgencyDetector()
        self.rec_engine = RecommendationEngine()
        self.engine = TriageEngine()

    def test_priority_critical_failure_scoring(self):
        """Verifies high score and P0 classification for crash and core dump."""
        title = "Fatal crash: segmentation fault in QuantizedConv2D"
        body = "Calling the op results in a core dumped segmentation fault in production."
        res = self.priority_scorer.calculate_priority(title, body, predicted_category="Bug")

        self.assertGreaterEqual(res.score, 80)
        self.assertEqual(res.level, "P0")
        self.assertEqual(res.name, "Critical")
        self.assertEqual(res.sla_hours, 2)
        self.assertTrue(res.is_heuristic)
        self.assertEqual(res.method, "rule_based")

    def test_priority_negation_handling(self):
        """Verifies that negated signals ('not a crash', 'no data loss') are NOT awarded points."""
        title = "Clarification on model export behavior"
        body = "This is not a crash and there is no data loss, just asking about config parameter."

        res_negated = self.priority_scorer.calculate_priority(title, body, predicted_category="Question")
        # Should not trigger critical failure or data loss points
        self.assertLess(res_negated.score, 60)
        self.assertIn(res_negated.level, ["P2", "P3", "P4"])
        # Check that reasoning mentions negation suppression
        reasoning_str = " ".join(res_negated.reasoning)
        self.assertTrue("negat" in reasoning_str.lower() or "suppress" in reasoning_str.lower())

    def test_priority_mitigation_by_workaround(self):
        """Verifies that a documented workaround reduces the priority score."""
        title = "Failure in JSON export"
        body_without_workaround = "Exporting JSON raises unhandled exception in background worker."
        body_with_workaround = "Exporting JSON raises unhandled exception. Temporary workaround: use dict serializer."

        score_without = self.priority_scorer.calculate_priority(title, body_without_workaround, "Bug").score
        score_with = self.priority_scorer.calculate_priority(title, body_with_workaround, "Bug").score

        self.assertLess(score_with, score_without)

    def test_severity_analysis_levels(self):
        """Verifies severity assignment based on blast radius."""
        # Critical crash
        crit_sev = self.severity_analyzer.evaluate_severity(
            title="Segmentation fault in production",
            body="Service down",
            priority_score=85,
            predicted_category="Bug"
        )
        self.assertEqual(crit_sev.level, "Critical")
        self.assertEqual(crit_sev.factors["availability"], "Down / Crash")

        # Low typo
        low_sev = self.severity_analyzer.evaluate_severity(
            title="Typo in README.md",
            body="Minor spelling mistake",
            priority_score=15,
            predicted_category="Documentation"
        )
        self.assertEqual(low_sev.level, "Low")

    def test_urgency_detection(self):
        """Verifies operational urgency detection."""
        imm_urg = self.urgency_detector.detect_urgency(
            title="Production outage in cluster",
            body="Broken in prod after v2.0 upgrade",
            priority_level="P0"
        )
        self.assertEqual(imm_urg.level, "Immediate")
        self.assertGreater(len(imm_urg.signals), 0)

        low_urg = self.urgency_detector.detect_urgency(
            title="Nice to have feature idea",
            body="Consider adding support for dark mode someday",
            priority_level="P4"
        )
        self.assertEqual(low_urg.level, "Low")

    def test_contextual_recommendations(self):
        """Verifies dynamic maintainer recommendations for bug without MRE vs security issue."""
        # Bug without stack trace
        recs_bug = self.rec_engine.generate_recommendations(
            predicted_category="Bug",
            priority_level="P2",
            severity_level="Medium",
            urgency_level="Medium",
            structural_features={"has_code_block": 0, "has_stack_trace": 0},
            priority_signals=[]
        )
        self.assertTrue(any("reproduc" in r.lower() for r in recs_bug))

        # Security issue
        recs_sec = self.rec_engine.generate_recommendations(
            predicted_category="Bug",
            priority_level="P0",
            severity_level="Critical",
            urgency_level="Immediate",
            structural_features={"has_code_block": 1, "has_stack_trace": 1},
            priority_signals=["Security concern: cve"]
        )
        self.assertTrue(any("security" in r.lower() for r in recs_sec))
        self.assertTrue(any("immediate" in r.lower() for r in recs_sec))

    def test_triage_engine_end_to_end_analysis(self):
        """Verifies full end-to-end analysis produces validated TriageResult."""
        result = self.engine.analyze_issue(
            title="CVE vulnerability: Remote denial of service via multipart form",
            body="Sending malformed boundary headers causes 100% CPU lockup in production server.",
            issue_id=404
        )

        self.assertIsInstance(result, TriageResult)
        self.assertEqual(result.issue_id, 404)
        self.assertIn(result.predicted_category, ["Bug", "Feature", "Documentation", "Question"])
        self.assertGreater(result.confidence, 0.0)
        self.assertIn(result.priority.level, ["P0", "P1"])
        self.assertEqual(result.priority.method, "rule_based")
        self.assertTrue(result.priority.is_heuristic)
        self.assertEqual(result.severity.level, "Critical")
        self.assertEqual(result.urgency.level, "Immediate")
        self.assertTrue(len(result.recommendations) > 0)
        self.assertTrue(len(result.explanation.limitations) > 0)

        # Test dictionary serialization
        result_dict = result.model_dump()
        self.assertEqual(result_dict["issue_id"], 404)
        self.assertIn("priority", result_dict)
        self.assertIn("severity", result_dict)

    def test_triage_engine_empty_input_fallback(self):
        """Verifies graceful handling of empty or blank issue inputs."""
        fallback_res = self.engine.analyze_issue(title="", body="", issue_id=999)
        self.assertTrue(fallback_res.is_fallback)
        self.assertEqual(fallback_res.predicted_category, "Unlabelled")
        self.assertEqual(fallback_res.confidence, 0.0)
        self.assertEqual(fallback_res.confidence_status, "Low")

    def test_triage_engine_batch_processing(self):
        """Verifies batch analysis processing without reloading model."""
        issues_batch = [
            {"id": 1, "title": "Deadlock in threadpool", "body": "Fatal crash deadlock"},
            {"id": 2, "title": "Add dark mode theme", "body": "Feature proposal for dark theme"},
            {"id": 3, "title": "Typo in installation docs", "body": "Spelling typo on line 42"}
        ]
        results = self.engine.analyze_batch(issues_batch)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].issue_id, 1)
        self.assertEqual(results[1].issue_id, 2)
        self.assertEqual(results[2].issue_id, 3)


if __name__ == "__main__":
    unittest.main()
