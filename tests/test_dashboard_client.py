"""
Phase 6 Dashboard Client & Session Logic Unit Tests.
Verifies APIClient request dispatching, session history storage,
deduplication logic, chart builders, and analytics calculations with mocked backend responses.
"""
import unittest
from unittest.mock import MagicMock, patch
from app.client import APIClientError, TriageAPIClient
from app.state import add_to_session_history, clear_session_history
from app.ui.charts import (
    create_category_donut_chart,
    create_priority_bar_chart,
    create_confidence_distribution_chart,
)
import altair as alt
import streamlit as st


class TestPhase6DashboardClient(unittest.TestCase):
    """Test suite for Streamlit API client, session management, and UI chart builders."""

    def setUp(self):
        self.client = TriageAPIClient(base_url="http://mock-backend:8000")
        # Initialize mock streamlit session_state
        if not hasattr(st, "session_state") or not isinstance(st.session_state, dict):
            st.session_state = {}
        st.session_state["session_history"] = []
        st.session_state["last_analyzed_hash"] = None
        st.session_state["api_url"] = "http://mock-backend:8000"
        st.session_state["api_timeout"] = 10

    @patch("requests.Session.get")
    def test_check_health_success(self, mock_get):
        """Verify APIClient parses healthy backend response."""
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = {
            "status": "healthy",
            "model_loaded": True,
            "triage_engine_status": "operational",
            "supported_classes": ["Bug", "Feature", "Docs", "Question"]
        }
        mock_get.return_value = mock_res

        data = self.client.check_health()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])

    @patch("requests.Session.get")
    def test_check_health_failure_raises_apierror(self, mock_get):
        """Verify connection errors raise APIClientError."""
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        with self.assertRaises(APIClientError) as ctx:
            self.client.check_health()
        self.assertEqual(ctx.exception.status_code, 503)

    @patch("requests.Session.get")
    def test_get_model_info_success(self, mock_get):
        """Verify model info parsing."""
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = {
            "model_name": "Complement Naive Bayes",
            "version": "2.0.0",
            "classes": ["Bug", "Feature", "Documentation", "Question"],
            "model_loaded": True
        }
        mock_get.return_value = mock_res

        info = self.client.get_model_info()
        self.assertEqual(info["model_name"], "Complement Naive Bayes")
        self.assertTrue(info["model_loaded"])

    @patch("requests.Session.post")
    def test_triage_issue_success(self, mock_post):
        """Verify successful single issue submission and parsing."""
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = {
            "issue_id": 1001,
            "title": "Crash in op",
            "predicted_category": "Bug",
            "confidence": 0.85,
            "priority": {"level": "P0", "score": 90, "method": "rule_based"}
        }
        mock_post.return_value = mock_res

        res = self.client.triage_issue(title="Crash in op", body="Fatal crash traceback")
        self.assertEqual(res["predicted_category"], "Bug")
        self.assertEqual(res["priority"]["level"], "P0")

    def test_session_history_management_and_deduplication(self):
        """Verify session state stores items and avoids duplicate appends."""
        item1 = {
            "title": "Issue A",
            "analyzed_at": "2026-11-26T10:00:00Z",
            "predicted_category": "Bug",
            "priority": {"level": "P0"}
        }
        item2 = {
            "title": "Issue B",
            "analyzed_at": "2026-11-26T10:05:00Z",
            "predicted_category": "Feature",
            "priority": {"level": "P2"}
        }

        add_to_session_history(item1)
        self.assertEqual(len(st.session_state["session_history"]), 1)

        # Duplicate append with same title and timestamp should NOT add second copy
        add_to_session_history(item1)
        self.assertEqual(len(st.session_state["session_history"]), 1)

        # Distinct item
        add_to_session_history(item2)
        self.assertEqual(len(st.session_state["session_history"]), 2)

        # Clear history
        clear_session_history()
        self.assertEqual(len(st.session_state["session_history"]), 0)

    def test_chart_generation(self):
        """Verify Altair charts generate valid chart specifications."""
        sample_history = [
            {
                "issue_id": 1,
                "title": "Crash on login",
                "predicted_category": "Bug",
                "confidence": 0.92,
                "priority": {"level": "P0", "score": 95}
            },
            {
                "issue_id": 2,
                "title": "Add dark mode",
                "predicted_category": "Feature",
                "confidence": 0.84,
                "priority": {"level": "P2", "score": 50}
            }
        ]

        donut = create_category_donut_chart(sample_history)
        self.assertIsInstance(donut, (alt.Chart, alt.TopLevelMixin))

        bar = create_priority_bar_chart(sample_history)
        self.assertIsInstance(bar, (alt.Chart, alt.LayerChart, alt.TopLevelMixin))

        hist = create_confidence_distribution_chart(sample_history)
        self.assertIsInstance(hist, (alt.Chart, alt.TopLevelMixin))


if __name__ == "__main__":
    unittest.main()
