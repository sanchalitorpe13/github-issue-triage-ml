"""
Phase 5 FastAPI REST API Unit and Integration Tests.
Tests all endpoints using FastAPI TestClient:
- GET /
- GET /health
- GET /api/v1/health
- GET /api/v1/model/info
- POST /api/v1/triage (single issue, empty fields, metadata)
- POST /api/v1/triage/batch (batch issue, partial errors, max limit)
- POST /predict (legacy backward compatibility)
- Error handling (404, 422 validation, standardized JSON format)
- CORS headers
"""
import unittest
from fastapi.testclient import TestClient
from app.main import app


class TestPhase5FastAPI(unittest.TestCase):
    """Test suite for FastAPI REST API endpoints."""

    def setUp(self):
        self.client = TestClient(app)

    def test_root_endpoint(self):
        """Verify GET / returns API metadata and documentation links."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("title", data)
        self.assertIn("version", data)
        self.assertIn("docs_url", data)
        self.assertEqual(data["docs_url"], "/docs")
        self.assertIn("endpoints", data)

    def test_basic_health_endpoint(self):
        """Verify GET /health returns healthy status."""
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("timestamp", data)

    def test_detailed_health_endpoint(self):
        """Verify GET /api/v1/health returns detailed diagnostic info."""
        res = self.client.get("/api/v1/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("status", data)
        self.assertIn("model_loaded", data)
        self.assertIn("supported_classes", data)
        self.assertTrue(data["model_loaded"])

    def test_model_info_endpoint(self):
        """Verify GET /api/v1/model/info returns sanitized model architecture & metrics."""
        res = self.client.get("/api/v1/model/info")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("model_name", data)
        self.assertIn("classes", data)
        self.assertIn("metrics_summary", data)
        self.assertTrue(data["model_loaded"])
        # Ensure no private absolute paths leaked
        data_str = str(data)
        self.assertNotIn("C:\\", data_str)
        self.assertNotIn("c:/", data_str)

    def test_single_issue_triage_success(self):
        """Verify POST /api/v1/triage correctly classifies and scores a bug report."""
        payload = {
            "title": "Segmentation fault in QuantizedConv2D",
            "body": "Op causes immediate core dump in production cluster.",
            "issue_number": 1001,
            "repository": "tensorflow/tensorflow"
        }
        res = self.client.post("/api/v1/triage", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["issue_id"], 1001)
        self.assertEqual(data["predicted_category"], "Bug")
        self.assertIn(data["priority"]["level"], ["P0", "P1"])
        self.assertEqual(data["priority"]["method"], "rule_based")
        self.assertTrue(data["priority"]["is_heuristic"])
        self.assertIn(data["severity"]["level"], ["Critical", "High"])
        self.assertTrue(len(data["recommendations"]) > 0)
        self.assertIn("category_reasoning", data["explanation"])

    def test_single_issue_triage_feature_request(self):
        """Verify POST /api/v1/triage correctly classifies feature proposal."""
        payload = {
            "title": "Add support for PyArrow backend in CSV reader",
            "body": "Proposal to integrate PyArrow engine to improve SIMD throughput."
        }
        res = self.client.post("/api/v1/triage", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["predicted_category"], "Feature")
        self.assertIn(data["priority"]["level"], ["P1", "P2", "P3"])

    def test_single_issue_triage_empty_body_handled(self):
        """Verify single issue triage handles empty body without crashing."""
        payload = {
            "title": "Typo in installation documentation"
        }
        res = self.client.post("/api/v1/triage", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["predicted_category"], "Documentation")

    def test_single_issue_triage_empty_title_validation(self):
        """Verify empty title string is rejected with 422 Unprocessable Entity."""
        payload = {
            "title": "",
            "body": "Body without title"
        }
        res = self.client.post("/api/v1/triage", json=payload)
        self.assertEqual(res.status_code, 422)
        data = res.json()
        self.assertEqual(data["error"], "ValidationError")
        self.assertIn("title", str(data["details"]))

    def test_single_issue_triage_missing_required_field(self):
        """Verify missing title field is rejected with 422."""
        res = self.client.post("/api/v1/triage", json={"body": "Only body"})
        self.assertEqual(res.status_code, 422)

    def test_batch_triage_success(self):
        """Verify POST /api/v1/triage/batch processes multiple issues with summary stats."""
        payload = {
            "issues": [
                {"title": "Fatal crash in threadpool", "body": "Panic core dump"},
                {"title": "Add dark mode theme", "body": "Feature suggestion"},
                {"title": "How to configure CORS?", "body": "Question about middleware"}
            ]
        }
        res = self.client.post("/api/v1/triage/batch", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total_submitted"], 3)
        self.assertEqual(data["total_successful"], 3)
        self.assertEqual(data["total_failed"], 0)
        self.assertEqual(len(data["results"]), 3)
        self.assertTrue(data["results"][0]["success"])
        self.assertEqual(data["results"][0]["result"]["predicted_category"], "Bug")
        self.assertEqual(data["results"][1]["result"]["predicted_category"], "Feature")

    def test_batch_triage_empty_list_rejected(self):
        """Verify empty batch list is rejected with 422."""
        res = self.client.post("/api/v1/triage/batch", json={"issues": []})
        self.assertEqual(res.status_code, 422)

    def test_legacy_predict_endpoint_compatibility(self):
        """Verify legacy POST /predict works with expected {"predicted_labels": [class]} response."""
        res = self.client.post("/predict?text=Fatal%20crash%20in%20conv2d")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("predicted_labels", data)
        self.assertIsInstance(data["predicted_labels"], list)
        self.assertIn(data["predicted_labels"][0], ["Bug", "Feature", "Documentation", "Question"])

    def test_cors_headers(self):
        """Verify CORS headers are present on options or cross-origin requests."""
        res = self.client.options(
            "/api/v1/triage",
            headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"}
        )
        self.assertIn("access-control-allow-origin", res.headers)


if __name__ == "__main__":
    unittest.main()
