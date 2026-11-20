"""
Phase 1 Foundation Verification Tests.
Uses Python's standard unittest framework for zero-dependency execution.
Verifies configuration integrity, logging, safe I/O, directory paths, legacy preservation, and FastAPI compatibility.
"""
import unittest
import tempfile
from pathlib import Path

from config.settings import settings
from src.triage_ml.utils.logger import get_logger
from src.triage_ml.utils.file_io import save_json, load_json, save_joblib, load_joblib


class TestPhase1Foundation(unittest.TestCase):
    """Test suite verifying Phase 1 foundation components."""

    def test_settings_initialization(self):
        """Verify settings loads properly with correct project metadata and directories."""
        self.assertEqual(settings.PROJECT_NAME, "GitHub Issue Triage ML")
        self.assertEqual(settings.VERSION, "2.0.0")
        self.assertTrue(settings.ROOT_PATH.exists())
        self.assertIn("Bug", settings.LABEL_TAXONOMY)
        self.assertIn("Feature", settings.LABEL_TAXONOMY)
        self.assertIn("P0", settings.PRIORITY_LEVELS)
        self.assertIn("critical", settings.SEVERITY_KEYWORDS)

    def test_settings_headers(self):
        """Verify GitHub headers generator produces valid User-Agent and Accept headers."""
        headers = settings.get_github_headers()
        self.assertIn("Accept", headers)
        self.assertIn("User-Agent", headers)
        self.assertEqual(headers["User-Agent"], "GitHub-Issue-Triage-ML-2.0")

    def test_logger_functionality(self):
        """Verify logger creation and proper naming."""
        logger = get_logger("test_module")
        self.assertEqual(logger.name, "test_module")
        self.assertTrue(logger.hasHandlers())

    def test_file_io_operations(self):
        """Verify atomic JSON and joblib serialization and deserialization."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # JSON Test
            test_json_file = tmp_path / "test.json"
            sample_data = {"test_key": "test_value", "number": 42, "items": [1, 2, 3]}
            saved_path = save_json(sample_data, test_json_file)
            self.assertTrue(saved_path.exists())
            loaded_data = load_json(saved_path)
            self.assertEqual(loaded_data, sample_data)

            # Joblib Test
            test_joblib_file = tmp_path / "test.joblib"
            sample_obj = {"model_name": "test_estimator", "weights": [0.1, 0.5, 0.9]}
            saved_joblib = save_joblib(sample_obj, test_joblib_file)
            self.assertTrue(saved_joblib.exists())
            loaded_obj = load_joblib(saved_joblib)
            self.assertEqual(loaded_obj, sample_obj)

    def test_legacy_files_preserved(self):
        """Verify that all original repository files remain untouched in their original locations."""
        expected_legacy_files = [
            Path("app/main.py"),
            Path("app/init.py"),
            Path("app/main11111.py"),
            Path("data/issues_clean.csv"),
            Path("data/issues_raw.csv"),
            Path("data/labels.csv"),
            Path("models/issue_classifier.pkl"),
            Path("models/tfidf_vectorizer.pkl"),
            Path("scripts/fetch_issues.py"),
            Path("scripts/preprocess.py"),
            Path("scripts/train_model.py"),
        ]
        for rel_path in expected_legacy_files:
            full_path = settings.ROOT_PATH / rel_path
            self.assertTrue(full_path.exists(), f"Original file missing: {rel_path}")

    def test_backup_integrity(self):
        """Verify that the verified backup directory exists and contains all backed-up files."""
        backup_root = settings.ROOT_PATH / "backup" / "original_repo"
        self.assertTrue(backup_root.exists(), "Backup directory does not exist")
        self.assertTrue((backup_root / "data" / "issues_raw.csv").exists())
        self.assertTrue((backup_root / "models" / "issue_classifier.pkl").exists())
        self.assertTrue((backup_root / "scripts" / "fetch_issues.py").exists())
        self.assertTrue((backup_root / "app" / "main.py").exists())

    def test_existing_fastapi_compatibility(self):
        """Verify that existing app.main still imports and runs without error."""
        try:
            from app.main import app as legacy_app
            self.assertIsNotNone(legacy_app)
            self.assertEqual(legacy_app.title, "AI Issue Triage System")
        except Exception as e:
            self.fail(f"Existing FastAPI app failed to import: {e}")


if __name__ == "__main__":
    unittest.main()
