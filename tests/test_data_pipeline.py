"""
Phase 2 Data Pipeline & Normalization Tests.
Verifies label parsing, character binarization fix, canonical taxonomy mapping,
genuine priority extraction, NLP text cleaning, structural features, and original dataset preservation.
"""
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from config.settings import settings
from src.triage_ml.data.label_normalizer import (
    parse_raw_labels,
    normalize_labels,
    extract_priority_label,
    CanonicalLabelEncoder
)
from src.triage_ml.nlp.text_cleaner import clean_text_for_nlp, extract_structural_features
from src.triage_ml.data.dataset_loader import (
    load_raw_dataset,
    preprocess_issues_dataframe,
    save_processed_datasets
)
from src.triage_ml.data.corpus_generator import (
    get_curated_benchmark_issues,
    save_curated_sample_issues,
    create_extended_multi_repo_dataset
)


class TestPhase2DataPipeline(unittest.TestCase):
    """Test suite for Phase 2 dataset inspection and preprocessing."""

    def test_parse_raw_labels_stringified_list(self):
        """CRITICAL REGRESSION TEST: Verifies stringified list is parsed as string tokens, NOT characters."""
        raw_str = "['type:bug', 'comp:keras']"
        parsed = parse_raw_labels(raw_str)
        self.assertEqual(parsed, ["type:bug", "comp:keras"])
        # Assert none of the parsed items are single characters
        self.assertNotIn("t", parsed)
        self.assertNotIn("[", parsed)
        self.assertNotIn("]", parsed)
        self.assertNotIn("'", parsed)

    def test_parse_raw_labels_edge_cases(self):
        """Verifies handling of empty, None, and comma-separated label inputs."""
        self.assertEqual(parse_raw_labels(""), [])
        self.assertEqual(parse_raw_labels(None), [])
        self.assertEqual(parse_raw_labels("[]"), [])
        self.assertEqual(parse_raw_labels(["already", "list"]), ["already", "list"])
        self.assertEqual(parse_raw_labels("bug, high, ui"), ["bug", "high", "ui"])

    def test_normalize_labels_canonical_mapping(self):
        """Verifies mapping of noisy community labels to canonical taxonomy."""
        # Bug tests
        self.assertEqual(normalize_labels(["type:bug"]), ["Bug"])
        self.assertEqual(normalize_labels(["kind/bug", "comp:xla"]), ["Bug"])
        self.assertEqual(normalize_labels(["defect"]), ["Bug"])

        # Feature tests
        self.assertEqual(normalize_labels(["enhancement"]), ["Feature"])
        self.assertEqual(normalize_labels(["feature request"]), ["Feature"])

        # Docs tests
        self.assertEqual(normalize_labels(["documentation", "type:docs"]), ["Documentation"])

        # Question tests
        self.assertEqual(normalize_labels(["question", "help wanted"]), ["Question"])

        # Multi-label simultaneous match
        multi = normalize_labels(["type:bug", "documentation"])
        self.assertIn("Bug", multi)
        self.assertIn("Documentation", multi)

    def test_extract_genuine_priority_labels(self):
        """Verifies that genuine priority labels are extracted without fabricating labels."""
        self.assertEqual(extract_priority_label(["severity:critical", "p0"]), "P0")
        self.assertEqual(extract_priority_label(["priority: high", "bug"]), "P1")
        self.assertEqual(extract_priority_label(["medium", "p2"]), "P2")
        self.assertEqual(extract_priority_label(["cosmetic", "p3"]), "P3")
        # When no priority label exists, MUST return None (no invention)
        self.assertIsNone(extract_priority_label(["type:bug", "comp:xla"]))

    def test_canonical_label_encoder(self):
        """Verifies multi-label encoding and inverse transformation preserves class names."""
        encoder = CanonicalLabelEncoder(["Bug", "Feature", "Documentation", "Question"])
        sample_targets = [["Bug"], ["Feature", "Documentation"], []]
        matrix = encoder.transform(sample_targets)

        self.assertEqual(matrix.shape, (3, 4))
        self.assertEqual(list(matrix[0]), [1, 0, 0, 0])
        self.assertEqual(list(matrix[1]), [0, 1, 1, 0])
        self.assertEqual(list(matrix[2]), [0, 0, 0, 0])

        reconstructed = encoder.inverse_transform(matrix)
        self.assertEqual(reconstructed[0], ["Bug"])
        self.assertEqual(sorted(reconstructed[1]), ["Documentation", "Feature"])
        self.assertEqual(reconstructed[2], [])

    def test_nlp_text_cleaner(self):
        """Verifies code block tokenization, URL removal, and Markdown cleaning."""
        raw_text = """
        ### Bug in TensorFlow
        <!-- issue template -->
        Here is a link: https://example.com/bug.
        Check this code:
        ```python
        def crash():
            raise RuntimeError("Segfault")
        ```
        And inline `var_x`.
        """
        cleaned = clean_text_for_nlp(raw_text)
        self.assertIn("[code_block]", cleaned)
        self.assertIn("[inline_code]", cleaned)
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("<!--", cleaned)
        self.assertIn("tensorflow", cleaned)

    def test_extract_structural_features(self):
        """Verifies structural metadata feature extraction."""
        title = "Fatal error: Segmentation fault in Conv2D!"
        body = "```python\ntf.crash()\n```\nTraceback (most recent call last):\n  File 'test.py', line 1\nRuntimeError"
        feats = extract_structural_features(title, body)

        self.assertEqual(feats["has_code_block"], 1)
        self.assertEqual(feats["has_stack_trace"], 1)
        self.assertGreater(feats["error_keyword_count"], 0)
        self.assertEqual(feats["has_exclamation"], 1)

    def test_preprocessing_preserves_originals(self):
        """Verifies that original CSV files in data/ remain completely unmodified."""
        raw_path = settings.DATA_PATH / "issues_raw.csv"
        self.assertTrue(raw_path.exists())
        raw_before = pd.read_csv(raw_path)

        # Execute preprocessing
        processed_df, target_df = preprocess_issues_dataframe(raw_before)

        # Check raw is unchanged
        raw_after = pd.read_csv(raw_path)
        self.assertEqual(len(raw_before), len(raw_after))
        self.assertEqual(list(raw_before.columns), list(raw_after.columns))

        # Save to data/processed/
        proc_path, lbl_path = save_processed_datasets(processed_df, target_df, prefix="legacy_verified")
        self.assertTrue(proc_path.exists())
        self.assertTrue(lbl_path.exists())

        # Verify target_df columns are real classes, NOT characters!
        self.assertIn("Bug", target_df.columns)
        self.assertNotIn(" ", target_df.columns)
        self.assertNotIn("[", target_df.columns)

    def test_curated_corpus_generation(self):
        """Verifies that the curated authentic GitHub issues corpus is valid."""
        issues = get_curated_benchmark_issues()
        self.assertGreaterEqual(len(issues), 20)
        json_path = save_curated_sample_issues()
        self.assertTrue(json_path.exists())

        extended_df = create_extended_multi_repo_dataset()
        self.assertGreater(len(extended_df), 40)


if __name__ == "__main__":
    unittest.main()
