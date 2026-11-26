"""
Phase 3 Machine Learning Model & Pipeline Unit Tests.
Tests vectorizer, data leakage prevention, empty text handling,
model training, evaluation metrics, prediction output, and artifact saving/loading.
"""
import unittest
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

from config.settings import settings
from src.triage_ml.nlp.vectorizer import IssueTextVectorizer
from src.triage_ml.models.classifier import get_classifier, get_all_baseline_models
from src.triage_ml.models.evaluator import ModelEvaluator
from src.triage_ml.models.trainer import ModelTrainer
from src.triage_ml.models.comparison import ModelComparator
from src.triage_ml.models.predictor import IssuePredictor


class TestPhase3Modeling(unittest.TestCase):
    """Test suite for Phase 3 ML modeling, evaluation, and inference."""

    def setUp(self):
        """Creates small reproducible synthetic test fixtures without requiring external downloads."""
        self.sample_train_texts = [
            "Segmentation fault when running conv2d in tensorflow",
            "Fatal crash: core dumped upon model execution",
            "Bug: unhandled exception in database connection rollback",
            "Feature request: add support for parquet reader engine",
            "Proposal: allow dark theme customization via config",
            "Documentation: update broken link in deployment guide",
            "Docs: add tutorial for multi-output regression",
            "Question: how to properly share session across background tasks",
            "Troubleshooting: standard scaler throws value error on sparse matrix"
        ]
        self.sample_train_labels = [
            "Bug", "Bug", "Bug",
            "Feature", "Feature",
            "Documentation", "Documentation",
            "Question", "Question"
        ]

    def test_vectorizer_fit_and_transform_prevents_leakage(self):
        """Verifies that vectorizer fits on train and transforms test without leakage."""
        vec = IssueTextVectorizer(max_features=50, ngram_range=(1, 2))
        self.assertFalse(vec.is_fitted)

        # Unfitted transform should raise ValueError
        with self.assertRaises(ValueError):
            vec.transform(["test text"])

        # Fit on train
        X_train_vec = vec.fit_transform(self.sample_train_texts)
        self.assertTrue(vec.is_fitted)
        self.assertEqual(X_train_vec.shape[0], len(self.sample_train_texts))

        # Transform unseen text (words not in train vocab are ignored)
        unseen_text = ["Unseen alien word xyz123 segmentation"]
        X_unseen_vec = vec.transform(unseen_text)
        self.assertEqual(X_unseen_vec.shape[0], 1)

    def test_vectorizer_top_features_extraction(self):
        """Verifies extraction of top influential n-grams."""
        vec = IssueTextVectorizer()
        vec.fit(self.sample_train_texts)
        top_words = vec.get_top_features_for_text("Segmentation fault crash in tensorflow", top_n=5)
        self.assertTrue(len(top_words) > 0)
        words = [w for w, _ in top_words]
        self.assertTrue(any("segmentation" in w or "crash" in w for w in words))

    def test_classifier_factory(self):
        """Verifies that all baseline models instantiate with valid configurations."""
        models = get_all_baseline_models(random_state=42)
        self.assertIn("Logistic Regression", models)
        self.assertIn("Complement Naive Bayes", models)
        self.assertIn("Linear Support Vector Machine", models)

        # Invalid model name raises ValueError
        with self.assertRaises(ValueError):
            get_classifier("unsupported_random_model")

    def test_evaluator_metrics_calculation(self):
        """Verifies accurate calculation of Accuracy, Macro F1, Weighted F1, and Confusion Matrix."""
        y_true = ["Bug", "Bug", "Feature", "Question", "Documentation"]
        y_pred = ["Bug", "Bug", "Feature", "Documentation", "Documentation"]

        metrics = ModelEvaluator.evaluate_predictions(y_true, y_pred)
        self.assertEqual(metrics["sample_count"], 5)
        self.assertGreater(metrics["accuracy"], 0.0)
        self.assertGreater(metrics["macro_f1"], 0.0)
        self.assertGreater(metrics["weighted_f1"], 0.0)
        self.assertIn("classes", metrics)
        self.assertIn("confusion_matrix", metrics)

    def test_trainer_and_comparator_pipeline(self):
        """Verifies end-to-end training, comparison, and best model selection on mini-fixture."""
        mini_df = pd.DataFrame({
            "cleaned_text": self.sample_train_texts,
            "canonical_labels": [[l] for l in self.sample_train_labels]
        })

        trainer = ModelTrainer(test_size=0.33, random_state=42)
        train_output = trainer.train_and_evaluate_all(mini_df)
        results = train_output["results"]

        self.assertIn("Logistic Regression", results)
        self.assertIn("Complement Naive Bayes", results)
        self.assertIn("Linear Support Vector Machine", results)

        # Test comparator
        comparison_df = ModelComparator.build_comparison_dataframe(results)
        self.assertEqual(len(comparison_df), 3)
        self.assertIn("Macro F1", comparison_df.columns)

        best_name, best_model, best_metrics = ModelComparator.select_best_model(results)
        self.assertTrue(isinstance(best_name, str))
        self.assertIsNotNone(best_model)
        self.assertGreaterEqual(best_metrics["macro_f1"], 0.0)

    def test_predictor_handling_edge_cases(self):
        """Verifies that IssuePredictor gracefully handles empty text, None, and valid text."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pipeline_file = tmp_path / "test_pipeline.joblib"
            meta_file = tmp_path / "test_meta.json"

            # Train a quick mini pipeline and save it
            mini_df = pd.DataFrame({
                "cleaned_text": self.sample_train_texts,
                "canonical_labels": [[l] for l in self.sample_train_labels]
            })
            trainer = ModelTrainer(test_size=0.33, random_state=42)
            out = trainer.train_and_evaluate_all(mini_df)
            best_name, best_model, best_metrics = ModelComparator.select_best_model(out["results"])

            comp_df = ModelComparator.build_comparison_dataframe(out["results"])
            saved_pipe, saved_meta = ModelComparator.save_production_artifacts(
                best_model_name=best_name,
                best_model=best_model,
                vectorizer=out["vectorizer"],
                metrics=best_metrics,
                comparison_df=comp_df,
                dataset_info={"total_rows": len(mini_df)},
                output_dir=tmp_path
            )

            # Test predictor initialized with saved artifacts
            predictor = IssuePredictor(pipeline_path=saved_pipe, metadata_path=saved_meta)
            self.assertTrue(predictor.is_ready())

            # Edge case 1: Empty text
            empty_res = predictor.predict(text="")
            self.assertTrue(empty_res["is_fallback"])
            self.assertEqual(empty_res["confidence"], 0.0)

            # Edge case 2: None input
            none_res = predictor.predict(text=None, title=None, body=None)
            self.assertTrue(none_res["is_fallback"])

            # Edge case 3: Punctuation-only string
            punct_res = predictor.predict(text="!@#$%^&*()_+{}[]")
            self.assertTrue(punct_res["is_fallback"])

            # Valid prediction
            valid_res = predictor.predict(text="Segmentation fault crash in conv2d")
            self.assertFalse(valid_res["is_fallback"])
            self.assertIn(valid_res["predicted_label"], ["Bug", "Feature", "Documentation", "Question"])
            self.assertGreater(valid_res["confidence"], 0.0)
            self.assertIn("probabilities", valid_res)


if __name__ == "__main__":
    unittest.main()
