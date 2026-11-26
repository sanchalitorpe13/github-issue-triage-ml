"""
Model Comparison & Best Model Selector.
Structures side-by-side benchmark comparison across models, justifies selection
based on Macro F1-score, and serializes the winning complete pipeline and metadata.
"""
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from sklearn.pipeline import Pipeline

from config.settings import settings
from src.triage_ml.utils.file_io import save_joblib, save_json
from src.triage_ml.utils.logger import get_logger

logger = get_logger("comparison")


class ModelComparator:
    """Compares candidate models and produces structured evaluation artifacts."""

    @staticmethod
    def build_comparison_dataframe(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Builds a tabular comparison DataFrame across trained models.

        Args:
            results: Results dictionary from ModelTrainer.train_and_evaluate_all().

        Returns:
            pd.DataFrame: Formatted comparison table.
        """
        rows = []
        for name, data in results.items():
            metrics = data.get("metrics", {})
            rows.append({
                "Model": name,
                "Macro F1": metrics.get("macro_f1", 0.0),
                "Weighted F1": metrics.get("weighted_f1", 0.0),
                "Accuracy": metrics.get("accuracy", 0.0),
                "Macro Precision": metrics.get("macro_precision", 0.0),
                "Macro Recall": metrics.get("macro_recall", 0.0),
                "Weighted Precision": metrics.get("weighted_precision", 0.0),
                "Weighted Recall": metrics.get("weighted_recall", 0.0),
                "Test Samples": metrics.get("sample_count", 0)
            })

        df = pd.DataFrame(rows)
        # Sort by Macro F1 descending (prioritizing balanced performance over accuracy)
        df = df.sort_values(by="Macro F1", ascending=False).reset_index(drop=True)
        return df

    @staticmethod
    def select_best_model(
        results: Dict[str, Dict[str, Any]],
        selection_metric: str = "macro_f1"
    ) -> Tuple[str, Any, Dict[str, Any]]:
        """
        Selects the winning model based on the chosen metric (default: Macro F1-score).

        Args:
            results: Results dictionary.
            selection_metric: Metric key to maximize.

        Returns:
            Tuple[best_model_name, best_model_object, best_metrics_dict]
        """
        best_name = ""
        best_score = -1.0
        best_data: Dict[str, Any] = {}

        for name, data in results.items():
            score = data.get("metrics", {}).get(selection_metric, -1.0)
            if score > best_score:
                best_score = score
                best_name = name
                best_data = data

        logger.info(f"Selected best model: '{best_name}' with {selection_metric} = {best_score:.4f}")
        return best_name, best_data["model"], best_data["metrics"]

    @staticmethod
    def save_production_artifacts(
        best_model_name: str,
        best_model: Any,
        vectorizer: Any,
        metrics: Dict[str, Any],
        comparison_df: pd.DataFrame,
        dataset_info: Dict[str, Any],
        output_dir: Optional[Path] = None
    ) -> Tuple[Path, Path]:
        """
        Saves the complete selected pipeline (vectorizer + classifier) and comprehensive metadata.

        Args:
            best_model_name: Name of selected model.
            best_model: Trained classifier instance.
            vectorizer: Fitted IssueTextVectorizer instance.
            metrics: Evaluation metrics for selected model.
            comparison_df: Comparison DataFrame across all models.
            dataset_info: Metadata about dataset size and distributions.
            output_dir: Optional destination directory (defaults to settings.MODELS_PATH).

        Returns:
            Tuple[pipeline_path, metadata_path]
        """
        target_dir = output_dir or settings.MODELS_PATH
        target_dir.mkdir(parents=True, exist_ok=True)

        # Build complete self-contained scikit-learn Pipeline
        full_pipeline = Pipeline([
            ("tfidf", vectorizer.vectorizer),
            ("classifier", best_model)
        ])

        pipeline_path = target_dir / "issue_type_classifier.joblib"
        save_joblib(full_pipeline, pipeline_path)

        # Build production metadata JSON
        metadata = {
            "model_name": best_model_name,
            "version": settings.VERSION,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "task": "Issue Type Classification",
            "classes": metrics.get("classes", []),
            "selection_metric": "macro_f1",
            "metrics": metrics,
            "dataset": dataset_info,
            "vectorizer_config": vectorizer.get_params(),
            "model_comparison": comparison_df.to_dict(orient="records")
        }

        metadata_path = target_dir / "metadata.json"
        save_json(metadata, metadata_path)

        logger.info(f"Saved production pipeline: {pipeline_path}")
        logger.info(f"Saved model metadata: {metadata_path}")

        return pipeline_path, metadata_path
