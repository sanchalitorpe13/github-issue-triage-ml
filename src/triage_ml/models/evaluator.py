"""
Model Evaluator.
Computes comprehensive metrics: Accuracy, Precision, Recall, F1 (macro and weighted),
Confusion Matrix, and detailed per-class classification breakdown.
"""
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    hamming_loss
)

from src.triage_ml.utils.logger import get_logger

logger = get_logger("evaluator")


class ModelEvaluator:
    """Evaluates multi-class and multi-label issue classification models."""

    @staticmethod
    def evaluate_predictions(
        y_true: Union[List[str], np.ndarray, pd.Series],
        y_pred: Union[List[str], np.ndarray, pd.Series],
        classes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculates a full suite of classification evaluation metrics.

        Args:
            y_true: Ground truth target labels.
            y_pred: Model predicted target labels.
            classes: Optional explicit list of class names.

        Returns:
            Dict[str, Any]: Evaluated metrics dictionary.
        """
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        unique_labels = classes or sorted(list(set(y_true_arr).union(set(y_pred_arr))))

        acc = float(accuracy_score(y_true_arr, y_pred_arr))
        prec_macro = float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))

        prec_weighted = float(precision_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
        rec_weighted = float(recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
        f1_weighted = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))

        # Per-class metrics
        report_dict = classification_report(
            y_true_arr,
            y_pred_arr,
            labels=unique_labels,
            output_dict=True,
            zero_division=0
        )

        # Confusion Matrix
        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=unique_labels)

        return {
            "accuracy": round(acc, 4),
            "macro_precision": round(prec_macro, 4),
            "macro_recall": round(rec_macro, 4),
            "macro_f1": round(f1_macro, 4),
            "weighted_precision": round(prec_weighted, 4),
            "weighted_recall": round(rec_weighted, 4),
            "weighted_f1": round(f1_weighted, 4),
            "classes": unique_labels,
            "confusion_matrix": cm.tolist(),
            "per_class_report": report_dict,
            "sample_count": len(y_true_arr)
        }

    @staticmethod
    def format_metrics_table(metrics: Dict[str, Any], model_name: str = "Model") -> str:
        """Formats evaluation metrics into a clean human-readable table."""
        lines = [
            f"=== Evaluation Report: {model_name} (N = {metrics.get('sample_count', 0)}) ===",
            f"  Accuracy:           {metrics.get('accuracy', 0):.4f}",
            f"  Macro F1-Score:     {metrics.get('macro_f1', 0):.4f}  <-- Primary Selection Metric",
            f"  Macro Precision:    {metrics.get('macro_precision', 0):.4f}",
            f"  Macro Recall:       {metrics.get('macro_recall', 0):.4f}",
            f"  Weighted F1-Score:  {metrics.get('weighted_f1', 0):.4f}",
            f"  Weighted Precision: {metrics.get('weighted_precision', 0):.4f}",
            f"  Weighted Recall:    {metrics.get('weighted_recall', 0):.4f}",
            "----------------------------------------------------------------",
            "  Per-Class Performance Breakdown:"
        ]

        per_class = metrics.get("per_class_report", {})
        classes = metrics.get("classes", [])
        for c in classes:
            c_info = per_class.get(c, {})
            p = c_info.get("precision", 0)
            r = c_info.get("recall", 0)
            f = c_info.get("f1-score", 0)
            sup = int(c_info.get("support", 0))
            lines.append(f"    • {c:15s} | Prec: {p:.2f} | Rec: {r:.2f} | F1: {f:.2f} | Support: {sup}")

        return "\n".join(lines)
