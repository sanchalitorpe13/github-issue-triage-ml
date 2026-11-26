"""
Production Prediction Engine.
Loads the serialized end-to-end pipeline and metadata, validates inputs,
handles edge cases (empty text, missing fields), generates calibrated confidence scores,
and extracts top influential keywords for decision explainability.
"""
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from config.settings import settings
from src.triage_ml.nlp.text_cleaner import clean_text_for_nlp
from src.triage_ml.utils.file_io import load_joblib, load_json
from src.triage_ml.utils.logger import get_logger

logger = get_logger("predictor")


class IssuePredictor:
    """Production inference engine for classifying GitHub issues."""

    def __init__(
        self,
        pipeline_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None
    ):
        self.pipeline_path = pipeline_path or (settings.MODELS_PATH / "issue_type_classifier.joblib")
        self.metadata_path = metadata_path or (settings.MODELS_PATH / "metadata.json")

        self.pipeline: Any = None
        self.metadata: Dict[str, Any] = {}
        self.classes: List[str] = []
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads serialized pipeline and metadata if they exist."""
        if self.pipeline_path.exists():
            self.pipeline = load_joblib(self.pipeline_path)
            logger.info(f"Loaded prediction pipeline from: {self.pipeline_path}")
        else:
            logger.warning(f"Pipeline not found at: {self.pipeline_path}. Predictor in uninitialized state.")

        if self.metadata_path.exists():
            self.metadata = load_json(self.metadata_path)
            self.classes = self.metadata.get("classes", ["Bug", "Feature", "Documentation", "Question"])
        else:
            self.classes = ["Bug", "Feature", "Documentation", "Question"]

    def is_ready(self) -> bool:
        """Checks whether the model pipeline is loaded and ready for inference."""
        return self.pipeline is not None

    def predict(
        self,
        text: Optional[str] = None,
        title: Optional[str] = None,
        body: Optional[str] = None,
        top_n_keywords: int = 5
    ) -> Dict[str, Any]:
        """
        Classifies an issue and outputs prediction, calibrated probabilities, and explanation.

        Args:
            text: Combined text, or pass title & body separately.
            title: Issue title string.
            body: Issue body string.
            top_n_keywords: Number of influential terms to return.

        Returns:
            Dict[str, Any]: Prediction result containing:
              - predicted_label: str
              - confidence: float (0.0 to 1.0)
              - probabilities: Dict[str, float]
              - top_keywords: List[Tuple[str, float]]
              - is_fallback: bool
        """
        # Input validation and aggregation
        if text:
            input_text = str(text)
        else:
            t = str(title) if title is not None else ""
            b = str(body) if body is not None else ""
            input_text = f"{t} {b}".strip()

        # Handle empty or missing text gracefully
        if not input_text or not input_text.strip():
            return {
                "predicted_label": "Unlabelled",
                "confidence": 0.0,
                "probabilities": {c: 0.0 for c in self.classes},
                "top_keywords": [],
                "is_fallback": True,
                "message": "Empty or missing issue text."
            }

        # Apply standard preprocessing
        cleaned = clean_text_for_nlp(input_text)
        if not cleaned.strip() or not re.search(r"[a-zA-Z0-9]", cleaned):
            return {
                "predicted_label": "Unlabelled",
                "confidence": 0.0,
                "probabilities": {c: 0.0 for c in self.classes},
                "top_keywords": [],
                "is_fallback": True,
                "message": "Text contained only punctuation, whitespace, or invalid tokens."
            }

        if not self.is_ready():
            raise RuntimeError("Model pipeline is not loaded. Train a model or provide valid artifact path.")

        # Inference
        pred_label = self.pipeline.predict([cleaned])[0]

        # Extract calibrated probabilities if supported
        probabilities: Dict[str, float] = {}
        confidence = 1.0

        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba([cleaned])[0]
            classes_in_model = getattr(self.pipeline, "classes_", self.classes)
            for c, p in zip(classes_in_model, probs):
                probabilities[str(c)] = round(float(p), 4)
            confidence = round(float(np.max(probs)), 4)
        else:
            # Fallback for models without predict_proba
            for c in self.classes:
                probabilities[c] = 1.0 if c == pred_label else 0.0

        # Extract top influential keywords via TF-IDF vectorizer step
        top_keywords: List[Tuple[str, float]] = []
        try:
            tfidf_step = self.pipeline.named_steps.get("tfidf")
            if tfidf_step:
                vec = tfidf_step.transform([cleaned])
                feature_names = tfidf_step.get_feature_names_out()
                cx = vec.tocoo()
                sorted_items = sorted(zip(cx.col, cx.data), key=lambda x: x[1], reverse=True)
                top_keywords = [
                    (feature_names[i], round(float(score), 4))
                    for i, score in sorted_items[:top_n_keywords]
                ]
        except Exception as e:
            logger.debug(f"Could not extract top keywords: {e}")

        return {
            "predicted_label": str(pred_label),
            "confidence": confidence,
            "probabilities": probabilities,
            "top_keywords": top_keywords,
            "is_fallback": False,
            "cleaned_text_preview": cleaned[:100] + ("..." if len(cleaned) > 100 else "")
        }
