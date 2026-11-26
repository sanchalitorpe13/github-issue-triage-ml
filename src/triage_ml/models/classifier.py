"""
Model Factory for GitHub Issue Classification.
Configures and initializes baseline classifiers:
  1. Logistic Regression (L2 regularized, balanced class weights, lbfgs solver)
  2. Complement Naive Bayes (Adapted for imbalanced text classification)
  3. Calibrated Linear Support Vector Machine (LinearSVC with calibrated decision probabilities)
"""
from typing import Any, Dict
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from config.settings import settings


class CalibratedLinearSVC(BaseEstimator, ClassifierMixin):
    """
    Linear Support Vector Classifier with softmax-scaled probability calibration.
    Inherits from BaseEstimator and ClassifierMixin for seamless scikit-learn Pipeline integration.
    Guarantees robust probability output without brittle cross-validation splits on small corpora.
    """

    def __init__(
        self,
        C: float = 1.0,
        class_weight: str = "balanced",
        random_state: int = settings.RANDOM_STATE
    ):
        self.C = C
        self.class_weight = class_weight
        self.random_state = random_state
        self.model = LinearSVC(
            C=self.C,
            class_weight=self.class_weight,
            random_state=self.random_state,
            max_iter=2000,
            dual="auto"
        )
        self.classes_: np.ndarray = np.array([])

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Computes calibrated class probabilities via softmax scaling of decision margins."""
        dists = self.model.decision_function(X)
        if len(self.classes_) == 2:
            probs_pos = 1.0 / (1.0 + np.exp(-np.clip(dists, -30, 30)))
            return np.vstack([1.0 - probs_pos, probs_pos]).T

        # Numerically stable softmax
        exp_d = np.exp(dists - np.max(dists, axis=1, keepdims=True))
        return exp_d / np.sum(exp_d, axis=1, keepdims=True)


def get_classifier(model_name: str, random_state: int = settings.RANDOM_STATE) -> Any:
    """
    Instantiates a configured classifier for issue text classification.

    Args:
        model_name: One of 'logistic_regression', 'naive_bayes', 'linear_svc'.
        random_state: Reproducible random seed.

    Returns:
        Configured scikit-learn compatible estimator.
    """
    name = model_name.lower().strip()

    if name in ["logistic_regression", "lr"]:
        return LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            random_state=random_state,
            max_iter=1000
        )

    elif name in ["naive_bayes", "complement_nb", "nb"]:
        return ComplementNB(
            alpha=1.0,
            norm=False
        )

    elif name in ["linear_svc", "svm", "svc"]:
        return CalibratedLinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=random_state
        )

    else:
        raise ValueError(
            f"Unknown model name '{model_name}'. Supported models: "
            f"['logistic_regression', 'naive_bayes', 'linear_svc']"
        )


def get_all_baseline_models(random_state: int = settings.RANDOM_STATE) -> Dict[str, Any]:
    """Returns all candidate baseline classifiers ready for training."""
    return {
        "Logistic Regression": get_classifier("logistic_regression", random_state=random_state),
        "Complement Naive Bayes": get_classifier("naive_bayes", random_state=random_state),
        "Linear Support Vector Machine": get_classifier("linear_svc", random_state=random_state),
    }
