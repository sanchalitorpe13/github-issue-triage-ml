"""
Model Training Pipeline.
Executes stratified train/test splitting, fits the TF-IDF vectorizer strictly on training data,
trains candidate classifiers, and records evaluation results.
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config.settings import settings
from src.triage_ml.nlp.vectorizer import IssueTextVectorizer
from src.triage_ml.models.classifier import get_all_baseline_models, get_classifier
from src.triage_ml.models.evaluator import ModelEvaluator
from src.triage_ml.utils.logger import get_logger

logger = get_logger("trainer")


class ModelTrainer:
    """Manages reproducible training and cross-evaluation across baseline models."""

    def __init__(
        self,
        test_size: float = settings.TEST_SPLIT_SIZE,
        random_state: int = settings.RANDOM_STATE
    ):
        self.test_size = test_size
        self.random_state = random_state

    def prepare_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Filters and splits dataset into stratified train and test partitions.
        Only issues with at least one canonical label are included.
        Prevents data leakage by separating texts prior to vectorization.

        Args:
            df: Processed DataFrame containing 'cleaned_text' and 'canonical_labels'.

        Returns:
            Tuple[X_train, X_test, y_train, y_test]
        """
        # Filter issues that have a canonical category
        valid_mask = df["canonical_labels"].apply(
            lambda x: len(eval(x) if isinstance(x, str) else x) > 0
        )
        filtered_df = df[valid_mask].copy()

        # Primary label: first canonical label
        def get_primary_label(x):
            labels = eval(x) if isinstance(x, str) else x
            return labels[0] if labels else "Unlabelled"

        filtered_df["target"] = filtered_df["canonical_labels"].apply(get_primary_label)

        X = filtered_df["cleaned_text"].tolist()
        y = filtered_df["target"].tolist()

        logger.info(f"Prepared {len(X)} labelled issues for training across classes: {sorted(list(set(y)))}")

        # Stratified train/test split (when minority classes have at least 2 samples and test size >= number of classes)
        class_counts = pd.Series(y).value_counts()
        n_classes = len(class_counts)
        test_count = int(len(y) * self.test_size) if isinstance(self.test_size, float) else self.test_size
        stratify_param = y if (class_counts.min() >= 2 and test_count >= n_classes) else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )

        logger.info(f"Split data into Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        return X_train, X_test, y_train, y_test

    def train_and_evaluate_all(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Trains all baseline models on the training split and evaluates on the held-out test split.

        Args:
            df: Processed DataFrame.

        Returns:
            Dict[str, Any]: Results dictionary containing metrics, models, and vectorizer.
        """
        X_train, X_test, y_train, y_test = self.prepare_data(df)

        # Fit vectorizer strictly on training data (Zero data leakage)
        vectorizer = IssueTextVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        candidate_models = get_all_baseline_models(random_state=self.random_state)
        results: Dict[str, Dict[str, Any]] = {}

        for name, model in candidate_models.items():
            logger.info(f"Training {name}...")
            # Fit model
            model.fit(X_train_vec, y_train)

            # Predict on unseen test data
            y_pred = model.predict(X_test_vec)

            # Evaluate
            metrics = ModelEvaluator.evaluate_predictions(y_test, y_pred)
            logger.info(
                f"{name} -> Accuracy: {metrics['accuracy']:.4f} | "
                f"Macro F1: {metrics['macro_f1']:.4f} | "
                f"Weighted F1: {metrics['weighted_f1']:.4f}"
            )

            results[name] = {
                "model": model,
                "metrics": metrics,
                "y_pred": y_pred.tolist()
            }

        return {
            "results": results,
            "vectorizer": vectorizer,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        }
