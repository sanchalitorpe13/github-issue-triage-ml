"""
TF-IDF Feature Extractor for Developer Text.
Extracts unigram and bigram features with sublinear TF scaling.
Strictly separates fit (train only) from transform (val/test/inference) to eliminate data leakage.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from config.settings import settings
from src.triage_ml.utils.logger import get_logger

logger = get_logger("vectorizer")


class IssueTextVectorizer:
    """Production wrapper around TfidfVectorizer for GitHub issue text."""

    def __init__(
        self,
        max_features: Optional[int] = None,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 1,
        sublinear_tf: bool = True
    ):
        self.max_features = max_features or settings.TFIDF_WORD_MAX_FEATURES
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=self.sublinear_tf,
            token_pattern=r"(?u)\b\w[\w\-\_]+\b"  # Preserves hyphens and underscores in code tokens
        )
        self.is_fitted: bool = False

    def fit(self, texts: List[str]) -> "IssueTextVectorizer":
        """
        Fits vocabulary and IDF weights on training texts ONLY.

        Args:
            texts: List of preprocessed text strings from training set.

        Returns:
            IssueTextVectorizer: Self instance.
        """
        clean_texts = [str(t) if t is not None else "" for t in texts]
        self.vectorizer.fit(clean_texts)
        self.is_fitted = True
        logger.info(f"Fitted TF-IDF vectorizer. Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        return self

    def transform(self, texts: List[str]) -> sp.csr_matrix:
        """
        Transforms text strings into TF-IDF document-term sparse matrix.

        Args:
            texts: List of text strings.

        Returns:
            scipy.sparse.csr_matrix: TF-IDF sparse matrix.
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transforming data. Call fit() first.")
        clean_texts = [str(t) if t is not None else "" for t in texts]
        return self.vectorizer.transform(clean_texts)

    def fit_transform(self, texts: List[str]) -> sp.csr_matrix:
        """Fits on training texts and returns transformed sparse matrix."""
        return self.fit(texts).transform(texts)

    def get_feature_names(self) -> List[str]:
        """Returns vocabulary feature names."""
        if not self.is_fitted:
            return []
        return self.vectorizer.get_feature_names_out().tolist()

    def get_top_features_for_text(self, text: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Returns top influential TF-IDF n-grams for an input text.

        Args:
            text: Single text string.
            top_n: Number of top features to return.

        Returns:
            List[Tuple[str, float]]: List of (feature_word, tfidf_weight) tuples.
        """
        if not self.is_fitted:
            return []
        vec = self.transform([text])
        feature_names = self.vectorizer.get_feature_names_out()
        cx = vec.tocoo()
        sorted_items = sorted(zip(cx.col, cx.data), key=lambda x: x[1], reverse=True)
        return [(feature_names[i], round(float(score), 4)) for i, score in sorted_items[:top_n]]

    def get_params(self) -> Dict[str, Any]:
        """Returns initialization hyperparameters."""
        return {
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "min_df": self.min_df,
            "sublinear_tf": self.sublinear_tf,
            "vocab_size": len(self.vectorizer.vocabulary_) if self.is_fitted else 0
        }
