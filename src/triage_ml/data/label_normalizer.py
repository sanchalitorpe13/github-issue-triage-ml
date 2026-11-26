"""
Label Normalizer & Taxonomy Mapper.
Solves the string-to-character binarization bug and maps heterogeneous labels
into canonical issue categories: Bug, Feature, Documentation, and Question.
"""
import ast
import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd
import numpy as np

from config.settings import settings
from src.triage_ml.utils.logger import get_logger

logger = get_logger("label_normalizer")


def parse_raw_labels(raw_labels: Any) -> List[str]:
    """
    Safely parses raw label values into a list of clean string tokens.
    Handles:
      - Stringified Python lists: "['type:bug', 'comp:keras']"
      - JSON-serialized arrays: '["type:bug", "comp:keras"]'
      - Comma-separated strings: "bug, frontend, high"
      - Pre-parsed Python lists or sets
      - Empty strings / NaNs / None

    Guarantees: NEVER splits a string into individual characters!

    Args:
        raw_labels: Raw input from CSV or API response.

    Returns:
        List[str]: Clean list of label strings.
    """
    if raw_labels is None or (isinstance(raw_labels, float) and np.isnan(raw_labels)):
        return []

    if isinstance(raw_labels, (list, tuple, set)):
        return [str(item).strip() for item in raw_labels if str(item).strip()]

    if not isinstance(raw_labels, str):
        return [str(raw_labels).strip()]

    cleaned_str = raw_labels.strip()
    if not cleaned_str or cleaned_str == "[]" or cleaned_str == "set()":
        return []

    # Attempt 1: Safe Python literal evaluation (e.g. "['type:bug', 'comp:keras']")
    if (cleaned_str.startswith("[") and cleaned_str.endswith("]")) or (
        cleaned_str.startswith("(") and cleaned_str.endswith(")")
    ):
        try:
            evaluated = ast.literal_eval(cleaned_str)
            if isinstance(evaluated, (list, tuple, set)):
                return [str(item).strip() for item in evaluated if str(item).strip()]
        except Exception:
            pass

        # Attempt 2: JSON parsing
        try:
            parsed_json = json.loads(cleaned_str)
            if isinstance(parsed_json, list):
                return [str(item).strip() for item in parsed_json if str(item).strip()]
        except Exception:
            pass

        # Attempt 3: Regex token extraction inside brackets
        tokens = re.findall(r"['\"]([^'\"]+)['\"]", cleaned_str)
        if tokens:
            return [t.strip() for t in tokens if t.strip()]

    # Attempt 4: Comma-separated values
    if "," in cleaned_str:
        return [item.strip() for item in cleaned_str.split(",") if item.strip()]

    # Single string label
    return [cleaned_str]


def normalize_labels(
    labels: List[str],
    taxonomy: Optional[Dict[str, List[str]]] = None
) -> List[str]:
    """
    Maps heterogeneous GitHub labels to standard canonical categories:
      - 'Bug'
      - 'Feature'
      - 'Documentation'
      - 'Question'

    Args:
        labels: List of raw label tokens.
        taxonomy: Optional dictionary mapping canonical classes to alias patterns.

    Returns:
        List[str]: List of canonical labels matched (e.g. ['Bug'], ['Bug', 'Documentation']).
    """
    active_taxonomy = taxonomy or settings.LABEL_TAXONOMY
    matched: Set[str] = set()

    for raw_label in labels:
        label_lower = raw_label.lower().strip()
        # Direct lookup or pattern match
        for canonical, patterns in active_taxonomy.items():
            for pattern in patterns:
                pattern_lower = pattern.lower().strip()
                # Exact match or prefix match (e.g. "type:bug", "kind/bug", "bug")
                if label_lower == pattern_lower or label_lower.startswith(f"{pattern_lower}:") or label_lower.startswith(f"{pattern_lower}/"):
                    matched.add(canonical)
                elif f":{pattern_lower}" in label_lower or f"/{pattern_lower}" in label_lower:
                    matched.add(canonical)
                elif pattern_lower in label_lower.split():
                    matched.add(canonical)

    return sorted(list(matched))


def extract_priority_label(labels: List[str]) -> Optional[str]:
    """
    Inspects whether any genuine priority or severity labels exist in the raw labels.
    Never invents or fabricates ground truth. Returns None if unlabelled.

    Standard mappings:
      - P0 / critical / blocker / severity:critical -> 'P0'
      - P1 / high / severity:high / priority:high   -> 'P1'
      - P2 / medium / severity:medium               -> 'P2'
      - P3 / low / severity:low / cosmetic          -> 'P3'

    Args:
        labels: List of raw label tokens.

    Returns:
        Optional[str]: Canonical priority tier ('P0', 'P1', 'P2', 'P3') or None.
    """
    priority_patterns = {
        "P0": ["p0", "priority: p0", "priority:p0", "critical", "blocker", "severity: critical", "severity:critical", "priority: critical"],
        "P1": ["p1", "priority: p1", "priority:p1", "high", "severity: high", "severity:high", "priority: high", "important"],
        "P2": ["p2", "priority: p2", "priority:p2", "medium", "severity: medium", "severity:medium", "priority: medium", "normal"],
        "P3": ["p3", "priority: p3", "priority:p3", "low", "severity: low", "severity:low", "priority: low", "minor", "cosmetic"]
    }

    for raw_label in labels:
        lbl = raw_label.lower().strip()
        for tier, patterns in priority_patterns.items():
            for p in patterns:
                if lbl == p or lbl.startswith(f"{p}/") or lbl.startswith(f"{p}:"):
                    return tier
                if f"/{p}" in lbl or f":{p}" in lbl:
                    return tier

    return None


class CanonicalLabelEncoder:
    """
    Multi-label and multi-class target encoder that preserves exact class names.
    Stores and restores class mappings cleanly to prevent index disconnect bugs.
    """

    def __init__(self, classes: Optional[List[str]] = None):
        self.classes_: List[str] = classes or ["Bug", "Feature", "Documentation", "Question"]
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes_)}

    def transform(self, label_series: List[List[str]]) -> np.ndarray:
        """
        Transforms a sequence of label lists into a binary indicator matrix.

        Args:
            label_series: List of label lists (e.g. [['Bug'], ['Feature', 'Documentation']])

        Returns:
            np.ndarray: 2D binary indicator array of shape (n_samples, n_classes).
        """
        matrix = np.zeros((len(label_series), len(self.classes_)), dtype=int)
        for row_idx, labels in enumerate(label_series):
            for label in labels:
                if label in self.class_to_idx:
                    col_idx = self.class_to_idx[label]
                    matrix[row_idx, col_idx] = 1
        return matrix

    def inverse_transform(self, binary_matrix: np.ndarray, threshold: float = 0.5) -> List[List[str]]:
        """
        Converts binary indicator or probability matrix back into class names.

        Args:
            binary_matrix: 2D array of predictions or probabilities.
            threshold: Probability threshold for positive label assignment.

        Returns:
            List[List[str]]: Reconstructed label lists.
        """
        result = []
        for row in binary_matrix:
            active_labels = [
                self.classes_[idx] for idx, val in enumerate(row) if val >= threshold
            ]
            result.append(active_labels)
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Serializes encoder metadata."""
        return {
            "classes": self.classes_,
            "class_to_idx": self.class_to_idx
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalLabelEncoder":
        """Reconstructs encoder from serialized metadata."""
        return cls(classes=data.get("classes"))
