"""
Dataset Loader & Processing Pipeline.
Loads raw datasets, executes clean preprocessing without mutating originals,
and produces model-ready datasets in data/processed/.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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
from src.triage_ml.utils.logger import get_logger

logger = get_logger("dataset_loader")


def load_raw_dataset(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads raw issues from CSV.

    Args:
        filepath: Optional path to the CSV file. Defaults to data/issues_raw.csv.

    Returns:
        pd.DataFrame: Loaded raw DataFrame.
    """
    target = filepath or (settings.DATA_PATH / "issues_raw.csv")
    if not target.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {target}")
    return pd.read_csv(target)


def preprocess_issues_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes reproducible, domain-aware preprocessing on an issues DataFrame.
    Guarantees:
      - Original DataFrame is not mutated.
      - Labels are safely parsed without character splitting.
      - Canonical categories (Bug, Feature, Documentation, Question) are mapped.
      - Structural metadata features are computed.
      - Cleaned NLP text is generated.

    Args:
        df: Input DataFrame containing columns ['id', 'title', 'body', 'labels'].

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (processed_df, multi_label_binary_df)
    """
    processed = df.copy()

    # Ensure required columns exist
    for col in ["title", "body", "labels"]:
        if col not in processed.columns:
            processed[col] = ""

    # Fill NaNs
    processed["title"] = processed["title"].fillna("")
    processed["body"] = processed["body"].fillna("")

    # 1. Parse raw labels safely
    processed["parsed_labels"] = processed["labels"].apply(parse_raw_labels)

    # 2. Map canonical categories
    processed["canonical_labels"] = processed["parsed_labels"].apply(normalize_labels)

    # 3. Extract genuine priority labels (if present)
    processed["priority_ground_truth"] = processed["parsed_labels"].apply(extract_priority_label)

    # 4. Generate combined and cleaned text for NLP
    processed["raw_combined_text"] = processed["title"] + " " + processed["body"]
    processed["cleaned_text"] = processed["raw_combined_text"].apply(clean_text_for_nlp)

    # 5. Extract structural features
    structural_records = []
    for _, row in processed.iterrows():
        feats = extract_structural_features(row["title"], row["body"])
        structural_records.append(feats)

    structural_df = pd.DataFrame(structural_records)
    for col in structural_df.columns:
        processed[col] = structural_df[col].values

    # 6. Encode multi-label target matrix
    encoder = CanonicalLabelEncoder()
    label_matrix = encoder.transform(processed["canonical_labels"].tolist())
    target_df = pd.DataFrame(label_matrix, columns=encoder.classes_)

    return processed, target_df


def save_processed_datasets(
    processed_df: pd.DataFrame,
    target_df: pd.DataFrame,
    prefix: str = "issues"
) -> Tuple[Path, Path]:
    """
    Saves processed data and target labels cleanly in data/processed/.
    Leaves all original datasets in data/ untouched.

    Args:
        processed_df: Preprocessed feature DataFrame.
        target_df: Binary indicator target DataFrame.
        prefix: Filename prefix.

    Returns:
        Tuple[Path, Path]: (path_to_processed_csv, path_to_labels_csv)
    """
    settings.PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    processed_path = settings.PROCESSED_DATA_PATH / f"{prefix}_processed.csv"
    labels_path = settings.PROCESSED_DATA_PATH / f"{prefix}_canonical_labels.csv"

    processed_df.to_csv(processed_path, index=False)
    target_df.to_csv(labels_path, index=False)

    logger.info(f"Saved processed dataset: {processed_path} ({len(processed_df)} rows)")
    logger.info(f"Saved canonical labels: {labels_path} ({target_df.shape[1]} classes)")

    return processed_path, labels_path
