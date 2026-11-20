"""
Safe File I/O Utilities for Data and Model Artifacts.
Provides atomic operations with directory creation and clean error handling.
"""
import json
from pathlib import Path
from typing import Any, Union
import joblib


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> Path:
    """
    Safely serializes data to a JSON file, automatically creating parent directories.

    Args:
        data: Data structure to serialize.
        path: Destination file path.
        indent: JSON indentation formatting.

    Returns:
        Path: Resolved file path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    return target


def load_json(path: Union[str, Path]) -> Any:
    """
    Loads JSON data from file.

    Args:
        path: Path to the JSON file.

    Returns:
        Deserialized JSON content.
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"JSON file not found at: {target}")
    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)


def save_joblib(obj: Any, path: Union[str, Path], compress: int = 3) -> Path:
    """
    Serializes a Python object or scikit-learn model using joblib.

    Args:
        obj: Object / model / vectorizer to serialize.
        path: Destination path.
        compress: Compression level (0-9).

    Returns:
        Path: Resolved file path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, target, compress=compress)
    return target


def load_joblib(path: Union[str, Path]) -> Any:
    """
    Loads a joblib serialized object or model.

    Args:
        path: Path to the serialized file.

    Returns:
        Deserialized Python object.
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Serialized file not found at: {target}")
    return joblib.load(target)
