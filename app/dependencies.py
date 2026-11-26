"""
FastAPI Dependencies & Service Providers.
Implements thread-safe singleton caching for the TriageEngine and IssuePredictor.
"""
from typing import Optional
from config.settings import settings
from src.triage_ml.prioritization.triage_engine import TriageEngine
from src.triage_ml.utils.logger import get_logger

logger = get_logger("dependencies")

# Module-level singleton
_triage_engine_instance: Optional[TriageEngine] = None


def get_triage_engine() -> TriageEngine:
    """
    Dependency provider returning the shared TriageEngine instance.
    Loads model artifacts into memory once and reuses across all requests.
    """
    global _triage_engine_instance
    if _triage_engine_instance is None:
        logger.info("Initializing TriageEngine singleton...")
        _triage_engine_instance = TriageEngine()
    return _triage_engine_instance


def get_settings():
    """Returns project configuration settings."""
    return settings
