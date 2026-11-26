"""
Centralized Configuration & Settings for GitHub Issue Triage ML 2.0.
Loads settings from environment variables with safe defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


class Settings:
    """Project-wide settings and configuration parameters."""

    # Project metadata
    PROJECT_NAME: str = "GitHub Issue Triage ML"
    VERSION: str = "2.0.0"

    # Filesystem Directories
    ROOT_PATH: Path = ROOT_DIR
    CONFIG_PATH: Path = ROOT_DIR / "config"
    DATA_PATH: Path = ROOT_DIR / "data"
    RAW_DATA_PATH: Path = ROOT_DIR / "data" / "raw"
    PROCESSED_DATA_PATH: Path = ROOT_DIR / "data" / "processed"
    MODELS_PATH: Path = ROOT_DIR / "models"
    NOTEBOOKS_PATH: Path = ROOT_DIR / "notebooks"
    APP_PATH: Path = ROOT_DIR / "app"
    BACKUP_PATH: Path = ROOT_DIR / "backup"
    LOGS_PATH: Path = ROOT_DIR / "logs"

    # Environment & Logging
    APP_ENV: str = os.getenv("APP_ENV", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # API Configuration
    API_TITLE: str = "AI Issue Triage System"
    API_DESCRIPTION: str = "Production AI Platform for Classifying and Prioritizing GitHub Issues using NLP and Machine Learning"
    API_VERSION: str = "2.0.0"
    CORS_ORIGINS: list = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",") if origin.strip()]
    MAX_BATCH_SIZE: int = 50
    MAX_TITLE_LENGTH: int = 500
    MAX_BODY_LENGTH: int = 50000

    # GitHub API Configuration
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
    GITHUB_API_BASE: str = "https://api.github.com"
    API_TIMEOUT: int = 15  # seconds
    DEFAULT_PER_PAGE: int = 100
    MAX_PAGES_DEFAULT: int = 10

    # Default Repositories for Multi-Repo Corpus
    DEFAULT_BENCHMARK_REPOSITORIES = [
        ("pallets", "flask"),
        ("psf", "requests"),
        ("tiangolo", "fastapi"),
        ("scikit-learn", "scikit-learn"),
        ("streamlit", "streamlit"),
        ("facebook", "react"),
    ]

    # Canonical Label Taxonomy for Issue Type Classification
    # Maps noisy, diverse community tags into 4 primary high-level categories
    LABEL_TAXONOMY = {
        "Bug": [
            "bug", "type:bug", "type: bug", "kind/bug", "defect",
            "regression", "type:regression", "bugfix", "error", "crash"
        ],
        "Feature": [
            "feature", "enhancement", "type:enhancement", "type: feature",
            "kind/feature", "feature request", "new feature", "proposal", "idea"
        ],
        "Documentation": [
            "documentation", "docs", "type:docs", "type:documentation",
            "kind/documentation", "doc-bug", "typo", "readme"
        ],
        "Question": [
            "question", "type:question", "support", "help wanted",
            "needs help", "discussion", "troubleshooting", "usage"
        ]
    }

    # Priority Levels & Operational SLA Definitions
    PRIORITY_LEVELS = {
        "P0": {
            "name": "Critical",
            "sla_hours": 2,
            "description": "Blocker, system crash, severe data loss, or active security advisory",
            "color": "#FF4B4B"
        },
        "P1": {
            "name": "High",
            "sla_hours": 8,
            "description": "Major feature broken or regression with no trivial workaround",
            "color": "#FFA726"
        },
        "P2": {
            "name": "Medium",
            "sla_hours": 24,
            "description": "Normal bug, performance issue, or important non-blocking feature",
            "color": "#29B5E8"
        },
        "P3": {
            "name": "Low",
            "sla_hours": 72,
            "description": "Minor defect, non-critical query, or standard enhancement",
            "color": "#00CC96"
        },
        "P4": {
            "name": "Trivial",
            "sla_hours": 168,
            "description": "Cosmetic defect, typo, minor formatting, or nice-to-have suggestion",
            "color": "#9E9E9E"
        }
    }

    # Priority Scoring Rule Weights (Heuristic Engine)
    PRIORITY_WEIGHTS = {
        "critical_failure": 35,       # crash, segfault, panic, core dump
        "security_vulnerability": 35, # CVE, exploit, remote code execution
        "data_loss": 30,              # data loss, database corruption
        "production_outage": 30,      # service down, 500 error in prod
        "regression": 20,             # regression introduced in recent version
        "major_breakage": 20,         # cannot start, fails to build, broken functionality
        "performance_leak": 15,       # memory leak, high CPU, freeze
        "has_stack_trace": 10,        # confirmed exception traceback
        "workaround_available": -15,  # mitigates urgency if user has working workaround
        "feature_penalty": -15,       # features are generally not urgent unless blocking
        "doc_typo_penalty": -25,      # docs and typos are low priority
    }

    # High-Urgency Heuristic Trigger Words (Used by the Hybrid Prioritization Engine)
    SEVERITY_KEYWORDS = {
        "critical": [
            "critical", "fatal", "crash", "segfault", "segmentation fault",
            "data loss", "corrupt", "panic", "exploit", "cve", "security",
            "vulnerability", "deadlock", "blocking", "broken in production"
        ],
        "high": [
            "regression", "failure", "cannot start", "fails to build",
            "unhandled exception", "memory leak", "timeout", "broken",
            "500 internal server error", "infinite loop"
        ],
        "low": [
            "typo", "cosmetic", "spelling", "docstring", "nice to have",
            "minor", "clarification", "formatting", "nitpick"
        ]
    }

    # ML Model Hyperparameter Defaults
    TFIDF_WORD_MAX_FEATURES: int = 5000
    TFIDF_CHAR_MAX_FEATURES: int = 2500
    RANDOM_STATE: int = 42
    TEST_SPLIT_SIZE: float = 0.20
    VAL_SPLIT_SIZE: float = 0.10

    def get_github_headers(self) -> dict:
        """Returns standard headers for GitHub API requests with token if configured."""
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "GitHub-Issue-Triage-ML-2.0"
        }
        if self.GITHUB_TOKEN and self.GITHUB_TOKEN.strip() and "your_github" not in self.GITHUB_TOKEN:
            headers["Authorization"] = f"token {self.GITHUB_TOKEN.strip()}"
        return headers

    def ensure_directories(self) -> None:
        """Ensures all essential directories exist on the filesystem."""
        for path in [
            self.RAW_DATA_PATH,
            self.PROCESSED_DATA_PATH,
            self.MODELS_PATH,
            self.NOTEBOOKS_PATH,
            self.LOGS_PATH,
            self.BACKUP_PATH
        ]:
            path.mkdir(parents=True, exist_ok=True)


# Global singleton instance
settings = Settings()
