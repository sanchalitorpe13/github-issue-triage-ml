"""
Production Logging Utility for GitHub Issue Triage ML.
Provides structured, formatted console and optional file logging.
"""
import logging
import sys
from typing import Optional
from config.settings import settings


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Creates or retrieves a standardized logger.

    Args:
        name: Name of the module/component requesting the logger.
        level: Optional log level override (e.g., 'DEBUG', 'INFO').

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if logger was already initialized
    if logger.handlers:
        return logger

    log_level_str = level or settings.LOG_LEVEL
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Console handler with clean, informative format
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)

    return logger
