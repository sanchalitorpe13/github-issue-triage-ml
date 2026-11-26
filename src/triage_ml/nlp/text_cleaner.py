"""
NLP Text Preprocessing & Structural Feature Extraction.
Specially engineered for developer issues, stack traces, and code blocks.
"""
import re
from typing import Any, Dict, Optional, Tuple
from config.settings import settings


def clean_text_for_nlp(text: Any, preserve_code_indicators: bool = True) -> str:
    """
    Cleans raw issue text for NLP vectorization while retaining technical context.

    Performs:
      - Null handling and string casting
      - Extraction/replacement of markdown code blocks with semantic placeholder tokens
      - Removal of markdown images, URLs, and HTML comments
      - Controlled normalization of whitespace and special symbols without stripping error codes

    Args:
        text: Raw issue title, body, or combined text.
        preserve_code_indicators: Whether to inject tokens like [CODE_BLOCK] for preserved structures.

    Returns:
        str: Normalized, model-ready text.
    """
    if text is None or not isinstance(text, str):
        return ""

    cleaned = text

    # Remove HTML comments often found in GitHub issue templates (<!-- ... -->)
    cleaned = re.sub(r"<!--[\s\S]*?-->", " ", cleaned)

    # Detect and replace fenced code blocks ```...```
    if preserve_code_indicators:
        cleaned = re.sub(r"```[\s\S]*?```", " [CODE_BLOCK] ", cleaned)
        cleaned = re.sub(r"`[^`\n]+`", " [INLINE_CODE] ", cleaned)
    else:
        cleaned = re.sub(r"```[\s\S]*?```", " ", cleaned)
        cleaned = re.sub(r"`[^`\n]+`", " ", cleaned)

    # Remove URLs (http://, https://)
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)

    # Remove Markdown image syntax: ![alt](url)
    cleaned = re.sub(r"!\[.*?\]\(.*?\)", " ", cleaned)

    # Remove Markdown link URLs, keeping link text: [text](url) -> text
    cleaned = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", cleaned)

    # Lowercase
    cleaned = cleaned.lower()

    # Normalize punctuation, keeping key programming and error tokens (hyphens, underscores, dots inside words)
    cleaned = re.sub(r"[^\w\s\-\_\[\]]", " ", cleaned)

    # Collapse multiple whitespaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def extract_structural_features(title: Any, body: Any) -> Dict[str, Any]:
    """
    Extracts informative structural metadata from issue title and body.
    These features serve both ML modeling and heuristic priority scoring.

    Features extracted:
      - has_code_block: bool (detected via ```)
      - has_stack_trace: bool (detected via traceback patterns like 'Traceback (most recent call last):')
      - title_length: int
      - body_length: int
      - error_keyword_count: int
      - exclamation_count: int
      - question_count: int

    Args:
        title: Issue title string.
        body: Issue body string.

    Returns:
        Dict[str, Any]: Extracted metadata feature dictionary.
    """
    title_str = str(title) if title is not None else ""
    body_str = str(body) if body is not None else ""
    full_text = f"{title_str} {body_str}"
    full_text_lower = full_text.lower()

    # Code block detection
    has_code_block = bool("```" in full_text or "<code>" in full_text.lower())

    # Stack trace patterns
    stack_trace_patterns = [
        r"traceback \(most recent call last\):",
        r"file \".*?\", line \d+",
        r"exception in thread",
        r"fatal error",
        r"segmentation fault",
        r"nullpointerexception",
        r"runtimeerror:",
        r"valueerror:",
        r"typeerror:"
    ]
    has_stack_trace = any(re.search(pat, full_text_lower) for pat in stack_trace_patterns)

    # Count error / severity keywords
    critical_words = settings.SEVERITY_KEYWORDS.get("critical", [])
    high_words = settings.SEVERITY_KEYWORDS.get("high", [])
    all_severity_words = set(critical_words + high_words)

    error_count = sum(full_text_lower.count(w) for w in all_severity_words)

    return {
        "has_code_block": int(has_code_block),
        "has_stack_trace": int(has_stack_trace),
        "title_length": len(title_str),
        "body_length": len(body_str),
        "error_keyword_count": error_count,
        "has_exclamation": int("!" in title_str),
        "has_question": int("?" in title_str),
    }
