"""Validate and filter raw text samples."""
import logging
from typing import Generator, Iterable

logger = logging.getLogger(__name__)

MIN_CHAR_LENGTH = 50
MAX_REJECTION_RATIO = 0.50  # abort if more than 50% of samples are filtered


def _is_english(text: str) -> bool:
    """Best-effort English detection without heavy dependencies."""
    try:
        from langdetect import detect
        return detect(text) == "en"
    except Exception:
        # If langdetect not installed or fails, assume English
        return True


def validate_stream(
    texts: Iterable[str],
    min_length: int = MIN_CHAR_LENGTH,
    check_language: bool = False,
) -> Generator[str, None, None]:
    """
    Filter text samples by length and optionally language.

    Raises RuntimeError if rejection ratio exceeds MAX_REJECTION_RATIO.
    """
    total = 0
    rejected = 0

    for text in texts:
        total += 1
        if len(text) < min_length:
            rejected += 1
            logger.debug(f"Rejected (too short, {len(text)} chars): {text[:40]!r}")
            continue
        if check_language and not _is_english(text):
            rejected += 1
            logger.debug(f"Rejected (non-English): {text[:40]!r}")
            continue
        yield text

    if total == 0:
        raise RuntimeError("No samples received by validator — check data loader.")

    ratio = rejected / total
    logger.info(f"Validation: {total} total, {rejected} rejected ({ratio:.1%})")

    if ratio > MAX_REJECTION_RATIO:
        raise RuntimeError(
            f"Rejection ratio {ratio:.1%} exceeds threshold {MAX_REJECTION_RATIO:.1%}. "
            "Check data source or preprocessing."
        )
