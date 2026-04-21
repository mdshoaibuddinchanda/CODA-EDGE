"""Clean and tokenize text into fixed-length sequences for LM evaluation."""
import logging
import re
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Domain-specific cleaning patterns
_WIKITEXT_HYPHEN = re.compile(r" @-@ ")
_WIKITEXT_HEADING = re.compile(r"^=+.*=+$", re.MULTILINE)
_MIMIC_PHI = re.compile(r"\[\*\*.*?\*\*\]")


def clean_text(text: str, domain: str) -> str:
    """Apply domain-specific cleaning rules."""
    if domain == "wikitext":
        text = _WIKITEXT_HYPHEN.sub("-", text)
        text = _WIKITEXT_HEADING.sub("", text)
    elif domain == "mimic":
        text = _MIMIC_PHI.sub("[REDACTED]", text)
    return text.strip()


def tokenize_and_chunk(
    texts: Iterable[str],
    tokenizer,
    domain: str,
    max_seq_length: int = 2048,
    stride: int = 512,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """
    Tokenize all texts, concatenate, and chunk into fixed-length sequences.

    Returns:
        np.ndarray of shape (n_chunks, max_seq_length) with token IDs.
        Optionally saves to output_path as .npy file.
    """
    all_ids: List[int] = []

    for text in texts:
        cleaned = clean_text(text, domain)
        if not cleaned:
            continue
        ids = tokenizer.encode(cleaned, add_special_tokens=False)
        all_ids.extend(ids)

    logger.info(f"Total tokens for '{domain}': {len(all_ids):,}")

    if len(all_ids) < max_seq_length:
        raise ValueError(
            f"Not enough tokens ({len(all_ids)}) to form a single chunk of {max_seq_length}."
        )

    # Chunk with stride
    chunks = []
    start = 0
    while start + max_seq_length <= len(all_ids):
        chunk = all_ids[start : start + max_seq_length]
        chunks.append(chunk)
        start += stride

    result = np.array(chunks, dtype=np.int32)
    logger.info(f"Created {len(chunks)} chunks of length {max_seq_length} (stride={stride})")

    # Validate: no unexpected padding tokens
    pad_id = tokenizer.pad_token_id
    if pad_id is not None:
        pad_count = (result == pad_id).sum()
        if pad_count > 0:
            logger.warning(f"Found {pad_count} pad tokens in chunked sequences.")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, result)
        logger.info(f"Saved tokenized sequences to {output_path}")

    return result
