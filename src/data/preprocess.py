"""Clean and tokenize text into fixed-length sequences for LM evaluation."""
import logging
import re
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_WIKITEXT_HYPHEN = re.compile(r" @-@ ")
_WIKITEXT_HEADING = re.compile(r"^=+.*=+$", re.MULTILINE)
_MIMIC_PHI = re.compile(r"\[\*\*.*?\*\*\]")

# Minimum characters before we bother tokenizing
_MIN_CHARS = 20


def clean_text(text: str, domain: str) -> str:
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
    batch_size: int = 1000,
) -> np.ndarray:
    """
    Tokenize texts in batches, concatenate all tokens, chunk into fixed-length sequences.

    Uses batch_encode_plus for ~10x speedup over per-document encode().
    """
    all_ids: List[int] = []
    batch: List[str] = []
    total_docs = 0
    skipped = 0

    def _flush(batch):
        if not batch:
            return
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        for ids in encoded["input_ids"]:
            all_ids.extend(ids)

    for text in texts:
        cleaned = clean_text(text, domain)
        if len(cleaned) < _MIN_CHARS:
            skipped += 1
            continue
        batch.append(cleaned)
        total_docs += 1
        if len(batch) >= batch_size:
            _flush(batch)
            batch = []
            if total_docs % 50000 == 0:
                logger.info(f"  [{domain}] tokenized {total_docs:,} docs, {len(all_ids):,} tokens so far")

    _flush(batch)  # flush remainder

    logger.info(
        f"[{domain}] {total_docs:,} docs tokenized, {skipped:,} skipped, "
        f"{len(all_ids):,} total tokens"
    )

    if len(all_ids) < max_seq_length:
        raise ValueError(
            f"Not enough tokens ({len(all_ids):,}) for chunk size {max_seq_length}. "
            f"Download more data or reduce max_seq_length."
        )

    # Chunk with stride
    chunks = []
    start = 0
    while start + max_seq_length <= len(all_ids):
        chunks.append(all_ids[start: start + max_seq_length])
        start += stride

    result = np.array(chunks, dtype=np.int32)
    logger.info(f"[{domain}] {len(chunks):,} chunks of length {max_seq_length} (stride={stride})")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, result)
        logger.info(f"[{domain}] Saved → {output_path}")

    return result
