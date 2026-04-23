"""Compute additional evaluation metrics: zero-shot accuracy, generation diversity."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from src.utils.device import get_model_device

logger = logging.getLogger(__name__)


def compute_lambada_accuracy(
    model,
    tokenizer,
    examples: List[Dict],
) -> float:
    """
    Zero-shot LAMBADA-style accuracy: predict the last word of each passage.

    Args:
        examples: List of dicts with 'context' and 'target' keys.

    Returns:
        Exact-match accuracy (float in [0, 1]).
    """
    device = get_model_device(model)
    model.eval()
    correct = 0

    with torch.no_grad():
        for ex in examples:
            context = ex["context"]
            target = ex["target"].strip()

            input_ids = tokenizer.encode(context, return_tensors="pt").to(device)
            output = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated = tokenizer.decode(
                output[0][input_ids.shape[1]:], skip_special_tokens=True
            )
            predicted = generated.strip().split()[0] if generated.strip() else ""

            if predicted.lower() == target.lower():
                correct += 1

    accuracy = correct / len(examples) if examples else 0.0
    logger.info(f"LAMBADA accuracy: {accuracy:.4f} ({correct}/{len(examples)})")
    return accuracy


def compute_distinct_n(texts: List[str], n: int = 2) -> float:
    """
    Compute Distinct-N: ratio of unique n-grams to total n-grams (generation diversity).
    """
    all_ngrams = []
    for text in texts:
        tokens = text.split()
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)

    if not all_ngrams:
        return 0.0
    distinct = len(set(all_ngrams)) / len(all_ngrams)
    logger.info(f"Distinct-{n}: {distinct:.4f}")
    return distinct


def collect_and_save_metrics(
    metrics: Dict,
    output_path: str,
) -> None:
    """Save metrics dict to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {output_path}")
