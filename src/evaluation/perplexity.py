"""Compute perplexity on tokenized test sequences."""
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from src.utils.device import get_model_device

logger = logging.getLogger(__name__)


def compute_perplexity(
    model,
    token_sequences: np.ndarray,
    batch_size: int = 4,
    output_path: Optional[str] = None,
) -> float:
    """
    Compute perplexity over tokenized sequences.

    Device handling
    ---------------
    Input tensors are placed on the model's own device via get_model_device(),
    so this works with single-GPU, multi-GPU (device_map="auto"), MPS, and CPU.

    Args:
        model:           HuggingFace causal LM (with or without CODA hooks).
        token_sequences: np.ndarray of shape (n_seqs, seq_len).
        batch_size:      Sequences per forward pass.
        output_path:     If set, save result as JSON.

    Returns:
        Perplexity (float).
    """
    device = get_model_device(model)
    logger.info(
        f"Computing perplexity: {len(token_sequences)} seqs, "
        f"batch_size={batch_size}, device={device}"
    )

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        ppl_bar = tqdm(
            range(0, len(token_sequences), batch_size),
            desc="  perplexity",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )
        for start in ppl_bar:
            batch = token_sequences[start : start + batch_size]
            # Place on model's device — handles device_map="auto" correctly
            input_ids = torch.tensor(batch, dtype=torch.long, device=device)

            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(
                    f"Invalid loss ({loss.item():.4f}) at batch {start}. Skipping."
                )
                continue

            n_tokens = input_ids.numel()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

            ppl_bar.set_postfix({"loss": f"{loss.item():.3f}"})

    if total_tokens == 0:
        raise RuntimeError("No valid tokens processed — all batches produced NaN/Inf loss.")

    mean_loss = total_loss / total_tokens
    ppl = float(np.exp(mean_loss))

    # CP7: sanity check
    if not (1.0 < ppl < 100_000):
        logger.warning(
            f"Perplexity {ppl:.2f} is outside expected range (1, 100 000). "
            "Review model outputs."
        )

    logger.info(
        f"Perplexity: {ppl:.4f}  "
        f"(mean_loss={mean_loss:.4f}, tokens={total_tokens:,})"
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "perplexity": ppl,
                    "mean_loss": mean_loss,
                    "total_tokens": total_tokens,
                    "device": str(device),
                },
                f,
                indent=2,
            )
        logger.info(f"Saved → {output_path}")

    return ppl
