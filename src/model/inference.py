"""Run forward pass and extract hidden states at specified layers."""
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from src.utils.device import get_model_device

logger = logging.getLogger(__name__)


@contextmanager
def _register_hooks(model, layer_indices: List[int]):
    """Context manager: register forward hooks, yield collected states, then clean up."""
    num_layers = model.config.num_hidden_layers
    hidden_states: Dict[int, List[torch.Tensor]] = {idx: [] for idx in layer_indices}
    hooks = []

    def _make_hook(idx: int):
        def hook_fn(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            # Always move to CPU immediately to avoid accumulating GPU tensors
            hidden_states[idx].append(hs.detach().cpu().float())
        return hook_fn

    for idx in layer_indices:
        resolved = idx if idx >= 0 else num_layers + idx
        if resolved < 0 or resolved >= num_layers:
            raise IndexError(
                f"Layer index {idx} (resolved={resolved}) out of range "
                f"for model with {num_layers} layers."
            )
        layer = model.model.layers[resolved]
        h = layer.register_forward_hook(_make_hook(idx))
        hooks.append(h)
        logger.debug(f"Hook registered on layer {idx} (resolved={resolved})")

    try:
        yield hidden_states
    finally:
        for h in hooks:
            h.remove()
        logger.debug(f"Removed {len(hooks)} forward hooks.")


def extract_hidden_states(
    model,
    token_sequences: np.ndarray,
    layer_indices: List[int],
    batch_size: int = 4,
    pooling: str = "last",
    save_dir: Optional[str] = None,
    domain: str = "unknown",
) -> Dict[int, np.ndarray]:
    """
    Extract hidden states from specified layers over token sequences.

    Device handling
    ---------------
    Input tensors are placed on the model's own device (resolved via
    get_model_device), so this works correctly with:
    - Single CUDA GPU
    - Multi-GPU (device_map="auto")
    - Apple MPS
    - CPU

    Args:
        model:           Loaded HuggingFace causal LM.
        token_sequences: np.ndarray of shape (n_seqs, seq_len).
        layer_indices:   Layers to extract from.
        batch_size:      Sequences per forward pass.
        pooling:         'last' token or 'mean' over sequence length.
        save_dir:        If set, save per-layer arrays to disk.
        domain:          Used for naming saved files.

    Returns:
        Dict mapping layer_idx -> np.ndarray of shape (n_seqs, hidden_dim).
    """
    # Resolve the device from the model itself — handles device_map="auto"
    device = get_model_device(model)
    logger.info(
        f"Extracting hidden states: {token_sequences.shape[0]} seqs, "
        f"layers={layer_indices}, device={device}, pooling={pooling}"
    )

    n_seqs = token_sequences.shape[0]
    results: Dict[int, List[np.ndarray]] = {idx: [] for idx in layer_indices}

    with torch.no_grad():
        with _register_hooks(model, layer_indices) as hidden_states:
            batch_bar = tqdm(
                range(0, n_seqs, batch_size),
                desc=f"  hidden states ({domain})",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
            )
            for start in batch_bar:
                batch = token_sequences[start : start + batch_size]
                # Place inputs on the model's device
                input_ids = torch.tensor(batch, dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_ids)

                model(input_ids=input_ids, attention_mask=attention_mask)

                for idx in layer_indices:
                    # Hook appended one tensor per forward call; pop it
                    batch_hs = hidden_states[idx].pop()  # (batch, seq_len, hidden_dim), CPU float32
                    if pooling == "last":
                        vec = batch_hs[:, -1, :].numpy()
                    else:
                        vec = batch_hs.mean(dim=1).numpy()
                    results[idx].append(vec)

    # Stack batches and validate
    final: Dict[int, np.ndarray] = {}
    for idx in layer_indices:
        stacked = np.concatenate(results[idx], axis=0)  # (n_seqs, hidden_dim)
        final[idx] = stacked
        logger.info(f"  Layer {idx}: shape={stacked.shape}, dtype={stacked.dtype}")

        if np.isnan(stacked).any():
            logger.error(f"NaN detected in hidden states at layer {idx}!")
        if np.isinf(stacked).any():
            logger.error(f"Inf detected in hidden states at layer {idx}!")

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            out_path = f"{save_dir}/{domain}_layer{idx}_hidden.npy"
            np.save(out_path, stacked)
            logger.info(f"  Saved → {out_path}")

    return final
