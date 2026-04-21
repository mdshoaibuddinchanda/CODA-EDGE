"""
Central device resolution for the CODA pipeline.

All modules import `get_device()` and `get_model_device()` from here
instead of scattering `torch.cuda.is_available()` checks everywhere.

Priority order
--------------
1. CUDA  — NVIDIA GPU (single or multi via device_map)
2. MPS   — Apple Silicon (M1/M2/M3)
3. CPU   — fallback
"""
import logging
from functools import lru_cache
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_device(preferred: Optional[str] = None) -> torch.device:
    """
    Return the best available device.

    Args:
        preferred: Force a specific device string (e.g. 'cpu', 'cuda:1').
                   If None, auto-selects best available.

    Returns:
        torch.device
    """
    if preferred is not None:
        device = torch.device(preferred)
        logger.info(f"Using forced device: {device}")
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        logger.info(
            f"CUDA device: {props.name}  "
            f"({props.total_memory / 1e9:.1f} GB VRAM, "
            f"compute {props.major}.{props.minor})"
        )
        return device

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("MPS device: Apple Silicon GPU")
        return device

    device = torch.device("cpu")
    logger.info("No GPU found — using CPU")
    return device


def get_model_device(model) -> torch.device:
    """
    Return the device of the first model parameter.

    For models loaded with device_map='auto' (multi-GPU or CPU offload),
    input tensors should go to the device of the embedding layer, not
    necessarily 'cuda:0'.  This handles that correctly.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return get_device()


def move_to_device(tensor: torch.Tensor, model) -> torch.Tensor:
    """
    Move a tensor to the same device as the model's first parameter.

    Use this instead of `.to("cuda")` everywhere so multi-GPU and MPS
    setups work without changes.
    """
    target = get_model_device(model)
    return tensor.to(target)


def log_device_info() -> None:
    """Log a full summary of available compute devices."""
    logger.info(f"PyTorch version : {torch.__version__}")
    logger.info(f"CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            logger.info(
                f"  cuda:{i}  {p.name}  "
                f"{p.total_memory / 1e9:.1f} GB  "
                f"compute {p.major}.{p.minor}"
            )
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    logger.info(f"MPS available   : {mps_ok}")
    logger.info(f"Selected device : {get_device()}")
