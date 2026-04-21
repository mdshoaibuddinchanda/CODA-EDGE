"""GPU/MPS memory utilities for the CODA pipeline."""
import logging

import torch

logger = logging.getLogger(__name__)


def log_gpu_memory(tag: str = "") -> None:
    """Log current GPU memory usage (CUDA or MPS)."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(
            f"[{tag}] CUDA memory — allocated: {allocated:.2f} GB, "
            f"reserved: {reserved:.2f} GB"
        )
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS does not expose per-allocation stats; log a note instead
        logger.info(f"[{tag}] MPS device active (memory stats not available via PyTorch API)")


def clear_gpu_cache() -> None:
    """Free unused GPU memory (CUDA or MPS)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("CUDA cache cleared.")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
        logger.debug("MPS cache cleared.")


def check_vram_limit(limit_bytes: float = 3.8e9) -> bool:
    """
    Return True if current GPU allocation is within limit.
    Always returns True on MPS/CPU (no reliable query available).
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        if allocated > limit_bytes:
            logger.warning(
                f"VRAM usage {allocated / 1e9:.2f} GB exceeds "
                f"limit {limit_bytes / 1e9:.2f} GB"
            )
            return False
    return True
