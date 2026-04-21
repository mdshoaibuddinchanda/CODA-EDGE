"""Apply whitening transform with partial strength alpha."""
import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def build_whitening_matrix(
    L_source: np.ndarray,
    L_target: np.ndarray,
) -> np.ndarray:
    """
    Compute W = L_source @ inv(L_target).

    This maps target-whitened space back to source-whitened space.
    Precompute once and reuse for all samples.
    """
    L_target_inv = np.linalg.inv(L_target)
    W = L_source @ L_target_inv
    logger.debug(f"Whitening matrix W shape: {W.shape}")
    return W


def apply_whitening(
    h: np.ndarray,
    W: np.ndarray,
    mu_source: np.ndarray,
    mu_target: np.ndarray,
    alpha: float = 0.8,
) -> np.ndarray:
    """
    Apply partial whitening transform to hidden state(s).

    Formula: h' = alpha * (W @ (h - mu_t) + mu_s) + (1 - alpha) * h

    Args:
        h: Shape (d,) or (batch, d).
        W: Whitening matrix (d, d).
        mu_source: Source mean (d,).
        mu_target: Target mean (d,).
        alpha: Interpolation strength in [0, 1].

    Returns:
        Transformed hidden state, same shape as h.
    """
    batched = h.ndim == 2
    if not batched:
        h = h[np.newaxis, :]  # (1, d)

    h_shifted = h - mu_target[np.newaxis, :]       # center to target
    h_mapped = (W @ h_shifted.T).T + mu_source      # map to source space
    h_out = alpha * h_mapped + (1.0 - alpha) * h

    # Checkpoint CP5: no NaN
    if np.isnan(h_out).any():
        logger.error("NaN detected in whitened hidden states! Falling back to identity (alpha=0).")
        h_out = h

    return h_out if batched else h_out[0]


def precompute_transform(
    L_source: np.ndarray,
    L_target: np.ndarray,
    mu_source: np.ndarray,
    mu_target: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute W and the combined bias term for efficient application.

    Returns:
        (W_scaled, bias) where h' = W_scaled @ h + bias + (1-alpha)*h
    """
    W = build_whitening_matrix(L_source, L_target)
    # h' = alpha*(W@h - W@mu_t + mu_s) + (1-alpha)*h
    #     = (alpha*W + (1-alpha)*I) @ h + alpha*(mu_s - W@mu_t)
    d = W.shape[0]
    W_eff = alpha * W + (1.0 - alpha) * np.eye(d)
    bias = alpha * (mu_source - W @ mu_target)
    return W_eff, bias
