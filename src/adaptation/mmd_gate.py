"""Compute MMD between source and target distributions; gate CODA application."""
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _rbf_kernel_matrix(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """Compute RBF kernel matrix K(X, Y) = exp(-||x-y||^2 / (2*sigma^2))."""
    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x.y
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    sq_dists = XX + YY.T - 2.0 * (X @ Y.T)
    sq_dists = np.maximum(sq_dists, 0.0)  # numerical safety
    return np.exp(-sq_dists / (2.0 * sigma ** 2))


def _median_bandwidth(X: np.ndarray, Y: np.ndarray, subsample: int = 200) -> float:
    """Estimate sigma as median pairwise distance (median heuristic)."""
    rng = np.random.default_rng(42)
    if len(X) > subsample:
        X = X[rng.choice(len(X), subsample, replace=False)]
    if len(Y) > subsample:
        Y = Y[rng.choice(len(Y), subsample, replace=False)]
    Z = np.concatenate([X, Y], axis=0)
    sq_dists = np.sum((Z[:, None] - Z[None, :]) ** 2, axis=-1)
    median_sq = np.median(sq_dists[sq_dists > 0])
    sigma = np.sqrt(median_sq / 2.0)
    return float(sigma) if sigma > 0 else 1.0


def compute_mmd_squared(
    H_source: np.ndarray,
    H_target: np.ndarray,
    sigma: Optional[float] = None,
    max_samples: int = 500,
) -> Tuple[float, float]:
    """
    Compute unbiased MMD^2 estimate between source and target hidden states.

    Args:
        H_source: Shape (n_s, d).
        H_target: Shape (n_t, d).
        sigma: RBF bandwidth. If None, uses median heuristic.
        max_samples: Subsample to this size to keep O(n^2) tractable.

    Returns:
        (mmd2, sigma) — MMD^2 value and bandwidth used.
    """
    rng = np.random.default_rng(0)
    if len(H_source) > max_samples:
        H_source = H_source[rng.choice(len(H_source), max_samples, replace=False)]
    if len(H_target) > max_samples:
        H_target = H_target[rng.choice(len(H_target), max_samples, replace=False)]

    if sigma is None:
        sigma = _median_bandwidth(H_source, H_target)
    logger.debug(f"MMD bandwidth sigma={sigma:.4f}")

    K_ss = _rbf_kernel_matrix(H_source, H_source, sigma)
    K_tt = _rbf_kernel_matrix(H_target, H_target, sigma)
    K_st = _rbf_kernel_matrix(H_source, H_target, sigma)

    n_s, n_t = len(H_source), len(H_target)

    # Unbiased estimator: zero out diagonal
    np.fill_diagonal(K_ss, 0.0)
    np.fill_diagonal(K_tt, 0.0)

    mmd2 = (
        K_ss.sum() / (n_s * (n_s - 1))
        + K_tt.sum() / (n_t * (n_t - 1))
        - 2.0 * K_st.mean()
    )
    logger.info(f"MMD^2 = {mmd2:.6f} (sigma={sigma:.4f})")
    return float(mmd2), float(sigma)


def should_apply_coda(
    H_source: np.ndarray,
    H_target: np.ndarray,
    threshold: Optional[float],
    sigma: Optional[float] = None,
) -> Tuple[bool, float]:
    """
    Decide whether to apply CODA based on MMD gate.

    If threshold is None, always apply CODA (no gate).

    Returns:
        (apply_coda: bool, mmd2: float)
    """
    mmd2, _ = compute_mmd_squared(H_source, H_target, sigma=sigma)

    if threshold is None:
        logger.info(f"MMD gate: no threshold set — applying CODA (MMD^2={mmd2:.6f})")
        return True, mmd2

    apply = mmd2 < threshold
    decision = "APPLY CODA" if apply else "SKIP CODA (fallback to base model)"
    logger.info(f"MMD gate: MMD^2={mmd2:.6f}, threshold={threshold:.6f} → {decision}")
    return apply, mmd2
