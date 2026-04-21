"""Compute covariance matrix and Cholesky decomposition of hidden states."""
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_covariance(
    H: np.ndarray,
    regularization: float = 1e-5,
    output_dir: Optional[str] = None,
    tag: str = "source_layer0",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute covariance, Cholesky factor, and mean of hidden state matrix H.

    Args:
        H: Shape (n_samples, d).
        regularization: Ridge term added to diagonal for numerical stability.
        output_dir: If set, save C, L, mu as .npy files.
        tag: Filename prefix (e.g., 'source_layer18' or 'scotus_layer18').

    Returns:
        (C, L, mu) — covariance (d,d), Cholesky lower-triangular (d,d), mean (d,).
    """
    n_samples, d = H.shape
    if n_samples < 2:
        raise ValueError(f"Need at least 2 samples, got {n_samples}.")

    if n_samples < 2 * d:
        logger.warning(
            f"n_samples ({n_samples}) < 2*d ({2*d}). Covariance may be ill-conditioned. "
            "Consider increasing calibration_samples or regularization."
        )

    mu = H.mean(axis=0)
    H_c = H - mu
    C = (H_c.T @ H_c) / (n_samples - 1)
    C += regularization * np.eye(d)

    # Log condition number
    cond = np.linalg.cond(C)
    logger.info(f"[{tag}] Covariance condition number: {cond:.3e}")
    if cond > 1e10:
        logger.warning(f"[{tag}] Ill-conditioned covariance (cond={cond:.2e}). Increasing regularization.")
        C += regularization * 10 * np.eye(d)

    # Checkpoint CP4: positive definite
    eigvals = np.linalg.eigvalsh(C)
    if not np.all(eigvals > 0):
        logger.error(f"[{tag}] Covariance is not positive definite! Min eigenvalue: {eigvals.min():.3e}")
        raise np.linalg.LinAlgError("Covariance matrix is not positive definite.")

    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError as e:
        logger.error(f"[{tag}] Cholesky failed: {e}. Trying stronger regularization.")
        C += 1e-3 * np.eye(d)
        L = np.linalg.cholesky(C)

    logger.info(f"[{tag}] Covariance shape: {C.shape}, Cholesky computed successfully.")

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        np.save(f"{output_dir}/{tag}_C.npy", C)
        np.save(f"{output_dir}/{tag}_L.npy", L)
        np.save(f"{output_dir}/{tag}_mu.npy", mu)
        logger.info(f"[{tag}] Saved C, L, mu to {output_dir}/")

    return C, L, mu
