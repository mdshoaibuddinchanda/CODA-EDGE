"""Generate figures for paper: robustness curves, ablation charts, layer-wise PPL."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

PLOTS_DIR = Path("outputs/plots")


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_alpha_ablation(
    domain: str,
    alphas: List[float],
    results_dir: str = "outputs/results",
    output_dir: Optional[str] = None,
) -> None:
    """Line plot: PPL vs. alpha for a given domain."""
    import matplotlib.pyplot as plt

    out_dir = Path(output_dir or PLOTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    ppls = []
    for alpha in alphas:
        path = f"{results_dir}/{domain}_coda_alpha{alpha}_ppl.json"
        try:
            data = _load_json(path)
            ppls.append(data["perplexity"])
        except (FileNotFoundError, KeyError) as e:
            logger.warning(f"Missing result for alpha={alpha}: {e}")
            ppls.append(None)

    valid = [(a, p) for a, p in zip(alphas, ppls) if p is not None]
    if not valid:
        logger.error(f"No valid alpha ablation data for domain '{domain}'.")
        return

    xs, ys = zip(*valid)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("Alpha (α)")
    ax.set_ylabel("Perplexity")
    ax.set_title(f"Alpha Ablation — {domain}")
    ax.grid(True)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"alpha_ablation_{domain}.{ext}", dpi=150)
    plt.close(fig)
    logger.info(f"Saved alpha ablation plot for '{domain}'.")


def plot_layer_ablation(
    domain: str,
    layer_indices: List[int],
    results_dir: str = "outputs/results",
    output_dir: Optional[str] = None,
) -> None:
    """Bar chart: PPL vs. layer index."""
    import matplotlib.pyplot as plt

    out_dir = Path(output_dir or PLOTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    ppls = []
    labels = []
    for idx in layer_indices:
        path = f"{results_dir}/{domain}_coda_layer{idx}_ppl.json"
        try:
            data = _load_json(path)
            ppls.append(data["perplexity"])
            labels.append(str(idx))
        except (FileNotFoundError, KeyError) as e:
            logger.warning(f"Missing result for layer={idx}: {e}")

    if not ppls:
        logger.error(f"No layer ablation data for domain '{domain}'.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, ppls)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Perplexity")
    ax.set_title(f"Layer Ablation — {domain}")
    ax.grid(axis="y")
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"layer_ablation_{domain}.{ext}", dpi=150)
    plt.close(fig)
    logger.info(f"Saved layer ablation plot for '{domain}'.")


def plot_robustness_curve(
    domain: str,
    shift_coefficients: List[float],
    ppls_base: List[float],
    ppls_coda: List[float],
    output_dir: Optional[str] = None,
) -> None:
    """Line plot: PPL vs. distribution shift coefficient for base vs. CODA."""
    import matplotlib.pyplot as plt

    out_dir = Path(output_dir or PLOTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(shift_coefficients, ppls_base, marker="s", label="Base Model")
    ax.plot(shift_coefficients, ppls_coda, marker="o", label="CODA")
    ax.set_xlabel("Shift Coefficient")
    ax.set_ylabel("Perplexity")
    ax.set_title(f"Robustness Curve — {domain}")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"robustness_curve_{domain}.{ext}", dpi=150)
    plt.close(fig)
    logger.info(f"Saved robustness curve for '{domain}'.")


def plot_mmd_comparison(
    domains: List[str],
    mmd_values: List[float],
    kl_values: Optional[List[float]] = None,
    cosine_values: Optional[List[float]] = None,
    output_dir: Optional[str] = None,
) -> None:
    """Bar chart comparing MMD, KL, and Cosine distance across domains."""
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir = Path(output_dir or PLOTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(domains))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, mmd_values, width, label="MMD²")
    if kl_values:
        ax.bar(x + width, kl_values, width, label="KL Divergence")
    if cosine_values:
        ax.bar(x + 2 * width, cosine_values, width, label="Cosine Distance")

    ax.set_xticks(x + width)
    ax.set_xticklabels(domains)
    ax.set_ylabel("Distance")
    ax.set_title("Distribution Distance Comparison")
    ax.legend()
    ax.grid(axis="y")
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"mmd_comparison.{ext}", dpi=150)
    plt.close(fig)
    logger.info("Saved MMD comparison plot.")
