"""Orchestrate the full CODA pipeline."""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.adaptation.covariance import compute_covariance
from src.adaptation.mmd_gate import should_apply_coda
from src.adaptation.whitening import precompute_transform
from src.utils.device import get_model_device

logger = logging.getLogger(__name__)


class CODAAdapter:
    """
    Applies covariance-based domain adaptation via forward hooks on a transformer model.

    Usage:
        adapter = CODAAdapter(model, config)
        adapter.calibrate(source_hidden, target_hidden, layer_idx)
        adapter.attach_hooks()
        # run inference ...
        adapter.remove_hooks()
    """

    def __init__(self, model, alpha: float = 0.8, regularization: float = 1e-5):
        self.model = model
        self.alpha = alpha
        self.regularization = regularization
        self._hooks = []
        self._transforms: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}  # layer -> (W_eff, bias)
        self._active = False

    def calibrate(
        self,
        source_hidden: Dict[int, np.ndarray],
        target_hidden: Dict[int, np.ndarray],
        layer_indices: List[int],
        covariance_dir: Optional[str] = None,
        domain: str = "target",
    ) -> None:
        """
        Compute source/target statistics and build whitening transforms.

        Args:
            source_hidden: Dict layer_idx -> (n_s, d) array.
            target_hidden: Dict layer_idx -> (n_t, d) array.
            layer_indices: Layers to calibrate.
            covariance_dir: If set, save covariance matrices here.
            domain: Name used for file naming.
        """
        for idx in layer_indices:
            if idx not in source_hidden or idx not in target_hidden:
                logger.warning(f"Layer {idx} missing from hidden states — skipping.")
                continue

            H_s = source_hidden[idx]
            H_t = target_hidden[idx]

            src_tag = f"source_layer{idx}"
            tgt_tag = f"{domain}_layer{idx}"

            _, L_s, mu_s = compute_covariance(
                H_s, self.regularization, output_dir=covariance_dir, tag=src_tag
            )
            _, L_t, mu_t = compute_covariance(
                H_t, self.regularization, output_dir=covariance_dir, tag=tgt_tag
            )

            W_eff, bias = precompute_transform(L_s, L_t, mu_s, mu_t, self.alpha)
            self._transforms[idx] = (W_eff, bias)
            logger.info(f"CODA transform calibrated for layer {idx}.")

    def check_mmd_gate(
        self,
        source_hidden: Dict[int, np.ndarray],
        target_hidden: Dict[int, np.ndarray],
        gate_layer: int = 18,
        threshold: Optional[float] = None,
    ) -> bool:
        """Run MMD gate check on a single representative layer."""
        H_s = source_hidden.get(gate_layer)
        H_t = target_hidden.get(gate_layer)
        if H_s is None or H_t is None:
            logger.warning(f"Gate layer {gate_layer} not available — defaulting to apply CODA.")
            return True
        apply, mmd2 = should_apply_coda(H_s, H_t, threshold=threshold)
        return apply

    def attach_hooks(self) -> None:
        """Register forward hooks on calibrated layers."""
        if self._active:
            logger.warning("Hooks already attached. Call remove_hooks() first.")
            return

        num_layers = self.model.config.num_hidden_layers
        # Resolve the model's actual device (handles device_map="auto", MPS, CPU)
        model_device = get_model_device(self.model)

        for idx, (W_eff, bias) in self._transforms.items():
            resolved = idx if idx >= 0 else num_layers + idx
            layer = self.model.model.layers[resolved]

            # Pre-move transform tensors to model device as float32.
            # Inside the hook we cast to match the hidden-state dtype (fp16/bf16/fp32).
            _W = torch.tensor(W_eff, dtype=torch.float32, device=model_device)
            _b = torch.tensor(bias, dtype=torch.float32, device=model_device)

            def _make_hook(W: torch.Tensor, b: torch.Tensor):
                def hook_fn(module, input, output):
                    hs = output[0] if isinstance(output, tuple) else output
                    # Move W/b to the exact device of this hidden state
                    # (can differ per-layer with device_map="auto")
                    W_dev = W.to(hs.device)
                    b_dev = b.to(hs.device)

                    orig_dtype = hs.dtype
                    # Compute in float32 for numerical stability, then cast back
                    hs_f = hs.float()
                    transformed = (
                        (W_dev @ hs_f.transpose(-1, -2)).transpose(-1, -2) + b_dev
                    ).to(orig_dtype)

                    if torch.isnan(transformed).any():
                        logger.error(
                            f"NaN in CODA output at layer {idx} — reverting to original."
                        )
                        transformed = hs

                    if isinstance(output, tuple):
                        return (transformed,) + output[1:]
                    return transformed

                return hook_fn

            h = layer.register_forward_hook(_make_hook(_W, _b))
            self._hooks.append(h)
            logger.info(
                f"CODA hook attached: layer {idx} (resolved={resolved}), "
                f"device={model_device}"
            )

        self._active = True

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._active = False
        logger.info("All CODA hooks removed.")

    def __enter__(self):
        self.attach_hooks()
        return self

    def __exit__(self, *args):
        self.remove_hooks()
