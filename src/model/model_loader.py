"""Load Phi-3-mini in 4-bit quantization with memory-efficient settings."""
import logging
from pathlib import Path
from typing import Tuple

import torch

from src.utils.device import get_device, log_device_info

logger = logging.getLogger(__name__)

MODEL_CACHE_DIR = Path("models/phi3-4bit")


def load_model_and_tokenizer(
    model_name: str = "microsoft/Phi-3-mini-4k-instruct",
    quantization: str = "4bit",
    device_map: str = "auto",
    torch_dtype_str: str = "float16",
    trust_remote_code: bool = True,
) -> Tuple:
    """
    Load model and tokenizer with optional 4-bit quantization.

    Device handling
    ---------------
    - CUDA  : 4-bit NF4 via bitsandbytes, device_map="auto"
    - MPS   : bitsandbytes not supported; falls back to float16 / bfloat16
    - CPU   : bitsandbytes not supported; falls back to float32

    Returns:
        (model, tokenizer) tuple.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log_device_info()

    device = get_device()
    is_cuda = device.type == "cuda"
    is_mps = device.type == "mps"

    # ── dtype selection ───────────────────────────────────────────────────────
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(torch_dtype_str, torch.float16)

    # MPS: float16 is supported for inference but bfloat16 is safer on M-series
    if is_mps and torch_dtype == torch.float16:
        logger.info("MPS detected — using bfloat16 for better numerical stability.")
        torch_dtype = torch.bfloat16

    # CPU: use float32 to avoid precision issues
    if not is_cuda and not is_mps:
        logger.info("CPU detected — using float32.")
        torch_dtype = torch.float32

    # ── tokenizer ─────────────────────────────────────────────────────────────
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(MODEL_CACHE_DIR),
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── quantization config (CUDA only) ───────────────────────────────────────
    quant_config = None
    if quantization == "4bit" and is_cuda:
        try:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            logger.info("4-bit NF4 quantization config created.")
        except Exception as e:
            logger.warning(f"BitsAndBytes 4-bit config failed: {e}. Falling back to full precision.")
            quant_config = None
    elif quantization == "4bit" and not is_cuda:
        logger.warning(
            f"4-bit quantization requires CUDA. "
            f"Running on {device.type} — loading in {torch_dtype} instead."
        )

    # ── device_map ────────────────────────────────────────────────────────────
    # device_map="auto" only makes sense with CUDA (multi-GPU / CPU offload).
    # For MPS and CPU, load directly onto the device.
    effective_device_map: object
    if is_cuda:
        effective_device_map = device_map  # "auto" or explicit map
    else:
        effective_device_map = {"": device.type}  # e.g. {"": "mps"} or {"": "cpu"}

    # ── model load ────────────────────────────────────────────────────────────
    logger.info(f"Loading model: {model_name}  (device_map={effective_device_map})")
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
        warnings.filterwarnings("ignore", message=".*resume_download.*deprecated.*")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map=effective_device_map,
            torch_dtype=torch_dtype,
            attn_implementation="eager",
            cache_dir=str(MODEL_CACHE_DIR),
            trust_remote_code=trust_remote_code,
        )
    model.eval()

    # ── CP3: VRAM check ───────────────────────────────────────────────────────
    if is_cuda:
        allocated_gb = torch.cuda.memory_allocated() / 1e9
        logger.info(f"VRAM after model load: {allocated_gb:.2f} GB")
        if allocated_gb > 3.8:
            logger.warning(
                "VRAM exceeds 3.8 GB — consider CPU offload or reducing batch size."
            )

    logger.info(
        f"Model ready.  dtype={torch_dtype}  "
        f"quantization={'4bit NF4' if quant_config else 'none'}  "
        f"device={effective_device_map}"
    )
    return model, tokenizer
