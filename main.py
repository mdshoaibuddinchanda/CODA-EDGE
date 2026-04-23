"""
CODA — Covariance-based Domain Adaptation Pipeline
====================================================
Entry point. Runs the full pipeline end-to-end, then verifies all outputs.

Usage
-----
    python main.py
    python main.py --config configs/experiment_legal.yaml
    python main.py --config configs/default.yaml --skip-verify
"""
import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

from src.utils.config import load_config
from src.utils.device import log_device_info
from src.utils.logging_utils import setup_logger
from src.utils.memory import clear_gpu_cache, log_gpu_memory


def _stage(bar: tqdm, label: str) -> None:
    bar.set_description(f"[{label}]")


def _load_or_tokenize(domain, split, tokenizer, cfg, output_path):
    """
    Load pre-tokenized .npy if it exists (e.g. tokenized by Colab Cell 4),
    otherwise tokenize from raw JSONL cache.
    """
    import logging
    import numpy as np
    from pathlib import Path as _Path
    _log = logging.getLogger("coda")
    npy = _Path(output_path)
    if npy.exists():
        _log.info(f"Loading pre-tokenized sequences from {output_path}")
        return np.load(output_path)

    from src.data.loader import load_domain
    from src.data.validation import validate_stream
    from src.data.preprocess import tokenize_and_chunk
    texts = validate_stream(load_domain(domain, split=split, use_cache=True))
    return tokenize_and_chunk(
        texts, tokenizer, domain,
        max_seq_length=cfg.data.max_seq_length,
        stride=cfg.data.stride,
        output_path=output_path,
    )


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    logger = setup_logger("coda")
    log_device_info()
    logger.info(f"Config : {config_path}")
    logger.info(f"Model  : {cfg.model.name}")
    logger.info(f"Alpha  : {cfg.coda.alpha}  |  Layers: {cfg.coda.layer_indices}")
    logger.info(f"Domains: {cfg.data.target_domains}")

    # Gate layer: use layer 18 if present, else middle layer, else first layer
    layer_indices = cfg.coda.layer_indices
    if 18 in layer_indices:
        gate_layer = 18
    else:
        gate_layer = layer_indices[len(layer_indices) // 2]
    logger.info(f"MMD gate layer: {gate_layer}")

    domains = cfg.data.target_domains
    n_stages = 3 + len(domains)

    overall = tqdm(
        total=n_stages,
        desc="CODA pipeline",
        unit="stage",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )

    # ── 1. Load model ─────────────────────────────────────────────────────────
    _stage(overall, "Loading model")
    from src.model.model_loader import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name=cfg.model.name,
        quantization=cfg.model.quantization,
        device_map=cfg.model.device_map,
        torch_dtype_str=cfg.model.torch_dtype,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    log_gpu_memory("after model load")
    overall.update(1)

    # ── 2. Source domain — train split only ───────────────────────────────────
    _stage(overall, f"Preprocessing {cfg.data.source_domain}")

    source_seqs = _load_or_tokenize(
        domain=cfg.data.source_domain,
        split="train",
        tokenizer=tokenizer,
        cfg=cfg,
        output_path=f"data/processed/{cfg.data.source_domain}_train_tokenized.npy",
    )
    overall.update(1)

    # ── 3. Source hidden states (calibration_samples sequences from train) ────
    # Using calibration_samples and source_sequences cap from config
    _stage(overall, "Source hidden states")
    from src.model.inference import extract_hidden_states

    n_source_calib = min(cfg.coda.calibration_samples, cfg.data.source_sequences, len(source_seqs))
    logger.info(f"Using {n_source_calib} source sequences for hidden state extraction.")

    source_hidden = extract_hidden_states(
        model, source_seqs[:n_source_calib], layer_indices,
        save_dir="data/calibration", domain="source",
    )
    overall.update(1)

    # ── 4–N. Per-domain loop ──────────────────────────────────────────────────
    from src.adaptation.coda import CODAAdapter
    from src.evaluation.perplexity import compute_perplexity

    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    for domain in domains:
        _stage(overall, f"Domain: {domain}")
        logger.info(f"\n{'='*60}\nDomain: {domain}\n{'='*60}")

        with tqdm(
            total=6, desc=f"  {domain}", unit="step",
            leave=False, dynamic_ncols=True,
        ) as step:

            # ── 4a: train split → calibration sequences ───────────────────────
            step.set_description(f"  {domain} | load train")
            train_seqs = _load_or_tokenize(
                domain=domain, split="train", tokenizer=tokenizer, cfg=cfg,
                output_path=f"data/processed/{domain}_train_tokenized.npy",
            )
            step.update(1)

            # ── 4b: test split → evaluation sequences ─────────────────────────
            step.set_description(f"  {domain} | load test")
            test_seqs = _load_or_tokenize(
                domain=domain, split="test", tokenizer=tokenizer, cfg=cfg,
                output_path=f"data/processed/{domain}_test_tokenized.npy",
            )
            step.update(1)

            # ── 4c: extract target hidden states from TRAIN only ──────────────
            step.set_description(f"  {domain} | hidden states")
            n_target_calib = cfg.coda.calibration_samples
            if len(train_seqs) < n_target_calib:
                logger.warning(
                    f"[{domain}] Only {len(train_seqs)} train sequences available, "
                    f"less than calibration_samples={n_target_calib}. Using all."
                )
                n_target_calib = len(train_seqs)

            # Calibration sequences: first N from TRAIN (never from test)
            calib_seqs = train_seqs[:n_target_calib]
            target_hidden = extract_hidden_states(
                model, calib_seqs, layer_indices,
                save_dir="data/calibration", domain=domain,
            )
            step.update(1)

            # ── 4d: baseline PPL on TEST split (capped at eval_sequences) ────
            step.set_description(f"  {domain} | baseline PPL")
            eval_seqs = test_seqs[:cfg.data.eval_sequences]
            logger.info(f"[{domain}] Evaluating PPL on {len(eval_seqs)}/{len(test_seqs)} test sequences.")
            base_ppl = compute_perplexity(
                model, eval_seqs,
                output_path=str(results_dir / f"{domain}_base_ppl.json"),
            )
            logger.info(f"[{domain}] Baseline PPL: {base_ppl:.4f}")
            step.update(1)

            # ── 4e: calibrate CODA + MMD gate ─────────────────────────────────
            step.set_description(f"  {domain} | calibrate CODA")
            adapter = CODAAdapter(
                model,
                alpha=cfg.coda.alpha,
                regularization=cfg.coda.covariance_regularization,
            )
            adapter.calibrate(
                source_hidden, target_hidden, layer_indices,
                covariance_dir="outputs/covariance", domain=domain,
            )
            apply_coda = adapter.check_mmd_gate(
                source_hidden, target_hidden,
                gate_layer=gate_layer,
                threshold=cfg.coda.mmd_threshold,
            )
            step.update(1)

            # ── 4f: CODA PPL on TEST split (same capped eval_seqs) ───────────
            step.set_description(f"  {domain} | CODA PPL")
            if apply_coda:
                with adapter:
                    coda_ppl = compute_perplexity(
                        model, eval_seqs,
                        output_path=str(results_dir / f"{domain}_coda_ppl.json"),
                    )
                logger.info(
                    f"[{domain}] CODA PPL: {coda_ppl:.4f}  "
                    f"(improvement: {base_ppl - coda_ppl:+.4f})"
                )
            else:
                logger.info(f"[{domain}] CODA skipped by MMD gate — using baseline.")
                coda_ppl = base_ppl
            step.update(1)

        # Save combined metrics
        metrics = {
            "domain": domain,
            "alpha": cfg.coda.alpha,
            "calibration_split": "train",
            "evaluation_split": "test",
            "calibration_sequences": n_target_calib,
            "test_sequences": len(eval_seqs),
            "base_ppl": base_ppl,
            "coda_ppl": coda_ppl,
            "ppl_improvement": base_ppl - coda_ppl,
            "coda_applied": apply_coda,
        }
        with open(results_dir / f"{domain}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"[{domain}] Metrics saved.")

        clear_gpu_cache()
        overall.update(1)

    overall.close()
    logger.info("Pipeline complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CODA — Covariance-based Domain Adaptation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --config configs/experiment_legal.yaml
  python main.py --config configs/default.yaml --skip-verify
        """,
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="Skip output verification after pipeline completes",
    )
    args = parser.parse_args()

    run(args.config)

    if not args.skip_verify:
        from verify_outputs import verify
        ok = verify(args.config)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
