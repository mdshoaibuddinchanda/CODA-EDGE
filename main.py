"""
CODA — Covariance-based Domain Adaptation Pipeline
====================================================
Entry point.  Runs the full pipeline end-to-end, then verifies all outputs.

Usage
-----
    python main.py                                          # default config
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


# ── helpers ──────────────────────────────────────────────────────────────────

def _stage(bar: tqdm, label: str) -> None:
    bar.set_description(f"[{label}]")


# ── main pipeline ─────────────────────────────────────────────────────────────

def run(config_path: str) -> None:
    cfg = load_config(config_path)
    logger = setup_logger("coda")
    log_device_info()
    logger.info(f"Config : {config_path}")
    logger.info(f"Model  : {cfg.model.name}")
    logger.info(f"Alpha  : {cfg.coda.alpha}  |  Layers: {cfg.coda.layer_indices}")
    logger.info(f"Domains: {cfg.data.target_domains}")

    domains = cfg.data.target_domains
    n_stages = 3 + len(domains)  # model + source-preprocess + source-hidden + N domains

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

    # ── 2. Source domain preprocessing ───────────────────────────────────────
    _stage(overall, f"Preprocessing {cfg.data.source_domain}")
    from src.data.loader import load_domain
    from src.data.validation import validate_stream
    from src.data.preprocess import tokenize_and_chunk

    source_texts = validate_stream(
        load_domain(cfg.data.source_domain, split="train", use_cache=True)
    )
    source_seqs = tokenize_and_chunk(
        source_texts, tokenizer, cfg.data.source_domain,
        max_seq_length=cfg.data.max_seq_length,
        stride=cfg.data.stride,
        output_path=f"data/processed/{cfg.data.source_domain}_tokenized.npy",
    )
    overall.update(1)

    # ── 3. Source hidden states ───────────────────────────────────────────────
    _stage(overall, "Source hidden states")
    from src.model.inference import extract_hidden_states
    source_hidden = extract_hidden_states(
        model, source_seqs[:200], cfg.coda.layer_indices,
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
            total=6,
            desc=f"  {domain}",
            unit="step",
            leave=False,
            dynamic_ncols=True,
        ) as step:

            # 4a — preprocess train
            step.set_description(f"  {domain} | preprocess train")
            target_texts = validate_stream(load_domain(domain, split="train", use_cache=True))
            target_seqs = tokenize_and_chunk(
                target_texts, tokenizer, domain,
                max_seq_length=cfg.data.max_seq_length,
                stride=cfg.data.stride,
                output_path=f"data/processed/{domain}_tokenized.npy",
            )
            step.update(1)

            # 4b — preprocess test
            step.set_description(f"  {domain} | preprocess test")
            target_test_texts = validate_stream(load_domain(domain, split="test", use_cache=True))
            target_test_seqs = tokenize_and_chunk(
                target_test_texts, tokenizer, domain,
                max_seq_length=cfg.data.max_seq_length,
                stride=cfg.data.stride,
            )
            step.update(1)

            # 4c — extract target hidden states
            step.set_description(f"  {domain} | hidden states")
            calib_seqs = target_seqs[: cfg.coda.calibration_samples]
            target_hidden = extract_hidden_states(
                model, calib_seqs, cfg.coda.layer_indices,
                save_dir="data/calibration", domain=domain,
            )
            step.update(1)

            # 4d — baseline perplexity
            step.set_description(f"  {domain} | baseline PPL")
            base_ppl = compute_perplexity(
                model, target_test_seqs,
                output_path=str(results_dir / f"{domain}_base_ppl.json"),
            )
            logger.info(f"[{domain}] Baseline PPL: {base_ppl:.4f}")
            step.update(1)

            # 4e — calibrate CODA + MMD gate
            step.set_description(f"  {domain} | calibrate CODA")
            adapter = CODAAdapter(
                model,
                alpha=cfg.coda.alpha,
                regularization=cfg.coda.covariance_regularization,
            )
            adapter.calibrate(
                source_hidden, target_hidden, cfg.coda.layer_indices,
                covariance_dir="outputs/covariance", domain=domain,
            )
            gate_layer = 18 if 18 in cfg.coda.layer_indices else cfg.coda.layer_indices[0]
            apply_coda = adapter.check_mmd_gate(
                source_hidden, target_hidden,
                gate_layer=gate_layer,
                threshold=cfg.coda.mmd_threshold,
            )
            step.update(1)

            # 4f — CODA perplexity
            step.set_description(f"  {domain} | CODA PPL")
            if apply_coda:
                with adapter:
                    coda_ppl = compute_perplexity(
                        model, target_test_seqs,
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
            "base_ppl": base_ppl,
            "coda_ppl": coda_ppl,
            "ppl_improvement": base_ppl - coda_ppl,
            "coda_applied": apply_coda,
        }
        with open(results_dir / f"{domain}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        clear_gpu_cache()
        overall.update(1)

    overall.close()
    logger.info("Pipeline complete.")


# ── entry point ───────────────────────────────────────────────────────────────

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
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip output verification step after pipeline completes",
    )
    args = parser.parse_args()

    # Run pipeline
    run(args.config)

    # Run output verification (CP8)
    if not args.skip_verify:
        from verify_outputs import verify
        ok = verify(args.config)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
