"""
Checkpoint CP8: Verify all expected output files exist after a pipeline run.

Usage:
    python verify_outputs.py --config configs/experiment_legal.yaml
"""
import json
import sys
from pathlib import Path

from src.utils.config import load_config


def verify(config_path: str) -> bool:
    cfg = load_config(config_path)
    domains = cfg.data.target_domains
    layers = cfg.coda.layer_indices

    missing = []
    warnings = []

    # ── Covariance files ──────────────────────────────────────────────────────
    cov_dir = Path("outputs/covariance")
    for layer in layers:
        for suffix in ("_C.npy", "_L.npy", "_mu.npy"):
            f = cov_dir / f"source_layer{layer}{suffix}"
            if not f.exists():
                missing.append(str(f))

    for domain in domains:
        for layer in layers:
            for suffix in ("_C.npy", "_L.npy", "_mu.npy"):
                f = cov_dir / f"{domain}_layer{layer}{suffix}"
                if not f.exists():
                    missing.append(str(f))

    # ── Results files ─────────────────────────────────────────────────────────
    res_dir = Path("outputs/results")
    for domain in domains:
        # base_ppl and metrics are always written
        for fname in (f"{domain}_base_ppl.json", f"{domain}_metrics.json"):
            f = res_dir / fname
            if not f.exists():
                missing.append(str(f))

        # coda_ppl.json is only written when CODA was actually applied
        # Check metrics.json to see if it was expected
        metrics_path = res_dir / f"{domain}_metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path, "r", encoding="utf-8") as mf:
                    m = json.load(mf)
                coda_applied = m.get("coda_applied", True)
                if coda_applied:
                    coda_ppl_file = res_dir / f"{domain}_coda_ppl.json"
                    if not coda_ppl_file.exists():
                        missing.append(str(coda_ppl_file))
                else:
                    warnings.append(
                        f"{domain}: CODA was skipped by MMD gate "
                        f"(base_ppl={m.get('base_ppl', '?'):.4f})"
                    )
            except (json.JSONDecodeError, KeyError):
                warnings.append(f"{domain}: could not parse metrics.json")

    # ── Report ────────────────────────────────────────────────────────────────
    if warnings:
        print("\n⚠️  Warnings:")
        for w in warnings:
            print(f"   - {w}")

    if missing:
        print(f"\n❌ Missing {len(missing)} expected output file(s):")
        for m in missing:
            print(f"   - {m}")
        return False

    print(f"\n✅ All expected output files present for domains: {domains}")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    ok = verify(args.config)
    sys.exit(0 if ok else 1)
