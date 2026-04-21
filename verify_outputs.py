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

    # Covariance files
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

    # Results files
    res_dir = Path("outputs/results")
    for domain in domains:
        for fname in (f"{domain}_base_ppl.json", f"{domain}_coda_ppl.json", f"{domain}_metrics.json"):
            f = res_dir / fname
            if not f.exists():
                missing.append(str(f))

    if missing:
        print(f"\n❌ Missing {len(missing)} expected output file(s):")
        for m in missing:
            print(f"   - {m}")
        return False
    else:
        print(f"\n✅ All expected output files present for domains: {domains}")
        return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    ok = verify(args.config)
    sys.exit(0 if ok else 1)
