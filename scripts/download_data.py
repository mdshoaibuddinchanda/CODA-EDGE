"""
Standalone data download script.

Downloads all datasets defined in the config to data/raw/ before running
the main experiment.  Safe to re-run — already-cached domains are skipped.

Usage:
    python scripts/download_data.py                          # all domains in default.yaml
    python scripts/download_data.py --config configs/experiment_legal.yaml
    python scripts/download_data.py --domains wikitext scotus
    python scripts/download_data.py --domains mimic --local-path /path/to/NOTEEVENTS.csv
"""
import argparse
import logging
import sys
from pathlib import Path

# Make sure project root is on the path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from src.data.loader import DATASET_REGISTRY, _cache_path, load_domain
from src.utils.config import load_config
from src.utils.logging_utils import setup_logger

logger = setup_logger("download_data")

SPLITS = ["train", "test"]  # validation used only for ablations; add manually if needed


def download_domain(
    domain: str,
    splits: list,
    local_path: str = None,
    max_samples: int = None,
) -> None:
    """Download (or verify cache for) all splits of a domain."""
    for split in tqdm(splits, desc=f"{domain}", unit="split", leave=False, dynamic_ncols=True):
        cache = _cache_path(domain, split)
        if cache.exists():
            size_mb = cache.stat().st_size / 1e6
            logger.info(f"  ✓ {domain}/{split} already cached ({size_mb:.1f} MB) — skipping.")
            continue

        logger.info(f"  ↓ Downloading {domain}/{split} …")
        count = 0
        # Consume the generator — loader handles caching internally
        for _ in load_domain(
            domain,
            split=split,
            streaming=True,
            local_path=local_path if split == "train" else None,
            use_cache=True,
            max_samples=max_samples,
        ):
            count += 1

        logger.info(f"  ✓ {domain}/{split}: {count:,} samples downloaded.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CODA datasets to data/raw/")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Override domains to download (default: all from config)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=SPLITS,
        choices=["train", "validation", "test"],
        help="Which splits to download (default: train test)",
    )
    parser.add_argument(
        "--local-path",
        default=None,
        help="Local file path for domains that require it (e.g. MIMIC-III CSV)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap samples per split (useful for quick smoke tests)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    all_domains = [cfg.data.source_domain] + cfg.data.target_domains

    domains = args.domains if args.domains else all_domains

    # Validate requested domains
    unknown = [d for d in domains if d not in DATASET_REGISTRY]
    if unknown:
        logger.error(f"Unknown domains: {unknown}. Available: {list(DATASET_REGISTRY.keys())}")
        sys.exit(1)

    logger.info(f"Downloading {len(domains)} domain(s): {domains}")
    logger.info(f"Splits: {args.splits}")
    logger.info(f"Output directory: data/raw/")

    domain_bar = tqdm(domains, desc="Overall", unit="domain", dynamic_ncols=True)
    for domain in domain_bar:
        domain_bar.set_description(f"Downloading: {domain}")
        info = DATASET_REGISTRY[domain]

        if info["hf_path"] is None and args.local_path is None:
            fallback = info.get("fallback")
            logger.warning(
                f"Domain '{domain}' requires credentialed access (e.g. MIMIC-III PhysioNet). "
                f"Provide --local-path to use your own file, "
                f"or the pipeline will fall back to '{fallback}'."
            )
            if fallback:
                logger.info(f"  → Downloading fallback '{fallback}' instead.")
                download_domain(fallback, args.splits, max_samples=args.max_samples)
            continue

        download_domain(
            domain,
            args.splits,
            local_path=args.local_path if domain == "mimic" else None,
            max_samples=args.max_samples,
        )

    domain_bar.close()
    logger.info("\nAll downloads complete. Files are in data/raw/")
    logger.info("Run the experiment with: python main.py --config " + args.config)


if __name__ == "__main__":
    main()
