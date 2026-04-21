"""
Load raw datasets from HuggingFace or local files.

Download behaviour
------------------
Each domain is fetched from HuggingFace (streaming or cached) and the raw
text is also written to  data/raw/<domain>_<split>.jsonl  so subsequent runs
can skip the network entirely.  Retry logic with exponential back-off handles
transient network failures.
"""
import json
import logging
import time
from pathlib import Path
from typing import Generator, Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")

# Registry: domain -> (hf_path, hf_config, text_field, hf_split_map)
# hf_split_map lets us translate generic split names to dataset-specific ones.
DATASET_REGISTRY = {
    "wikitext": {
        "hf_path": "Salesforce/wikitext",
        "hf_config": "wikitext-103-v1",
        "text_field": "text",
        # wikitext has train / validation / test
        "splits": {"train": "train", "validation": "validation", "test": "test"},
    },
    # pile-of-law only has train / validation (75/25 split, no separate test)
    # courtlistener_opinions covers all US federal courts including SCOTUS
    "scotus": {
        "hf_path": "pile-of-law/pile-of-law",
        "hf_config": "courtlistener_opinions",
        "text_field": "text",
        "splits": {"train": "train", "validation": "validation", "test": "validation"},
    },
    # Same underlying config — federal circuit opinions are a subset of courtlistener
    "federal_circuit": {
        "hf_path": "pile-of-law/pile-of-law",
        "hf_config": "courtlistener_opinions",
        "text_field": "text",
        "splits": {"train": "train", "validation": "validation", "test": "validation"},
    },
    # scientific_papers has train / validation only
    "arxiv": {
        "hf_path": "scientific_papers",
        "hf_config": "arxiv",
        "text_field": "abstract",
        "splits": {"train": "train", "validation": "validation", "test": "validation"},
    },
    # MIMIC-III requires credentialed PhysioNet access.
    # Fallback: PubMed abstracts (open access).
    "mimic": {
        "hf_path": None,  # local only — pass local_path= or use fallback
        "hf_config": None,
        "text_field": "text",
        "splits": {},
        "fallback": "pubmed",
    },
    "pubmed": {
        "hf_path": "pubmed_qa",
        "hf_config": "pqa_unlabeled",
        "text_field": "long_answer",
        "splits": {"train": "train", "validation": "train", "test": "train"},
    },
}

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0  # seconds


def _hf_load_with_retry(hf_path: str, hf_config: str, split: str, streaming: bool):
    """Call load_dataset with exponential back-off on transient failures."""
    from datasets import load_dataset

    last_exc = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return load_dataset(
                hf_path,
                hf_config,
                split=split,
                streaming=streaming,
                trust_remote_code=True,
            )
        except Exception as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    f"Download attempt {attempt}/{_MAX_RETRIES} failed: {exc}. "
                    f"Retrying in {delay:.0f}s…"
                )
                time.sleep(delay)
            else:
                logger.error(f"All {_MAX_RETRIES} download attempts failed for {hf_path}/{hf_config}.")
    raise last_exc


def _cache_path(domain: str, split: str) -> Path:
    return RAW_DIR / f"{domain}_{split}.jsonl"


def _write_cache(path: Path, records: list) -> None:
    """Write list of text strings to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for text in records:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    logger.info(f"Cached {len(records):,} records → {path}  ({path.stat().st_size / 1e6:.1f} MB)")


def _read_cache(path: Path) -> Generator[str, None, None]:
    """Yield text strings from a JSONL cache file."""
    logger.info(f"Loading from local cache: {path}")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)["text"]


def load_domain(
    domain: str,
    split: str = "train",
    streaming: bool = True,
    local_path: Optional[str] = None,
    use_cache: bool = True,
    max_samples: Optional[int] = None,
) -> Generator[str, None, None]:
    """
    Yield cleaned text strings for a given domain.

    Download flow
    -------------
    1. If local_path is given, read directly from that file.
    2. If a local cache exists at data/raw/<domain>_<split>.jsonl, use it.
    3. Otherwise download from HuggingFace (with retry), cache to data/raw/,
       then yield.

    Args:
        domain:      Key in DATASET_REGISTRY.
        split:       'train', 'validation', or 'test'.
        streaming:   Use HF streaming (avoids full download for large datasets).
                     When streaming=False the full dataset is downloaded and cached.
        local_path:  Override — read from this local file instead.
        use_cache:   If True, read/write the data/raw/ JSONL cache.
        max_samples: Stop after this many samples (useful for quick tests).
    """
    # --- 1. Local file override ---
    if local_path:
        yield from _load_local(local_path, max_samples=max_samples)
        return

    if domain not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown domain '{domain}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    info = DATASET_REGISTRY[domain]

    # --- MIMIC-III: no HF path, try local or fallback ---
    if info["hf_path"] is None:
        fallback = info.get("fallback")
        logger.warning(
            f"Domain '{domain}' requires credentialed local access. "
            f"Pass local_path= to load_domain(), or using fallback '{fallback}'."
        )
        if fallback:
            yield from load_domain(
                fallback, split=split, streaming=streaming,
                use_cache=use_cache, max_samples=max_samples,
            )
        return

    # --- 2. Local cache hit ---
    cache_file = _cache_path(domain, split)
    if use_cache and cache_file.exists():
        count = 0
        for text in _read_cache(cache_file):
            yield text
            count += 1
            if max_samples and count >= max_samples:
                return
        return

    # --- 3. Download from HuggingFace ---
    hf_split = info["splits"].get(split, split)
    text_field = info["text_field"]

    logger.info(
        f"Downloading '{domain}' ({split}) from "
        f"{info['hf_path']}/{info['hf_config']} …"
    )

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    ds = _hf_load_with_retry(
        info["hf_path"], info["hf_config"], hf_split, streaming=streaming
    )

    records: list = []
    count = 0

    desc = f"Downloading {domain}/{split}"
    with tqdm(desc=desc, unit=" docs", dynamic_ncols=True) as pbar:
        for sample in ds:
            text = sample.get(text_field, "") or ""
            text = text.strip()
            if not text:
                continue
            records.append(text)
            yield text
            count += 1
            pbar.update(1)
            if max_samples and count >= max_samples:
                break

    logger.info(f"Downloaded {count:,} samples from '{domain}' ({split})")

    # Write cache (skip when streaming with a cap — partial data)
    if use_cache and (not streaming or max_samples is None):
        _write_cache(cache_file, records)


def _load_local(path: str, max_samples: Optional[int] = None) -> Generator[str, None, None]:
    """
    Load text from a local file.

    Supports:
    - .jsonl  — one JSON object per line, reads 'text' or 'plain_text' field
    - .json   — list of objects or single object with 'text' field
    - .txt    — one document per non-empty line
    - .csv    — reads 'text' column via csv.DictReader
    """
    import csv

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Local file not found: {path}")

    logger.info(f"Loading local file: {path}  ({p.stat().st_size / 1e6:.1f} MB)")
    count = 0
    suffix = p.suffix.lower()

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        if suffix == ".jsonl":
            for line in tqdm(f, desc=f"Reading {p.name}", unit=" lines", dynamic_ncols=True):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = (obj.get("text") or obj.get("plain_text") or "").strip()
                if text:
                    yield text
                    count += 1
                    if max_samples and count >= max_samples:
                        break

        elif suffix == ".json":
            data = json.load(f)
            items = data if isinstance(data, list) else [data]
            for obj in tqdm(items, desc=f"Reading {p.name}", unit=" docs", dynamic_ncols=True):
                text = (obj.get("text") or obj.get("plain_text") or "").strip()
                if text:
                    yield text
                    count += 1
                    if max_samples and count >= max_samples:
                        break

        elif suffix == ".csv":
            reader = csv.DictReader(f)
            for row in tqdm(reader, desc=f"Reading {p.name}", unit=" rows", dynamic_ncols=True):
                text = (row.get("text") or row.get("plain_text") or "").strip()
                if text:
                    yield text
                    count += 1
                    if max_samples and count >= max_samples:
                        break

        else:  # .txt or anything else — one doc per line
            for line in tqdm(f, desc=f"Reading {p.name}", unit=" lines", dynamic_ncols=True):
                text = line.strip()
                if text:
                    yield text
                    count += 1
                    if max_samples and count >= max_samples:
                        break

    logger.info(f"Loaded {count:,} records from {path}")
