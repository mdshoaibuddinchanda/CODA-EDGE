"""Logging setup for CODA experiments."""
import logging
import sys
from datetime import datetime
from pathlib import Path


LOG_DIR = Path("outputs/logs")


def setup_logger(experiment_name: str) -> logging.Logger:
    """Create a logger that writes to file and stdout."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"{experiment_name}_{timestamp}.log"

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
