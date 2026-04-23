"""Load and validate YAML configuration."""
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelConfig:
    name: str = "microsoft/Phi-3-mini-4k-instruct"
    quantization: str = "4bit"
    device_map: str = "auto"
    torch_dtype: str = "float16"
    trust_remote_code: bool = True


@dataclass
class CodaConfig:
    calibration_samples: int = 100
    layer_indices: List[int] = field(default_factory=lambda: [0, 6, 12, 18, 24, -1])
    alpha: float = 0.8
    mmd_threshold: Optional[float] = None
    covariance_regularization: float = 1e-5


@dataclass
class DataConfig:
    source_domain: str = "wikitext"
    target_domains: List[str] = field(default_factory=lambda: ["scotus"])
    max_seq_length: int = 2048
    stride: int = 512


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    coda: CodaConfig = field(default_factory=CodaConfig)
    data: DataConfig = field(default_factory=DataConfig)
    config_path: Optional[str] = None


def _dict_to_config(d: Dict[str, Any]) -> Config:
    model = ModelConfig(**d.get("model", {}))
    coda_d = d.get("coda", {})
    if "mmd_threshold" not in coda_d:
        coda_d["mmd_threshold"] = None
    coda = CodaConfig(**coda_d)
    data = DataConfig(**d.get("data", {}))
    return Config(model=model, coda=coda, data=data)


def _validate(cfg: Config) -> None:
    assert 0.0 <= cfg.coda.alpha <= 1.0, "alpha must be in [0, 1]"
    assert cfg.coda.calibration_samples > 0, "calibration_samples must be positive"
    assert cfg.data.max_seq_length > 0, "max_seq_length must be positive"
    assert cfg.data.stride > 0, "stride must be positive"
    assert len(cfg.coda.layer_indices) > 0, "layer_indices must not be empty"


def load_config(path: str) -> Config:
    """Load YAML config from path and return a Config object."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = _dict_to_config(raw)
    _validate(cfg)
    cfg.config_path = str(Path(path).resolve())
    return cfg


def parse_args_and_load() -> Config:
    """Parse --config CLI argument and load config."""
    parser = argparse.ArgumentParser(description="CODA experiment runner")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    args, _ = parser.parse_known_args()
    return load_config(args.config)
