# CODA — Covariance-based Domain Adaptation

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2+cu121-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.2-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Platform](https://img.shields.io/badge/Platform-Kaggle%20GPU-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Adapt a frozen language model to any target domain at inference time — no fine-tuning, no gradient updates, ~100 calibration samples.

---

## What is CODA?

CODA aligns hidden-state distributions between a source domain (WikiText-103) and a target domain (legal, medical, technical) using a **whitening transform** applied via a forward hook. No weights are modified.

The core transform at layer `L`:

```text
h' = α · (W · (h − μ_t) + μ_s) + (1 − α) · h
     where  W = L_source @ inv(L_target)
```

An **MMD gate** decides whether the distribution shift is large enough to warrant adaptation. If not, the base model runs unchanged.

---

## Running on Kaggle

This project is pinned to the Kaggle GPU environment: **CUDA 12.2, Python 3.10, Ubuntu 20.04**.

### Step 1 — Enable GPU accelerator

In your Kaggle notebook, go to **Settings → Accelerator → GPU T4 x2** (or P100).

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

All packages are pinned to exact versions tested on Kaggle. PyTorch is installed from the CUDA 12.1 wheel, which is fully compatible with Kaggle's CUDA 12.2 runtime. FAISS uses the `faiss-gpu-cu12` wheel for GPU-accelerated similarity search.

### Step 3 — Download datasets

```bash
python scripts/download_data.py
```

Downloads all domains and caches them to `data/raw/`. Re-running skips already-downloaded domains.

```bash
# Specific domains only
python scripts/download_data.py --domains wikitext scotus

# Quick smoke test — 1000 samples per split
python scripts/download_data.py --max-samples 1000
```

> **MIMIC-III** requires credentialed [PhysioNet](https://physionet.org/) access. Without it the pipeline falls back to **PubMed abstracts** automatically. To use your own file: `--local-path /path/to/NOTEEVENTS.csv`

### Step 4 — Run the full pipeline

```bash
python main.py
```

This runs the complete pipeline end-to-end and verifies all outputs at the end.

```bash
# Specific experiment
python main.py --config configs/experiment_legal.yaml

# Skip output verification
python main.py --skip-verify
```

### Step 5 — Verify outputs

```bash
python verify_outputs.py --config configs/experiment_legal.yaml
```

---

## Project Structure

```text
coda/
├── configs/
│   ├── default.yaml                  All hyperparameters
│   ├── experiment_legal.yaml         SCOTUS + Federal Circuit
│   ├── experiment_medical.yaml       MIMIC-III / PubMed fallback
│   └── experiment_technical.yaml    ArXiv CS abstracts
│
├── data/
│   ├── raw/                          Downloaded JSONL cache
│   ├── processed/                    Tokenised .npy sequences
│   ├── calibration/                  Hidden-state .npy files
│   └── splits/                       Train/val/test index files
│
├── models/
│   └── phi3-4bit/                    HuggingFace model cache
│
├── outputs/
│   ├── covariance/                   C, L, mu .npy matrices
│   ├── results/                      PPL and metrics JSON
│   ├── plots/                        PNG/PDF figures
│   └── logs/                         Timestamped experiment logs
│
├── scripts/
│   └── download_data.py              Standalone dataset downloader
│
├── src/
│   ├── adaptation/
│   │   ├── coda.py                   Pipeline orchestrator + hook manager
│   │   ├── covariance.py             Covariance + Cholesky computation
│   │   ├── mmd_gate.py               MMD² gate (RBF kernel, median bandwidth)
│   │   └── whitening.py              Whitening transform (partial α blend)
│   ├── data/
│   │   ├── loader.py                 HF download, local file, JSONL cache
│   │   ├── preprocess.py             Domain cleaning + tokenise + chunk
│   │   └── validation.py             Length / language / rejection-ratio filter
│   ├── evaluation/
│   │   ├── metrics.py                LAMBADA accuracy, Distinct-N
│   │   ├── perplexity.py             Cross-entropy PPL with NaN guard
│   │   └── plotting.py               Alpha/layer ablation, robustness, MMD plots
│   ├── model/
│   │   ├── inference.py              Forward hooks, hidden-state extraction
│   │   └── model_loader.py           4-bit NF4 load, VRAM checkpoint
│   └── utils/
│       ├── config.py                 YAML loader + dataclass validation
│       ├── device.py                 Central GPU/MPS/CPU device resolution
│       ├── logging_utils.py          File + stdout logger setup
│       └── memory.py                 GPU memory logging + cache clear
│
├── main.py                           ← Run this. Full pipeline + verify.
├── verify_outputs.py                 Output completeness checker (CP8)
├── requirements.txt                  Pinned for Kaggle GPU (CUDA 12.2)
├── LICENSE
├── PROGRESS.md
└── README.md
```

---

## Datasets

| Domain | HuggingFace ID | Access | Fallback |
| --- | --- | --- | --- |
| WikiText-103 | `Salesforce/wikitext` | Open | — |
| SCOTUS Opinions | `pile-of-law/pile-of-law` (scotus) | Open | — |
| Federal Circuit | `pile-of-law/pile-of-law` (federal_courts_opinions) | Open | — |
| MIMIC-III Clinical Notes | PhysioNet (credentialed) | Restricted | PubMed abstracts |
| ArXiv CS Abstracts | `scientific_papers/arxiv` | Open | — |

---

## Model

**Phi-3-mini-4k-instruct** (`microsoft/Phi-3-mini-4k-instruct`) loaded in 4-bit NF4 quantisation via `bitsandbytes`. Requires ~2.5 GB VRAM. Cached to `models/phi3-4bit/` on first run.

---

## Configuration

All hyperparameters live in `configs/default.yaml`.

| Parameter | Default | Description |
| --- | --- | --- |
| `coda.alpha` | `0.8` | Blend strength — 0 = no adaptation, 1 = full whitening |
| `coda.layer_indices` | `[0,6,12,18,24,-1]` | Transformer layers to adapt |
| `coda.calibration_samples` | `100` | Target-domain samples for covariance estimation |
| `coda.mmd_threshold` | `null` | MMD gate threshold (null = always apply CODA) |
| `coda.covariance_regularization` | `1e-5` | Ridge term added to covariance diagonal |
| `data.max_seq_length` | `2048` | Token sequence length |
| `data.stride` | `512` | Sliding-window stride for chunking |

---

## Outputs

| File | Location | Description |
| --- | --- | --- |
| `{domain}_base_ppl.json` | `outputs/results/` | Baseline perplexity (no adaptation) |
| `{domain}_coda_ppl.json` | `outputs/results/` | Perplexity after CODA |
| `{domain}_metrics.json` | `outputs/results/` | All metrics + improvement delta |
| `source_layer{L}_C.npy` | `outputs/covariance/` | Source covariance matrix (d×d) |
| `{domain}_layer{L}_L.npy` | `outputs/covariance/` | Cholesky factor of target covariance |
| `alpha_ablation_{domain}.png` | `outputs/plots/` | PPL vs α |
| `layer_ablation_{domain}.png` | `outputs/plots/` | PPL vs layer index |
| `robustness_curve_{domain}.png` | `outputs/plots/` | PPL vs distribution shift |
| `mmd_comparison.png` | `outputs/plots/` | MMD² vs KL vs Cosine across domains |
| `experiment_{timestamp}.log` | `outputs/logs/` | Full timestamped run log |

---

## Validation Checkpoints

| ID | Location | Check |
| --- | --- | --- |
| CP1 | After `loader.py` | Text non-empty, UTF-8 valid |
| CP2 | After `preprocess.py` | Sequence length == `max_seq_length` |
| CP3 | After `model_loader.py` | VRAM < 3.8 GB |
| CP4 | After `covariance.py` | Covariance is positive definite |
| CP5 | After `whitening.py` | No NaN in transformed states |
| CP6 | After `mmd_gate.py` | MMD value logged; gate decision recorded |
| CP7 | After `perplexity.py` | PPL in range (1, 100 000) |
| CP8 | End of pipeline | All expected output files present |

---

## Technology Stack

| Library | Pinned Version | Role |
| --- | --- | --- |
| PyTorch | 2.2.2+cu121 | Model inference, CUDA tensors |
| Transformers | 4.40.2 | Phi-3-mini loading |
| bitsandbytes | 0.43.1 | 4-bit NF4 quantisation (CUDA) |
| accelerate | 0.29.3 | `device_map="auto"`, multi-GPU |
| datasets | 2.19.1 | HuggingFace streaming data |
| NumPy | 1.26.4 | Covariance + Cholesky (OpenBLAS) |
| SciPy | 1.13.0 | LAPACK linalg ops |
| scikit-learn | 1.4.2 | Covariance utilities |
| faiss-gpu-cu12 | 1.7.3 | GPU similarity search (CUDA 12) |
| Matplotlib | 3.8.4 | Plotting |
| Seaborn | 0.13.2 | Plot styling |
| tqdm | 4.66.2 | Progress bars |
| PyYAML | 6.0.1 | Config parsing |
| langdetect | 1.0.9 | Language detection |

---

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-change`
3. Commit: `git commit -m "feat: description"`
4. Push and open a pull request.

---

## License

MIT — see [LICENSE](LICENSE).
