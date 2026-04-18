# NanoChat Package Dependencies

This document provides a comprehensive breakdown of all third-party packages required for the nanochat pipeline, from tokenization through inference.

## Core Framework (All Stages)

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.9.1 | Core deep learning framework |
| `torch.nn.functional` | built-in | Neural network operations |

---

## 1. Tokenization Stage

| Package | Version | Purpose |
|---------|---------|---------|
| `tokenizers` | >=0.22.0 | HuggingFace BPE tokenizer training |
| `tiktoken` | >=0.11.0 | GPT-4 style tokenization (inference) |
| `rustbpe` | >=0.1.0 | Fast BPE training in Rust |

### Critical Indirect Dependencies
- `regex` (via tiktoken/tokenizers) - Pattern matching for tokenization

---

## 2. Data Loading & Preprocessing

| Package | Version | Purpose |
|---------|---------|---------|
| `datasets` | >=4.0.0 | HuggingFace datasets library |
| `pyarrow` | (via datasets) | Parquet file reading |
| `requests` | (std lib) | HTTP downloads |

### Critical Indirect Dependencies
- `pyarrow` - Required for reading parquet data shards
- `fsspec` (via datasets) - File system abstraction

---

## 3. Pretraining (base_train)

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.9.1 | Model training, distributed |
| `torch.distributed` | built-in | Multi-GPU training |
| `wandb` | >=0.21.3 | Experiment logging |

### Critical Indirect Dependencies
- `numpy` (via torch) - Numerical operations
- `psutil` - System resource monitoring

---

## 4. Fine-Tuning (SFT/RL)

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.9.1 | Model training |
| `datasets` | >=4.0.0 | Loading MMLU, GSM8K, etc. |
| `wandb` | >=0.21.3 | Experiment logging |

### Direct Task Dependencies
Uses `datasets` library for loading:
- MMLU (Multiple choice questions)
- GSM8K (Math word problems)
- ARC (Science reasoning)
- HumanEval (Code generation)
- SmolTalk (Conversational data)

---

## 5. Evaluation

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.9.1 | Model inference |
| `datasets` | >=4.0.0 | Loading eval datasets |
| `jinja2` | (via core_eval) | Prompt templating |
| `yaml` | (via base_eval) | Config parsing |
| `csv` | std lib | Results export |

---

## 6. Inference (Web/CLI)

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.9.1 | Model inference |
| `fastapi` | >=0.117.1 | Web API framework |
| `uvicorn` | >=0.36.0 | ASGI server |
| `pydantic` | (via fastapi) | Data validation |
| `tiktoken` | >=0.11.0 | Token counting |

### Critical Indirect Dependencies
- `starlette` (via fastapi) - Web framework components
- `anyio` (via uvicorn) - Async I/O
- `httpx` (optional, via fastapi) - HTTP client

---

## 7. Utilities (Cross-Cutting)

| Package | Version | Purpose |
|---------|---------|---------|
| `psutil` | >=7.1.0 | System monitoring |
| `filelock` | (via common.py) | File locking for downloads |
| `kernels` | >=0.11.7 | Custom CUDA kernels (optional) |

---

## Installation Profiles

### Minimal Install (Inference Only)

For running pre-trained models only:

```toml
torch==2.9.1
tiktoken>=0.11.0
fastapi>=0.117.1
uvicorn>=0.36.0
tokenizers>=0.22.0
```

### Training Install (Pretraining + SFT)

For training models from scratch:

```toml
torch==2.9.1
datasets>=4.0.0
wandb>=0.21.3
tokenizers>=0.22.0
tiktoken>=0.11.0
rustbpe>=0.1.0
pyarrow  # via datasets
```

### Full Development

For development, testing, and model conversion:

```toml
# All above plus:
pytest>=8.0.0
transformers>=4.57.3  # For model conversion
matplotlib>=3.10.8   # For plotting
ipykernel>=7.1.0     # For notebooks
```

---

## Key Design Decisions

1. **No `transformers` in core** - Library avoids HuggingFace transformers dependency for core functionality to minimize bloat

2. **Dual tokenizers** - Uses both `tiktoken` (fast inference) and `tokenizers` (training) for optimal performance in different contexts

3. **Modular torch** - Works with both CPU and GPU torch variants via uv extras

4. **Optional distributed** - `torch.distributed` only used when available (multi-GPU setups)

5. **Minimal web stack** - FastAPI + Uvicorn provides modern async web serving without heavy frameworks

6. **Lazy loading** - Many dataset/task dependencies are loaded only when specific evaluations are run

---

## Dependency Tree Summary

```
nanochat
├── Core: torch (2.9.1)
│   └── numpy, psutil (indirect)
├── Tokenization
│   ├── tiktoken (inference)
│   ├── tokenizers (training)
│   └── rustbpe (fast BPE)
├── Data
│   ├── datasets (HuggingFace)
│   │   └── pyarrow, fsspec
│   └── requests (downloads)
├── Training
│   ├── torch.distributed
│   └── wandb (logging)
├── Web
│   ├── fastapi
│   │   ├── pydantic
│   │   └── starlette
│   └── uvicorn
│       └── anyio
└── Utils
    ├── psutil
    └── filelock
```

**Total core footprint:** ~8 direct dependencies for full pipeline

---

## Platform-Specific Notes

### Mac Studio (MPS)
- Use `torch` with MPS backend
- All packages work on Apple Silicon
- **Known issue:** SFT training has numerical instability (nan loss)
- Base training and inference work correctly

### CUDA (NVIDIA GPU)
- Use `torch` with CUDA 12.8
- All functionality works correctly
- FP8 support available on H100+

### CPU-Only
- Use `torch` CPU variant
- All functionality works but slower
- Suitable for development/testing

---

*Generated: April 2026*
