# nanochat AGENTS.md

**BUILD AND TRAIN** LLMs from scratch. This is a training harness by Andrej Karpathy - not a demo, but a complete pipeline for building GPT-2 grade models.

**Goal:** Train GPT-2 grade model for <$100 on 8xH100, or build on your own hardware (Mac Studio, etc.).

## Quick Start

```bash
# GPU node setup
uv sync --extra gpu && source .venv/bin/activate

# CPU/Mac setup  
uv sync --extra cpu && source .venv/bin/activate

# Run full pipeline (~3 hours on 8xH100)
bash runs/speedrun.sh

# Quick CPU/MPS demo (~30 min on MacBook)
bash runs/runcpu.sh
```

## Key Architecture Principle

**`--depth` is the single complexity dial.** Everything else auto-scales:
- Model width = depth × 64 (aspect ratio)
- Learning rates, batch sizes, training horizon all derived from depth
- GPT-2 capability ≈ depth 24-26

Common quick iteration scale: `--depth=12` (~5 min runs)

## Entry Points

All scripts are modules, run as `python -m scripts.<name>` or `torchrun --nproc_per_node=8 -m scripts.<name>`:

| Script | Purpose |
|--------|---------|
| `base_train.py` | Pretraining (main focus) |
| `base_eval.py` | CORE metric, BPB, sampling |
| `chat_sft.py` | Supervised fine-tuning |
| `chat_rl.py` | Reinforcement learning |
| `chat_cli.py` | CLI chat interface |
| `chat_web.py` | Web UI (FastAPI) |
| `tok_train.py` | Train tokenizer |
| `tok_eval.py` | Eval tokenizer compression |

## Precision / dtype

Auto-detected in `nanochat/common.py`:
- CUDA SM 80+ (A100/H100): `bfloat16`
- Older CUDA / CPU / MPS: `float32`

Override: `NANOCHAT_DTYPE=float32 torchrun ...`

**No `torch.amp.autocast`.** Explicit dtype via `COMPUTE_DTYPE` global. Custom `Linear` layer casts weights in forward pass.

## Testing

```bash
# Run all tests
pytest

# Skip slow tests
pytest -m "not slow"

# Specific test
python -m pytest tests/test_engine.py -v
```

## Key Env Vars

| Var | Purpose |
|-----|---------|
| `NANOCHAT_BASE_DIR` | Cache dir (default: `~/.cache/nanochat`) |
| `NANOCHAT_DTYPE` | Force dtype: `bfloat16`, `float16`, `float32` |
| `WANDB_RUN` | Set run name (`dummy` = disable logging) |
| `OMP_NUM_THREADS=1` | Required for DataLoader efficiency |

## Distributed Training

Always use `torchrun`, even single-GPU tests work fine:

```bash
# 8x GPU
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=12 --run=test

# Single GPU (auto gradient accumulation)
python -m scripts.base_train -- --depth=12 --run=test
```

## Project Layout

```
nanochat/          # Core library (GPT model, tokenizer, data, optimizers)
scripts/           # Entry points (all runnable as -m modules)
tasks/             # Evaluation tasks (MMLU, GSM8K, HumanEval, etc.)
dev/               # Utilities, notebooks, leaderboard docs
tests/             # pytest tests
runs/              # Reference run scripts (speedrun.sh, runcpu.sh, etc.)
```

## OOM Troubleshooting

If you run out of VRAM, reduce `--device-batch-size` (default 32) to 16, 8, 4, 2, or 1. Gradient accumulation auto-compensates to maintain total batch size.

## Leaderboard

Primary metric: **wall-clock time to beat GPT-2 CORE score (0.256525)** on 8xH100. See `dev/LEADERBOARD.md` for submission process.

## Dependencies

- Python 3.10+
- PyTorch 2.9.1
- uv (package manager)
- See `pyproject.toml` for full list

## Mac Studio Setup

For Mac Studio Ultra (M1/M2/M3) deployment, use the skills:

1. **`setup` skill**: One-time environment setup (installs uv, clones repo, creates venv)
2. **`train` skill**: Train models - explains datasets, options, time estimates, expected results
3. **`nanochat` skill**: Run training scripts or start web server

See `skills/setup/SKILL.md`, `skills/train/SKILL.md`, `skills/nanochat/SKILL.md` and `skills/debug/SKILL.md` for detailed instructions.
