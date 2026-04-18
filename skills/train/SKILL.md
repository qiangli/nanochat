---
name: train
description: Train nanochat models on Mac Studio Ultra - covers datasets, training options, time estimates, and expected results
---

> **Goal:** Build full GPT-2 grade models (depth=24), not demo models. The pipeline is designed for training from scratch.
>
> **Troubleshooting:** If you encounter `loss: nan` during SFT training, see the **debug** skill for the solution. This is a known dataloader bug with a documented fix.

You need to train a nanochat model. This skill explains what training is required, what datasets are used, how long it takes, and what results to expect.

## What Training Do You Need?

NanoChat requires **two training stages**:

### Stage 1: Pretraining (Base Model)
- **What**: Trains the model on general text data (internet text)
- **Purpose**: Learns language, facts, reasoning, world knowledge
- **Output**: Base model that can complete text

### Stage 2: Supervised Fine-Tuning (SFT)
- **What**: Trains on conversational data with question-answer pairs
- **Purpose**: Teaches the model to chat, follow instructions, use tools
- **Output**: Chat-ready model you can talk to

## Datasets Required

### 1. Pretraining Data: ClimbMix-400B

**Source**: NVIDIA ClimbMix (curated web text)

| Shards | Size | Training Time | Use Case |
|--------|------|---------------|----------|
| 8 shards | ~800 MB | ~30-60 min | Quick demo/testing |
| 170 shards | ~17 GB | ~12-24 hours | Full GPT-2 model |

**Download**:
```bash
# Demo (8 shards = ~2B characters)
python -m nanochat.dataset -n 8

# Full GPT-2 (170 shards = ~42B characters)
python -m nanochat.dataset -n 170
```

**Location**: `~/.cache/nanochat/base_data_climbmix/`

### 2. SFT Data: Identity Conversations

**Source**: Synthetic chat data

- **Size**: ~2.3 MB
- **Format**: JSONL (conversations)
- **Content**: Q&A pairs, tool use examples, personality
- **Purpose**: Makes model chat-ready

**Download**:
```bash
curl -L -o ~/.cache/nanochat/identity_conversations.jsonl \
    https://karpathy-public.s3-west-2.amazonaws.com/identity_conversations.jsonl
```

## Training Options

### Option A: Quick Demo Model (Recommended First)

**Specs**: `--depth=6` (small model)
**Time**: ~30-60 minutes on Mac Studio M3 Ultra
**Quality**: Basic conversational ability, simple facts
**Use for**: Testing, development, learning

```bash
cat > ~/train_demo.sh << 'EOF'
#!/bin/bash
set -e

echo "======================================"
echo "Training Demo Model (depth=6)"
echo "Estimated time: 30-60 minutes"
echo "======================================"

cd ~/nanochat
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
export OMP_NUM_THREADS=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

echo ""
echo "Step 1/5: Downloading datasets..."
python -m nanochat.dataset -n 8

echo ""
echo "Step 2/5: Training tokenizer..."
python -m scripts.tok_train --max-chars=2000000000

echo ""
echo "Step 3/5: Evaluating tokenizer..."
python -m scripts.tok_eval

echo ""
echo "Step 4/5: Training base model (depth=6)..."
python -m scripts.base_train \
    --depth=6 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=100 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=100 \
    --num-iterations=5000 \
    --device-type mps \
    --run=demo_d6

echo ""
echo "Step 5/5: Training SFT..."
curl -sL -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3-west-2.amazonaws.com/identity_conversations.jsonl

python -m scripts.chat_sft \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=200 \
    --eval-tokens=524288 \
    --num-iterations=1500 \
    --device-type mps \
    --run=demo_sft

echo ""
echo "======================================"
echo "Training Complete!"
echo "======================================"
echo "Model location: $NANOCHAT_BASE_DIR/sft/"
echo ""
echo "Start the server: bash ~/run_nanochat_server.sh"
echo "Or use the 'nanochat' skill"
EOF

chmod +x ~/train_demo.sh
bash ~/train_demo.sh
```

### Option B: Medium Model (Better Quality)

**Specs**: `--depth=12` (GPT-1 size)
**Time**: ~2-4 hours
**Quality**: Better reasoning, more knowledge
**Use for**: Personal assistant, better conversations

```bash
cat > ~/train_medium.sh << 'EOF'
#!/bin/bash
set -e

echo "======================================"
echo "Training Medium Model (depth=12)"
echo "Estimated time: 2-4 hours"
echo "======================================"

cd ~/nanochat
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
export OMP_NUM_THREADS=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

echo "Downloading datasets (more data for better model)..."
python -m nanochat.dataset -n 32

echo "Training tokenizer..."
python -m scripts.tok_train --max-chars=8000000000

echo "Training base model (depth=12)..."
python -m scripts.base_train \
    --depth=12 \
    --device-batch-size=16 \
    --max-seq-len=1024 \
    --num-iterations=10000 \
    --eval-every=500 \
    --core-metric-every=-1 \
    --device-type mps \
    --run=medium_d12

echo "Training SFT..."
curl -sL -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3.amazonaws.com/identity_conversations.jsonl

python -m scripts.chat_sft \
    --max-seq-len=1024 \
    --device-batch-size=16 \
    --num-iterations=3000 \
    --device-type mps \
    --run=medium_sft

echo "Training complete!"
EOF

chmod +x ~/train_medium.sh
bash ~/train_medium.sh
```

### Option C: Full GPT-2 Model (Best Quality)

**Specs**: `--depth=24` (GPT-2 size)
**Time**: ~15-30 hours
**Quality**: Near GPT-2 level, good reasoning
**Use for**: Production use, serious projects

```bash
cat > ~/train_gpt2.sh << 'EOF'
#!/bin/bash
set -e

echo "======================================"
echo "Training GPT-2 Model (depth=24)"
echo "Estimated time: 15-30 hours"
echo "======================================"

cd ~/nanochat
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
export OMP_NUM_THREADS=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

echo "Downloading full dataset (170 shards)..."
python -m nanochat.dataset -n 170

echo "Training tokenizer..."
python -m scripts.tok_train

echo "Training base model (depth=24)..."
python -m scripts.base_train \
    --depth=24 \
    --device-batch-size=8 \
    --max-seq-len=1024 \
    --num-iterations=16704 \
    --eval-every=1000 \
    --core-metric-every=-1 \
    --device-type mps \
    --run=gpt2_d24

echo "Training SFT..."
curl -sL -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3.amazonaws.com/identity_conversations.jsonl

python -m scripts.chat_sft \
    --max-seq-len=1024 \
    --device-batch-size=8 \
    --num-iterations=5000 \
    --device-type mps \
    --run=gpt2_sft

echo "Training complete!"
EOF

chmod +x ~/train_gpt2.sh
```

## Time Estimates Summary

| Model | Depth | Dataset | Base Training | SFT Training | Total |
|-------|-------|---------|---------------|--------------|-------|
| **Demo** | 6 | 8 shards | ~25-45 min | ~5-10 min | **30-60 min** |
| **Medium** | 12 | 32 shards | ~1.5-3 hr | ~20-40 min | **2-4 hours** |
| **GPT-2** | 24 | 170 shards | ~12-24 hr | ~2-4 hr | **15-30 hours** |

## What to Expect

### Demo Model (depth=6)
- Can answer simple questions
- Knows basic facts (Paris is capital of France)
- Understands simple instructions
- Limited reasoning ability
- May hallucinate or give wrong answers

**Example chat**:
```
User: What is the capital of France?
Assistant: Paris

User: What color is the sky?
Assistant: Blue

User: Write a poem about cats
Assistant: [Simple short poem]
```

### Medium Model (depth=12)
- Better reasoning and explanations
- More factual knowledge
- Can follow multi-step instructions
- Better conversation flow
- Still makes mistakes but fewer

### GPT-2 Model (depth=24)
- Coherent long-form text
- Good reasoning on many topics
- Creative writing capability
- Can explain concepts
- Comparable to early GPT-2 quality

## Storage Requirements

| Model | Downloads | Checkpoints | Total Space |
|-------|-----------|-------------|-------------|
| Demo | ~1 GB | ~500 MB | **~2 GB** |
| Medium | ~4 GB | ~1 GB | **~6 GB** |
| GPT-2 | ~20 GB | ~3 GB | **~25 GB** |

## Monitoring Training

```bash
# Watch training progress (in another terminal)
tail -f ~/nanochat_training.log

# Check GPU/memory usage
while true; do
    echo "Memory pressure: $(memory_pressure | grep 'System-wide memory free' | awk '{print $5}')"
    sleep 30
done

# Keep Mac awake during long training
# System Settings > Battery > Options > Prevent automatic sleeping
```

## Background Training

For long training (GPT-2), run in background:

```bash
# Start training in background
nohup bash ~/train_gpt2.sh > ~/nanochat_training.log 2>&1 &

# Monitor progress
tail -f ~/nanochat_training.log

# Check if still running
pgrep -f "scripts.base_train"

# View final model
ls -lh ~/.cache/nanochat/sft/
```

## Check Training Results

```bash
# Check model exists
ls -la ~/.cache/nanochat/sft/

# Check checkpoints
ls -la ~/.cache/nanochat/base/

# View training logs
ls -la ~/.cache/nanochat/

# Test the model
cd ~/nanochat
source .venv/bin/activate
python -m scripts.chat_cli -p "Hello!"
```

## Recommendations

1. **First time?** Start with Demo model (30-60 min)
2. **Want better quality?** Train Medium model (2-4 hours)
3. **Need production quality?** Train GPT-2 model (overnight)
4. **Keep Mac awake**: Disable sleep during training
5. **Monitor temperature**: Ensure good ventilation

## Troubleshooting

### Training too slow
- This is normal on MPS vs CUDA
- Reduce `--depth` for faster training
- Reduce `--num-iterations`

### Out of memory
```bash
# Reduce batch size
python -m scripts.base_train --device-batch-size=8 ...

# Reduce sequence length
python -m scripts.base_train --max-seq-len=512 ...
```

### Download fails
```bash
# Retry dataset download
python -m nanochat.dataset -n 8 --force

# Check internet
curl -I https://huggingface.co
```

### Training interrupted
```bash
# Resume from checkpoint (if available)
python -m scripts.base_train \
    --resume-from-step 1000 \
    ...
```