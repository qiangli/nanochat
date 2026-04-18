---
name: nanochat
description: Run nanochat training or start the web server for chat on Mac Studio Ultra
---

You need to train a nanochat model or start the web server for chatting. Use this skill after the one-time 'setup' skill has been completed.

## Quick Start

```bash
# SSH to the machine
ssh noviadmin@novicortex.local

# Option 1: Train a model (takes hours)
bash ~/run_nanochat_train.sh

# Option 2: Auto-start web server (checks if running, starts if needed)
bash ~/nanochat_auto.sh

# Option 3: Manual server control
bash ~/run_nanochat_server.sh  # Explicit start
pkill -f "python -m scripts.chat_web"  # Stop
```

## Prerequisites

- Setup skill has been run (uv installed, nanochat cloned, venv created)
- SSH access to the Mac Studio
- For training: Several hours of compute time
- For server: A trained model in `~/.cache/nanochat/sft/`

## Option 1: Train a Model

Training on Mac Studio Ultra (M3, 96GB, 60 GPU cores) takes:
- **Demo model (depth=6)**: ~30-60 minutes
- **GPT-2 grade (depth=24)**: 12-24+ hours

### Training Script

```bash
cat > ~/run_nanochat_train.sh << 'EOF'
#!/bin/bash
set -e

echo "======================================"
echo "NanoChat Training - Mac Studio Ultra"
echo "======================================"

# Setup environment
cd ~/nanochat
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate

# Detect hardware
HOSTNAME=$(hostname)
CHIP=$(sysctl -n machdep.cpu.brand_string)
MEMORY_GB=$(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024))
GPU_CORES=$(system_profiler SPDisplaysDataType | grep "Total Number of Cores" | awk '{print $5}')

echo "Hardware:"
echo "  Host: $HOSTNAME"
echo "  Chip: $CHIP"
echo "  Memory: ${MEMORY_GB}GB"
echo "  GPU Cores: $GPU_CORES"
echo ""

# Required env vars
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p $NANOCHAT_BASE_DIR

# Verify MPS
python -c "import torch; assert torch.backends.mps.is_available(), 'MPS not available'; print('✓ MPS ready')"

# Check if model already exists
if [ -d "$NANOCHAT_BASE_DIR/sft" ] && [ "$(ls -A $NANOCHAT_BASE_DIR/sft)" ]; then
    echo ""
    echo "⚠ Model already exists at $NANOCHAT_BASE_DIR/sft/"
    echo "To retrain, delete it first: rm -rf $NANOCHAT_BASE_DIR/sft"
    echo ""
    read -p "Continue training anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Training cancelled. To start server, use run_nanochat_server.sh"
        exit 0
    fi
fi

echo ""
echo "======================================"
echo "Starting Training"
echo "======================================"
echo ""
echo "This will use the Mac Studio optimized script (runs/runcpu.sh)"
echo "Training a depth=6 model (~30-60 min on Mac Studio Ultra)"
echo ""
echo "Started at: $(date)"
echo ""

# Run training
python -m nanochat.dataset -n 8
python -m scripts.tok_train --max-chars=2000000000
python -m scripts.tok_eval

# Train base model (MPS optimized)
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
    --run=macstudio_d6

# Evaluate
python -m scripts.base_eval \
    --device-batch-size=1 \
    --split-tokens=16384 \
    --max-per-task=16 \
    --device-type mps

# SFT training
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

python -m scripts.chat_sft \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=200 \
    --eval-tokens=524288 \
    --num-iterations=1500 \
    --device-type mps \
    --run=macstudio_sft

echo ""
echo "======================================"
echo "Training Complete!"
echo "======================================"
echo "Completed at: $(date)"
echo ""
echo "Model location: $NANOCHAT_BASE_DIR/sft/"
echo ""
echo "Next step: Start the web server with run_nanochat_server.sh"
EOF

chmod +x ~/run_nanochat_train.sh
bash ~/run_nanochat_train.sh
```

Training runs in foreground. For background training:
```bash
nohup bash ~/run_nanochat_train.sh > ~/nanochat_training.log 2>&1 &
tail -f ~/nanochat_training.log
```

## Option 2: Start Web Server (Auto-Start if Not Running)

### Quick Auto-Start Script

This script automatically checks if the server is running and starts it if needed:

```bash
cat > ~/nanochat_auto.sh << 'EOF'
#!/bin/bash

# Setup environment
cd ~/nanochat
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate

HOSTNAME=$(hostname)
URL="http://${HOSTNAME}.local:8000/"

# Required env vars
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "======================================"
echo "NanoChat Auto-Start"
echo "======================================"
echo ""

# Check if model exists
if [ ! -d "$NANOCHAT_BASE_DIR/sft" ] || [ -z "$(ls -A $NANOCHAT_BASE_DIR/sft 2>/dev/null)" ]; then
    echo "❌ No trained model found at $NANOCHAT_BASE_DIR/sft/"
    echo ""
    echo "Please run training first:"
    echo "  bash ~/run_nanochat_train.sh"
    echo ""
    exit 1
fi

echo "✓ Model found"
echo ""

# Check if server is already running
if pgrep -f "python -m scripts.chat_web" > /dev/null; then
    echo "✓ Server is already running!"
    echo ""
    echo "======================================"
    echo "Access your nanochat at:"
    echo "  $URL"
    echo "======================================"
    echo ""
    echo "To restart: pkill -f 'python -m scripts.chat_web' && bash ~/nanochat_auto.sh"
    exit 0
fi

# Server not running, start it
echo "Starting web server..."
echo ""

nohup python -m scripts.chat_web \
    --host 0.0.0.0 \
    --port 8000 \
    --device-type mps \
    > ~/nanochat_server.log 2>&1 &

SERVER_PID=$!
sleep 3

# Verify server started
if pgrep -f "python -m scripts.chat_web" > /dev/null; then
    echo "✓ Server started successfully (PID: $SERVER_PID)"
    echo ""
    echo "======================================"
    echo "🎉 Your nanochat is ready!"
    echo ""
    echo "  $URL"
    echo ""
    echo "======================================"
    echo ""
    echo "Commands:"
    echo "  View logs:  tail -f ~/nanochat_server.log"
    echo "  Stop:       pkill -f 'python -m scripts.chat_web'"
    echo "  Health:     curl http://localhost:8000/health"
    echo ""
else
    echo "❌ Server failed to start!"
    echo "Check logs: tail -f ~/nanochat_server.log"
    exit 1
fi
EOF

chmod +x ~/nanochat_auto.sh
bash ~/nanochat_auto.sh
```

**This script will:**
1. ✅ Check if a trained model exists
2. ✅ Check if server is already running (and show URL if yes)
3. ✅ Auto-start server if not running
4. ✅ Provide the access URL

### Manual Server Script (Legacy)

If you prefer the original explicit server script:

```bash
cat > ~/run_nanochat_server.sh << 'EOF'
#!/bin/bash
set -e

echo "======================================"
echo "NanoChat Web Server"
echo "======================================"

# Setup environment
cd ~/nanochat
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate

# Detect hardware
HOSTNAME=$(hostname)
echo "Host: $HOSTNAME"
echo "URL: http://${HOSTNAME}.local:8000/"
echo ""

# Required env vars
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Verify model exists
if [ ! -d "$NANOCHAT_BASE_DIR/sft" ] || [ -z "$(ls -A $NANOCHAT_BASE_DIR/sft 2>/dev/null)" ]; then
    echo "ERROR: No trained model found!"
    echo "Expected: $NANOCHAT_BASE_DIR/sft/"
    echo ""
    echo "Run training first with: bash ~/run_nanochat_train.sh"
    exit 1
fi

echo "✓ Model found"
echo ""

# Check if server already running
if pgrep -f "python -m scripts.chat_web" > /dev/null; then
    echo "⚠ Server is already running!"
    echo "To restart: pkill -f 'python -m scripts.chat_web'"
    echo "Then run this script again."
    exit 1
fi

echo "======================================"
echo "Starting Web Server"
echo "======================================"
echo ""

# Start server with MPS support
# Using nohup so it keeps running after SSH disconnect
nohup python -m scripts.chat_web \
    --host 0.0.0.0 \
    --port 8000 \
    --device-type mps \
    > ~/nanochat_server.log 2>&1 &

SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"
echo ""

# Wait for server to start
sleep 3

# Check if running
if pgrep -f "python -m scripts.chat_web" > /dev/null; then
    echo "✓ Server is running!"
    echo ""
    echo "======================================"
    echo "Access your nanochat at:"
    echo "  http://${HOSTNAME}.local:8000/"
    echo "======================================"
    echo ""
    echo "To monitor: tail -f ~/nanochat_server.log"
    echo "To stop:    pkill -f 'python -m scripts.chat_web'"
    echo ""
    echo "Health check:"
    sleep 2
    curl -s http://localhost:8000/health 2>/dev/null && echo "✓ Health check passed" || echo "⚠ Health check failed (server may still be starting)"
else
    echo "ERROR: Server failed to start!"
    echo "Check logs: tail -f ~/nanochat_server.log"
    exit 1
fi
EOF

chmod +x ~/run_nanochat_server.sh
bash ~/run_nanochat_server.sh
```

## Management Commands

### Check Status
```bash
# Check if training is running
pgrep -a -f "scripts.base_train|scripts.chat_sft"

# Check if server is running
pgrep -a -f "scripts.chat_web"

# Check what ports are in use
lsof -i :8000
```

### View Logs
```bash
# Training logs
tail -f ~/nanochat_training.log

# Server logs
tail -f ~/nanochat_server.log

# Real-time log during active session
tail -f ~/nanochat/chat_web.log
```

### Stop Services
```bash
# Stop training
pkill -f "scripts.base_train"
pkill -f "scripts.chat_sft"

# Stop server
pkill -f "python -m scripts.chat_web"

# Stop all nanochat processes
pkill -f "nanochat|scripts.base|scripts.chat"
```

### Restart Server
```bash
# Quick restart
pkill -f "python -m scripts.chat_web"
bash ~/run_nanochat_server.sh
```

## Training Larger Models

The default script trains a depth=6 model (demo size). For better quality:

```bash
# Depth 12 (GPT-1 size) - ~2-4 hours on Mac Studio
cd ~/nanochat
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1

python -m scripts.base_train \
    --depth=12 \
    --device-batch-size=16 \
    --max-seq-len=1024 \
    --num-iterations=10000 \
    --device-type mps \
    --run=macstudio_d12

# Depth 24 (GPT-2 size) - 12-24+ hours
python -m scripts.base_train \
    --depth=24 \
    --device-batch-size=8 \
    --max-seq-len=1024 \
    --num-iterations=16704 \
    --device-type mps \
    --run=macstudio_d24
```

## Using the Chat Interface

Once the server is running:

1. **Web Browser**: Open `http://hostname.local:8000/`
2. **API Access**:
   ```bash
   curl -X POST http://novicortex.local:8000/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "messages": [{"role": "user", "content": "Hello!"}],
       "temperature": 0.8,
       "max_tokens": 100
     }'
   ```

## Troubleshooting

### "No module named 'nanochat'"
```bash
cd ~/nanochat
source .venv/bin/activate
```

### "MPS not available"
```bash
# Check macOS version
sw_vers -productVersion  # Need 13.0+

# Verify PyTorch
python -c "import torch; print(torch.backends.mps.is_available())"
```

### Server not accessible from network
```bash
# Check firewall
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate

# Test locally first
curl http://localhost:8000/health

# Check binding
lsof -i :8000 | grep LISTEN
```

### Training too slow
- This is expected on MPS vs CUDA
- Reduce `--depth` for faster training
- Reduce `--num-iterations` for shorter runs
- Use `--eval-every=-1` to skip evaluations

### Out of memory
```bash
# Reduce batch size
python -m scripts.base_train --device-batch-size=8 ...  # instead of 32

# Reduce sequence length
python -m scripts.base_train --max-seq-len=256 ...  # instead of 512
```