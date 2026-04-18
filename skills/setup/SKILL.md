---
name: setup
description: One-time setup for nanochat on Mac Studio Ultra (installs uv, clones repo, sets up environment)
---

> **Purpose:** This repository is for **building and training** nanochat models from scratch, not just running demos. We build, fix issues, and train full models.
>
> **Note:** If you encounter NaN losses during training, see the **debug** skill for troubleshooting.

You need to set up nanochat on a Mac Studio Ultra from scratch. This is a one-time setup that installs required tools, clones the repository, and prepares the environment.

## When to Use This Skill

- Fresh machine with no nanochat installation
- Setting up a new Mac Studio for nanochat
- Preparing environment before training

## Prerequisites

- macOS 13.0+ (Ventura or later)
- SSH access to the Mac (e.g., `novicortex.local`)
- Internet connection
- At least 20GB free disk space

## One-Time Setup Steps

### Step 1: SSH to the Mac Studio

```bash
ssh username@hostname.local
# Example: ssh noviadmin@novicortex.local
```

### Step 2: Run the complete setup script

```bash
cat > ~/setup_nanochat.sh << 'EOF'
#!/bin/bash
set -e

echo "======================================"
echo "NanoChat One-Time Setup"
echo "======================================"

# Detect system
HOSTNAME=$(hostname)
echo "Setting up on: $HOSTNAME"
echo "Target URL will be: http://${HOSTNAME}.local:8000/"
echo ""

# Check macOS version
MACOS_VERSION=$(sw_vers -productVersion)
echo "macOS Version: $MACOS_VERSION"

# Check memory
MEMORY_GB=$(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024))
echo "Memory: ${MEMORY_GB} GB"

# Check chip
CHIP=$(sysctl -n machdep.cpu.brand_string)
echo "Chip: $CHIP"

# Check GPU cores
GPU_CORES=$(system_profiler SPDisplaysDataType | grep "Total Number of Cores" | awk '{print $5}')
echo "GPU Cores: $GPU_CORES"

echo ""
echo "======================================"
echo "Installing uv Package Manager"
echo "======================================"

if command -v uv &> /dev/null; then
    echo "uv already installed: $(uv --version)"
else
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "uv installed successfully"
fi

# Add uv to PATH for this session
export PATH="$HOME/.local/bin:$PATH"

echo ""
echo "======================================"
echo "Cloning NanoChat Repository"
echo "======================================"

if [ -d "$HOME/nanochat" ]; then
    echo "nanochat directory already exists"
    cd ~/nanochat
    echo "Pulling latest changes..."
    git pull
else
    cd ~
    git clone https://github.com/karpathy/nanochat.git
    cd nanochat
    echo "Repository cloned"
fi

echo ""
echo "======================================"
echo "Setting Up Python Environment"
echo "======================================"

# Create virtual environment and install dependencies
echo "Creating venv and installing dependencies (CPU/MPS build)..."
uv sync --extra cpu

echo ""
echo "======================================"
echo "Verifying Installation"
echo "======================================"

source .venv/bin/activate

# Check Python version
PYTHON_VERSION=$(python --version)
echo "Python: $PYTHON_VERSION"

# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check MPS availability
python -c "
import torch
mps_available = torch.backends.mps.is_available()
mps_built = torch.backends.mps.is_built()
print(f'MPS available: {mps_available}')
print(f'MPS built: {mps_built}')
if mps_available:
    print('✓ MPS (Metal Performance Shaders) is ready for GPU acceleration')
else:
    print('⚠ MPS not available - will use CPU only')
"

# Check key packages
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
python -c "import tiktoken; print('tiktoken: OK')"

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "  1. Train a model: use the 'nanochat' skill"
echo "  2. Or manually run: cd ~/nanochat && source .venv/bin/activate"
echo ""
echo "Your nanochat will be available at:"
echo "  http://${HOSTNAME}.local:8000/"
echo ""
EOF

chmod +x ~/setup_nanochat.sh
bash ~/setup_nanochat.sh
```

### Step 3: Verify the setup

After the script completes, verify everything is ready:

```bash
# Check uv is in PATH
which uv
uv --version

# Check nanochat directory
ls -la ~/nanochat/

# Check venv exists
ls -la ~/nanochat/.venv/

# Test Python environment
cd ~/nanochat
source .venv/bin/activate
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

## What Gets Installed

| Component | Location | Purpose |
|-----------|----------|---------|
| `uv` | `~/.local/bin/uv` | Fast Python package manager |
| `nanochat/` | `~/nanochat/` | Main repository |
| `.venv/` | `~/nanochat/.venv/` | Python virtual environment |
| `~/.cache/nanochat/` | Cache dir | Models, data, checkpoints |

## Environment Details

- **Python**: 3.10+ (from pyproject.toml)
- **PyTorch**: 2.9.1 with MPS support
- **Device**: MPS (Metal Performance Shaders) for Apple Silicon GPU
- **Backend**: CPU/MPS (no CUDA on Mac)

## Post-Setup

After this one-time setup, use the `nanochat` skill to:
- Train models
- Start the web server
- Run inference

## Troubleshooting

### uv command not found after install
```bash
# Add to shell profile
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Git clone fails
```bash
# Check git is installed
which git

# Alternative: download ZIP from GitHub
# Extract to ~/nanochat manually
```

### uv sync fails
```bash
# Try with verbose output
cd ~/nanochat
uv sync --extra cpu -v

# Or force reinstall
rm -rf .venv
uv sync --extra cpu
```

### MPS not available
```bash
# Check macOS version (need 13.0+)
sw_vers -productVersion

# Verify PyTorch installation
python -c "import torch; print(torch.__version__)"
```

## Verification Checklist

After setup, confirm these work:

- [ ] `uv --version` shows version
- [ ] `~/nanochat/` exists with git repo
- [ ] `~/nanochat/.venv/` exists
- [ ] `source .venv/bin/activate` works
- [ ] `python -c "import torch; print(torch.backends.mps.is_available())"` returns True

## Next Steps

Once setup is complete:
1. Use the **'nanochat'** skill to train models
2. Use the **'nanochat'** skill to start the web server
3. Access at `http://hostname.local:8000/`