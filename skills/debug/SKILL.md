---
name: debug
description: Debugging guide for nanochat issues - lessons learned from fixing MPS NaN and dataloader bugs
---

# NanoChat Debugging Guide

This skill documents the debugging process and solutions for issues encountered when setting up nanochat on Mac Studio Ultra.

## Issue 1: MPS/CPU SFT Training NaN Loss

### Symptoms
- SFT training starts with normal loss (~2.5)
- Step 2 immediately shows `loss: nan`
- Affects both MPS and CPU devices
- Base model training works fine

### Root Cause Analysis

**Step 1: Verify Base Model Works**
```bash
# Test base model inference
python -c "from nanochat.checkpoint_manager import load_model; model, _, _ = load_model('base', 'cpu', model_tag='d6', step=3000); print('Model loads OK')"
```
✓ Base model is fine

**Step 2: Test Individual Components**
```bash
# Create test script to isolate components
python debug_training.py  # Test simple training loop
python debug_sft_tasks.py  # Test each SFT task
python debug_grad_accum.py  # Test gradient accumulation
python debug_compile.py  # Test torch.compile
```
✓ All components work individually

**Step 3: Test Exact SFT Loop**
```bash
# Create debug_sft_loop.py - mimic exact SFT training
python debug_sft_loop.py
```
✓ Works with simple data

**Step 4: Debug Actual SFT Script**
```bash
# Add debug output to chat_sft.py to print batch contents
# Found: x range [32759, 32759], y range [-1, -1]
# All inputs are BOS tokens, all targets are -1 (masked)
```

**Root Cause Identified:**
The SFT dataloader's `bestfit` packing algorithm creates empty batches when the conversation buffer is depleted. Rows get filled with only BOS tokens, and all targets are masked as -1. Cross-entropy with all ignore_index values returns NaN.

### Solution

**File:** `scripts/chat_sft.py`  
**Function:** `sft_data_generator_bos_bestfit()`

**Fix 1 - Lines 258-274:** Prevent empty rows
```python
# When no conversation fits and row is empty, force a conversation
if content_len == 0 and len(conv_buffer) > 0:
    # Take smallest conversation even if it doesn't fit
    smallest_idx = min(range(len(conv_buffer)), key=lambda i: len(conv_buffer[i][0]))
    conv, conv_mask = conv_buffer.pop(smallest_idx)
    # Truncate if necessary
    conv = conv[:row_capacity]
    conv_mask = conv_mask[:row_capacity]
    row.extend(conv)
    mask_row.extend(conv_mask)
    consumed += ddp_world_size
    content_len = len(row)
else:
    # Normal padding
    row.extend([bos_token] * remaining)
    mask_row.extend([0] * remaining)
```

**Fix 2 - Lines 279-286:** Ensure minimum content length
```python
# Ensure content_len is at least 1 to avoid masking everything
actual_content_len = content_len if padded else row_capacity
if actual_content_len == 0:
    # If somehow we have no content, count non-BOS tokens
    actual_content_len = sum(1 for t in row if t != bos_token)
    if actual_content_len == 0:
        actual_content_len = 1  # Force at least 1 token
row_lengths.append(actual_content_len)
```

### Verification
```bash
# Test with debug version
python -m scripts.chat_sft_debug --num-iterations=10 --device-type mps
# Output: All losses finite (2.5-6.0 range), no NaN
```

## Issue 2: Multiprocessing Crash During Evaluation

### Symptoms
- Training crashes at step 400 with `EOFError`
- Error in `multiprocessing.Manager()` during HumanEval
- Happens during `--eval-every` validation

### Root Cause
HumanEval evaluation uses multiprocessing to execute generated code safely. On Mac/MPS, this causes semaphore/resource tracker issues.

### Solution
Disable evaluation during training:
```bash
python -m scripts.chat_sft \
    --eval-every=-1 \        # Disable validation
    --chatcore-every=-1 \    # Disable ChatCORE metric
    ...
```

Run evaluation separately after training completes.

## Issue 3: Slow Log Output with nohup

### Symptoms
- nohup log files don't update in real-time
- Process running but log appears stuck
- Can't see training progress

### Solutions

**Option 1: Use unbuffered Python output**
```bash
export PYTHONUNBUFFERED=1
python -u -m scripts.chat_sft ...  # -u flag for unbuffered
```

**Option 2: Use stdbuf**
```bash
stdbuf -oL python -m scripts.chat_sft ...
```

**Option 3: Run in foreground for debugging**
```bash
# For initial testing, run without nohup
timeout 300 python -m scripts.chat_sft ...
```

## Issue 4: MPS vs CPU Performance

### Comparison

| Device | Speed | NaN Issue | Notes |
|--------|-------|-----------|-------|
| CPU | ~5k tok/sec | Same bug | Slower but stable |
| MPS | ~50k tok/sec | Same bug | 10x faster, same fix |

### Key Insight
The NaN issue was **NOT** MPS-specific. It was a dataloader bug that affected both CPU and MPS equally. After fixing the dataloader, both work perfectly.

**Recommendation:** Use MPS for 10x faster training.

## Debugging Methodology

### 1. Isolate the Problem
```bash
# Test individual components
- Base model loads? ✓
- Optimizer works? ✓
- Simple training loop? ✓
- Individual tasks? ✓
- Gradient accumulation? ✓
- torch.compile? ✓
```

### 2. Add Debug Output
```python
# In training loop, add:
print(f"DEBUG: x shape: {x.shape}, range: [{x.min()}, {x.max()}]")
print(f"DEBUG: y shape: {y.shape}, has -1: {(y == -1).sum()}")
print(f"DEBUG: loss: {loss.item()}, NaN: {torch.isnan(loss).item()}")
```

### 3. Create Minimal Reproduction
Create standalone scripts that test specific functionality without full pipeline.

### 4. Check Data Pipeline
Most training issues are data issues:
- Empty batches
- All masked targets
- Wrong tensor shapes
- Invalid token IDs

### 5. Verify Fix
- Test on small scale (10-20 steps)
- Check for NaN losses
- Verify checkpoints saved
- Test inference with trained model

## Common Pitfalls

### Pitfall 1: Optimizer State Corruption
**Don't** load optimizer state from base model when starting SFT:
```bash
# Use --load-optimizer=0 for fresh start
python -m scripts.chat_sft --load-optimizer=0 ...
```

### Pitfall 2: Wrong Device Specification
```bash
# Explicitly specify device
python -m scripts.chat_sft --device-type mps ...  # or cpu
```

### Pitfall 3: dtype Issues on MPS
```bash
# Force float32 for MPS stability
export NANOCHAT_DTYPE=float32
```

### Pitfall 4: Running Out of Memory
Reduce batch size:
```bash
python -m scripts.chat_sft --device-batch-size=4 ...  # instead of 32
```

## Tools for Debugging

### 1. Process Monitoring
```bash
# Watch process
ps aux | grep chat_sft

# Check CPU/Memory
top -pid <PID>

# Monitor log
tail -f ~/training.log
```

### 2. Log Analysis
```bash
# Check for NaN
grep "loss: nan" training.log

# Count steps
grep -c "loss:" training.log

# View last N steps
grep "loss:" training.log | tail -20
```

### 3. Checkpoint Verification
```bash
# Check checkpoint saved
ls -lh ~/.cache/nanochat/chatsft_checkpoints/d6/

# Verify model loads
python -c "from nanochat.checkpoint_manager import load_model; model, _, _ = load_model('sft', 'mps', step=500)"
```

## Success Verification Checklist

- [ ] Training completes without NaN
- [ ] Checkpoints saved successfully
- [ ] Model loads for inference
- [ ] Web server starts
- [ ] Health check passes
- [ ] Chat responds to input
- [ ] No memory leaks
- [ ] Loss trending down/stable

## References

- Original issue: PyTorch MPS #121209, nanoGPT #217
- Fix commit: `b285271 Fix SFT dataloader NaN issue`
- Location: `scripts/chat_sft.py`, lines 258-286
