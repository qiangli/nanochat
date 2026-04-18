import torch
import os
import sys

sys.path.insert(0, "/Users/noviadmin/nanochat")

from nanochat.checkpoint_manager import load_model_from_dir
from nanochat.engine import Engine

device = torch.device("mps")
print("Loading SFT model...")

# Load directly from checkpoint dir
checkpoint_dir = os.path.expanduser("~/.cache/nanochat/chatsft_checkpoints/d6")
model, tokenizer, meta = load_model_from_dir(
    checkpoint_dir, device, phase="train", step=500
)

print(f"Model loaded: depth={model.config.n_layer}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
print(f"Vocab size: {tokenizer.get_vocab_size()}")

# Test simple completion
print("\n=== Test 1: Simple completion ===")
engine = Engine(model, tokenizer)
prompt = "The sky is blue because"
print(f"Prompt: '{prompt}'")

# Encode
tokens = tokenizer.encode(prompt)
print(f"Token IDs: {tokens}")

# Generate a few tokens
print("\nGenerating 10 tokens...")
result = []
for i, token in enumerate(engine.generate(tokens, max_tokens=10, temperature=0.8)):
    word = tokenizer.decode([token])
    result.append(word)
    print(f"Token {i}: ID={token}, Text='{word}'")

full_output = prompt + "".join(result)
print(f"\nFull output: '{full_output}'")

# Test 2: Chat format
print("\n=== Test 2: Chat format ===")
conversation = [{"role": "user", "content": "Why is the sky blue?"}]
ids, mask = tokenizer.render_conversation(conversation)
print(f"Conversation tokens: {len(ids)}")
print(f"First 20 tokens: {ids[:20]}")

# Continue generation
print("\nGenerating response...")
result2 = []
for i, token in enumerate(engine.generate(ids, max_tokens=20, temperature=0.8)):
    word = tokenizer.decode([token])
    result2.append(word)
    if i < 5:
        print(f"Token {i}: '{word}'")

print(f"\nResponse: '{''.join(result2[:50])}...'")
