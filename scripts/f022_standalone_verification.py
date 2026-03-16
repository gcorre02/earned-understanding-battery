#!/usr/bin/env python3
"""
F-022 Standalone Verification — No Battery Framework

This script answers ONE question with NO abstractions:

    Does a fresh, untrained DistilGPT-2 + LoRA change its adapter weights
    when exposed to novel text?

If YES → the generativity measured by the battery is a received property
         of the pre-trained LLM, not earned by the architecture.
If NO  → F-022 was wrong and the finding should be retracted.

Every intermediate value is printed. Nothing is hidden behind a framework.
Read this script top to bottom. If you understand each print statement,
you understand what F-022 claims.

Requirements: torch, transformers, peft
Run: python f022_standalone_verification.py
"""

import torch
import copy
import hashlib
import json

# ──────────────────────────────────────────────────────────────────
# STEP 0: Print environment
# ──────────────────────────────────────────────────────────────────
print("=" * 70)
print("F-022 STANDALONE VERIFICATION")
print("=" * 70)

import transformers, peft
print(f"torch:        {torch.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"peft:         {peft.__version__}")
print(f"device:       cpu (deterministic)")
print()

# ──────────────────────────────────────────────────────────────────
# STEP 1: Load DistilGPT-2 + LoRA (fresh, no training)
# ──────────────────────────────────────────────────────────────────
print("STEP 1: Loading fresh DistilGPT-2 + LoRA")
print("-" * 50)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig

torch.manual_seed(42)

base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["c_attn"],  # Same as FoxworthyF
    lora_dropout=0.0,
    bias="none",
)
model = get_peft_model(base_model, lora_config)
model.eval()  # No dropout, deterministic

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params:     {total_params:,}")
print(f"Trainable (LoRA): {trainable_params:,}")
print()

# ──────────────────────────────────────────────────────────────────
# STEP 2: Measure fresh adapter state
# ──────────────────────────────────────────────────────────────────
print("STEP 2: Measuring fresh adapter state (before any input)")
print("-" * 50)

def adapter_norm(mdl):
    """L2 norm of all trainable (LoRA) parameters."""
    total = 0.0
    for name, param in mdl.named_parameters():
        if param.requires_grad:
            total += param.data.norm(2).item() ** 2
    return total ** 0.5

def adapter_hash(mdl):
    """SHA256 of all trainable parameter bytes — detects ANY change."""
    h = hashlib.sha256()
    for name, param in sorted(mdl.named_parameters()):
        if param.requires_grad:
            h.update(param.data.cpu().numpy().tobytes())
    return h.hexdigest()[:16]

def print_adapter_details(mdl, label):
    """Print per-module norms for full transparency."""
    print(f"  [{label}]")
    for name, param in mdl.named_parameters():
        if param.requires_grad:
            print(f"    {name}: shape={list(param.shape)}, "
                  f"norm={param.data.norm(2).item():.6f}, "
                  f"mean={param.data.mean().item():.8f}")
    norm = adapter_norm(mdl)
    h = adapter_hash(mdl)
    print(f"  Total adapter norm: {norm:.6f}")
    print(f"  Adapter hash:       {h}")
    return norm, h

norm_before, hash_before = print_adapter_details(model, "FRESH — no input, no training")
print()

# ──────────────────────────────────────────────────────────────────
# STEP 3: Deep copy the model (to compare later)
# ──────────────────────────────────────────────────────────────────
print("STEP 3: Creating deep copy for comparison")
print("-" * 50)
model_snapshot = copy.deepcopy(model)
snapshot_hash = adapter_hash(model_snapshot)
print(f"  Snapshot hash: {snapshot_hash}")
assert snapshot_hash == hash_before, "Deep copy should be identical"
print(f"  Verified: snapshot == original")
print()

# ──────────────────────────────────────────────────────────────────
# STEP 4: Feed novel text through the model WITH gradient
# ──────────────────────────────────────────────────────────────────
print("STEP 4: Feeding novel text WITH gradient computation")
print("-" * 50)

novel_text = "The quantum properties of crystalline structures in metamaterials"
print(f"  Input text: '{novel_text}'")

inputs = tokenizer(novel_text, return_tensors="pt", padding=True, truncation=True)
print(f"  Token IDs: {inputs['input_ids'].tolist()}")
print(f"  Num tokens: {inputs['input_ids'].shape[1]}")

# Forward pass with gradient
model.train()  # Enable gradient computation
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss
print(f"  Loss: {loss.item():.6f}")

# Backward pass
loss.backward()
print(f"  Backward pass complete")

# Check gradients exist
grad_norms = {}
for name, param in model.named_parameters():
    if param.requires_grad and param.grad is not None:
        grad_norms[name] = param.grad.norm(2).item()
        
print(f"  Gradients computed for {len(grad_norms)} parameters:")
for name, gnorm in grad_norms.items():
    print(f"    {name}: grad_norm={gnorm:.8f}")
print()

# ──────────────────────────────────────────────────────────────────
# STEP 5: Apply one optimizer step (simulate what FoxworthyF does)
# ──────────────────────────────────────────────────────────────────
print("STEP 5: Applying ONE optimizer step (lr=5e-4, same as FoxworthyF)")
print("-" * 50)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=5e-4  # Same as FoxworthyF default
)
optimizer.step()
optimizer.zero_grad()
model.eval()

norm_after, hash_after = print_adapter_details(model, "AFTER one gradient step on novel text")
print()

# ──────────────────────────────────────────────────────────────────
# STEP 6: Compare — did the adapter change?
# ──────────────────────────────────────────────────────────────────
print("STEP 6: COMPARISON — Did the adapter change?")
print("=" * 70)

norm_delta = abs(norm_after - norm_before)
hash_changed = hash_after != hash_before

print(f"  Norm before:  {norm_before:.6f}")
print(f"  Norm after:   {norm_after:.6f}")
print(f"  Norm delta:   {norm_delta:.6f}")
print(f"  Hash before:  {hash_before}")
print(f"  Hash after:   {hash_after}")
print(f"  Hash changed: {hash_changed}")
print()

# Per-parameter comparison against snapshot
print("  Per-parameter changes:")
max_change = 0.0
for (name1, p1), (name2, p2) in zip(
    sorted(model.named_parameters()),
    sorted(model_snapshot.named_parameters())
):
    if p1.requires_grad:
        diff = (p1.data - p2.data).norm(2).item()
        max_change = max(max_change, diff)
        print(f"    {name1}: delta_norm={diff:.8f}")
print(f"  Max parameter change: {max_change:.8f}")
print()

# ──────────────────────────────────────────────────────────────────
# STEP 7: Now do the SAME thing with a TRAINED model
# ──────────────────────────────────────────────────────────────────
print("STEP 7: Repeat with a TRAINED model (50 steps on domain A)")
print("-" * 50)

# Reload fresh
torch.manual_seed(42)
base_model2 = AutoModelForCausalLM.from_pretrained("distilgpt2")
model2 = get_peft_model(base_model2, lora_config)

# Train on domain A (50 steps of simple text)
domain_a_texts = [
    "cooking pasta requires boiling water",
    "add salt to the water before the pasta",
    "drain the pasta when al dente",
    "tomato sauce goes well with spaghetti",
    "olive oil is essential for italian cooking",
] * 10  # 50 steps

model2.train()
opt2 = torch.optim.AdamW(
    [p for p in model2.parameters() if p.requires_grad],
    lr=5e-4
)

for i, text in enumerate(domain_a_texts):
    inp = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    out = model2(**inp, labels=inp["input_ids"])
    out.loss.backward()
    opt2.step()
    opt2.zero_grad()

model2.eval()
norm_trained, hash_trained = print_adapter_details(model2, "AFTER 50 training steps on domain A")
print()

# Now expose trained model to novel text (domain B)
print("STEP 8: Exposing TRAINED model to novel text (domain B)")
print("-" * 50)

norm_trained_before = adapter_norm(model2)
hash_trained_before = adapter_hash(model2)

model2.train()
inp_novel = tokenizer(novel_text, return_tensors="pt", padding=True, truncation=True)
out_novel = model2(**inp_novel, labels=inp_novel["input_ids"])
print(f"  Loss on novel text: {out_novel.loss.item():.6f}")
out_novel.loss.backward()

# Check gradient magnitude
trained_grad_norms = {}
for name, param in model2.named_parameters():
    if param.requires_grad and param.grad is not None:
        trained_grad_norms[name] = param.grad.norm(2).item()
print(f"  Gradient norms on novel text (trained model):")
for name, gnorm in trained_grad_norms.items():
    print(f"    {name}: grad_norm={gnorm:.8f}")

opt2.step()
opt2.zero_grad()
model2.eval()

norm_trained_after = adapter_norm(model2)
hash_trained_after = adapter_hash(model2)

print(f"  Norm before novel: {norm_trained_before:.6f}")
print(f"  Norm after novel:  {norm_trained_after:.6f}")
print(f"  Norm delta:        {abs(norm_trained_after - norm_trained_before):.6f}")
print(f"  Hash changed:      {hash_trained_after != hash_trained_before}")
print()

# ──────────────────────────────────────────────────────────────────
# STEP 9: VERDICT
# ──────────────────────────────────────────────────────────────────
print("=" * 70)
print("VERDICT")
print("=" * 70)
print()

fresh_changed = hash_after != hash_before
fresh_delta = abs(norm_after - norm_before)
trained_delta = abs(norm_trained_after - norm_trained_before)

print(f"  Fresh system + novel text:")
print(f"    Adapter changed: {fresh_changed}")
print(f"    Norm delta:      {fresh_delta:.6f}")
print()
print(f"  Trained system + novel text:")
print(f"    Adapter changed: {hash_trained_after != hash_trained_before}")
print(f"    Norm delta:      {trained_delta:.6f}")
print()

if fresh_changed and fresh_delta > 0.01:
    print("  ✓ F-022 CONFIRMED: Fresh DistilGPT-2+LoRA changes adapter weights")
    print("    on novel text. This is a RECEIVED property of the pre-trained LLM.")
    print()
    if trained_delta < fresh_delta:
        print(f"  ✓ BONUS: Trained system changes LESS ({trained_delta:.4f}) than")
        print(f"    fresh system ({fresh_delta:.4f}). Training CONSUMES responsiveness.")
    elif trained_delta >= fresh_delta:
        print(f"  ? Trained system changes MORE ({trained_delta:.4f}) than")
        print(f"    fresh system ({fresh_delta:.4f}). Investigate further.")
else:
    print("  ✗ F-022 NOT CONFIRMED: Fresh adapter does not meaningfully change.")
    print("    The finding should be retracted.")

print()
print("=" * 70)
print("Script complete. Every intermediate value is above.")
print("If you disagree with the verdict, point to the step that's wrong.")
print("=" * 70)
