"""
Diagnostic script to understand why train_megat.py produces garbage text.
This is a copy with added diagnostics — the original train_megat.py is untouched.
"""

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# ============================================================
# MODEL DEFINITION (identical to train_megat.py)
# ============================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel
        print(f"Loading weights from pretrained gpt: {model_type}")

        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('attn.bias') or 'c_attn' in k]  # FIXED filter to keep c_attn.bias

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys) == len(sd_keys_hf), f"mismatched keys: {len(sd_keys)} != {len(sd_keys_hf)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model, model_hf  # Return BOTH for comparison

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits


# ============================================================
# DIAGNOSTICS
# ============================================================

import tiktoken

print("=" * 70)
print("DIAGNOSIS: Why is the generated text garbage?")
print("=" * 70)

# Load both models
model, model_hf = GPT.from_pretrained("gpt2")
model.eval()
model_hf.eval()

device = 'cpu'  # Use CPU so MPS doesn't interfere
model.to(device)
model_hf.to(device)

enc = tiktoken.get_encoding("gpt2")
prompt = "Hello, I'm a language model, "
tokens = enc.encode(prompt)
tokens_t = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("TEST 1: Compare logits (our model vs HuggingFace)")
print("=" * 70)

with torch.no_grad():
    our_logits = model(tokens_t)
    hf_logits = model_hf(tokens_t).logits

print(f"Our logits shape: {our_logits.shape}")
print(f"HF  logits shape: {hf_logits.shape}")

max_diff = (our_logits - hf_logits).abs().max().item()
mean_diff = (our_logits - hf_logits).abs().mean().item()
print(f"Max  absolute difference: {max_diff:.6f}")
print(f"Mean absolute difference: {mean_diff:.6f}")

# Show top predictions from each
our_last = our_logits[0, -1, :]
hf_last = hf_logits[0, -1, :]
our_top5 = torch.topk(our_last, 5)
hf_top5 = torch.topk(hf_last, 5)
print(f"\nOur model top-5 next tokens: {[enc.decode([t]) for t in our_top5.indices.tolist()]}")
print(f"HF  model top-5 next tokens: {[enc.decode([t]) for t in hf_top5.indices.tolist()]}")

if max_diff > 1.0:
    print("\n>>> LARGE DIFFERENCE — model output differs significantly from HuggingFace!")
elif max_diff > 0.01:
    print("\n>>> Small numerical differences (floating point). Model seems OK.")
else:
    print("\n>>> Logits match almost perfectly.")

# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("TEST 2: Weight tying check (lm_head.weight vs wte.weight)")
print("=" * 70)

our_lm = model.lm_head.weight.data
our_wte = model.transformer.wte.weight.data
are_equal = torch.equal(our_lm, our_wte)
print(f"lm_head.weight == wte.weight (values)?  {are_equal}")
print(f"lm_head.weight is wte.weight (identity)? {our_lm.data_ptr() == our_wte.data_ptr()}")

if not are_equal:
    diff = (our_lm - our_wte).abs().max().item()
    print(f"Max difference between lm_head and wte: {diff}")

hf_lm = model_hf.lm_head.weight.data
hf_wte = model_hf.transformer.wte.weight.data
print(f"HF lm_head.weight is wte.weight (identity)? {hf_lm.data_ptr() == hf_wte.data_ptr()}")

# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("TEST 3: Key mapping check")
print("=" * 70)

sd = model.state_dict()
sd_hf = model_hf.state_dict()
our_keys = set(k for k in sd.keys() if not k.endswith('attn.bias'))
hf_keys = set(k for k in sd_hf.keys() if not k.endswith('attn.masked_bias') and not k.endswith('attn.bias'))
only_ours = our_keys - hf_keys
only_hf = hf_keys - our_keys
print(f"Our keys: {len(our_keys)}, HF keys: {len(hf_keys)}")
if only_ours: print(f"Keys ONLY in ours: {only_ours}")
if only_hf: print(f"Keys ONLY in HF:   {only_hf}")
if not only_ours and not only_hf: print("All key names match.")

# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("TEST 4: Weight value comparison (all layers)")
print("=" * 70)

transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
mismatches = []
for k in sorted(hf_keys):
    our_w = sd[k]
    hf_w = sd_hf[k]
    if any(k.endswith(w) for w in transposed):
        hf_w = hf_w.t()
    diff = (our_w.float() - hf_w.float()).abs().max().item()
    if diff > 1e-5:
        mismatches.append((k, diff))

if mismatches:
    print(f"MISMATCHES found ({len(mismatches)}):")
    for k, d in mismatches:
        print(f"  {k}: max diff = {d}")
else:
    print("All weights match within 1e-5.")

# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("TEST 5: Generate with OUR model on CPU")
print("=" * 70)

num_return_sequences = 5
max_length = 30

torch.manual_seed(42)
x = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0).repeat(num_return_sequences, 1).to(device)
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)

print("Our model output (CPU):")
for i in range(num_return_sequences):
    print(f"  > {enc.decode(x[i, :max_length].tolist())}")

# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("TEST 6: Generate with HUGGINGFACE model on CPU (ground truth)")
print("=" * 70)

torch.manual_seed(42)
x_hf = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0).repeat(num_return_sequences, 1).to(device)
while x_hf.size(1) < max_length:
    with torch.no_grad():
        logits_hf = model_hf(x_hf).logits[:, -1, :]
        probs_hf = F.softmax(logits_hf, dim=-1)
        topk_probs, topk_indices = torch.topk(probs_hf, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x_hf = torch.cat((x_hf, xcol), dim=1)

print("HuggingFace model output (CPU, ground truth):")
for i in range(num_return_sequences):
    print(f"  > {enc.decode(x_hf[i, :max_length].tolist())}")

# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("TEST 7: MPS vs CPU comparison")
print("=" * 70)

if torch.backends.mps.is_available():
    model.to('mps')
    tokens_mps = tokens_t.to('mps')
    with torch.no_grad():
        logits_mps = model(tokens_mps).to('cpu')
    model.to('cpu')
    with torch.no_grad():
        logits_cpu = model(tokens_t)
    mps_diff = (logits_mps - logits_cpu).abs().max().item()
    print(f"Max logit diff MPS vs CPU: {mps_diff:.6f}")
    if mps_diff > 1.0:
        print(">>> MPS introduces SIGNIFICANT numerical differences!")
    else:
        print(">>> MPS and CPU produce similar results.")
else:
    print("MPS not available.")

# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("TEST 8: ROOT CAUSE — the 'endswith(attn.bias)' filter bug")
print("=" * 70)

# Show the filtering bug
test_keys = [
    'transformer.h.0.attn.bias',        # causal mask buffer (SHOULD be filtered)
    'transformer.h.0.attn.c_attn.bias',  # attention projection bias (should NOT be filtered)
    'transformer.h.0.attn.c_proj.bias',  # projection bias (should NOT be filtered)
]
print("The filter: k.endswith('attn.bias')")
print()
for k in test_keys:
    filtered = k.endswith('attn.bias')
    print(f"  '{k}'")
    print(f"    → endswith('attn.bias') = {filtered}")
    if 'c_attn' in k and filtered:
        print(f"    >>> BUG! This is a REAL parameter, not a buffer! It should NOT be filtered!")

print()

# Show the actual bias values (our random init vs HF pretrained)
our_bias = model.transformer.h[0].attn.c_attn.bias.data
hf_bias = model_hf.transformer.h[0].attn.c_attn.bias.data
print(f"Our c_attn.bias[:10] (random init, never overwritten):")
print(f"  {our_bias[:10]}")
print(f"HF  c_attn.bias[:10] (pretrained):")
print(f"  {hf_bias[:10]}")
print(f"Max difference: {(our_bias - hf_bias).abs().max().item():.4f}")

# Count how many biases are affected 
sd_our = model.state_dict()
affected_keys = [k for k in sd_our.keys() if 'c_attn.bias' in k]
print(f"\nAffected parameters ({len(affected_keys)} total):")
for k in affected_keys[:3]:
    print(f"  {k}")
print(f"  ... and {len(affected_keys) - 3} more across all layers")

print()
print("EXPLANATION:")
print("  In from_pretrained(), this line:")
print("    sd_keys = [k for k in sd_keys if not k.endswith('attn.bias')]")
print()
print("  is meant to filter out the causal mask BUFFER: 'transformer.h.X.attn.bias'")
print("  But 'transformer.h.X.attn.c_attn.bias' ALSO ends with 'attn.bias'!")
print("  So the c_attn.bias PARAMETER keys are filtered out from both sides.")
print("  The len() assertion still passes (both lose the same keys).")
print("  But c_attn.bias is never copied → stays at random init → garbage output!")

print("\n" + "=" * 70)
print("TEST 9: VERIFY FIX — reload with corrected filter")
print("=" * 70)

# Reload with the CORRECT filter
from transformers import GPT2LMHeadModel as HFModel
config_fix = GPTConfig(n_layer=12, n_head=12, n_embd=768, vocab_size=50257, block_size=1024)
model_fixed = GPT(config_fix)
sd_fixed = model_fixed.state_dict()

# CORRECT filter: match ONLY the buffer key (not c_attn.bias)
sd_keys_fixed = [k for k in sd_fixed.keys() if k != 'transformer.h.0.attn.bias' 
                 and not (k.endswith('.attn.bias') and '.c_attn' not in k and '.c_proj' not in k)]
# Actually simpler: filter keys ending with exactly ".attn.bias" but NOT ".c_attn.bias"
sd_keys_fixed = [k for k in sd_fixed.keys() 
                 if not (k.endswith('attn.bias') and 'c_attn' not in k)]

model_hf_fix = HFModel.from_pretrained("gpt2")
sd_hf_fix = model_hf_fix.state_dict()
sd_keys_hf_fix = [k for k in sd_hf_fix.keys() 
                  if not k.endswith('attn.masked_bias')
                  and not (k.endswith('attn.bias') and 'c_attn' not in k)]

transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

print(f"Fixed key counts: ours={len(sd_keys_fixed)}, HF={len(sd_keys_hf_fix)}")
assert len(sd_keys_fixed) == len(sd_keys_hf_fix), f"Key mismatch: {len(sd_keys_fixed)} vs {len(sd_keys_hf_fix)}"

for k in sd_keys_hf_fix:
    if any(k.endswith(w) for w in transposed):
        with torch.no_grad():
            sd_fixed[k].copy_(sd_hf_fix[k].t())
    else:
        with torch.no_grad():
            sd_fixed[k].copy_(sd_hf_fix[k])

model_fixed.eval().to(device)

# Verify bias is now correct
our_bias_fixed = model_fixed.transformer.h[0].attn.c_attn.bias.data
hf_bias_check = model_hf_fix.transformer.h[0].attn.c_attn.bias.data
print(f"c_attn.bias matches after fix? {torch.allclose(our_bias_fixed, hf_bias_check, atol=1e-5)}")

# Compare logits
with torch.no_grad():
    fixed_logits = model_fixed(tokens_t)
    hf_logits_check = model_hf_fix(tokens_t).logits
max_diff_fixed = (fixed_logits - hf_logits_check).abs().max().item()
print(f"Logit diff after fix: {max_diff_fixed:.6f}")

# Generate text with fixed model
torch.manual_seed(42)
x_fix = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0).repeat(num_return_sequences, 1).to(device)
while x_fix.size(1) < max_length:
    with torch.no_grad():
        logits_fix = model_fixed(x_fix)[:, -1, :]
        probs_fix = F.softmax(logits_fix, dim=-1)
        topk_probs, topk_indices = torch.topk(probs_fix, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x_fix = torch.cat((x_fix, xcol), dim=1)

print("\nFIXED model output (CPU):")
for i in range(num_return_sequences):
    print(f"  > {enc.decode(x_fix[i, :max_length].tolist())}")

# HF reference for comparison
torch.manual_seed(42)
x_hf2 = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0).repeat(num_return_sequences, 1).to(device)
while x_hf2.size(1) < max_length:
    with torch.no_grad():
        logits_hf2 = model_hf_fix(x_hf2).logits[:, -1, :]
        probs_hf2 = F.softmax(logits_hf2, dim=-1)
        topk_probs, topk_indices = torch.topk(probs_hf2, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x_hf2 = torch.cat((x_hf2, xcol), dim=1)

print("\nHuggingFace reference output (CPU):")
for i in range(num_return_sequences):
    print(f"  > {enc.decode(x_hf2[i, :max_length].tolist())}")

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print("""
ROOT CAUSE FOUND:
================
In from_pretrained(), the filter:
    sd_keys = [k for k in sd_keys if not k.endswith('attn.bias')]

is meant to skip the causal mask buffer ('transformer.h.X.attn.bias'),
but it ALSO skips 'transformer.h.X.attn.c_attn.bias' because that
string also ends with 'attn.bias'!

Result: The c_attn.bias parameters (12 layers × 2304 values each) are
NEVER copied from the pretrained model. They keep their random init
values, corrupting every attention computation → garbage text.

FIX (in train_megat.py, line ~108):
  Change:
    sd_keys = [k for k in sd_keys if not k.endswith('attn.bias')]
  To:
    sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias') or 'c_attn' in k]

  And similarly for sd_keys_hf (line ~113):
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('attn.bias')]
  To:
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias') or 'c_attn' in k]

MINOR ISSUES (non-critical):
  1. Line 2: 'from logging import config' — unused import, remove it
  2. Line 98: print uses %s with print() not logging — use f-string instead
""")

