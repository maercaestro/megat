# ________________________________________________
# config.py — All hyperparameters for Megat GPT-2 355M pretraining
#
# Everything that controls training behaviour lives here.
# When you want to experiment (different LR, bigger batch, etc.),
# this is the only file you need to touch.
# ________________________________________________


# ─── MODEL ARCHITECTURE ───────────────────────────────────────────────────────
#
# These are the exact dimensions for GPT-2 Medium (355M parameters).
# GPT-2 comes in four sizes:
#   small:  124M  (12 layers, 12 heads, 768 dim)
#   medium: 355M  (24 layers, 16 heads, 1024 dim)  ← we are here
#   large:  774M  (36 layers, 20 heads, 1280 dim)
#   xl:    1558M  (48 layers, 25 heads, 1600 dim)
#
# Why 355M?
#   With ~72 hours on an A40, a 124M model would be massively overtrained
#   (~700% Chinchilla-optimal — wasted compute). The 355M reaches ~89%
#   Chinchilla-optimal, which is the sweet spot for our budget.
#
# Quick parameter count sanity check:
#   Token embeddings: vocab_size × n_embd = 32000 × 1024 ≈ 32M
#   Per layer: roughly 4 × n_embd² ≈ 4M params (attention + MLP)
#   Total: 32M + 24 × ~13.5M ≈ 355M ✓

N_LAYER = 24      # number of transformer blocks stacked (depth of the model)
N_HEAD  = 16      # number of attention heads per block; each head = n_embd // n_head = 64 dims
N_EMBD  = 1024    # embedding dimension — the "width" of the model at every layer

VOCAB_SIZE  = 32000  # our custom Malay BPE tokenizer (NOT the English GPT-2 50257)
BLOCK_SIZE  = 1024   # maximum context length — how many tokens the model sees at once


# ─── TRAINING HYPERPARAMETERS ─────────────────────────────────────────────────
#
# These are drawn from the GPT-2 / GPT-3 papers and Chinchilla scaling laws.
# Ref: "Language Models are Few-Shot Learners" (Brown et al., 2020), Table 2.

MAX_LR = 3e-4   # peak learning rate
                # Why 3e-4 and not 6e-4 (the 124M value)?
                # Larger models are more sensitive to high LRs — the gradient
                # updates are amplified across 24 layers instead of 12.
                # The original GPT-2 Medium paper used 2.5e-4. We use 3e-4
                # as a conservative middle ground.

MIN_LR = 3e-5   # floor LR = 10% of peak. The cosine schedule decays down
                # to this but never below. Keeps the model learning slightly
                # even at the tail end of training.

WARMUP_STEPS = 2000  # linearly ramp LR from 0 → MAX_LR over these first steps.
                     # Without warmup, the model starts with random weights and
                     # a high LR — the first few batches produce large, chaotic
                     # gradients that can permanently damage the early layers.
                     # Warmup gives the optimizer time to "get oriented" first.

MAX_STEPS = 10000    # total optimizer steps.
                     # At ~512K tokens/step → ~5.12B total tokens.
                     # That's roughly 2.5 epochs over a 2B-token corpus.
                     # The RTX 5090 does ~35–45K tokens/sec → each step takes ~12–15s
                     # → 10000 steps × 13s ≈ 36 hours. Fits in the ~42h budget
                     # (at $0.69/hr on RunPod) with buffer for setup and uploads.


# ─── BATCH SIZE ───────────────────────────────────────────────────────────────
#
# In LLM training, "batch size" means tokens per optimizer step, not sequences.
# We target ~500K tokens/step, following Chinchilla recommendations for 355M.
#
# The problem: the A40 has 48GB VRAM, but a 355M model takes ~6GB for weights
# + optimizer states. That leaves ~42GB for activations, enough for 8 sequences
# of 1024 tokens per forward pass. But 8 × 1024 = 8,192 tokens per step is far
# too small — gradients are noisy at this scale.
#
# Solution: Gradient Accumulation.
# Run 64 forward passes (micro-steps), accumulate (ADD) the gradients from each,
# then take ONE optimizer step. The effect is identical to a true batch of
# 64 × 8,192 = 524,288 tokens — but without needing 64× the VRAM.
#
# MICRO_BATCH × SEQ_LEN × GRAD_ACCUM = TOTAL_BATCH
#      8       ×  1024   ×     64     = 524,288  ✓

MICRO_BATCH  = 4        # sequences per GPU forward pass (VRAM-limited)
                        # A40 (48GB) could fit 8; RTX 5090 (32GB) needs 4.
                        # If you get an OOM, drop to 2. If VRAM headroom looks
                        # comfortable after step 1, you can try bumping to 6.
SEQ_LEN      = 1024     # tokens per sequence = the model's context window
TOTAL_BATCH  = 524288   # target tokens per optimizer step (2^19 ≈ 512K)
                        # GRAD_ACCUM = TOTAL_BATCH // (MICRO_BATCH × SEQ_LEN) = 128


# ─── OPTIMIZER ────────────────────────────────────────────────────────────────

WEIGHT_DECAY = 0.1      # L2 regularization applied only to weight matrices.
                        # Acts as a "pull toward zero" — prevents individual weights
                        # from growing arbitrarily large, which causes overfitting.
                        # NOT applied to biases or LayerNorm parameters (see model.py).

BETAS = (0.9, 0.95)     # AdamW momentum hyperparameters.
                        # beta1=0.9: exponential decay for gradient momentum (standard).
                        # beta2=0.95: decay for squared gradient (variance estimate).
                        # The default beta2 is 0.999 — we use 0.95 because it makes
                        # the optimizer more responsive to recent gradient changes,
                        # which is better for LLMs with varying-length dependencies.

GRAD_CLIP = 1.0         # gradient norm clipping threshold.
                        # If the global gradient norm exceeds 1.0, all gradients are
                        # scaled down proportionally. Prevents "gradient explosions"
                        # — rare but catastrophic large updates when the model sees
                        # an unusual batch (e.g., very long repeated sequences).


# ─── LOGGING & CHECKPOINTING ──────────────────────────────────────────────────

LOG_INTERVAL    = 100   # print loss / lr / throughput to console every N steps
VAL_INTERVAL    = 2000  # run validation loss + save checkpoint every N steps
                        # 2000 steps × 20s/step ≈ every 11 hours — enough to
                        # catch problems early without wasting too much compute
SAVE_INTERVAL   = 1000  # save a checkpoint every N steps regardless of val loss
                        # keeps the last 5 checkpoints on disk (older ones deleted)
SAMPLE_INTERVAL = 2000  # generate a short text sample every N steps
                        # loss numbers alone can't tell you if Malay is coherent —
                        # generating actual text is the real qualitative check


# ─── PATHS ────────────────────────────────────────────────────────────────────
#
# These paths assume you pull the HuggingFace dataset repo into the working dir.
# Adjust if your RunPod layout is different.

DATA_DIR       = "data/"            # pre-tokenized binary files (train_000.bin, val_000.bin, ...)
TOKENIZER_DIR  = "megat_tokenizer/" # trained BPE tokenizer files (vocab.json + merges.txt)
CHECKPOINT_DIR = "checkpoints/"     # saved model states
LOG_DIR        = "logs/"            # optional log files


# ─── WANDB (experiment tracking) ──────────────────────────────────────────────
#
# wandb (Weights & Biases) streams your loss curves, LR schedule, and generated
# samples to a browser dashboard — very useful for monitoring a 72-hour run
# without needing to SSH in constantly.
# Sign up free at wandb.ai, then run: wandb login
# If wandb is not installed, train.py degrades gracefully to console-only logging.

WANDB_PROJECT  = "megat-gpt2"
WANDB_RUN_NAME = "355m-malay-pretrain-5090"
