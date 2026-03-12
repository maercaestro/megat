# ________________________________________________
# train.py — Production Training Script for Megat GPT-2 355M
#
# This is the main script you run on RunPod.
# It handles the full training loop:
#   - Memory-mapped data loading from pre-tokenized binary files
#   - Cosine LR schedule with warmup
#   - Gradient accumulation (simulates large batch without large VRAM)
#   - bfloat16 mixed precision (2× throughput on A40 Tensor Cores)
#   - torch.compile (fused CUDA kernels, ~15% faster)
#   - Periodic validation + checkpointing
#   - Sample generation to qualitatively track progress
#   - Optional wandb logging
#
# Usage:
#   python train.py
#
# To resume from the latest checkpoint, just run the same command —
# the script auto-detects and resumes from the most recent checkpoint.
# ________________________________________________

import os
import glob
import math
import time
import numpy as np
import torch

import config as C
from model import GPT, GPTConfig


# ─── OPTIONAL: wandb ──────────────────────────────────────────────────────────
#
# wandb streams loss curves, LR schedule, and generated samples to a web
# dashboard — very useful for checking on a 72-hour run without SSH.
# Install: pip install wandb  then  wandb login
# If not installed, everything still works — just logs to console only.

try:
    import wandb
    USE_WANDB = True
except ImportError:
    USE_WANDB = False
    print("[INFO] wandb not found — console logging only. (pip install wandb)")


# ─── DATA LOADER ──────────────────────────────────────────────────────────────

class MemmapDataLoader:
    """
    Loads pre-tokenized binary data from disk using memory mapping.

    WHAT IS MEMORY MAPPING?
    ────────────────────────
    Instead of reading the full 5GB binary file into RAM upfront, np.memmap
    tells the OS to map the file into virtual memory. The OS then loads only
    the pages (chunks) you actually read, on demand. Benefits:
    - Near-zero startup time (no waiting for a 5GB file to load)
    - RAM usage is bounded — the OS manages what stays in the cache
    - Still fast because the OS prefetches sequential reads ahead of time

    DATA FORMAT:
    ─────────────
    Files are flat binary arrays of uint16 values (2 bytes per token).
    uint16 supports values 0–65,535, which comfortably covers our 32K vocab.
    At 2B tokens → ~4GB on disk.

    EXPECTED FILE NAMES:
    ─────────────────────
    train_000.bin, train_001.bin, ...  (training data, chunked at ~50M tokens each)
    val_000.bin                        (validation data, one file is usually enough)

    HOW BATCHING WORKS:
    ────────────────────
    We read B×T + 1 consecutive tokens from the current position:
      x = tokens[pos   : pos + B*T]  reshaped to (B, T)  — the inputs
      y = tokens[pos+1 : pos + B*T+1] reshaped to (B, T)  — the targets

    y is x shifted right by 1: each token in x is paired with the token that
    follows it in y. The model learns to predict y[b, t] given x[b, 0..t].
    """

    def __init__(self, data_dir, split, B, T):
        self.B = B
        self.T = T

        pattern = os.path.join(data_dir, f"{split}_*.bin")
        self.files = sorted(glob.glob(pattern))
        assert len(self.files) > 0, (
            f"No data files found matching '{pattern}'.\n"
            f"Did you run the pre-tokenization step on Colab and upload to HF Hub?"
        )

        self.file_idx = 0
        self.pos      = 0
        self._load_file(self.file_idx)

        # Print total token count as a sanity check
        total_tokens = sum(os.path.getsize(f) // 2 for f in self.files)
        print(f"  [{split:5s}] {len(self.files)} file(s)  |  ~{total_tokens / 1e9:.2f}B tokens")

    def _load_file(self, idx):
        """Memory-map one binary file. Switching files is near-instant."""
        self.tokens = np.memmap(self.files[idx], dtype=np.uint16, mode='r')

    def next_batch(self):
        """Return the next (x, y) training pair, advancing the position."""
        B, T = self.B, self.T

        # If we're too close to the end of the current file, move to the next
        if self.pos + B * T + 1 > len(self.tokens):
            self.file_idx = (self.file_idx + 1) % len(self.files)
            self.pos = 0
            self._load_file(self.file_idx)

        # Read B*T+1 tokens and form the input/target pair
        buf = torch.tensor(
            self.tokens[self.pos : self.pos + B * T + 1].astype(np.int64),
            dtype=torch.long
        )
        x = buf[:-1].view(B, T)  # inputs: all but the last token
        y = buf[1:].view(B, T)   # targets: all but the first token (shifted right)

        self.pos += B * T
        return x, y


# ─── LEARNING RATE SCHEDULE ───────────────────────────────────────────────────

def get_lr(step):
    """
    Cosine LR schedule with linear warmup. Three phases:

    Phase 1 — Warmup (steps 0 → WARMUP_STEPS):
        LR increases linearly from 0 to MAX_LR.
        Why: at step 0 the weights are random and gradients are chaotic.
        Starting with a low LR gives the optimizer time to "find its footing"
        before taking large steps.

    Phase 2 — Cosine decay (WARMUP_STEPS → MAX_STEPS):
        LR follows a cosine curve from MAX_LR down to MIN_LR.
        The cosine shape decays quickly at first (fast progress in early training)
        then slowly near the end (careful refinement). Better than linear decay
        empirically — especially for LLMs where late training still matters.

        Formula: LR = MIN_LR + 0.5 × (MAX_LR - MIN_LR) × (1 + cos(π × progress))
        where progress goes from 0.0 (start of decay) to 1.0 (end of training).

    Phase 3 — Floor (after MAX_STEPS):
        Returns MIN_LR. Shouldn't occur in normal training.
    """
    if step < C.WARMUP_STEPS:
        return C.MAX_LR * (step + 1) / C.WARMUP_STEPS

    if step > C.MAX_STEPS:
        return C.MIN_LR

    progress = (step - C.WARMUP_STEPS) / (C.MAX_STEPS - C.WARMUP_STEPS)  # 0.0 → 1.0
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))                # 1.0 → 0.0
    return C.MIN_LR + coeff * (C.MAX_LR - C.MIN_LR)


# ─── CHECKPOINTING ────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, step, val_loss, path):
    """
    Save model + optimizer state to disk.

    WHY SAVE THE OPTIMIZER STATE?
    ──────────────────────────────
    AdamW maintains two momentum buffers per parameter:
      m (first moment):  exponential moving average of gradients
      v (second moment): exponential moving average of squared gradients

    These encode the "history" of recent gradient updates. If you only save
    model weights and resume training, the optimizer starts cold — it takes
    hundreds of steps to rebuild its momentum estimates, causing a noticeable
    loss spike. Saving the full optimizer state = seamless resume with no dip.

    The checkpoint file contains everything needed to resume exactly:
    step, model weights, optimizer state, and the best val loss so far.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save({
        'step':      step,
        'val_loss':  val_loss,
        'model':     model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': {
            'n_layer': C.N_LAYER, 'n_head': C.N_HEAD, 'n_embd': C.N_EMBD,
            'vocab_size': C.VOCAB_SIZE, 'block_size': C.BLOCK_SIZE,
        },
    }, path)
    print(f"  Checkpoint saved → {path}")


def load_checkpoint(path, model, optimizer):
    """Load checkpoint, restore model + optimizer state, return step and val_loss."""
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    print(f"  Resumed from step {ckpt['step']}  (val_loss: {ckpt['val_loss']:.4f})")
    return ckpt['step'], ckpt['val_loss']


def get_latest_checkpoint(checkpoint_dir):
    """Return the path of the most recent step checkpoint, or None if none exist."""
    files = sorted(glob.glob(os.path.join(checkpoint_dir, "ckpt_step_*.pt")))
    return files[-1] if files else None


# ─── VALIDATION ───────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_val_loss(model, val_loader, val_steps, device):
    """
    Estimate validation loss over a fixed number of batches.

    @torch.no_grad() tells PyTorch not to store activations for backpropagation
    during this function. This halves memory usage and speeds up inference —
    we don't need gradients when we're just measuring loss, not training.

    We run val_steps batches (not the full val set) to keep it fast.
    50 batches × 8 sequences × 1024 tokens = 409,600 tokens — enough to get
    a statistically reliable loss estimate in under 30 seconds on an A40.

    Val loss is your most honest measure of training progress:
    - Train loss can look great while the model is just memorizing
    - Val loss tells you how well the model generalises to unseen text
    Expected trajectory for a healthy run:
      Hour 0:  ~10.5 (random weights, uniform over 32K tokens → log(32000) ≈ 10.4)
      Hour 12: ~3.5–4.0
      Hour 36: ~3.0–3.5
      Hour 72: ~2.8–3.2
    """
    model.eval()
    losses = []
    for _ in range(val_steps):
        x, y = val_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():

    # ── Device setup ──────────────────────────────────────────────────────────
    assert torch.cuda.is_available(), (
        "No CUDA GPU found. This script is designed for RunPod (A40 or RTX 5090).\n"
        "If testing locally on Mac, use train_megat.py (the CPU/MPS version)."
    )
    device = 'cuda'
    print(f"\nGPU:  {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Reproducibility — same seed = same random weight init every run.
    # Useful for comparing runs with different hyperparameters.
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)


    # ── Gradient accumulation setup ───────────────────────────────────────────
    # Verify the batch size math: TOTAL_BATCH must divide evenly.
    assert C.TOTAL_BATCH % (C.MICRO_BATCH * C.SEQ_LEN) == 0, \
        "TOTAL_BATCH must be divisible by MICRO_BATCH × SEQ_LEN"
    grad_accum_steps = C.TOTAL_BATCH // (C.MICRO_BATCH * C.SEQ_LEN)

    print(f"\nBatch configuration:")
    print(f"  Micro batch size:     {C.MICRO_BATCH} sequences")
    print(f"  Sequence length:      {C.SEQ_LEN} tokens")
    print(f"  Grad accum steps:     {grad_accum_steps}")
    print(f"  Effective batch size: {C.TOTAL_BATCH:,} tokens/step")


    # ── Data loaders ──────────────────────────────────────────────────────────
    print("\nLoading data loaders...")
    train_loader = MemmapDataLoader(C.DATA_DIR, 'train', C.MICRO_BATCH, C.SEQ_LEN)
    val_loader   = MemmapDataLoader(C.DATA_DIR, 'val',   C.MICRO_BATCH, C.SEQ_LEN)


    # ── Tokenizer (used only for sample generation) ────────────────────────────
    from tokenizers import ByteLevelBPETokenizer
    tokenizer = ByteLevelBPETokenizer(
        os.path.join(C.TOKENIZER_DIR, "vocab.json"),
        os.path.join(C.TOKENIZER_DIR, "merges.txt"),
    )
    print(f"\nTokenizer loaded  |  vocab size: {tokenizer.get_vocab_size()}")


    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nInitializing model...")
    model_config = GPTConfig(
        block_size=C.BLOCK_SIZE,
        vocab_size=C.VOCAB_SIZE,
        n_layer=C.N_LAYER,
        n_head=C.N_HEAD,
        n_embd=C.N_EMBD,
    )
    model = GPT(model_config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params / 1e6:.1f}M")

    # torch.compile JIT-compiles the model into optimised CUDA kernels.
    # The first 1–2 steps will be slower (compilation overhead), then ~15% faster.
    # Occasionally causes issues on some PyTorch versions — comment out if needed.
    print("  Compiling with torch.compile() ...")
    model = torch.compile(model)


    # ── Optimizer ─────────────────────────────────────────────────────────────
    print("\nConfiguring optimizer...")
    optimizer = model.configure_optimizers(
        weight_decay=C.WEIGHT_DECAY,
        learning_rate=C.MAX_LR,
        device=device,
    )


    # ── Resume from checkpoint (if one exists) ────────────────────────────────
    start_step    = 0
    best_val_loss = float('inf')

    latest_ckpt = get_latest_checkpoint(C.CHECKPOINT_DIR)
    if latest_ckpt:
        print(f"\nFound checkpoint: {latest_ckpt}")
        start_step, best_val_loss = load_checkpoint(latest_ckpt, model, optimizer)
        start_step += 1  # resume from the NEXT step, not the one we already ran
    else:
        print("\nNo checkpoint found — starting from scratch.")

    os.makedirs(C.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(C.LOG_DIR, exist_ok=True)


    # ── wandb ─────────────────────────────────────────────────────────────────
    if USE_WANDB:
        wandb.init(
            project=C.WANDB_PROJECT,
            name=C.WANDB_RUN_NAME,
            resume='allow',  # if a run with this name exists, resume it
            config={
                'n_layer': C.N_LAYER, 'n_head': C.N_HEAD, 'n_embd': C.N_EMBD,
                'vocab_size': C.VOCAB_SIZE, 'block_size': C.BLOCK_SIZE,
                'max_lr': C.MAX_LR, 'min_lr': C.MIN_LR,
                'warmup_steps': C.WARMUP_STEPS, 'max_steps': C.MAX_STEPS,
                'total_batch': C.TOTAL_BATCH, 'grad_accum': grad_accum_steps,
            },
        )
        print("wandb initialized.")


    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training: step {start_step} → {C.MAX_STEPS}")
    print(f"  Tokens/step: {C.TOTAL_BATCH:,}  |  Total: ~{C.TOTAL_BATCH * C.MAX_STEPS / 1e9:.1f}B tokens")
    print(f"{'='*60}\n")

    model.train()

    for step in range(start_step, C.MAX_STEPS + 1):
        t0 = time.time()


        # ── Validation + checkpoint ───────────────────────────────────────────
        #
        # Every VAL_INTERVAL steps, measure val loss and save a checkpoint.
        # Doing both together means every checkpoint has a paired val_loss —
        # useful for picking the best checkpoint after training ends.
        if step % C.VAL_INTERVAL == 0:
            val_loss = estimate_val_loss(model, val_loader, val_steps=50, device=device)
            is_best  = val_loss < best_val_loss

            print(f"\n── Validation ──────────────────────────────")
            print(f"   Step {step:,}  |  val_loss: {val_loss:.4f}  |  best: {best_val_loss:.4f}")

            # Save a numbered checkpoint at every val step
            ckpt_path = os.path.join(C.CHECKPOINT_DIR, f"ckpt_step_{step:06d}.pt")
            save_checkpoint(model, optimizer, step, val_loss, ckpt_path)

            # Separately save the best checkpoint so far
            if is_best:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, step, val_loss,
                                os.path.join(C.CHECKPOINT_DIR, "ckpt_best.pt"))
                print(f"   *** New best model! ***")

            # Keep only the last 5 numbered checkpoints (saves disk space on RunPod)
            all_numbered = sorted(glob.glob(os.path.join(C.CHECKPOINT_DIR, "ckpt_step_*.pt")))
            for old in all_numbered[:-5]:
                os.remove(old)

            print(f"────────────────────────────────────────────\n")

            if USE_WANDB:
                wandb.log({'val_loss': val_loss, 'best_val_loss': best_val_loss}, step=step)


        # ── Sample generation ─────────────────────────────────────────────────
        #
        # Loss numbers alone don't tell you if the model is writing coherent Malay.
        # Generating actual text every SAMPLE_INTERVAL steps gives you a qualitative
        # check — you can see when the model starts forming real Malay words,
        # then real sentences, then coherent passages.
        #
        # Expected progression:
        #   Early (hour 0–6):   mostly noise, some common short words appear
        #   Mid (hour 12–24):   Malay words form, grammar is rough
        #   Late (hour 48–72):  sentences are grammatical, paragraphs are coherent
        if step % C.SAMPLE_INTERVAL == 0 and step > 0:
            prompt       = "Pada suatu hari"  # "One fine day" — common story opener
            prompt_ids   = tokenizer.encode(prompt).ids
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

            generated   = model.generate(prompt_tensor, max_new_tokens=80,
                                         temperature=0.8, top_p=0.95)
            sample_text = tokenizer.decode(generated[0].tolist())

            print(f"\n── Sample (step {step:,}) ──────────────────────────")
            print(f"   {sample_text}")
            print(f"────────────────────────────────────────────\n")

            if USE_WANDB:
                wandb.log({'sample': wandb.Html(f"<pre>{sample_text}</pre>")}, step=step)


        # ── Forward pass with gradient accumulation ───────────────────────────
        #
        # We want 512K tokens per optimizer step, but can only fit 8K per forward
        # pass. Solution: run grad_accum_steps=64 micro-steps, accumulate gradients,
        # then take ONE optimizer update.
        #
        # IMPORTANT: loss must be divided by grad_accum_steps before .backward().
        # PyTorch's .backward() ADDS gradients to existing ones (it doesn't zero
        # them). So after 64 micro-steps, accumulated gradient = sum of 64 losses.
        # Dividing each loss by 64 makes the accumulated gradient = MEAN of 64
        # losses — equivalent to computing the loss on the full 512K-token batch.

        optimizer.zero_grad()  # clear gradients from the previous step
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            # bfloat16 mixed precision:
            # The forward pass and loss computation run in bfloat16 (16-bit).
            # bfloat16 has the same exponent range as float32 but 3× fewer mantissa
            # bits — so it can represent very large/small numbers but with less
            # precision. That's fine for neural network activations.
            # Benefits: ~2× throughput on A40 Tensor Cores, ~2× less VRAM for
            # activations. The optimizer still runs in float32 for stability.
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, loss = model(x, y)

            loss = loss / grad_accum_steps  # normalize so accumulated grad = mean
            loss_accum += loss.detach()     # track total loss (detached from graph)
            loss.backward()                 # accumulate gradients


        # ── Gradient clipping ─────────────────────────────────────────────────
        #
        # If the global gradient norm (sqrt of sum of squares across all params)
        # exceeds GRAD_CLIP, scale all gradients down proportionally.
        # This prevents rare but catastrophic large updates ("gradient explosions")
        # that can occur when the model encounters an unusual batch.
        # clip_grad_norm_ returns the pre-clip norm — useful to log and watch.
        # If norm is consistently > 1.0 early in training, that's normal.
        # If it's very large (10+) in mid-training, something may be wrong.
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)


        # ── LR schedule update ────────────────────────────────────────────────
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


        # ── Optimizer step ────────────────────────────────────────────────────
        optimizer.step()


        # ── Timing ────────────────────────────────────────────────────────────
        # synchronize() waits for all GPU operations to finish before measuring time.
        # Without it, the timer would measure only CPU-side scheduling, not actual
        # GPU work — giving misleadingly fast numbers.
        torch.cuda.synchronize()
        t1 = time.time()


        # ── Console logging ───────────────────────────────────────────────────
        if step % C.LOG_INTERVAL == 0:
            dt              = t1 - t0
            tokens_per_sec  = C.TOTAL_BATCH / dt
            hours_remaining = (C.MAX_STEPS - step) * dt / 3600

            print(
                f"step {step:5d}/{C.MAX_STEPS} | "
                f"loss: {loss_accum.item():.4f} | "
                f"lr: {lr:.2e} | "
                f"norm: {norm:.2f} | "
                f"tok/s: {tokens_per_sec:,.0f} | "
                f"dt: {dt:.1f}s | "
                f"ETA: {hours_remaining:.1f}h"
            )

            if USE_WANDB:
                wandb.log({
                    'train_loss':    loss_accum.item(),
                    'lr':            lr,
                    'grad_norm':     norm,
                    'tokens_per_sec': tokens_per_sec,
                }, step=step)


    # ── Training complete ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Training complete!")
    print("="*60)

    final_path = os.path.join(C.CHECKPOINT_DIR, "ckpt_final.pt")
    save_checkpoint(model, optimizer, C.MAX_STEPS, best_val_loss, final_path)
    print(f"\nFinal checkpoint: {final_path}")
    print(f"Best val loss achieved: {best_val_loss:.4f}")
    print("\nRemember: upload checkpoints to BOTH HuggingFace Hub AND Google Drive!")

    if USE_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
