# GPT-2 355M Bahasa Melayu — Training Plan

## Project Overview

**Goal:** Train a GPT-2 Medium (355M) model from scratch to generate fictional stories in Bahasa Melayu, then finetune it on your own creative works.

**Resources:**

| Resource | Spec | Role | Budget |
|----------|------|------|--------|
| RunPod A40 | 48GB VRAM, bf16 | Pretraining | $29 → ~72 hours @ $0.40/hr |
| Google Colab Pro | T4/A100 | Finetuning + experimentation | Colab subscription |
| Finetuning data | 1–10 MB of your fiction | Style transfer | — |

**Why A40 over 5090:** The A40 at $0.40/hr gives ~72 hours of compute vs ~42 hours on a 5090 at $0.69/hr. For a 355M model, raw speed matters less than total training time. The A40's 48GB VRAM also gives comfortable headroom for larger batch sizes.

**Why 355M over 124M:** With ~72 hours on an A40, a 124M model would be massively overtrained (12+ epochs, ~700% Chinchilla-optimal). That compute is better spent on a 355M model that reaches ~89% Chinchilla-optimal — a genuinely well-trained model, not just an overtrained small one. The 355M has nearly 3x the capacity, which translates directly into longer coherent passages, better narrative structure, and richer vocabulary usage.

---

## Phase 0: Local Preparation (Before Spending Any GPU Time)

This phase is **critical**. Every minute spent on RunPod doing data prep is money wasted. Budget 3–5 days for this.

### 0.1 — Download the Pretraining Corpus: FineWeb2 Malay

**Primary source: FineWeb2 `zsm_Latn` (Standard Malay, Latin script)**

The FineWeb2 Malay subset contains ~9.46 million documents — far more data than you need. This is already cleaned, globally deduplicated, and quality-filtered by HuggingFace, so you can skip most manual data cleaning. It outperforms mC4, OSCAR, CulturaX, and HPLT on downstream benchmarks.

**Target: ~2–2.5 billion tokens** (enough for ~3 epochs during the 70-hour training run).
This is roughly 3–4 million documents, or about 5–8 GB of raw text. You do NOT need the full 9.46M documents.

**How to download:**

```python
from datasets import load_dataset

# Option A: Stream and save a subset (recommended — avoids downloading all 9.46M docs)
ds = load_dataset(
    "HuggingFaceFW/fineweb-2",
    "zsm_Latn",
    split="train",
    streaming=True
)

# Take ~4M documents (adjust based on your target token count)
# Save to disk as you stream — don't try to load everything into RAM
import json
count = 0
with open("fineweb2_malay.jsonl", "w") as f:
    for example in ds:
        f.write(json.dumps({"text": example["text"]}) + "\n")
        count += 1
        if count >= 4_000_000:
            break
        if count % 100_000 == 0:
            print(f"Downloaded {count:,} documents...")

# Option B: Download the full subset (if you have disk space and want to explore)
ds_full = load_dataset("HuggingFaceFW/fineweb-2", "zsm_Latn", split="train")
ds_full.save_to_disk("fineweb2_malay_full")
```

**Supplementary sources (optional, only if you want more diversity):**

| Source | When to use | How to Get It |
|--------|-------------|---------------|
| Malay Wikipedia | Always good to add — high quality prose | `load_dataset("wikipedia", "20231101.ms")` |
| FineWeb2 Indonesian (`ind_Latn`) | Only if you want to expand beyond Malay | `load_dataset("HuggingFaceFW/fineweb-2", "ind_Latn", ...)` |

Wikipedia is worth adding (~200–400 MB) for high-quality, well-structured prose that complements the web-crawled FineWeb2 data. Indonesian data is NOT needed — FineWeb2 Malay alone is more than sufficient.

### 0.2 — Light Cleaning Pass

FineWeb2 is already heavily cleaned, but a light pass is still worthwhile:

1. **Spot-check quality** — sample 100–200 random documents and read them. Look for systematic issues: documents that are actually English, garbled text, HTML artifacts, or auto-generated content that slipped through.

2. **Optional length filter** — remove very short documents (<30 words) if present. FineWeb2's filters should have caught most of these, but a quick pass doesn't hurt.

3. **Optional: light language verification** — run `fasttext` language ID on a random sample to check that the vast majority (>95%) are genuinely Malay/Indonesian. If the false positive rate is low, skip this for the full corpus.

4. **Normalize whitespace** — collapse multiple newlines, strip trailing spaces. FineWeb2 handles most encoding issues, but a quick pass with `ftfy` can catch remaining edge cases.

**Key point:** Don't over-clean. FineWeb2's pipeline has already done the heavy lifting (language filtering via GlotLID, global MinHash deduplication, heuristic quality filters tuned per language). Your light pass is just a sanity check, not a full cleaning pipeline.

Save as chunked `.txt` files (one per ~100MB) or keep as a HuggingFace Arrow dataset.

### 0.3 — Train a Custom BPE Tokenizer

**Do not use the original GPT-2 English tokenizer.** It was trained on English text and will fragment Malay words catastrophically — common words like "pembangunan" or "masyarakat" would become 4–6 tokens instead of 1–2, effectively halving your model's context window and wasting capacity.

```
Tokenizer config:
- Algorithm: BPE (byte-level, like GPT-2)
- Vocab size: 32,000
- Training data: sample ~500MB–1GB from your FineWeb2 Malay download
- Special tokens: <|endoftext|>, <|story|>, <|end|>, <|pad|>
- Library: HuggingFace `tokenizers` (trains in minutes on CPU)
```

After training, verify:
- Common Malay words tokenize into 1–2 tokens (e.g., "masyarakat" → 1 token, "pembangunan" → 1–2 tokens)
- Your fiction-specific vocabulary (character names, places) tokenizes reasonably
- Average tokens-per-word ratio is ~1.3–1.5 (vs ~2.0+ with English GPT-2 tokenizer on Malay)
- The tokenizer handles Jawi script or Arabic loanwords gracefully if present in your data

### 0.4 — Pre-tokenize the Entire Corpus

Convert all text into token ID sequences and save as binary files (flat binary `uint16`). This eliminates tokenization overhead during training.

```
For a 5GB text corpus with 32k vocab → ~2.5B tokens → ~5GB as uint16 binary
```

Organize into train/val split (99.5/0.5 is fine — you need barely any validation data, just enough to sanity-check).

**For 355M:** Pre-tokenization is even more important than for 124M. The model trains slower per token, so any I/O or tokenization bottleneck has a larger relative impact.

### 0.5 — Prepare Your Fiction Dataset

1. Collect all your stories/novellas
2. Clean and normalize formatting consistently
3. Wrap each piece with delimiters:
   ```
   <|story|>
   Tajuk: Hujan Di Petang Hari
   Genre: Cerpen

   [full story text]
   <|end|>
   ```
4. Save separately — this goes to Colab, not RunPod
5. Also pre-tokenize this dataset using your custom tokenizer

**Tip:** Include genre or mood tags if your fiction spans different styles. The 355M model has enough capacity to learn conditional generation based on these tags during finetuning.

### 0.6 — Prepare and Test Your Training Script

Write your full training script and **test it locally on CPU for 10–20 steps**. Verify:
- Data loading works (memory-mapped binary files)
- Model initializes correctly (~355M params, custom vocab size)
- Loss decreases over a few steps
- Checkpointing saves and loads correctly
- Logging (wandb or tensorboard) works
- Gradient accumulation works correctly at the intended scale

**Package everything** into a clean directory:
```
gpt2-malay/
├── data/
│   ├── train_000.bin, train_001.bin, ...
│   └── val_000.bin
├── tokenizer/
│   ├── tokenizer.json
│   └── vocab.json
├── finetune_data/
│   └── stories.bin
├── train.py
├── model.py
├── config.py
└── requirements.txt
```

**Upload this to HuggingFace Hub or Google Drive ahead of time** so you can pull it instantly when the RunPod instance starts.

---

## Phase 1: Pretraining on RunPod A40 (~72 Hours)

### 1.1 — Environment Setup (Budget: ≤30 minutes)

The moment the instance spins up:

1. Pull your prepared data from HuggingFace Hub or Google Drive (RunPod has fast network — a 5GB dataset should download in minutes)
2. Install dependencies: `pip install torch transformers tokenizers wandb flash-attn` (most should be pre-installed on RunPod PyTorch templates)
3. Verify: GPU detected, bf16 works, flash attention available
4. Launch training inside `tmux` (critical — protects against SSH disconnects)
5. Start training immediately

**Important:** The A40 supports bf16 via Ampere Tensor Cores. Verify this works in your script — some older PyTorch builds on RunPod templates may default to fp16. Both work, but bf16 is slightly more numerically stable.

### 1.2 — Model Architecture

GPT-2 Medium, adjusted for your custom tokenizer:

```
n_layer:        24
n_head:         16
n_embd:         1024
vocab_size:     32,000 (your custom tokenizer)
block_size:     1024 (context length)
dropout:        0.0 (for pretraining, dropout hurts more than it helps at this scale)
bias:           False (slightly cleaner, following modern practice)

Total params:   ~355M
```

**Initialize from scratch** — do not load English GPT-2 Medium weights. The embedding layer won't match your tokenizer, and the learned representations are English-specific.

**Weight initialization:** Use the standard GPT-2 init scheme — normal distribution with std=0.02 for most layers, with output projection scaled by 1/√(2*n_layer) to stabilize deep residual streams.

### 1.3 — Training Hyperparameters

```
Optimizer:          AdamW
Learning rate:      3e-4 (peak) — slightly lower than 124M due to larger model
LR schedule:        Cosine decay to ~3e-5 (10% of peak)
Warmup:             2,000 steps (linear)
Weight decay:       0.1
Betas:              (0.9, 0.95)
Grad clip:          1.0
Precision:          bf16

Micro batch size:   8 (per GPU — A40 has 48GB, 355M model is ~6GB for weights+optimizer)
Gradient accum:     ~40–64 steps
Effective batch:    ~320–512 sequences = ~327k–524k tokens/batch

Total tokens:       ~6B tokens (3–4 epochs over a 2B-token corpus)
Total steps:        ~12,000–18,000 steps
```

**Why 3e-4 instead of 6e-4:** Larger models are more sensitive to learning rate. The 355M benefits from a slightly more conservative peak LR. This is well-established in the GPT-2/GPT-3 literature — the original GPT-2 Medium used 2.5e-4.

**Why this batch size:** The effective batch of ~500k tokens is roughly Chinchilla-optimal for this model size. With ~25k tokens/second throughput, each step takes ~20 seconds, giving ~4,300 steps per day, ~13,000 steps over 3 days.

### 1.4 — Performance Optimizations

Every one of these matters across a 72-hour run:

1. **bf16 mixed precision** — ~2x throughput over fp32, native on A40 Tensor Cores
2. **Flash Attention 2** — `pip install flash-attn` or use `torch.nn.functional.scaled_dot_product_attention`. The 355M model's 24 layers make attention a larger fraction of compute than in 124M — Flash Attention has more impact here.
3. **torch.compile** — wrap the model: `model = torch.compile(model)`. Expect 10–20% speedup after initial compilation overhead.
4. **Pre-tokenized binary data** — zero tokenization overhead
5. **Memory-mapped data loading** — `np.memmap` for the binary token files
6. **Gradient checkpointing** — probably NOT needed (355M fits comfortably in 48GB), but keep it as a fallback if you want to increase batch size
7. **Pin memory + num_workers=4** — standard PyTorch dataloader optimizations

Expected throughput: **20,000–30,000 tokens/second** on a single A40 with bf16 and flash attention.

### 1.5 — Monitoring and Checkpointing

- Log training loss every 100 steps to wandb (lightweight, doesn't slow training)
- Run a **quick validation loss check every 2,000 steps** (unlike with 124M where we skipped this, the 72-hour run is long enough that catching problems early saves significant money)
- Save checkpoints every **4 hours** (~18 checkpoints total, ~700MB each for 355M)
- Keep the last 5 checkpoints on disk; upload the best to Google Drive every 12 hours

**What to watch for:**
- Loss should drop steeply in the first 1,000 steps (from ~10+ down to ~6)
- By hour 10–12, loss should be around 3.5–4.0
- By hour 36 (halfway), loss should be around 3.0–3.5
- By hour 60–72, loss should plateau around 2.8–3.2
- Val loss should track train loss closely — if val loss diverges upward while train loss drops, you're memorizing (unlikely with a 2B+ token corpus, but watch for it after epoch 3)
- If loss spikes suddenly, your learning rate is too high or you have a data corruption issue

**Loss curve shape:** Expect most of the learning to happen in the first 30 hours. The remaining 40+ hours yield diminishing but real improvements — each additional epoch refines the model's grasp of Malay grammar, rare vocabulary, and longer-range patterns.

### 1.6 — 72-Hour Run Management

Unlike a 24-hour sprint, a 3-day run requires active management:

**Every 12 hours:**
- SSH in, check training is still running (tmux attach)
- Verify loss is progressing (check wandb dashboard)
- Upload the latest checkpoint to Google Drive as insurance

**At hour 48 (2 days in):**
- Evaluate whether continuing is worthwhile. If loss has fully plateaued (no improvement in last 12 hours), consider stopping early and pocketing the remaining credit for a future run.
- Generate a few test samples from the latest checkpoint to gauge quality.

**With 2 hours remaining (~hour 70):**
1. Stop training cleanly
2. Save the final checkpoint
3. Upload to persistent storage (both HuggingFace Hub AND Google Drive — belt and suspenders)
4. Also save: tokenizer files, training logs, config, wandb run ID
5. **Verify downloads are complete and not corrupted** before terminating
6. Generate a few samples as a final sanity check

**This checkpoint is your $29 investment. Upload it to at least two locations.**

---

## Phase 2: Finetuning on Google Colab Pro (No Time Pressure)

### 2.1 — Setup

1. Load your pretrained 355M checkpoint and custom tokenizer from Google Drive
2. Upload your pre-tokenized fiction dataset
3. Verify model loads correctly — generate a few samples to confirm the base model produces coherent Malay

**Colab Pro GPU notes:**
- T4 (16GB): Fits 355M comfortably. Training will be slower but perfectly adequate for finetuning.
- A100 (40GB): If Colab assigns you one, great — faster finetuning. Don't rely on getting one.
- L4: Also fine if available.

### 2.2 — Finetuning Hyperparameters

```
Optimizer:          AdamW
Learning rate:      1e-5 to 3e-5 (peak) — 10–30x lower than pretraining
LR schedule:        Cosine decay
Warmup:             100–200 steps
Weight decay:       0.1
Precision:          fp16 (T4) or bf16 (A100/L4)

Micro batch size:   2–4 (T4) or 4–8 (A100)
Gradient accum:     16–32 steps
Effective batch:    64–128 sequences

Epochs:             3–8 (for 1–10MB dataset)
```

**Why even lower LR than for 124M finetuning:** The 355M model has learned richer representations during pretraining. You want to nudge, not shove. A learning rate of 1e-5 to 3e-5 preserves the model's fluency while adapting it to your style. Start at 2e-5 and adjust based on results.

**Finetuning time estimate:** With 1–10MB of text on a T4, expect 30–90 minutes per full run. On an A100, 10–30 minutes. This is fast enough to iterate many times.

### 2.3 — Overfitting Management

With 1–10MB of text and a 355M model, overfitting is your primary enemy. The model has 355M parameters trying to learn from perhaps 1–5M tokens — that's a massive capacity-to-data ratio.

**Signs of overfitting:**
- Training loss drops below 0.5 and keeps falling toward 0
- Generated text reproduces exact passages from your stories
- Model loses ability to generate novel content — outputs feel like patchworks of your training data
- Perplexity on a held-out portion of your fiction spikes while training loss drops

**Mitigations (in order of importance):**
1. **Save a checkpoint after every epoch** — this is your most powerful tool
2. **Enable dropout (0.1)** during finetuning — unlike pretraining, dropout helps here because the dataset is small
3. **Monitor by generating samples** at each checkpoint (temperature 0.8, top-p 0.95) — human evaluation is the real test
4. **Data augmentation** — mix in 10–20% general Malay text from your pretraining corpus into each batch. This acts as a regularizer, preventing the model from "forgetting" general Malay in favor of memorizing your specific stories.
5. **Early stopping** — pick the checkpoint where generations are stylistically similar to your work but not verbatim copies. This is typically epoch 2–4 for the lower end of your data range, epoch 4–6 for the upper end.

### 2.4 — Iterative Experimentation

The beauty of Colab Pro is you can iterate freely:

- **Round 1:** Baseline finetune (LR=2e-5, dropout=0.1, 5 epochs)
- **Round 2:** Lower LR (1e-5, 8 epochs) — slower adaptation, potentially cleaner
- **Round 3:** Higher LR (5e-5, 3 epochs) — faster adaptation, riskier
- **Round 4:** Mixed data (80% your fiction, 20% general Malay, LR=2e-5, 5 epochs)
- **Round 5:** Experiment with prompt formats — genre tags, mood descriptors, character lists
- **Round 6:** Try LoRA finetuning (rank 16–32) as an alternative to full finetuning — may generalize better with small data

Each round takes 30–90 minutes on a T4. You can do 5–10 experiments in a single Colab session.

**LoRA consideration:** For a 355M model with only 1–10MB of finetuning data, LoRA (Low-Rank Adaptation) is worth trying. It freezes the base model and only trains small adapter matrices, which acts as a strong regularizer. Use rank 16–32, alpha=32–64, applied to attention Q/V projections. This gives you a ~1–5MB adapter file instead of a full 700MB checkpoint, and often generalizes better on small datasets.

---

## Phase 3: Evaluation and Generation

### 3.1 — Qualitative Evaluation

Generate 20–30 story openings with varied prompts and settings. Evaluate on:

- **Fluency:** Is the Bahasa Melayu grammatically correct? Are Malay particles (lah, kah, pun, etc.) used naturally?
- **Coherence:** Do paragraphs follow logically? Can the model sustain a scene for 3–5 paragraphs? (This is where 355M should noticeably outperform 124M.)
- **Style match:** Does it sound like your writing? Vocabulary, sentence rhythm, tone?
- **Creativity:** Does it produce novel ideas, or just remix your training data?
- **Narrative structure:** Can it maintain a character or setting across paragraphs? Handle dialogue?
- **Malay-specific:** Does it handle Malay idioms, cultural references, and sentence structure naturally?

### 3.2 — Generation Settings

```
Best starting point for fiction:
- Temperature: 0.75–0.90
- Top-p: 0.92–0.95
- Top-k: 50 (optional, as secondary filter)
- Repetition penalty: 1.1–1.2 (helps prevent loops — more important for 355M as it's more confident)
- Max length: 512–1024 tokens per generation
```

**For longer stories:** Generate in chunks with overlapping context. Feed the last 768 tokens as context for the next generation (leaving 256 tokens of the 1024 window for new content). This gives the model enough context to maintain coherence across segments.

**Interactive writing mode:** Generate 1–2 paragraphs, edit/select your preferred continuation, then feed it back as context. The 355M model is strong enough to maintain stylistic consistency across these human-in-the-loop iterations.

---

## Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| RunPod instance crashes mid-training | Medium | Run in `tmux`; checkpoint every 4h; upload to GDrive every 12h |
| A40 not available on RunPod | Low | RTX A6000 ($0.79/hr, 48GB) as backup — same VRAM, slightly less compute; or L40 if available |
| SSH disconnect during 72h run | High | `tmux` is non-negotiable; training continues even if SSH drops |
| Loss plateaus early (by hour 30) | Low-Medium | Generate samples to evaluate; if quality is good, stop early and save credit |
| Corpus too small for 355M | Low | FineWeb2 zsm_Latn has 9.46M docs — more than enough; add Indonesian subset if needed |
| Overfitting during finetuning | High | Dropout, mixed data batches, LoRA, multiple checkpoints |
| Checkpoint corrupted/lost | Low-Critical | Upload to HF Hub + Google Drive; verify before terminating |
| RunPod auto-stops on low balance | Medium | Monitor credit balance; RunPod stops pod when ~10min of credit remains |

---

## Realistic Timeline

```
WEEK BEFORE RUNPOD:
  Day 1:    Download FineWeb2 zsm_Latn (~4M docs, streaming to JSONL)
            Add Malay Wikipedia if desired
            Light cleaning pass — spot check, remove outliers
  Day 2:    Train BPE tokenizer on ~1GB sample of downloaded data
            Pre-tokenize entire corpus to binary uint16 files
  Day 3:    Write and debug training script (test on CPU for 20 steps)
  Day 4:    Prepare fiction dataset for finetuning
            Upload pre-tokenized data to HF Hub or GDrive
            Final dry run of training script on CPU

RUNPOD (3 days):
  Hour 0–0.5:     Spin up A40, pull data, verify environment
  Hour 0.5–24:    Pretraining day 1 — loss should drop to ~3.5–4.0
  Hour 24:        Check in — verify progress, upload checkpoint
  Hour 24–48:     Pretraining day 2 — loss should reach ~3.0–3.5
  Hour 48:        Check in — generate samples, evaluate quality
                  Decision point: continue or stop early?
  Hour 48–70:     Pretraining day 3 — loss refines to ~2.8–3.2
  Hour 70–72:     Final checkpoint save, upload to 2 locations, verify

AFTER RUNPOD (Colab Pro, at your own pace):
  Session 1:  Load checkpoint, sanity check, baseline finetune
  Session 2:  Hyperparameter experiments (LR sweep, dropout)
  Session 3:  LoRA experiments
  Session 4:  Mixed-data finetuning
  Session 5:  Best checkpoint selection + generation testing
  Session 6+: Prompt format experiments, interactive writing tests
```

---

## Expected Outcomes

**What this 355M model WILL do well:**
- Generate grammatically correct, natural-sounding Bahasa Melayu
- Maintain coherence across 3–5 paragraphs (noticeable improvement over 124M)
- Capture your vocabulary, sentence rhythm, and thematic preferences after finetuning
- Handle dialogue and scene-setting with reasonable quality
- Produce genuinely useful story seeds, character sketches, and narrative fragments
- Respond to genre/mood tags if trained with them during finetuning

**What this model WON'T do:**
- Write full coherent multi-page stories without human intervention
- Match GPT-4 / Claude quality in coherence or factual accuracy
- Handle complex multi-character plot threads or long-range narrative arcs
- Perfectly distinguish Malay from Indonesian in all cases (though it will be predominantly Malay)
- Produce consistently high-quality output — expect ~40–60% of generations to be usable, with the rest needing editing or regeneration

**Compared to 124M:** The 355M model should produce noticeably better output — longer coherent passages, more natural Malay, better handling of your writing style, and fewer degenerate outputs (repetitive loops, grammatical breakdowns). The difference is most apparent in longer generations (>200 tokens) where the larger model's deeper representation keeps the narrative more focused.

---

## Cost Breakdown

```
RunPod A40:     ~72 hours × $0.40/hr = $28.80
Storage:        ~10GB × $0.10/GB    = ~$1.00 (container disk while running)
                                     ─────────
Total estimate:                       ~$29.80

Buffer:         Keep ~$1–2 of credit as safety margin against
                unexpected charges (stopped pod storage, etc.)

Actual training budget:  ~$27–28 → ~68–70 hours of A40 compute
```

**Note:** RunPod requires at least 1 hour of credit balance to launch a pod and auto-stops when balance drops to ~10 minutes remaining. Plan accordingly — don't let the balance surprise you.
