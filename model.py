# ________________________________________________
# model.py — GPT-2 Architecture for Megat
#
# This file contains the full model definition:
#   GPTConfig        → dataclass holding all architecture dimensions
#   CausalSelfAttention → multi-head attention (the "communication" step)
#   MLP              → feed-forward network (the "computation" step)
#   Block            → one transformer layer = Attention + MLP + residuals
#   GPT              → the full model: embeddings + N blocks + output head
#
# Architecture is GPT-2 Medium (355M) trained from scratch on Malay.
# Based on Karpathy's nanoGPT, modernised with Flash Attention and scaled init.
# ________________________________________________

import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


# ─── CONFIG ───────────────────────────────────────────────────────────────────

@dataclass
class GPTConfig:
    """
    Holds all the architectural dimensions of the model.
    These must match exactly between training and inference —
    you cannot change the architecture after training has started.

    Default values here are for GPT-2 Medium (355M parameters).
    They get overridden by config.py when the model is instantiated.
    """
    block_size: int = 1024   # maximum sequence length (context window in tokens)
    vocab_size: int = 32000  # number of unique tokens in our Malay BPE tokenizer
    n_layer:    int = 24     # number of transformer blocks (the "depth")
    n_head:     int = 16     # number of attention heads per block
    n_embd:     int = 1024   # embedding dimension (the "width" of every layer)


# ─── ATTENTION ────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    KEY CONCEPTS:
    ─────────────
    "Self-attention" means each token attends to other tokens in the SAME sequence
    (as opposed to cross-attention, where one sequence attends to a different one).

    "Causal" means each token can only attend to tokens at the SAME position or
    EARLIER — never to future tokens. This is what makes the model autoregressive:
    it predicts token N by only looking at tokens 0..N-1. At inference time, we
    generate one token at a time, left to right.

    "Multi-head" means we run H independent attention operations in parallel (H=16
    for 355M). Each head sees all tokens but focuses on different aspects — one head
    might learn syntactic patterns, another semantic relationships, etc. The outputs
    are concatenated and projected back to n_embd.

    IMPLEMENTATION NOTE:
    ────────────────────
    For efficiency, Q, K, V are computed in a single linear projection of size
    3×n_embd, then split. This is equivalent to three separate projections but
    uses one matrix multiply (faster on GPU).

    We use PyTorch's scaled_dot_product_attention (Flash Attention 2 under the hood)
    instead of the naive manual implementation. Flash Attention avoids materializing
    the full (T×T) attention matrix in VRAM — for T=1024 and B=8, that matrix would
    be 1024×1024×8×16 heads = 128M floats = 256MB per batch. Flash Attention tiles
    the computation to fit in GPU L2 cache, saving ~4× VRAM and running ~2× faster.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, \
            "Embedding dim must be divisible by num heads — each head gets n_embd // n_head dims"

        # Single projection that computes Q, K, V simultaneously.
        # Input: (B, T, n_embd) → Output: (B, T, 3 × n_embd)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Output projection: after merging all heads, project back to n_embd.
        # Marked for scaled initialization (see GPT._init_weights).
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Causal mask — lower-triangular matrix of 1s.
        # Stored as a buffer (not a trainable parameter).
        # NOTE: Flash Attention handles causality internally via is_causal=True,
        # so this mask isn't used in the forward pass. It's kept here as documentation
        # of the concept, and as a fallback if you switch to the manual implementation.
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
                  .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dim (C = n_embd)

        # ── Step 1: Project input to Q, K, V ──────────────────────────────────
        # One linear layer → split into three equal chunks along the last dim.
        qkv = self.c_attn(x)                         # (B, T, 3×C)
        q, k, v = qkv.split(self.n_embd, dim=2)      # each: (B, T, C)

        # ── Step 2: Reshape for multi-head attention ───────────────────────────
        # Split the embedding dim C into n_head heads, each of size head_size.
        # Then transpose so shape is (B, n_head, T, head_size) — this groups
        # the sequence positions next to head_size for the dot-product computation.
        head_size = C // self.n_head
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        # ── Step 3: Flash Attention ────────────────────────────────────────────
        # Computes softmax(Q×Kᵀ / √head_size) × V with is_causal=True (masks future).
        # This is a fused CUDA kernel — much faster and more memory-efficient than
        # the manual equivalent you see commented out in train_megat.py.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, n_head, T, head_size)

        # ── Step 4: Merge heads + output projection ────────────────────────────
        # Transpose back and reshape: (B, n_head, T, head_size) → (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Final linear projection
        y = self.c_proj(y)  # (B, T, C)
        return y


# ─── MLP ──────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """
    Feed-forward network (FFN) applied to each token independently after attention.

    WHY BOTH ATTENTION AND MLP?
    ───────────────────────────
    Attention handles "communication": tokens look at each other and decide what
    information to gather. But it doesn't transform that information — it just
    does weighted averaging of value vectors.

    The MLP handles "computation": it transforms the gathered information using
    two linear layers with a non-linear activation in between. This is where the
    model actually processes what it learned from attention.

    Each transformer block alternates: attend (communicate) → mlp (think).

    ARCHITECTURE DETAILS:
    ─────────────────────
    Linear(n_embd → 4×n_embd) → GELU → Linear(4×n_embd → n_embd)

    The 4× expansion comes from the original Transformer paper (Vaswani 2017)
    and has been kept in all GPT variants. It gives the MLP enough "room" to
    learn complex non-linear transformations.

    WHY GELU INSTEAD OF RELU?
    ─────────────────────────
    ReLU hard-zeroes all negative activations: f(x) = max(0, x).
    This can cause "dead neurons" — if a neuron's pre-activation is always
    negative, its gradient is always zero and it never learns.

    GELU (Gaussian Error Linear Unit) is smoother: it gently suppresses negative
    values rather than cutting them off entirely. Empirically better for LLMs.
    We use approximate='tanh' which is slightly faster with near-identical results.
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)  # expand
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)  # contract
        self.c_proj.NANOGPT_SCALE_INIT = 1  # flag for scaled initialization

    def forward(self, x):
        x = self.c_fc(x)    # (B, T, n_embd) → (B, T, 4×n_embd)
        x = self.gelu(x)    # non-linearity
        x = self.c_proj(x)  # (B, T, 4×n_embd) → (B, T, n_embd)
        return x


# ─── BLOCK ────────────────────────────────────────────────────────────────────

class Block(nn.Module):
    """
    One transformer block.

    Structure (GPT-2 "Pre-LN" variant):
        x → LayerNorm → Attention → + x  (residual)
          → LayerNorm → MLP       → + x  (residual)

    WHY RESIDUAL CONNECTIONS?
    ─────────────────────────
    x = x + sublayer(x)  — the "+" is the residual.

    Without residuals, gradients in deep networks shrink to near-zero as they
    backpropagate through many layers (vanishing gradient problem). Residuals
    give gradients a "highway" to flow directly to earlier layers, bypassing
    the non-linear sublayers. This is what makes 24-layer networks trainable.

    An intuitive way to think about it: each block adds a small "delta" to the
    residual stream rather than fully overwriting it. The residual stream carries
    a summary of everything processed so far, and each block refines it slightly.

    WHY PRE-LN (LayerNorm BEFORE the sublayer)?
    ────────────────────────────────────────────
    The original Transformer used Post-LN (normalize AFTER attention/MLP).
    GPT-2 switched to Pre-LN (normalize BEFORE) and it's been standard since.
    Pre-LN is more stable to train, especially at large scale, because it
    normalizes the input to each sublayer — preventing the attention/MLP from
    seeing wildly different magnitude inputs at different training stages.
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)  # normalize before attention
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)  # normalize before MLP
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # attention sublayer + residual
        x = x + self.mlp(self.ln_2(x))   # MLP sublayer + residual
        return x


# ─── FULL MODEL ───────────────────────────────────────────────────────────────

class GPT(nn.Module):
    """
    Full GPT-2 model.

    Components:
    ───────────
    wte  (word token embedding):    maps each token ID → n_embd vector
    wpe  (word position embedding): maps each position index → n_embd vector
    h    (transformer blocks):      stack of N Block layers
    ln_f (final LayerNorm):         normalizes the output of the last block
    lm_head (language model head):  projects n_embd → vocab_size (the logits)

    TOKEN + POSITION EMBEDDINGS:
    ─────────────────────────────
    The model needs to know both WHAT each token is and WHERE in the sequence
    it appears. We learn separate embedding tables for each and add them together.
    Both are (n_embd,) vectors, so addition gives a (n_embd,) combined embedding.

    Note: Transformers have no built-in sense of order — without position
    embeddings, "cat sat on mat" and "mat on sat cat" would look identical.
    GPT-2 uses learned absolute position embeddings (up to block_size positions).

    WEIGHT TYING:
    ─────────────
    wte.weight and lm_head.weight are THE SAME TENSOR (they share memory).
    This is standard in GPT-2 and saves vocab_size × n_embd = 32K × 1024 ≈ 32M params.

    The intuition: the embedding that "represents" a token should be similar to
    the output logit that "votes for" that token. Both encode the same concept
    (token identity in semantic space), just from different directions.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            'wte':  nn.Embedding(config.vocab_size, config.n_embd),  # token embeddings
            'wpe':  nn.Embedding(config.block_size, config.n_embd),  # position embeddings
            'h':    nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd),                     # final layer norm
        })

        # Language model head: projects from embedding space to vocabulary logits.
        # bias=False — the final LayerNorm already provides a bias-like offset.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying — share the token embedding table with the output projection.
        # This means wte.weight and lm_head.weight point to the same memory.
        self.transformer.wte.weight = self.lm_head.weight

        # Apply custom weight initialization to all submodules
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Custom weight initialization following the GPT-2 paper.

        STANDARD INIT:
        Normal(mean=0, std=0.02) for all Linear layers and Embeddings.
        std=0.02 is a conservative choice that keeps initial activations
        in a reasonable range before training begins.

        SCALED INIT FOR OUTPUT PROJECTIONS:
        The c_proj layers (output of attention and MLP) are scaled down by
        1 / sqrt(2 × n_layer). Here's why:

        Each transformer block ADDS to the residual stream via the residual
        connection: x = x + sublayer(x). With 24 blocks, this addition happens
        48 times (2 per block — once for attention, once for MLP). If each
        sublayer starts at std=0.02, the residual stream's variance after 24
        blocks would be ~24× larger than at initialization.

        Scaling down the output projections by 1/sqrt(2×n_layer) = 1/sqrt(48)
        keeps the initial residual stream variance stable regardless of depth.
        This prevents the loss from starting at an unusually high value and
        makes the first few training steps more stable.
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # This flag is set on attn.c_proj and mlp.c_proj — the layers
                # that write to the residual stream.
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, weight_decay, learning_rate, device):
        """
        Set up AdamW optimizer with separate weight decay groups.

        WHY SEPARATE WEIGHT DECAY GROUPS?
        ───────────────────────────────────
        Weight decay is L2 regularization: it adds a penalty proportional to
        the squared magnitude of each weight, effectively pulling weights toward
        zero. This prevents individual weights from growing too large and
        memorizing training examples.

        BUT — not all parameters should be decayed:

        Weight matrices (2D tensors, e.g., attention projections, MLP layers,
        embeddings): YES — these are the "heavy" parameters that encode learned
        knowledge. Regularizing them prevents overfitting.

        Biases and LayerNorm parameters (1D tensors): NO — these are small
        scale/shift parameters. Regularizing them hurts performance because they
        have too few degrees of freedom to overfit.

        We separate them by tensor dimensionality: dim >= 2 → decay, dim < 2 → no decay.

        FUSED ADAMW:
        ─────────────
        PyTorch has a CUDA-fused version of AdamW that performs the full parameter
        update in a single GPU kernel launch instead of many small operations.
        About 10% speed improvement on CUDA. Only available on CUDA, not MPS.
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        decay_params    = [p for _, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for _, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params,    "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        n_decay    = sum(p.numel() for p in decay_params)
        n_no_decay = sum(p.numel() for p in no_decay_params)
        print(f"  Decayed parameter tensors:     {len(decay_params):3d}  →  {n_decay:,} params")
        print(f"  Non-decayed parameter tensors: {len(no_decay_params):3d}  →  {n_no_decay:,} params")

        use_fused = ('fused' in inspect.signature(torch.optim.AdamW).parameters) and (device == 'cuda')
        print(f"  Fused AdamW: {use_fused}")

        return torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused,
        )

    def forward(self, idx, targets=None):
        """
        Forward pass through the model.

        idx:     (B, T) — input token IDs
        targets: (B, T) — target token IDs, shifted one position right from idx
                          (only needed during training to compute loss)

        HOW NEXT-TOKEN PREDICTION WORKS:
        ─────────────────────────────────
        The model sees a sequence of T tokens and simultaneously predicts the
        next token at every single position. For a sequence [t0, t1, t2, t3]:
            - At position 0, predict t1
            - At position 1, predict t2
            - At position 2, predict t3
            - At position 3, predict t4 (next token to generate)

        So targets is just idx shifted one step right. The loss is computed as
        the average cross-entropy across all B×T positions — this is standard
        "teacher forcing" for autoregressive training.

        During inference (targets=None), we only care about the logits at the
        LAST position (the model's prediction for the next token).
        """
        _, T = idx.size()
        assert T <= self.config.block_size, \
            f"Sequence length {T} exceeds block_size {self.config.block_size}"

        # Position indices: [0, 1, 2, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # Token embeddings + position embeddings, added together
        # wte: (B, T) → (B, T, n_embd)
        # wpe: (T,)   → (T, n_embd), broadcast over batch dim
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        x = tok_emb + pos_emb               # (B, T, n_embd)

        # Pass through all transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final LayerNorm
        x = self.transformer.ln_f(x)  # (B, T, n_embd)

        # Project to vocabulary: logits[b, t, v] = score for token v being
        # next after position t in batch b
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten (B, T, vocab_size) → (B*T, vocab_size) and (B, T) → (B*T,)
            # then compute cross-entropy loss over all positions at once.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_p=0.95):
        """
        Autoregressive text generation.

        At each step:
        1. Forward pass → get logits for the next token
        2. Apply temperature scaling
        3. Apply top-p (nucleus) sampling
        4. Sample one token from the filtered distribution
        5. Append it to idx and repeat

        TEMPERATURE:
        ─────────────
        We divide logits by temperature before softmax.
        - temperature = 1.0 → original distribution
        - temperature < 1.0 → sharper distribution (more confident, less creative)
        - temperature > 1.0 → flatter distribution (more random, more creative)
        Good starting point for fiction: 0.75–0.90

        TOP-P (NUCLEUS SAMPLING):
        ──────────────────────────
        Instead of sampling from the full 32K-token vocabulary, only consider
        the smallest set of tokens whose cumulative probability exceeds top_p.
        This cuts off the long tail of very unlikely tokens that would produce
        nonsensical text, while keeping enough diversity for creative writing.
        top_p=0.95 means: use the top tokens that together account for 95% of
        the probability mass.

        For longer stories: generate in chunks, feeding the last 768 tokens as
        context for each new chunk (leave 256 tokens of headroom in the 1024
        context window for new content).
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Trim context to block_size if the sequence has grown too long
            idx_cond = idx if idx.size(1) <= self.config.block_size \
                           else idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # only care about last position: (B, vocab_size)

            # Temperature scaling
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)  # (B, vocab_size)

            # Top-p (nucleus) sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Mask tokens beyond the nucleus (where cumulative prob > top_p)
            # Shift right by 1 so the token that PUSHES cumprob over top_p is still included
            remove_mask = (cumulative_probs - sorted_probs) > top_p
            sorted_probs[remove_mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)  # renormalize

            # Sample from the filtered distribution, then unsort back to original vocab indices
            sampled = torch.multinomial(sorted_probs, num_samples=1)  # (B, 1)
            next_token = sorted_indices.gather(dim=-1, index=sampled)  # (B, 1)

            idx = torch.cat([idx, next_token], dim=1)  # append to sequence

        self.train()
        return idx
