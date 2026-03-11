"""
Implementation of GPT-2 in Pytorch, inspired by Karpathy's tutorial.
Plan to add more features and modern LLM techniques in the implementation.
Plan to train on my own tokenizer, and see whether there's a difference 
in training dynamics and final performance compared to using the gpt2 tokenizer.

"""


from dataclasses import dataclass
from logging import config
import math
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
    

    def forward(self, x): #read this properly
        B, T, C = x.size()
        qkv = self.c_attn(x) # (B, T, 3 * C)
        q, k, v = qkv.split(self.n_embd, dim=2) # (B, T, C) * 3

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_size)

        #att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) 
        #att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        #att = F.softmax(att, dim=-1)
        #y = att @ v # (B, n_head, T, head_size)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # (B, n_head, T, head_size)


        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)
        y = self.c_proj(y) # (B, T, C)
        return y



class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.gelu = nn.GELU(approximate='tanh')
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x) #prevent dead neurons that might comes from relu
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
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        #weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        #imit params
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Load a pretrained GPT-2  model weights from huggingface
        """
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}, "model_type must be one of 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'"
        from transformers import GPT2LMHeadModel
        print("Loading weights from pretrained gpt: %s",model_type)

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
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('attn.bias') or 'c_attn' in k]
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
        
        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        #start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        #create optimizer groups. Any parameters that is 2D will be weight decayed, otherwise not.
        # i.e All weight tensor in matmuls + embeddings decay, all biases and layer norms don't

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)
        print(f"num decay params tensors: {len(decay_params)}, num no decay params tensors: {len(no_decay_params)}")
        print(f"num decay params: {num_decay_params}, num no decay params: {num_no_decay_params}")

        #create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == 'cuda'
        print(f"Using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits,loss

import time
import tiktoken    

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f" loaded {len(self.tokens)} tokens from input.txt")
        print(f" 1 epoch = {len(self.tokens) // (B*T)} batches")

        #state
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B*T + 1]
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T)

        self.current_position += B*T
        if self.current_position + (B*T + 1) >= len(self.tokens):
            self.current_position = 0
        
        return x, y
    
device = "cpu"

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"Using {device} device")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

#get a data batch
train_loader = DataLoaderLite(B=4, T=512)

"""
If we have issues withh high batch size, but no GPU due to miskin,
we can use gradient accumulation to simulate larger batch size.
This is done by accumulating gradients over multiple forward passes before performing an optimizer step.

why batch size must match? Because all the learning rate and hyperparameters tie in to that
batch size. 

"""
total_batch_size = 4096 # ~16K tokens per step, suitable for small dataset + M1
B = 4 #micro batch size (keep small for M1 8GB)
T= 512 #sequence length

assert total_batch_size % (B * T) == 0, "total_batch_size must be divisible by B*T"
grad_accum_steps = total_batch_size // (B * T)
print(f"Total desired batch sizes: {total_batch_size}")
print(f"Calculated grad_accum_steps: {grad_accum_steps}")


model = GPT(GPTConfig())
model.to(device)
# model = torch.compile(model, mode = "default") # disabled: saves memory on M1 8GB
# logits, loss = model(x,y)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


"""
Hyperparameter optimization for this model is taken from GPT3 paper, refer to 
Details of Model Training. 

Gradient norm clipping is used to prevent exploding gradients, which can occur when training large models.


"""


# optimizer = torch.optim.AdamW(model.parameters(), lr = max_lr, betas = (0.9, 0.95), eps= 1e-8) #old one
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device = device)
for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize() 
    elif device == "mps":
        torch.mps.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000 # ms
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps) / (t1-t0)
    print(f"Step {step:4d} | Loss: {loss_accum.item():.4f} | dt: {dt:.2f}ms | lr: {lr:4e} | norm: {norm:.2f} | tok/sec: {tokens_per_sec:.2f}")


import sys; sys.exit(0)
    