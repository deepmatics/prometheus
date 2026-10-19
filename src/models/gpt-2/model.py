from dataclasses import dataclass
import math
from typing import Any #Configuration settings
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(2026)

@dataclass
class config:
    batch_size: int = 16
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 4
    n_embd: int = 64
    n_head: int = 4
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    device_type: str = 'cuda' if torch.cuda.is_available() else 'cpu'

cfg = config()

with open('./data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

""" Character Level Tokenization """
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    # Fixed scoping: pulled block_size, batch_size, and device from cfg
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - cfg.block_size, (cfg.batch_size,))
    x = torch.stack([data_split[i:i+cfg.block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+cfg.block_size+1] for i in ix])
    x, y = x.to(cfg.device_type), y.to(cfg.device_type)
    return x, y

class SelfAttention(nn.Module):
    '''
    Implementation of Self-Attention using PyTorch
    '''
    tril: torch.Tensor

    def __init__(self, config):
        super().__init__() # Inherit and boot up the base class properties
        assert config.n_embd % config.n_head == 0

        # --- ARCHITECTURE NOTES: PROJECTIONS ---
        # 1. c_attn (3 * n_embd): Fused projection creating Q, K, and V vectors for 
        #    all attention heads in a single GPU operation (split apart during forward).
        # 2. c_proj (n_embd): Linear layer that mixes and synthesizes perspectives 
        #    across all independent attention heads after concatenation (concatenation != communication).
        self.c_attn = nn.Linear(in_features = config.n_embd, out_features = 3 * config.n_embd, bias = config.bias) # Q, K, V for all heads
        self.c_proj = nn.Linear(in_features = config.n_embd, out_features = 3 * config.n_embd, bias = config.bias) # Output Projections

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape()  # batch size, sequence length, embedding dimensionality (n_embd)

        # --- ARCHITECTURE NOTES: MULTI-HEAD RESHAPING ---
        # 1. split: Divides the fused (3 * n_embd) tensor back into independent Q, K, and V streams.
        # 2. view: Reshapes the flat embedding dimension (C) into separate heads: (B, T, n_head, head_size).
        # 3. transpose(1, 2): Swaps dimensions to (B, n_head, T, head_size) so the GPU can 
        #    process all attention heads in parallel as independent mini-batches.
        # Fused QKV projection: c_attn outputs [B, T, 3 * n_embd], 
        # which is split along dim=2 into three separate [B, T, n_embd] tensors for Q, K, and V.
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1 ,2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1 ,2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1 ,2) # (B, nh, T, hs)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask = None, dropout_p= config.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.resid_dropout(self.c_proj(y))

        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()

        # --- ARCHITECTURE NOTES: MLP (FEED-FORWARD NETWORK) ---
        # 1. c_fc (Expansion): Expands the embedding dimension by 4x (4 * n_embd) to give 
        #    the model a wide scratchpad space for complex non-linear reasoning.
        # 2. Why 4x?: An empirical sweet spot from "Attention Is All You Need" that 
        #    balances model capacity against parameter explosion and diminishing returns.
        # 3. c_proj (Contraction): Shrinks features back down to n_embd so it can 
        #    be safely added back to the main residual stream.
        self.c_fc = nn.Linear(in_features= config.n_embd, out_features= 4 * config.n_embd, bias = config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(in_features= 4 * config.n_embd, out_features= config.n_embd, bias = config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.ln1 = LayerNorm(ndim= config.n_embd, bias= config.bias)
        self.attn = SelfAttention(config)
        self.ln2 = LayerNorm(ndim= config.n_emdb, bias= config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.block_size is not None
        assert config.vocab_size is not None

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed), 
            wpe = nn.Embedding(config.block_size, config.n_embed),
            dropout = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]), 
            lnf = LayerNorm(ndim= config.n_embed, bias= config.bias)
        ))

        self.lm_head = nn.Linear(in_features= config.n_embd, out_features= config.vocab_size, bias= False)
        self.transformer.wte.weight = self.lm_head.weight # type: ignore https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights) # type: ignore
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets= None):
        device = idx.device
        b, t = idx.size()
        assert t <= config.block_size, f"Cannot forward sequence of length {t}, block size is only {config.block_size}"
        pos = torch.arange(0, t, dtype= torch.long, device=device)

        tok_emb = self.transformer.wte(idx) # type: ignore
        pos_emb = self.transformer.wpe(pos) # type: ignore
        x = self.transformer.dropout(tok_emb + pos_emb) # type: ignore

        for block in self.transformer.h: # type: ignore
            x = block(x)
        x = self.transformer.lnf(x) # type: ignore

        


