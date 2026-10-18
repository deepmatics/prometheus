from dataclasses import dataclass #Configuration settings
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(2026)

@dataclass
class config:
    batch_size: int = 16
    block_size: int = 1024
    n_layer: int = 4
    n_embd: int = 64
    n_head: int = 4
    dropout: float = 0.0
    bias: bool = True
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

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.bias = config.bias

        # --- ARCHITECTURE NOTES: PROJECTIONS ---
        # 1. c_attn (3 * n_embd): Fused projection creating Q, K, and V vectors for 
        #    all attention heads in a single GPU operation (split apart during forward).
        # 2. c_proj (n_embd): Linear layer that mixes and synthesizes perspectives 
        #    across all independent attention heads after concatenation (concatenation != communication).
        self.c_attn = nn.Linear(in_features = self.n_embd, out_features = 3 * self.n_embd, bias = self.bias) # Q, K, V for all heads
        self.c_proj = nn.Linear(in_features = self.n_embd, out_features = 3 * self.n_embd, bias = self.bias) # Output Projections

        # Dropout
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

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

        y = F.scaled_dot_product_attention(q, k, v, attn_mask = None, dropout_p= self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.resid_dropout(self.c_proj(y))

        return y

    