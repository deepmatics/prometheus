from dataclasses import dataclass
import math
from typing import Any #Configuration settings
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(2026)

@dataclass
class gpt2_config:
    batch_size: int = 16
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 4
    n_embd: int = 64
    n_head: int = 4
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    device_type: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    learning_rate = 3e-4

cfg = gpt2_config()

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
        
        # --- ARCHITECTURE NOTES: PROJECTIONS ---
        # 1. c_attn (3 * n_embd): Fused projection creating Q, K, and V vectors for 
        #    all attention heads in a single GPU operation (split apart during forward).
        # 2. c_proj (n_embd): Linear layer that mixes and synthesizes perspectives 
        #    across all independent attention heads after concatenation (concatenation != communication).
        self.c_attn = nn.Linear(in_features = config.n_embd, out_features = 3 * config.n_embd, bias = config.bias) # Q, K, V for all heads
        self.c_proj = nn.Linear(in_features = config.n_embd, out_features = config.n_embd, bias = config.bias) # Output Projections

        # Dropout residual connection
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape  # batch size, sequence length, embedding dimensionality (n_embd)

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

        y = F.scaled_dot_product_attention(q, k, v, attn_mask = None, dropout_p= cfg.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.resid_dropout(self.c_proj(y))

        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()

        # --- ARCHITECTURE NOTES: MLP (FEED-FORWARD NETWORK) ---
        # 1. c_fc: Expands the embedding dimension by 4x (4 * n_embd) to give the model a wide scratchpad space for complex non-linear reasoning.
        # 2. Why 4x?: An empirical sweet spot from "Attention Is All You Need" that balances model capacity against parameter explosion and diminishing returns.
        # 3. c_proj : Shrinks features back down to n_embd so it can be safely added back to the main residual stream.
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
        self.ln2 = LayerNorm(ndim= config.n_embd, bias= config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # tokens look at each other 
        x = x + self.mlp(self.ln2(x))   # each token thinks on its own
        return x

class GPT(nn.Module):
 
    def __init__(self, config: gpt2_config = cfg):
        super().__init__()
        self.config = config
 
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)   # token id
        self.wpe = nn.Embedding(config.block_size, config.n_embd)   # position
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(ndim=config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
 
        # weight tying: input embedding and output layer share one matrix
        self.lm_head.weight = self.wte.weight
 
        # init all weights, then apply GPT-2's scaled init to residual projections
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
 
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
 
    def get_num_params(self, non_embedding=True):
        """Number of parameters. Position embeddings are subtracted by default."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wpe.weight.numel()
        return n_params
 
    def forward(self, idx, targets=None):
        # idx: batch of token IDs, shape (B, T) = (batch size, sequence length).
        # The model can't read text, so the tokenizer turns it into integers
        # (e.g. "Hello world" -> [15496, 995]); wte then looks up a vector for each ID.
        # targets is the same as idx shifted one token right (what to predict at each position).

        B, T = idx.shape
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
 
        x = self.dropout(self.wte(idx) + self.wpe(pos))     # (B, T, C)
        for block in self.blocks:
            x = block(x)                                    # (B, T, C) vectors after all the blocks
        x = self.ln_f(x)                                    # (B, T, vocab_size)  one score per possible next token

        logits = self.lm_head(x)                            # (B, T, vocab_size) one score per possible next token

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss
 
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """idx: LongTensor (B, T). Returns (B, T + max_new_tokens). Call model.eval() first."""
        block_size = self.config.block_size
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
 
    @torch.no_grad()
    def generate_text(self, prompt, max_new_tokens=200, temperature=0.8, top_k=50):
        """Text in, text out."""
        enc = tiktoken.get_encoding("gpt2")
        device = self.config.device_type
        idx = torch.tensor([enc.encode_ordinary(prompt)], dtype=torch.long, device=device)
        out = self.generate(idx, max_new_tokens, temperature, top_k)
        return enc.decode(out[0].tolist())
 
    @classmethod
    def from_checkpoint(cls, path):
        """Rebuild a trained model from a checkpoint saved by train.py."""
        ckpt = torch.load(path, map_location="cpu")
        saved = dict(ckpt["config"])
        saved.pop("device_type", None)            # machine-specific: auto-detected by the config default
        config = gpt2_config(**saved)
        model = cls(config)
        model.load_state_dict(ckpt["model"])
        return model.to(config.device_type).eval()