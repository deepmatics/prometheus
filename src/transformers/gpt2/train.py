import contextlib
import json
import math
import os
import time
from dataclasses import asdict

import numpy as np
import torch
import gc

from src.transformers.gpt2.model import GPT, gpt2_config

# ---- config ----
cfg = gpt2_config()
device = cfg.device_type
max_iters = 5
eval_interval = 1000
eval_iters = 250
log_interval = 1000
grad_accum_steps = 16                 # raise (e.g. 4) for a bigger effective batch without more VRAM
grad_clip = 1.0                       # cap on gradient norm; protects against loss spikes
warmup_iters = 715                    # LR ramps up from ~0 over these steps
min_lr = cfg.learning_rate / 10       # LR decays down to this by the end
resume = True                         # continue from last.pt if it exists
optimizer_beta1 = 0.9                 # GPT-3 Paper
optimizer_beta2 = 0.95                # GPT-3 Paper
weight_decay = 0.1                    # GPT-3 Paper; applied to 2D+ params only
use_compile = True                    # set False if Triton isn't working
log_interval = 100                    # must differ from eval_interval to see tokens/s

best_path = "/assets/model-checkpoints/ckpt.pt"                 # best validation loss; used for generation
last_path = "/assets/model-checkpoints/last.pt"                 # latest state incl. optimizer; used to resume
os.makedirs(os.path.dirname(last_path), exist_ok=True)

# ---- data: token files made by prepare_data.py ----
data_dir = "data/output/tinystories"         # folder containing train.bin, val.bin, meta.json

meta = json.load(open(f"{data_dir}/meta.json"))
assert meta["vocab_size"] <= cfg.vocab_size, "tokenizer vocab is larger than the model's vocab_size"
np_dtype = np.dtype(meta["dtype"])    # uint16 for GPT-2 tokens

tokens_per_iter = cfg.batch_size * cfg.block_size * grad_accum_steps
print(f"tokens per step: {tokens_per_iter:,}")
print(f"planned: {max_iters * tokens_per_iter:,} tokens "
      f"= {max_iters * tokens_per_iter / meta['train_tokens']:.1f} epochs of the data")

def get_batch(split):
    # memmap reads only the slices we index, not the whole file. Re-creating it each call
    # avoids a slow memory build-up over very long runs.
    data = np.memmap(f"{data_dir}/{split}.bin", dtype=np_dtype, mode="r")
    ix = torch.randint(len(data) - cfg.block_size - 1, (cfg.batch_size,))
    # astype(int64) because the embedding layer needs "long" indices, not uint16
    x = torch.stack([torch.from_numpy(data[i : i + cfg.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + cfg.block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

# ---- mixed precision: bf16 halves activation memory on GPUs that support it (RTX 30-series does) ----
use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
print(f"bf16 autocast: {use_amp}")

# ---- learning-rate schedule: linear warmup, then cosine decay to min_lr ----
def get_lr(it):
    if it < warmup_iters:
        return cfg.learning_rate * (it + 1) / warmup_iters
    if it >= max_iters:
        return min_lr
    ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1 + math.cos(math.pi * ratio))      # goes 1 -> 0
    return min_lr + coeff * (cfg.learning_rate - min_lr)

# ---- loss estimate on train and val ----
@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            with amp_ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def save_checkpoint(model, optimizer, it, best_val, is_best):
    saved = asdict(cfg)
    saved.pop("device_type")          # machine-specific, re-detected on load
    state = {"model": model.state_dict(), "config": saved}
    if is_best:
        torch.save(state, best_path)  # model + config only: what generation needs
    state.update(optimizer=optimizer.state_dict(), iter=it, best_val=best_val)
    torch.save(state, last_path)      # everything needed to resume training

# Speed-up: let float32 matmuls run on GPU tensor cores using TF32
# (rounds inputs to ~10 mantissa bits, accumulates in full fp32).
# Tensors stay float32; only the multiply step is less precise.
# Safe for deep learning training; ~1.5-3x faster on Ampere+ GPUs (A100/H100/RTX 30+).
# Options: 'highest' (default, full fp32) | 'high' (TF32) | 'medium' (bf16, fastest).
torch.set_float32_matmul_precision('high')

# ---- model + optimizer ----
model = GPT(cfg).to(device)
print(f"parameters: {model.get_num_params():,}")

def configure_optimizers(model, weight_decay, learning_rate, betas, device):
    # start with all of the candidate parameters (that require grad)
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    # 2D+ tensors (matmul weights, embeddings) get weight decay; 1D (biases, LayerNorm) don't
    decay_params   = [p for p in param_dict.values() if p.dim() >= 2]
    nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
    groups = [
        {"params": decay_params,   "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    print(f"decay tensors: {len(decay_params)}, no-decay tensors: {len(nodecay_params)}")
    # fused AdamW does the update in one GPU kernel (faster)
    return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, fused=(device == "cuda"))

optimizer = configure_optimizers(model, weight_decay, cfg.learning_rate, (optimizer_beta1, optimizer_beta2), device)
# Compile the model into optimized fused GPU kernels (less Python overhead and memory traffic)
cmodel = torch.compile(model) if use_compile else model

# ---- resume ----
start_it, best_val = 0, float("inf")
if resume and os.path.exists(last_path):
    ck = torch.load(last_path, map_location=device)
    # Load the saved training state onto the current device.
    # torch.compile() wraps a model and prefixes every state_dict key with "_orig_mod."
    # (e.g. "_orig_mod.wte.weight"), but our raw, uncompiled `model` expects plain keys
    # ("wte.weight"). Checkpoints saved from a compiled model therefore fail to load with
    # missing/unexpected key errors. Here we strip that prefix from every key before loading.
    # removeprefix() is a no-op for keys that don't have it, so clean checkpoints still work.
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()})
    optimizer.load_state_dict(ck["optimizer"])
    start_it, best_val = ck["iter"], ck["best_val"]
    print(f"resumed from step {start_it}, best val {best_val:.4f}")

# ---- training loop ----
t_last = time.time()
for it in range(start_it, max_iters + 1):
    lr = get_lr(it)
    for group in optimizer.param_groups:
        group["lr"] = lr

    if it % eval_interval == 0:
        losses = estimate_loss(cmodel)
        print(f"step {it}: train {losses['train']:.4f}, val {losses['val']:.4f}, lr {lr:.2e}")
        is_best = losses["val"] < best_val
        if is_best:
            best_val = losses["val"]
        save_checkpoint(model, optimizer, it, best_val, is_best)
        t_last = time.time()          # don't count eval time in the speed numbers

    if it == max_iters:
        break

    # gradient accumulation: several small batches add up to one optimizer step
    for _ in range(grad_accum_steps):
        x, y = get_batch("train")
        with amp_ctx:
            _, loss = cmodel(x, y)
        (loss / grad_accum_steps).backward()   # divide so the summed gradient is an average and prevent gradient explosion

    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if it % log_interval == 0 and it % eval_interval != 0:
        dt = time.time() - t_last
        t_last = time.time()
        print(f"step {it}: loss {loss.item():.4f}, {tokens_per_iter * log_interval / dt:,.0f} tokens/s")

print(f"done. best val loss {best_val:.4f}, saved to {best_path}")

# ---- sample from the best checkpoint ----
model = GPT.from_checkpoint(best_path)
print(model.generate_text("What is the colour of the sky?", max_new_tokens=10))

del model, cmodel, optimizer
gc.collect()
torch.cuda.empty_cache()