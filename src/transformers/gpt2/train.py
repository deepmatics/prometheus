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
cfg = gpt2_config(block_size=256)     # smaller context = much faster on a small machine
device = cfg.device_type
max_iters = 50000
eval_interval = 200
eval_iters = 200
log_interval = 100
grad_accum_steps = 4                  # raise (e.g. 4) for a bigger effective batch without more VRAM
grad_clip = 1.0                       # cap on gradient norm; protects against loss spikes
warmup_iters = 500                    # LR ramps up from ~0 over these steps
min_lr = cfg.learning_rate / 10       # LR decays down to this by the end
resume = True                         # continue from last.pt if it exists

best_path = "ckpt.pt"                 # best validation loss; used for generation
last_path = "last.pt"                 # latest state incl. optimizer; used to resume

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

# ---- model + optimizer ----
model = GPT(cfg).to(device)
print(f"parameters: {model.get_num_params():,}")
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.95))

# ---- resume ----
start_it, best_val = 0, float("inf")
if resume and os.path.exists(last_path):
    ck = torch.load(last_path, map_location=device)
    model.load_state_dict(ck["model"])
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
        losses = estimate_loss(model)
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
            _, loss = model(x, y)
        (loss / grad_accum_steps).backward()   # divide so the summed gradient is an average

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
print(model.generate_text("What is the colour of the sky?", max_new_tokens=5000))

del model, optimizer
gc.collect()
torch.cuda.empty_cache()