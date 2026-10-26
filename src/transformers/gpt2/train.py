from dataclasses import asdict
import torch
import tiktoken
from src.transformers.gpt2.model import GPT, gpt2_config

# ---- config ----
cfg = gpt2_config(block_size=256)     # smaller context = much faster on a small machine
device = cfg.device_type
max_iters = 50000
eval_interval = 200
eval_iters = 20

# ---- 1. data: text -> token ids ----
enc = tiktoken.get_encoding("gpt2")
text = open("data/input.txt", encoding="utf-8").read()
data = torch.tensor(enc.encode_ordinary(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - cfg.block_size - 1, (cfg.batch_size,))
    x = torch.stack([d[i : i + cfg.block_size] for i in ix])          # inputs
    y = torch.stack([d[i + 1 : i + cfg.block_size + 1] for i in ix])  # targets: shifted by one
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def save_checkpoint(model, path="ckpt.pt"):
    saved = asdict(cfg)
    saved.pop("device_type")          # machine-specific, re-detected on load
    torch.save({"model": model.state_dict(), "config": saved}, path)

# ---- 2. model + optimizer ----
model = GPT(cfg).to(device)
print(f"parameters: {model.get_num_params():,}")
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

# ---- 3. training loop ----
best_val = float("inf")
for it in range(max_iters + 1):
    if it % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {it}: train {losses['train']:.4f}, val {losses['val']:.4f}")
        if losses["val"] < best_val:          # checkpoint whenever validation improves
            best_val = losses["val"]
            save_checkpoint(model)

    if it == max_iters:
        break

    x, y = get_batch("train")
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"done. best val loss {best_val:.4f}, saved to ckpt.pt")

# ---- 4. sample from the best checkpoint ----
model = GPT.from_checkpoint("ckpt.pt")
print(model.generate_text("ROMEO:", max_new_tokens=150))