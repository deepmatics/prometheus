"""
data_prep.py: turn any Hugging Face text dataset into flat token files for pre-training.
 
Output (in --out-dir):
    train.bin   all training tokens as one long array of integers
    val.bin     held-out tokens for measuring validation loss
    meta.json   what was used to build the files (tokenizer, dtype, token counts)
 
"""

import json
import os
from typing import List 
import numpy as np
import tiktoken
from datasets import load_dataset
from dataclasses import dataclass

@dataclass(slots=True)
class PrepConfig:
    dataset: str = "roneneldan/TinyStories"
    config: str | None = None
    split: str = "train"
    text_column: str = "text"
    out_dir: str = "data/output/tinystories"
    max_tokens: int = 5_000_000      # start small to test, then scale up
    val_tokens: int = 500_000
    tokenizer: str = "gpt2"
    shuffle: bool = False
    no_eot: bool = False
    batch_docs: int = 1000
    threads: int = 8

args = PrepConfig()
 
 
def main():

    # tokenizer

    enc = tiktoken.get_encoding(args.tokenizer)
    # Store tokens in the smallest integer type that fits the vocabulary:
    # GPT-2's 50257 IDs fit in uint16 (2 bytes/token) instead of int64 (8 bytes/token).
    dtype = np.uint16 if enc.max_token_value < 2**16 else np.uint32
    eot = enc.eot_token  # "end of text" ID (50256 for GPT-2): marks where one document ends
    # print(enc.n_vocab, dtype, eot)
 
    # dataset streamed
    # streaming=True reads examples on the fly instead of downloading the whole dataset first.
    # Combined with --max-tokens, only ever needed is fetched.
    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    if args.shuffle:
        # approximate shuffle: documents are mixed inside a rolling buffer of 10k examples
        ds = ds.shuffle(seed=2026, buffer_size=10_000)
 
    def tokenize(texts):
        """list[str] -> one flat numpy array of token IDs, with an EOT after each document."""
        # encode_ordinary ignores special-token strings that may appear inside the text
        # (so a document containing "<|endoftext|>" can't inject a fake separator).
        id_lists = enc.encode_ordinary_batch(texts, num_threads=args.threads)
        out = []
        for ids in id_lists:
            out.extend(ids)
            if not args.no_eot:
                out.append(eot)
        return np.array(out, dtype=dtype)
 
    def token_chunks():
        """Yield arrays of tokens, args.batch_docs documents at a time."""
        texts = []
        for example in ds:
            text = example[args.text_column]
            if text:  # skip empty rows
                texts.append(text)
            if len(texts) == args.batch_docs:
                yield tokenize(texts)
                texts = []
        if texts:
            yield tokenize(texts)
 
    # stream -> tokenize -> append to files
    os.makedirs(args.out_dir, exist_ok=True)
    n_val = n_train = 0
    with open(os.path.join(args.out_dir, "val.bin"), "wb") as val_f, \
         open(os.path.join(args.out_dir, "train.bin"), "wb") as train_f:
 
        for chunk in token_chunks():
            # The first --val-tokens tokens of the stream become the validation set.
            # They are never written to train.bin, so train and val cannot overlap.
            if n_val < args.val_tokens:
                take = min(len(chunk), args.val_tokens - n_val)
                val_f.write(chunk[:take].tobytes())
                n_val += take
                chunk = chunk[take:]
 
            # Whatever remains goes to the training file, up to the --max-tokens budget.
            room = args.max_tokens - n_train
            if len(chunk) > 0 and room > 0:
                train_f.write(chunk[:room].tobytes())
                n_train += min(len(chunk), room)
 
            print(f"\rval: {n_val:,}  train: {n_train:,} / {args.max_tokens:,}", end="", flush=True)
            if n_train >= args.max_tokens:
                break
    print()
 
    # record how the files were made
    # The training script reads this to know how to decode train.bin.
    meta = dict(
        dataset=args.dataset, config=args.config, split=args.split,
        tokenizer=args.tokenizer, vocab_size=enc.n_vocab, dtype=np.dtype(dtype).name,
        train_tokens=n_train, val_tokens=n_val, eot_appended=not args.no_eot,
    )
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"done. wrote {n_train:,} train and {n_val:,} val tokens to {args.out_dir}")
 
 
if __name__ == "__main__":
    main()
     