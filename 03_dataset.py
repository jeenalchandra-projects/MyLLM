"""
STEP 3: Dataset — Preparing Batches for Training
=================================================
Before we can train the model, we need to slice our giant text file into
small, manageable pieces called BATCHES.

KEY CONCEPT: Context Window (Block Size)
The model reads a fixed-length window of characters at a time.
  - BLOCK_SIZE = 128 means the model sees 128 characters at once
  - This is the model's "working memory" or "attention span"
  - GPT-4's context window is ~128,000 tokens — our's is tiny, but the idea is the same

KEY CONCEPT: Input / Target pairs
For EVERY position in the text, the model's job is to predict the NEXT character.
Given the sequence:  F  o  r  d     M  u  s  t  a  n  g
                     0  1  2  3  4  5  6  7  8  9  10 11

The model sees position 0 (F) and must predict position 1 (o).
It sees positions 0-1 (Fo) and must predict position 2 (r).
...and so on.

So:
  input  = [F, o, r, d, ' ', M, u, s, t, a, n]
  target = [o, r, d, ' ', M, u, s, t, a, n, g]

The target is just the input shifted by one position.

KEY CONCEPT: Batch
Processing one example at a time is slow. Instead we process BATCH_SIZE
examples simultaneously. This is like grading 32 student papers at once
vs. one at a time — same total work, but 32x more efficient.

Run: python 03_dataset.py
"""

import torch
import json
import os

# ── Hyperparameters ──────────────────────────────────────────────────
# "Hyperparameters" are settings we choose before training begins.
# Unlike model weights (which are learned), we set these manually.

BLOCK_SIZE = 128    # Context window: how many characters the model sees at once
BATCH_SIZE = 32     # How many independent sequences we process in parallel
TRAIN_SPLIT = 0.9   # 90% of data for training, 10% for validation

DATA_FILE = "data/vehicles.txt"
VOCAB_FILE = "data/vocab.json"


def load_data():
    """Load and encode the full training text into a tensor of integers."""
    # Load raw text
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    # Load vocabulary (built in step 2)
    with open(VOCAB_FILE, "r") as f:
        char_to_int = json.load(f)

    # Encode entire text to a list of integers
    data = [char_to_int[c] for c in text if c in char_to_int]

    # Convert to a PyTorch tensor
    # KEY CONCEPT: Tensor
    # A tensor is PyTorch's fundamental data structure — essentially a
    # multi-dimensional array (like a numpy array) that can live on GPU
    # and supports automatic differentiation.
    data_tensor = torch.tensor(data, dtype=torch.long)

    return data_tensor, len(char_to_int)


def get_splits(data_tensor):
    """Split data into training and validation sets."""
    n = int(TRAIN_SPLIT * len(data_tensor))
    train_data = data_tensor[:n]
    val_data = data_tensor[n:]
    return train_data, val_data


def get_batch(split_data, device="cpu"):
    """
    Randomly sample a batch of (input, target) pairs.

    Returns two tensors of shape [BATCH_SIZE, BLOCK_SIZE]:
      - x (inputs):  batch of input sequences
      - y (targets): same sequences shifted by 1 (what comes next)
    """
    # Pick BATCH_SIZE random starting positions
    # Each starting position must leave room for BLOCK_SIZE characters
    ix = torch.randint(len(split_data) - BLOCK_SIZE, (BATCH_SIZE,))

    # Stack slices into tensors
    x = torch.stack([split_data[i     : i + BLOCK_SIZE    ] for i in ix])
    y = torch.stack([split_data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])

    return x.to(device), y.to(device)


if __name__ == "__main__":
    print("Loading and preparing data...")
    data, vocab_size = load_data()
    train_data, val_data = get_splits(data)

    print(f"\nDataset stats:")
    print(f"  Total tokens (characters): {len(data):,}")
    print(f"  Training tokens:           {len(train_data):,}")
    print(f"  Validation tokens:         {len(val_data):,}")
    print(f"  Vocabulary size:           {vocab_size}")

    # ----- VERIFICATION -----
    print("\n--- VERIFICATION ---")
    x, y = get_batch(train_data)
    print(f"  Input batch shape:  {x.shape}   (expected: [{BATCH_SIZE}, {BLOCK_SIZE}])")
    print(f"  Target batch shape: {y.shape}  (expected: [{BATCH_SIZE}, {BLOCK_SIZE}])")

    # Show the first sequence decoded
    with open(VOCAB_FILE, "r") as f:
        char_to_int = json.load(f)
    int_to_char = {int(v): k for k, v in char_to_int.items()}

    sample_input = "".join(int_to_char[i.item()] for i in x[0])
    sample_target = "".join(int_to_char[i.item()] for i in y[0])
    print(f"\n  Sample input  (seq 0): {repr(sample_input[:60])}...")
    print(f"  Sample target (seq 0): {repr(sample_target[:60])}...")
    print("  (Target = Input shifted 1 position forward — model predicts target from input)")

    print("\nDataset ready!")
    print("Next step: python 04_model.py")
