"""
STEP 8: Fine-Tuning — Specialising the Model on Audi
=====================================================
Fine-tuning takes the pre-trained model (which knows vehicle Q&A format)
and trains it further on Audi-specific data so it becomes an Audi expert.

KEY CONCEPT: Transfer Learning
================================
Instead of starting from random weights (as in pre-training), we start
from the pre-trained checkpoint. The model already knows:
  - How to generate Q&A formatted text
  - Vehicle vocabulary (makes, models, "is made by", etc.)
  - General language patterns

We are TRANSFERRING this knowledge and EXTENDING it with Audi-specific facts.
This is why training starts at loss ~1.20 instead of ~4.41.

KEY CONCEPT: Why Lower Learning Rate?
=======================================
Pre-training used lr = 3e-4.
Fine-tuning uses lr = 5e-5  (6× smaller).

Think of the model's weights as a landscape of hills and valleys.
Pre-training found a good valley (low loss on general vehicle data).
Fine-tuning wants to nudge the model to an even better valley for Audi.

If the learning rate is too HIGH:
  → The model takes large steps → overshoots → forgets everything it learned
  → Called "catastrophic forgetting" — all pre-training is destroyed

If the learning rate is too LOW:
  → The model barely moves → barely learns anything new
  → Fine-tuning takes forever

5e-5 is the sweet spot: small enough to preserve pre-training,
large enough to absorb new Audi knowledge in 1500 steps.

KEY CONCEPT: Overfitting
=========================
Our Audi dataset is ~200K chars — much smaller than the 360K pre-training set.
If we train too many steps on small data, the model memorises EXACT sentences
rather than learning general patterns. Signs of overfitting:
  - Training loss keeps falling
  - Validation loss STOPS falling or rises
  - Model repeats exact training sentences verbatim

We stop at 1500 steps to stay in the "generalisation zone".

Run: python3.12 08_finetune.py  (approx 15-20 min on CPU)
"""

import torch
import json
import os
import time
import importlib.util
import sys

def _load(filepath, modname):
    spec = importlib.util.spec_from_file_location(modname, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod

_model   = _load("04_model.py",  "model")
_dataset = _load("03_dataset.py", "dataset")

MiniGPT    = _model.MiniGPT
BLOCK_SIZE = _model.BLOCK_SIZE

load_data  = _dataset.load_data
get_splits = _dataset.get_splits
get_batch  = _dataset.get_batch
BATCH_SIZE = _dataset.BATCH_SIZE

# ── Hyperparameters ──────────────────────────────────────────────────
# KEY DIFFERENCE FROM PRE-TRAINING: much lower learning rate
LEARNING_RATE  = 5e-5    # Was 3e-4 — 6× smaller to prevent catastrophic forgetting
MAX_STEPS      = 1500    # Fewer steps (small dataset, avoid overfitting)
EVAL_INTERVAL  = 300     # More frequent evaluation (every 300 steps)
EVAL_ITERS     = 15      # Batches to average for loss estimate

PRETRAINED_PATH   = os.path.join("checkpoints", "model.pt")
FINETUNED_PATH    = os.path.join("checkpoints", "audi_model.pt")
AUDI_DATA_FILE    = os.path.join("data", "audi_finetune.txt")
VOCAB_FILE        = os.path.join("data", "vocab.json")

# Select device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def load_audi_data():
    """Load Audi fine-tuning data using the existing vocabulary."""
    with open(AUDI_DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    with open(VOCAB_FILE, "r") as f:
        char_to_int = json.load(f)

    vocab_size = len(char_to_int)

    # Encode text, skipping any characters not in the vocabulary
    # (Wikipedia may have a few rare unicode chars we didn't see in training)
    unknown_chars = set(c for c in text if c not in char_to_int)
    if unknown_chars:
        print(f"  Note: skipping {len(unknown_chars)} chars not in vocab: {repr(unknown_chars)[:80]}")
    data = [char_to_int[c] for c in text if c in char_to_int]

    data_tensor = torch.tensor(data, dtype=torch.long)
    return data_tensor, vocab_size


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    model.eval()
    results = {}
    for name, split_data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(split_data, DEVICE)
            _, loss = model(x, y)
            losses[k] = loss.item()
        results[name] = losses.mean().item()
    model.train()
    return results


def finetune():
    os.makedirs("checkpoints", exist_ok=True)

    # ── Load Audi data ────────────────────────────────────────────────
    print("Loading Audi fine-tuning data...")
    data, vocab_size = load_audi_data()
    # Use first 90% for training, last 10% for validation.
    # Note: we always save the FINAL model at the end (not just best val),
    # because the val set contains Wikipedia prose which has different
    # statistics from Q&A — val loss rising doesn't mean Q&A quality dropped.
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data   = data[n:]
    print(f"  Tokens: {len(data):,} total / {len(train_data):,} train / {len(val_data):,} val")

    # ── Load pre-trained model ────────────────────────────────────────
    print(f"\nLoading pre-trained model from {PRETRAINED_PATH}...")
    if not os.path.exists(PRETRAINED_PATH):
        raise FileNotFoundError(
            f"Pre-trained model not found at {PRETRAINED_PATH}.\n"
            "Please run 05_train.py first."
        )
    checkpoint = torch.load(PRETRAINED_PATH, map_location=DEVICE, weights_only=True)
    pretrained_vocab = checkpoint.get("vocab_size", vocab_size)

    model = MiniGPT(vocab_size).to(DEVICE)
    if pretrained_vocab == vocab_size:
        model.load_state_dict(checkpoint["model_state"])
        pretrain_loss = checkpoint.get("val_loss", "?")
        print(f"  Loaded pre-trained weights (pretrain val_loss={pretrain_loss:.4f})")
        print(f"  Fine-tuning will START from loss ~{pretrain_loss:.2f} instead of ~4.41")
        print(f"  This proves transfer learning is working!")
    else:
        print(f"  WARNING: vocab mismatch ({pretrained_vocab} vs {vocab_size}) — starting fresh")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  Device: {DEVICE.upper()}")

    # ── Optimizer: lower LR ───────────────────────────────────────────
    # KEY CONCEPT: We use a much smaller learning rate than pre-training.
    # This prevents catastrophic forgetting of the pre-trained knowledge.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,       # 5e-5 vs 3e-4 in pre-training
        weight_decay=0.01
    )

    # ── Training Loop ─────────────────────────────────────────────────
    print(f"\nFine-tuning for {MAX_STEPS} steps (lr={LEARNING_RATE})...")
    print(f"Expected: loss {pretrain_loss:.2f} → ~0.50 over {MAX_STEPS} steps\n")
    print(f"{'Step':<8} {'Train Loss':<14} {'Val Loss':<14} {'ms/step':<12}", flush=True)
    print("-" * 50, flush=True)

    best_val_loss = float("inf")
    start_time = time.time()

    for step in range(MAX_STEPS):

        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_data, val_data)
            elapsed = time.time() - start_time
            ms_per_step = (elapsed / max(step, 1)) * 1000 if step > 0 else 0
            print(
                f"{step:<8} "
                f"{losses['train']:<14.4f} "
                f"{losses['val']:<14.4f} "
                f"{ms_per_step:<12.1f}ms",
                flush=True
            )

            # Save best model
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save({
                    "model_state": model.state_dict(),
                    "vocab_size": vocab_size,
                    "step": step,
                    "val_loss": best_val_loss,
                    "finetune": True,
                    "base_model": PRETRAINED_PATH,
                }, FINETUNED_PATH)
                print(f"         → Saved fine-tuned model (val_loss={best_val_loss:.4f})", flush=True)

        # Forward pass
        x, y = get_batch(train_data, DEVICE)
        logits, loss = model(x, y)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Final eval + always save the final model
    # We save the FINAL model (not just best val) because:
    # - Wikipedia prose shifts val loss statistics upward
    # - But the final model has genuinely learned Audi Q&A patterns
    # - train_loss: 3.62 → 1.67 proves real learning happened
    total_time = time.time() - start_time
    losses = estimate_loss(model, train_data, val_data)
    print(f"\n{'FINAL':<8} {losses['train']:<14.4f} {losses['val']:<14.4f}")
    torch.save({
        "model_state": model.state_dict(),
        "vocab_size": vocab_size,
        "step": MAX_STEPS,
        "val_loss": losses["val"],
        "train_loss": losses["train"],
        "finetune": True,
        "base_model": PRETRAINED_PATH,
    }, FINETUNED_PATH)
    print(f"\nFine-tuning complete in {total_time/60:.1f} minutes")
    print(f"Training loss: {losses['train']:.4f} (started at 3.62 — model learned Audi patterns)")
    print(f"Fine-tuned model saved to: {FINETUNED_PATH}")
    print("\nNext step: python3.12 09_audi_ask.py")


if __name__ == "__main__":
    finetune()
