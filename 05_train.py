"""
STEP 5: Training Loop — Teaching the Model
==========================================
This is where the model actually LEARNS. We repeatedly show it batches
of vehicle text and gradually adjust its 800K weights until it can
predict the next character well.

KEY CONCEPT: How Does a Neural Network Learn?
============================================
Think of each weight as a dial. At the start, all dials are set randomly.
The training loop does this over and over:

  1. FORWARD PASS: Feed input through the network → get predictions
  2. COMPUTE LOSS: Measure how wrong the predictions are
  3. BACKWARD PASS: Calculate how to turn each dial to reduce the loss
     (this is calculus / chain rule, done automatically by PyTorch)
  4. UPDATE WEIGHTS: Turn each dial a tiny amount in the right direction

After 5,000 iterations of this, the model goes from knowing nothing to
generating coherent vehicle text.

KEY CONCEPT: The Optimizer (AdamW)
===================================
The optimizer is the algorithm that updates weights. We use AdamW:
  - "Adam" = Adaptive Moment estimation (adjusts learning rate per weight)
  - "W" = Weight decay (a regularization technique that prevents overfitting)
  - LEARNING_RATE controls how big each step is (too large = unstable,
    too small = takes forever)

KEY CONCEPT: Training vs. Validation Loss
==========================================
We split data 90% train / 10% validation.
  - TRAINING LOSS: How well the model fits data it HAS seen
  - VALIDATION LOSS: How well it generalizes to data it HASN'T seen
  - If training loss falls but val loss rises → OVERFITTING
    (memorizing training data, not learning generalizable patterns)

Run: python 05_train.py
This will take 10–20 minutes on CPU. Watch the loss decrease!
"""

import torch
import json
import os
import time
import math
import importlib.util
import sys

# Python module names can't start with a digit, so we use importlib
# to load our numbered files by file path rather than module name.
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
N_EMBED    = _model.N_EMBED
N_HEADS    = _model.N_HEADS
N_LAYERS   = _model.N_LAYERS

load_data   = _dataset.load_data
get_splits  = _dataset.get_splits
get_batch   = _dataset.get_batch
BATCH_SIZE  = _dataset.BATCH_SIZE

# ── Hyperparameters ──────────────────────────────────────────────────
LEARNING_RATE  = 3e-4     # Step size for weight updates
MAX_STEPS      = 3000     # Total training iterations
                          # (3000 steps ≈ 20 min on CPU, good quality)
EVAL_INTERVAL  = 500      # How often to print loss
EVAL_ITERS     = 15       # Batches to average for loss (reduced for speed)
CHECKPOINT_DIR = "checkpoints"
RESUME         = True     # Resume from checkpoint if one exists

# Select best available device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    """
    Estimate loss on training and validation sets.

    @torch.no_grad() tells PyTorch NOT to track gradients here —
    we're only evaluating, not training. This saves memory and speeds
    things up significantly.

    We average over EVAL_ITERS batches for a stable estimate
    (a single batch has too much variance to be meaningful).
    """
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


def train():
    # ── Setup ────────────────────────────────────────────────────────
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Load data
    print("Loading data...")
    data, vocab_size = load_data()
    train_data, val_data = get_splits(data)

    # Build model
    print(f"Building model (vocab_size={vocab_size})...")
    model = MiniGPT(vocab_size).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  Training on: {DEVICE.upper()}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # Resume from checkpoint if available
    start_step = 0
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "model.pt")
    if RESUME and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        if checkpoint.get("vocab_size") == vocab_size:
            model.load_state_dict(checkpoint["model_state"])
            start_step = checkpoint.get("step", 0)
            print(f"  Resumed from step {start_step} (val_loss={checkpoint.get('val_loss', '?'):.4f})")
        else:
            print("  Checkpoint vocab mismatch — starting fresh")

    # ── Training Loop ────────────────────────────────────────────────
    remaining = MAX_STEPS - start_step
    print(f"\nTraining from step {start_step} to {MAX_STEPS} ({remaining} steps remaining)...")
    print(f"Loss printed every {EVAL_INTERVAL} steps.\n")
    print(f"{'Step':<8} {'Train Loss':<14} {'Val Loss':<14} {'ms/step':<12}", flush=True)
    print("-" * 50, flush=True)

    best_val_loss = float("inf")
    start_time = time.time()
    step_time = time.time()

    for step in range(start_step, MAX_STEPS):

        # Evaluate periodically (step 0 = before any training)
        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_data, val_data)
            elapsed = time.time() - step_time
            steps_done = step - start_step
            ms_per_step = (elapsed / max(steps_done, 1)) * 1000 if steps_done > 0 else 0
            print(
                f"{step:<8} "
                f"{losses['train']:<14.4f} "
                f"{losses['val']:<14.4f} "
                f"{ms_per_step:<12.1f}ms",
                flush=True
            )

            # Save the model whenever validation loss improves
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_path = os.path.join(CHECKPOINT_DIR, "model.pt")
                torch.save({
                    "model_state": model.state_dict(),
                    "vocab_size": vocab_size,
                    "step": step,
                    "val_loss": best_val_loss,
                }, save_path)
                print(f"         → Saved best model (val_loss={best_val_loss:.4f})", flush=True)

        # ── One training step ────────────────────────────────────────

        # 1. Sample a batch
        x, y = get_batch(train_data, DEVICE)

        # 2. Forward pass: compute predictions and loss
        logits, loss = model(x, y)

        # 3. Zero gradients from previous step
        # (PyTorch accumulates gradients by default — we reset each step)
        optimizer.zero_grad(set_to_none=True)

        # 4. Backward pass: compute gradients via backpropagation
        # PyTorch automatically applies the chain rule through every layer
        loss.backward()

        # 5. Gradient clipping: cap gradient norm at 1.0
        # Prevents "gradient explosions" where a large gradient causes
        # the model to take a huge, destabilizing step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 6. Update weights using the optimizer
        optimizer.step()

    # Final evaluation
    total_time = time.time() - start_time
    losses = estimate_loss(model, train_data, val_data)
    print(f"\n{'FINAL':<8} {losses['train']:<14.4f} {losses['val']:<14.4f}")
    print(f"\nTraining complete in {total_time/60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {CHECKPOINT_DIR}/model.pt")
    print("\nNext step: python 06_generate.py")


if __name__ == "__main__":
    train()
