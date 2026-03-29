"""
STEP 9: Ask Your Audi Expert LLM
=================================
This script demonstrates the fine-tuned Audi model and includes a
BEFORE/AFTER comparison to clearly show what fine-tuning achieved.

KEY CONCEPT: What Did Fine-Tuning Actually Do?
===============================================
The base model (model.pt) knows vehicle Q&A format but answers with
random/hallucinated makes because it never saw Audi-specific facts.

The fine-tuned model (audi_model.pt) has the SAME architecture and
the SAME number of parameters — but its weights have been nudged to
encode Audi-specific knowledge:
  - Which models Audi makes
  - What quattro means
  - What RS stands for
  - When models were introduced
  - Model categories (sedan, SUV, sports car, electric)

The comparison mode lets you SEE this difference side by side.

Run: python3.12 09_audi_ask.py
"""

import torch
import json
import os
import importlib.util
import sys

def _load(filepath, modname):
    spec = importlib.util.spec_from_file_location(modname, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod

_model = _load("04_model.py", "model")
MiniGPT    = _model.MiniGPT
BLOCK_SIZE = _model.BLOCK_SIZE

BASE_CHECKPOINT   = os.path.join("checkpoints", "model.pt")
AUDI_CHECKPOINT   = os.path.join("checkpoints", "audi_model.pt")
VOCAB_FILE        = os.path.join("data", "vocab.json")

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def load_model(checkpoint_path):
    """Load a model from a checkpoint file."""
    with open(VOCAB_FILE, "r") as f:
        char_to_int = json.load(f)
    int_to_char = {int(v): k for k, v in char_to_int.items()}
    vocab_size = len(char_to_int)

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model = MiniGPT(vocab_size).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, char_to_int, int_to_char


def generate(model, prompt, char_to_int, int_to_char,
             max_new_chars=180, temperature=0.7):
    """Generate text from a prompt."""
    prompt_clean = "".join(c for c in prompt if c in char_to_int)
    ids = [char_to_int[c] for c in prompt_clean]
    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        out = model.generate(idx, max_new_chars, temperature=temperature)

    full = "".join(int_to_char[i] for i in out[0].tolist())
    return full[len(prompt_clean):]


def compare(question_prompt, base_model, audi_model, char_to_int, int_to_char):
    """Run the same question through both models and print side-by-side."""
    base_answer  = generate(base_model,  question_prompt, char_to_int, int_to_char)
    audi_answer  = generate(audi_model,  question_prompt, char_to_int, int_to_char)

    print(f"QUESTION: {question_prompt.strip()}")
    print()
    print("BASE MODEL (pre-training only):")
    print(f"  {(question_prompt + base_answer).strip()[:200]}")
    print()
    print("FINE-TUNED MODEL (Audi expert):")
    print(f"  {(question_prompt + audi_answer).strip()[:200]}")
    print("-" * 65)


if __name__ == "__main__":
    # ── Load both models ─────────────────────────────────────────────
    print("Loading models...")
    if not os.path.exists(AUDI_CHECKPOINT):
        raise FileNotFoundError(
            f"Fine-tuned model not found at {AUDI_CHECKPOINT}.\n"
            "Please run 08_finetune.py first."
        )

    with open(VOCAB_FILE, "r") as f:
        char_to_int = json.load(f)
    int_to_char = {int(v): k for k, v in char_to_int.items()}

    base_ckpt = torch.load(BASE_CHECKPOINT, map_location=DEVICE, weights_only=True)
    audi_ckpt = torch.load(AUDI_CHECKPOINT, map_location=DEVICE, weights_only=True)

    base_model, _, _ = load_model(BASE_CHECKPOINT)
    audi_model, _, _ = load_model(AUDI_CHECKPOINT)

    print(f"Base model:        step {base_ckpt['step']}, val_loss={base_ckpt['val_loss']:.4f}")
    print(f"Fine-tuned model:  step {audi_ckpt['step']}, val_loss={audi_ckpt['val_loss']:.4f}")
    print(f"Device: {DEVICE.upper()}\n")

    # ── Before / After Comparison ────────────────────────────────────
    print("=" * 65)
    print("BEFORE vs AFTER FINE-TUNING — SIDE BY SIDE")
    print("=" * 65)
    print("This shows exactly what fine-tuning added.\n")

    comparison_prompts = [
        "Q: What models does Audi make?\nA:",
        "Q: What does RS stand for in Audi models?\nA:",
        "Q: What is Audi quattro?\nA:",
        "Q: What is the Audi R8?\nA:",
        "Q: What are Audi's electric vehicles?\nA:",
    ]

    for prompt in comparison_prompts:
        compare(prompt, base_model, audi_model, char_to_int, int_to_char)
        print()

    # ── Fine-tuned model only: more detailed questions ───────────────
    print("=" * 65)
    print("AUDI EXPERT MODEL — DETAILED QUESTIONS")
    print("=" * 65)
    print("(Temperature 0.7 — balanced creativity/accuracy)\n")

    detailed_questions = [
        "Q: What is the difference between the Audi A4 and S4?\nA:",
        "Q: When was the Audi Q7 first produced?\nA:",
        "Q: What is Audi's flagship sedan?\nA:",
        "Q: Where is Audi headquartered?\nA:",
        "Q: What platform does the Audi e-tron GT share?\nA:",
        "Q: What is the performance hierarchy in Audi models?\nA:",
        "Q: What body styles does the Audi A4 come in?\nA:",
    ]

    for prompt in detailed_questions:
        answer = generate(audi_model, prompt, char_to_int, int_to_char,
                          max_new_chars=180, temperature=0.7)
        print(f"{prompt}{answer}")
        print("-" * 65)

    # ── Interactive mode ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("INTERACTIVE MODE — Ask Your Audi Expert")
    print("Format: 'Q: Your question here?\\nA:'")
    print("Type 'compare <question>' to compare base vs fine-tuned")
    print("Type 'quit' to exit")
    print("=" * 65)

    while True:
        try:
            user_input = input("\nPrompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in ("quit", "exit"):
            break

        user_input = user_input.replace("\\n", "\n")

        if user_input.lower().startswith("compare "):
            # Compare mode
            question = user_input[8:].strip()
            if not question.startswith("Q:"):
                question = f"Q: {question}\nA:"
            compare(question, base_model, audi_model, char_to_int, int_to_char)
        else:
            try:
                temp_str = input("Temperature [0.7]: ").strip()
                temp = float(temp_str) if temp_str else 0.7
            except ValueError:
                temp = 0.7

            result = generate(audi_model, user_input, char_to_int, int_to_char,
                              max_new_chars=200, temperature=temp)
            print(f"\n{user_input}{result}")

    print("\nGoodbye! Your Audi Expert LLM is ready.")
