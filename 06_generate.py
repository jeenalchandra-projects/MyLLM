"""
STEP 6: Generate Text — Ask Your Vehicle LLM Questions
=======================================================
The model is trained. Now we use it to generate text.

KEY CONCEPT: Inference vs. Training
=====================================
During TRAINING: we feed known input+target pairs, compute loss,
                 update weights (model is "studying")

During INFERENCE (generation): we feed a PROMPT and the model
                                generates continuations (model is "thinking")

KEY CONCEPT: Autoregressive Generation
=======================================
The model generates one character at a time:
  1. Feed prompt to model → get probability for next character
  2. Sample a character from that distribution
  3. Append it to the sequence
  4. Feed extended sequence back → get next character probability
  5. Repeat

This is called AUTOREGRESSIVE because each new output becomes
part of the input for the next step.

KEY CONCEPT: Temperature
=========================
Temperature controls how "creative" or "confident" the model is:

  Before sampling we divide the logits by temperature T:
    logits_scaled = logits / T

  Low T (e.g. 0.2):
    → Logits are amplified → probability concentrates on top tokens
    → Output is predictable, repetitive, factual-sounding

  High T (e.g. 1.5):
    → Logits are dampened → probability spreads more evenly
    → Output is more random, creative, sometimes nonsensical

  T = 1.0 is the "neutral" setting — no modification.
  T = 0.8 is a common practical default.

Run: python 06_generate.py
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

CHECKPOINT_FILE = "checkpoints/model.pt"
VOCAB_FILE = "data/vocab.json"

# ── Pick best available device ────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def load_model_and_vocab():
    """Load trained model weights and vocabulary from disk."""
    if not os.path.exists(CHECKPOINT_FILE):
        raise FileNotFoundError(
            f"No checkpoint found at {CHECKPOINT_FILE}\n"
            "Please run 05_train.py first to train the model."
        )

    # Load vocabulary
    with open(VOCAB_FILE, "r") as f:
        char_to_int = json.load(f)
    int_to_char = {int(v): k for k, v in char_to_int.items()}
    vocab_size = len(char_to_int)

    # Load model
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE, weights_only=True)
    model = MiniGPT(vocab_size).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"Loaded model from step {checkpoint['step']} "
          f"(val_loss={checkpoint['val_loss']:.4f})")
    return model, char_to_int, int_to_char


def ask(prompt, model, char_to_int, int_to_char,
        max_new_chars=200, temperature=0.8):
    """
    Generate a continuation from a prompt string.

    prompt:         The starting text (e.g. "Q: Who makes the Mustang?\nA:")
    max_new_chars:  How many new characters to generate
    temperature:    Creativity control (0.2=conservative, 0.8=balanced, 1.5=creative)
    """
    # Filter out any characters not in our vocabulary
    prompt_clean = "".join(c for c in prompt if c in char_to_int)
    if len(prompt_clean) < len(prompt):
        dropped = set(prompt) - set(char_to_int.keys())
        print(f"  (Note: dropped {len(prompt)-len(prompt_clean)} unknown chars: {dropped})")

    # Encode prompt to token IDs
    prompt_ids = [char_to_int[c] for c in prompt_clean]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(idx, max_new_chars, temperature=temperature)

    # Decode full output (prompt + generated)
    full_output = "".join(int_to_char[i] for i in output_ids[0].tolist())

    # Return only the generated part (after the prompt)
    generated = full_output[len(prompt_clean):]
    return generated


if __name__ == "__main__":
    print("Loading model...")
    model, char_to_int, int_to_char = load_model_and_vocab()
    print(f"Device: {DEVICE.upper()}")
    print()

    # ── Demo questions ───────────────────────────────────────────────
    # These prompts follow the Q&A format the model was trained on.
    # The model will try to complete them in the same format.

    demo_questions = [
        "Q: What models does Ford make?\nA:",
        "Q: Who makes the Mustang?\nA:",
        "Q: What models does Toyota make?\nA:",
        "Q: Is the Camry made by Honda?\nA:",
        "Q: What models does BMW make?\nA:",
    ]

    print("=" * 60)
    print("VEHICLE LLM — QUESTION ANSWERING")
    print("=" * 60)
    print(f"Temperature: 0.8  |  Max new chars: 150\n")

    for question in demo_questions:
        answer = ask(question, model, char_to_int, int_to_char,
                     max_new_chars=150, temperature=0.8)
        print(f"{question}{answer}")
        print("-" * 60)

    # ── Interactive mode ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("Type your own prompts! Format: 'Q: Your question here?\\nA:'")
    print("Type 'quit' to exit.")
    print("=" * 60)

    while True:
        try:
            user_prompt = input("\nPrompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_prompt.lower() in ("quit", "exit", "q"):
            break
        if not user_prompt:
            continue

        # Replace literal \n with actual newline
        user_prompt = user_prompt.replace("\\n", "\n")

        try:
            temp = float(input("Temperature [0.8]: ").strip() or "0.8")
        except ValueError:
            temp = 0.8

        result = ask(user_prompt, model, char_to_int, int_to_char,
                     max_new_chars=200, temperature=temp)
        print(f"\n{user_prompt}{result}")

    print("\nGoodbye! Your Vehicle LLM is ready.")
    print("\nNEXT PROJECT: Fine-tuning this model on specific vehicle data")
    print("(e.g. only EVs, only luxury brands, detailed specs)")
