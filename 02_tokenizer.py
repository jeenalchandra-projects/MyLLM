"""
STEP 2: Tokenizer — Turning Text into Numbers
==============================================
Neural networks only work with numbers, never raw text. A TOKENIZER is the
bridge between human-readable text and the numbers the model processes.

KEY CONCEPT: What is a Token?
A token is the atomic unit the model thinks in. We use CHARACTER-LEVEL
tokenization: every individual character is one token.

   "Ford" → [12, 33, 27, 19]   (4 tokens)
   [12, 33, 27, 19] → "Ford"

VOCABULARY: The set of all unique tokens the model knows about.
Our vocabulary = every unique character found in vehicles.txt.
Typical size: ~70–100 characters (letters, digits, spaces, punctuation).

REAL-WORLD NOTE: GPT-4 uses Byte-Pair Encoding (BPE) which merges common
character sequences into single tokens. "Ford" might be ONE token in GPT-4,
not four. This makes it more efficient. But character-level is much simpler
to understand and implement — the concept is exactly the same.

Run: python 02_tokenizer.py
"""

import os
import json

DATA_FILE = "data/vehicles.txt"
VOCAB_FILE = "data/vocab.json"


def build_vocab(text):
    """
    Scan all characters in text and build a vocabulary.
    Returns:
        chars: sorted list of unique characters
        char_to_int: dict mapping char → int
        int_to_char: dict mapping int → char
    """
    # Find every unique character
    chars = sorted(set(text))
    vocab_size = len(chars)

    # Build lookup dictionaries
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}

    return chars, char_to_int, int_to_char, vocab_size


def encode(text, char_to_int):
    """Convert a string to a list of integers."""
    return [char_to_int[c] for c in text]


def decode(integers, int_to_char):
    """Convert a list of integers back to a string."""
    return "".join(int_to_char[i] for i in integers)


def save_vocab(char_to_int, vocab_file):
    """Save vocabulary to disk so training and generation use the same mapping."""
    os.makedirs(os.path.dirname(vocab_file), exist_ok=True)
    with open(vocab_file, "w") as f:
        json.dump(char_to_int, f)
    print(f"Vocabulary saved to {vocab_file}")


def load_vocab(vocab_file):
    """Load vocabulary from disk."""
    with open(vocab_file, "r") as f:
        char_to_int = json.load(f)
    int_to_char = {int(v): k for k, v in char_to_int.items()}
    return char_to_int, int_to_char


if __name__ == "__main__":
    # Load the training text
    print(f"Loading text from {DATA_FILE}...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"  Text length: {len(text):,} characters")

    # Build the vocabulary
    chars, char_to_int, int_to_char, vocab_size = build_vocab(text)

    print(f"\nVocabulary size: {vocab_size} unique characters")
    print(f"Characters in vocab: {repr(''.join(chars))}")

    # Save vocabulary so other scripts can reuse the exact same mapping
    save_vocab(char_to_int, VOCAB_FILE)

    # ----- VERIFICATION -----
    print("\n--- VERIFICATION ---")
    test_strings = [
        "Ford Mustang",
        "Q: Who makes the Camry?\nA: Toyota makes the Camry.",
        "BMW 3 Series",
    ]
    for s in test_strings:
        # Check for any characters not in our vocab
        unknown = [c for c in s if c not in char_to_int]
        if unknown:
            print(f"WARNING: unknown characters {unknown} in test string: {repr(s)}")
            continue
        encoded = encode(s, char_to_int)
        decoded = decode(encoded, int_to_char)
        match = decoded == s
        print(f"  '{s}'")
        print(f"  → encoded: {encoded[:10]}{'...' if len(encoded) > 10 else ''}")
        print(f"  → decoded: '{decoded}' ✓" if match else f"  → MISMATCH! decoded: '{decoded}'")
        print()

    print(f"Tokenizer ready. Vocabulary size = {vocab_size}")
    print("Next step: python 03_dataset.py")
