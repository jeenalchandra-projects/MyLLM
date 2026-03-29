"""
STEP 0: Environment Setup
=========================
This script checks that everything is installed and working before we begin.

KEY CONCEPT: Why PyTorch?
PyTorch is a "deep learning framework" — a library that handles all the math
behind neural networks. The most important thing it does is AUTOMATIC DIFFERENTIATION:
when we tell it to compute a loss (how wrong the model is), it automatically
figures out how to adjust every single weight in the network to reduce that loss.
Without PyTorch, we'd have to derive and implement calculus by hand.

Run this first: python 00_setup.py
"""

import sys

print("=" * 50)
print("Checking Python version...")
print(f"Python {sys.version}")
assert sys.version_info >= (3, 8), "Need Python 3.8+ (use python3.12)"
print("Python version OK!\n")

# Try importing required packages
missing = []

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"  - CUDA (GPU) available: {torch.cuda.is_available()}")
    print(f"  - MPS (Apple Silicon GPU) available: {torch.backends.mps.is_available()}")
    # Pick the best available device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"  - We will train on: {device.upper()}")
    print("PyTorch OK!\n")
except ImportError:
    missing.append("torch")

try:
    import numpy as np
    print(f"NumPy version: {np.__version__} — OK")
except ImportError:
    missing.append("numpy")

try:
    import requests
    print(f"Requests version: {requests.__version__} — OK")
except ImportError:
    missing.append("requests")

if missing:
    print("\nMISSING PACKAGES. Run this command:")
    print(f"  pip install {' '.join(missing)}")
    sys.exit(1)
else:
    print("\n" + "=" * 50)
    print("All dependencies are installed!")
    print("You're ready to build an LLM.")
    print("Next step: python 01_fetch_data.py")
    print("=" * 50)
