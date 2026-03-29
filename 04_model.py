"""
STEP 4: The Transformer Model — GPT Architecture from Scratch
=============================================================
This is the heart of the project. We build a GPT-style neural network
from scratch using PyTorch. Every component is explained.

KEY CONCEPT: What is a Neural Network?
A neural network is a function with millions of tunable numbers (WEIGHTS).
We adjust these weights during training until the function's outputs
(predicted next characters) match the actual next characters in our text.

ARCHITECTURE OVERVIEW:
  Input (character IDs)
    ↓
  [Token Embedding + Position Embedding]
    ↓
  [Transformer Block] × N_LAYERS
      └── LayerNorm → Self-Attention → LayerNorm → FeedForward
    ↓
  Final LayerNorm
    ↓
  Linear projection → Logits (one score per vocabulary character)

The logits tell us: "how likely is each character to come next?"

Run: python 04_model.py   (prints parameter count and does a forward pass test)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math

# ── Hyperparameters ──────────────────────────────────────────────────
BLOCK_SIZE  = 128   # Context window (must match 03_dataset.py)
N_EMBED     = 64    # Size of each token's embedding vector
N_HEADS     = 4     # Number of attention heads
N_LAYERS    = 4     # Number of transformer blocks stacked
DROPOUT     = 0.1   # Dropout rate (regularization — explained below)


# ════════════════════════════════════════════════════════════════════
# COMPONENT 1: Self-Attention Head
# ════════════════════════════════════════════════════════════════════
class AttentionHead(nn.Module):
    """
    KEY CONCEPT: Self-Attention
    ============================
    Self-attention lets every token "look at" every other token
    (that came before it) and decide how relevant each one is.

    Example: When processing "Ford Mustang is a sports car",
    the word "sports" benefits from knowing "Mustang" was mentioned —
    they are highly relevant to each other.

    HOW IT WORKS:
    Each token creates 3 vectors from its embedding:
      - Query (Q):  "What am I looking for?"
      - Key   (K):  "What do I offer?"
      - Value (V):  "What information do I carry?"

    Attention score = Q · Kᵀ  (dot product = similarity)
    High score = "these two tokens are highly relevant to each other"

    The scores are normalized (softmax) → attention WEIGHTS (sum to 1)
    Output = weighted sum of all Value vectors

    CAUSAL MASKING: We mask out future positions (tokens not yet generated).
    Token at position 5 can only look at positions 0-5, not 6,7,8...
    This is essential for text generation — you can't "cheat" by
    looking at the answer before predicting it.
    """

    def __init__(self, head_size):
        super().__init__()
        # Linear layers that project embeddings into Q, K, V spaces
        # bias=False is standard for attention projections
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.key   = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.dropout = nn.Dropout(DROPOUT)

        # Causal mask: a lower-triangular matrix of ones
        # register_buffer = not a learned parameter, but saved with model
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        )

    def forward(self, x):
        B, T, C = x.shape  # Batch, Time (sequence length), Channels (embed dim)
        head_size = self.query.out_features

        q = self.query(x)  # [B, T, head_size]
        k = self.key(x)    # [B, T, head_size]
        v = self.value(x)  # [B, T, head_size]

        # Compute attention scores: Q · Kᵀ
        # Scale by 1/√head_size — prevents scores from becoming too large
        # (large scores → extreme softmax → vanishing gradients)
        scale = head_size ** -0.5
        scores = q @ k.transpose(-2, -1) * scale  # [B, T, T]

        # Apply causal mask: set future positions to -infinity
        # After softmax, -inf → 0 (ignored completely)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))

        # Softmax: convert scores to probabilities (sum to 1)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Weighted sum of values
        out = weights @ v  # [B, T, head_size]
        return out


# ════════════════════════════════════════════════════════════════════
# COMPONENT 2: Multi-Head Attention
# ════════════════════════════════════════════════════════════════════
class MultiHeadAttention(nn.Module):
    """
    KEY CONCEPT: Multi-Head Attention
    ==================================
    Instead of one attention computation, we run N_HEADS in PARALLEL.
    Each head learns to attend to different types of relationships:
      - Head 1 might learn make ↔ model associations
      - Head 2 might learn Q&A format patterns
      - Head 3 might learn word co-occurrences
      - Head 4 might learn positional patterns

    The outputs from all heads are CONCATENATED and then projected
    back to the original embedding size.

    head_size = N_EMBED // N_HEADS
    So with N_EMBED=64, N_HEADS=4: each head has size 16.
    Concatenating 4 heads of size 16 = 64. Same size in, same size out.
    """

    def __init__(self):
        super().__init__()
        head_size = N_EMBED // N_HEADS
        self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(N_HEADS)])
        # Final projection: blend the concatenated head outputs
        self.proj = nn.Linear(N_EMBED, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # Run all heads, concatenate along the last dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# ════════════════════════════════════════════════════════════════════
# COMPONENT 3: Feed-Forward Network
# ════════════════════════════════════════════════════════════════════
class FeedForward(nn.Module):
    """
    KEY CONCEPT: Feed-Forward Layer
    ================================
    After attention mixes information ACROSS tokens, the feed-forward
    layer processes each token INDEPENDENTLY.

    Think of attention as: "gather relevant information from context"
    Think of feed-forward as: "now think about what that means"

    Structure: Linear → ReLU → Linear → Dropout
    The inner layer is 4× wider (this is standard in transformer design).
    ReLU = "Rectified Linear Unit" — sets negative values to 0.
    It introduces non-linearity, letting the network learn complex patterns.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBED, 4 * N_EMBED),  # expand
            nn.ReLU(),
            nn.Linear(4 * N_EMBED, N_EMBED),  # compress back
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


# ════════════════════════════════════════════════════════════════════
# COMPONENT 4: Transformer Block
# ════════════════════════════════════════════════════════════════════
class TransformerBlock(nn.Module):
    """
    KEY CONCEPT: Transformer Block = Attention + FeedForward + Residuals
    ======================================================================
    One transformer block combines:
      1. Layer Normalization → Multi-Head Attention → Residual connection
      2. Layer Normalization → Feed-Forward → Residual connection

    RESIDUAL CONNECTIONS (skip connections):
    Instead of: output = layer(x)
    We do:      output = layer(x) + x

    Why? In deep networks (many layers), gradients can vanish (become ~0)
    before reaching early layers — the network stops learning.
    Residual connections give gradients a "highway" to flow backwards
    without passing through every layer. This was the key insight of
    ResNet (2015) that made deep networks trainable.

    LAYER NORMALIZATION:
    Rescales each token's representation to have mean=0, std=1.
    Prevents numbers from exploding or vanishing as they flow through layers.
    Note: we apply LayerNorm BEFORE each sub-layer (this is "Pre-LN", slightly
    different from the original 2017 transformer paper which used Post-LN).
    Pre-LN is more stable and used in GPT-2/3.
    """

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBED)
        self.attn = MultiHeadAttention()
        self.ln2 = nn.LayerNorm(N_EMBED)
        self.ff = FeedForward()

    def forward(self, x):
        # Attention sub-layer with residual
        x = x + self.attn(self.ln1(x))
        # Feed-forward sub-layer with residual
        x = x + self.ff(self.ln2(x))
        return x


# ════════════════════════════════════════════════════════════════════
# THE FULL GPT MODEL
# ════════════════════════════════════════════════════════════════════
class MiniGPT(nn.Module):
    """
    KEY CONCEPT: The Complete GPT Architecture
    ===========================================
    Putting it all together:

    1. TOKEN EMBEDDING: Look up each character's embedding vector.
       Each of the VOCAB_SIZE characters gets a unique 64-dimensional vector.
       These vectors are LEARNED — they start random and slowly adjust
       to encode meaning through training.

    2. POSITION EMBEDDING: Add positional information.
       The same character at position 0 vs position 50 should be treated
       differently. We add a learned position vector (one per position
       up to BLOCK_SIZE) to the token embedding.

    3. N TRANSFORMER BLOCKS: The main processing stack.

    4. FINAL LAYER NORM + LINEAR: Convert the final embedding to
       LOGITS — one score per vocabulary character.
       High logit = "this character is likely to come next"

    DROPOUT: Randomly zeroes out 10% of activations during training.
    This is REGULARIZATION — it prevents the model from memorizing
    the training data and forces it to learn more robust patterns.
    Like studying without always having notes to refer to.
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

        # Embedding tables
        self.token_embedding = nn.Embedding(vocab_size, N_EMBED)
        self.pos_embedding = nn.Embedding(BLOCK_SIZE, N_EMBED)

        # Stack of transformer blocks
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(N_LAYERS)])

        # Final normalization and output projection
        self.ln_final = nn.LayerNorm(N_EMBED)
        self.lm_head = nn.Linear(N_EMBED, vocab_size)

        # Initialize weights (important for stable training)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Weight initialization: start weights from a small normal distribution.
        If weights start too large, training can diverge (explode).
        If too small, gradients vanish. Small normal values work well.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass: convert token IDs → logits (and optionally compute loss).

        idx:     [B, T] tensor of token IDs
        targets: [B, T] tensor of target token IDs (optional, for training)
        Returns: logits [B, T, vocab_size], loss (scalar or None)
        """
        B, T = idx.shape
        assert T <= BLOCK_SIZE, f"Sequence length {T} exceeds block size {BLOCK_SIZE}"

        # 1. Token + Position embeddings
        tok_emb = self.token_embedding(idx)                             # [B, T, N_EMBED]
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.pos_embedding(pos)                               # [T, N_EMBED]
        x = tok_emb + pos_emb                                           # [B, T, N_EMBED]

        # 2. Pass through transformer blocks
        x = self.blocks(x)                                              # [B, T, N_EMBED]

        # 3. Final norm + project to vocabulary
        x = self.ln_final(x)                                            # [B, T, N_EMBED]
        logits = self.lm_head(x)                                        # [B, T, vocab_size]

        # 4. Compute loss if targets are provided (training mode)
        loss = None
        if targets is not None:
            # KEY CONCEPT: Cross-Entropy Loss
            # For each position, the model predicts a probability distribution
            # over all vocab_size characters. Cross-entropy measures how
            # "surprised" the model was by the correct answer.
            # Loss = -log(probability assigned to correct character)
            # Low loss = model was confident AND correct (good)
            # High loss = model was confident but wrong, or uncertain (bad)
            #
            # We reshape to [B*T, vocab_size] because F.cross_entropy
            # expects 2D predictions.
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, V),
                targets.view(B * T)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8):
        """
        Generate new tokens one at a time.

        idx:            [1, T] tensor of starting token IDs (the prompt)
        max_new_tokens: how many new characters to generate
        temperature:    controls randomness (explained in 06_generate.py)
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context to BLOCK_SIZE (the model can't handle longer input)
            idx_cond = idx[:, -BLOCK_SIZE:]

            # Forward pass (no loss needed during generation)
            logits, _ = self(idx_cond)

            # Take only the logits for the LAST position (next token prediction)
            logits = logits[:, -1, :]  # [1, vocab_size]

            # Temperature scaling: divide logits before softmax
            # Low temp (0.2) → sharper distribution → more predictable output
            # High temp (1.5) → flatter distribution → more random output
            logits = logits / temperature

            # Convert logits → probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample the next token from the probability distribution
            next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]

            # Append to running sequence
            idx = torch.cat([idx, next_token], dim=1)  # [1, T+1]

        return idx


if __name__ == "__main__":
    # Load vocab to get vocab_size
    with open("data/vocab.json", "r") as f:
        char_to_int = json.load(f)
    vocab_size = len(char_to_int)

    # Instantiate the model
    model = MiniGPT(vocab_size)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 50)
    print("MiniGPT — Vehicle LLM")
    print("=" * 50)
    print(f"Vocabulary size:    {vocab_size} characters")
    print(f"Embedding size:     {N_EMBED}")
    print(f"Attention heads:    {N_HEADS}")
    print(f"Transformer layers: {N_LAYERS}")
    print(f"Context window:     {BLOCK_SIZE} characters")
    print(f"Total parameters:   {total_params:,}")
    print(f"Trainable params:   {trainable_params:,}")
    print()

    # Test a forward pass
    dummy_input = torch.zeros((2, 10), dtype=torch.long)
    dummy_targets = torch.zeros((2, 10), dtype=torch.long)
    logits, loss = model(dummy_input, dummy_targets)

    print(f"Forward pass test:")
    print(f"  Input shape:  {dummy_input.shape}")
    print(f"  Output shape: {logits.shape}  (batch=2, seq=10, vocab={vocab_size})")
    print(f"  Initial loss: {loss.item():.4f}")
    print(f"  Expected loss ≈ {-1 * (1/vocab_size) * (1/vocab_size) :.1f} (random baseline: {math.log(vocab_size):.2f})")
    print()
    print("Model architecture looks good!")
    print("Next step: python 05_train.py")
