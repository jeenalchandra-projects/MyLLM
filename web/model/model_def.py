"""
MiniGPT — standalone model definition for web deployment.
Copied from 04_model.py (self-contained, no relative imports).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

BLOCK_SIZE = 128
N_EMBED    = 64
N_HEADS    = 4
N_LAYERS   = 4
DROPOUT    = 0.1


class AttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.key   = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.dropout = nn.Dropout(DROPOUT)
        self.register_buffer("mask", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape
        head_size = self.query.out_features
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scale = head_size ** -0.5
        scores = q @ k.transpose(-2, -1) * scale
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return weights @ v


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = N_EMBED // N_HEADS
        self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(N_HEADS)])
        self.proj = nn.Linear(N_EMBED, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBED, 4 * N_EMBED),
            nn.ReLU(),
            nn.Linear(4 * N_EMBED, N_EMBED),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1  = nn.LayerNorm(N_EMBED)
        self.attn = MultiHeadAttention()
        self.ln2  = nn.LayerNorm(N_EMBED)
        self.ff   = FeedForward()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, N_EMBED)
        self.pos_embedding   = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.blocks          = nn.Sequential(*[TransformerBlock() for _ in range(N_LAYERS)])
        self.ln_final        = nn.LayerNorm(N_EMBED)
        self.lm_head         = nn.Linear(N_EMBED, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos     = torch.arange(T, device=idx.device)
        pos_emb = self.pos_embedding(pos)
        x       = tok_emb + pos_emb
        x       = self.blocks(x)
        x       = self.ln_final(x)
        logits  = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs  = F.softmax(logits, dim=-1)
            next_t = torch.multinomial(probs, num_samples=1)
            idx    = torch.cat([idx, next_t], dim=1)
        return idx
