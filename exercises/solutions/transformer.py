"""Reference solutions: a decoder-only transformer, parameter counting, FLOPs (page 04)."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * weight, computed in float32, returned in x.dtype."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()                                                # statistics in fp32, even for bf16 input
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)   # eps INSIDE the sqrt
        return (self.weight.float() * x).to(dtype)


class SwiGLU(nn.Module):
    """FFN(x) = (SiLU(x W1) * (x W3)) W2, with W1, W3: d -> d_ff and W2: d_ff -> d."""

    def __init__(self, d_model: int, d_ff: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)    # gate
        self.w3 = nn.Linear(d_model, d_ff, bias=bias)    # up
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)    # down

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CausalSelfAttention(nn.Module):
    """Causal multi-head / grouped-query self-attention without positional information."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: Optional[int] = None,
                 head_dim: Optional[int] = None, bias: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.head_dim = d_model // n_heads if head_dim is None else head_dim
        assert n_heads % self.n_kv_heads == 0
        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=bias)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)      # (B, n_h, T, d_h)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)   # (B, n_kv, T, d_h)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        g = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(g, dim=1)                                           # (B, n_h, T, d_h)
        v = v.repeat_interleave(g, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)                 # (B, n_h, T, d_h)
        return self.wo(y.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim))


class Block(nn.Module):
    """Pre-norm decoder block: x + Attn(Norm(x)), then + FFN(Norm(.))."""

    def __init__(self, cfg: dict):
        super().__init__()
        d, eps, bias = cfg["d_model"], cfg.get("norm_eps", 1e-5), cfg.get("bias", False)
        self.attn_norm = RMSNorm(d, eps)
        self.attn = CausalSelfAttention(d, cfg["n_heads"], cfg.get("n_kv_heads"), cfg.get("head_dim"), bias)
        self.ffn_norm = RMSNorm(d, eps)
        self.ffn = SwiGLU(d, cfg["d_ff"], bias)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))      # each sublayer ADDS to the residual stream
        return x + self.ffn(self.ffn_norm(x))


class GPT(nn.Module):
    """Decoder-only language model: embed -> L blocks -> final norm -> LM head."""

    def __init__(self, cfg: dict):
        super().__init__()
        d, V, L = cfg["d_model"], cfg["vocab_size"], cfg["n_layers"]
        self.cfg = cfg
        self.tok_emb = nn.Embedding(V, d)
        max_len = cfg.get("max_seq_len", 0)
        self.pos_emb = nn.Embedding(max_len, d) if max_len > 0 else None
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(L)])
        self.final_norm = RMSNorm(d, cfg.get("norm_eps", 1e-5))
        self.lm_head = nn.Linear(d, V, bias=False)
        # init: N(0, 0.02) everywhere, zero biases, residual output projections scaled by 1/sqrt(2L)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
        for blk in self.blocks:                   # 2 residual additions per block -> 2L in total
            for w in (blk.attn.wo.weight, blk.ffn.w2.weight):
                nn.init.normal_(w, mean=0.0, std=0.02 / math.sqrt(2 * L))
        if cfg.get("tie_embeddings", False):
            self.lm_head.weight = self.tok_emb.weight     # one shared (V, d) matrix

    def forward(self, idx: Tensor, targets: Optional[Tensor] = None):
        B, T = idx.shape
        x = self.tok_emb(idx)                                                # (B, T, d)
        if self.pos_emb is not None:
            assert T <= self.pos_emb.num_embeddings
            x = x + self.pos_emb(torch.arange(T, device=idx.device))         # broadcast over batch
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(self.final_norm(x))                            # (B, T, V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def _dims(cfg: dict):
    d, nh = cfg["d_model"], cfg["n_heads"]
    nkv = cfg.get("n_kv_heads") or nh
    dh = cfg.get("head_dim") or d // nh
    return d, nh, nkv, dh


def count_params(cfg: dict) -> int:
    """Closed-form parameter count of GPT(cfg); equals sum(p.numel()) with shared weights counted once."""
    d, nh, nkv, dh = _dims(cfg)
    V, L, dff = cfg["vocab_size"], cfg["n_layers"], cfg["d_ff"]
    bias = cfg.get("bias", False)
    attn = d * nh * dh + 2 * d * nkv * dh + nh * dh * d          # Wq, Wk, Wv, Wo
    mlp = 3 * d * dff                                            # W1, W3, W2
    if bias:
        attn += nh * dh + 2 * nkv * dh + d
        mlp += 2 * dff + d
    per_layer = attn + mlp + 2 * d                               # + two RMSNorm gains
    embeddings = V * d * (1 if cfg.get("tie_embeddings", False) else 2)
    positions = cfg.get("max_seq_len", 0) * d
    return embeddings + positions + L * per_layer + d            # + final RMSNorm


def train_flops(n_params: float, n_tokens: float) -> float:
    """C ~ 6 N D: 2N forward + 4N backward per token."""
    return 6.0 * n_params * n_tokens


def forward_flops_per_token(cfg: dict, seq_len: int, causal: bool = True) -> float:
    """2 * (matmul weights incl. LM head) + attention score/value matmuls."""
    d, nh, nkv, dh = _dims(cfg)
    V, L, dff = cfg["vocab_size"], cfg["n_layers"], cfg["d_ff"]
    matmul_weights = L * (d * nh * dh + 2 * d * nkv * dh + nh * dh * d + 3 * d * dff) + V * d
    attn = 4.0 * L * seq_len * nh * dh                           # QK^T and AV, 2 FLOPs per MAC each
    if causal:
        attn *= 0.5                                              # average query sees half the keys
    return 2.0 * matmul_weights + attn


def attention_flop_fraction(cfg: dict, seq_len: int, causal: bool = True) -> float:
    """Share of forward FLOPs per token spent in the QK^T and AV matmuls."""
    total = forward_flops_per_token(cfg, seq_len, causal)
    return (total - forward_flops_per_token(cfg, 0, causal)) / total
