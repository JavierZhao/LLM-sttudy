"""Reference solutions: attention from scratch (page 03)."""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    m = x.amax(dim=dim, keepdim=True)
    # A slice that is entirely -inf has max = -inf, and (-inf) - (-inf) = nan. Use 0 as the shift there.
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    e = torch.exp(x - m)                        # all exponents <= 0, so e is in [0, 1]: no overflow
    s = e.sum(dim=dim, keepdim=True)
    # s == 0 only for fully masked slices, where e is all zeros; divide by 1 to return zeros
    return e / torch.where(s > 0, s, torch.ones_like(s))


def causal_mask(T: int, S: Optional[int] = None, device=None) -> Tensor:
    S = T if S is None else S
    if S < T:
        raise ValueError(f"need S >= T, got T={T}, S={S}")
    i = torch.arange(T, device=device)[:, None] + (S - T)   # absolute position of each query
    j = torch.arange(S, device=device)[None, :]
    return j <= i                                            # (T, S)


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    d = q.shape[-1]
    z = (q @ k.transpose(-1, -2)) / d**0.5                  # (..., T, S) logits
    if mask is not None:
        z = z.masked_fill(~mask, float("-inf"))              # block BEFORE the softmax so rows renormalize
    return softmax(z, dim=-1) @ v                            # (..., T, d_v); fully masked rows give zeros


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, causal: bool = True) -> Tensor:
        B, T, d = x.shape
        # (B, T, d) -> (B, T, n_h, d_h) -> (B, n_h, T, d_h): head h owns columns h*d_h ... (h+1)*d_h - 1
        split = lambda t: t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q, k, v = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))
        mask = causal_mask(T, device=x.device) if causal else None   # (T, T) broadcasts over (B, n_h)
        o = scaled_dot_product_attention(q, k, v, mask)              # (B, n_h, T, d_h)
        o = o.transpose(1, 2).reshape(B, T, d)                       # concat heads: (B, T, n_h * d_h)
        return self.o_proj(o)
