"""Reference solutions: QK-Clip from Kimi K2's MuonClip (page 32)."""
from __future__ import annotations

import torch
from torch import Tensor


def max_attention_logit(q: Tensor, k: Tensor, causal: bool = True) -> Tensor:
    """(H,) signed max of q_i . k_j / sqrt(d) over batch, i and (j <= i if causal)."""
    d = q.shape[-1]
    scores = torch.einsum("bhid,bhjd->bhij", q, k) / d**0.5          # (B, H, T, T)
    if causal:
        T = q.shape[-2]
        keep = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~keep, float("-inf"))            # masked pairs never reach softmax
    return scores.amax(dim=(0, 2, 3))                                # reduce over batch, i, j


def _gamma(max_logits: Tensor, tau: float) -> Tensor:
    """gamma_h = tau / S_max^h if S_max^h > tau else 1 (also 1 for zero or negative logits)."""
    over = max_logits > tau
    safe = torch.where(over, max_logits, torch.ones_like(max_logits))   # avoid dividing by <= 0
    return torch.where(over, tau / safe, torch.ones_like(max_logits))


def qk_clip(W_q_heads: Tensor, W_k_heads: Tensor, max_logits: Tensor, tau: float,
            alpha: float = 0.5) -> tuple[Tensor, Tensor, Tensor]:
    """W_q <- gamma^alpha W_q, W_k <- gamma^(1-alpha) W_k, per head; the logit scales by gamma."""
    if tau <= 0 or not 0.0 <= alpha <= 1.0:
        raise ValueError("need tau > 0 and 0 <= alpha <= 1")
    gamma = _gamma(max_logits.to(W_q_heads.dtype), tau)                  # (H,)
    sq = (gamma ** alpha).view(-1, 1, 1)                                 # broadcast over (D, d)
    sk = (gamma ** (1.0 - alpha)).view(-1, 1, 1)
    return W_q_heads * sq, W_k_heads * sk, gamma                         # new tensors, inputs untouched


def qk_clip_mla(W_qc: Tensor, W_kc: Tensor, W_qr: Tensor, W_kr: Tensor, max_logits: Tensor,
                tau: float) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """q^C and k^C by sqrt(gamma), q^R by gamma, shared k^R untouched: the whole logit scales by gamma."""
    if tau <= 0:
        raise ValueError("need tau > 0")
    gamma = _gamma(max_logits.to(W_qc.dtype), tau)
    root = gamma.sqrt().view(-1, 1, 1)
    full = gamma.view(-1, 1, 1)
    # content term: sqrt(g) * sqrt(g) = g ; rotary term: g * 1 = g (k^R is shared, so it is left alone)
    return W_qc * root, W_kc * root, W_qr * full, W_kr.clone(), gamma
