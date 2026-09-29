"""Reference solutions: stability primitives and multi-token prediction (page 11)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def rms_norm(x: Tensor, gain: Tensor | None = None, eps: float = 1e-6) -> Tensor:
    """x / sqrt(mean(x^2) + eps) * gain over the last axis, statistics in float32."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if gain is not None:
        y = y * gain.float()
    return y.to(x.dtype)


def qk_norm_attention(q: Tensor, k: Tensor, v: Tensor, g_q: Tensor, g_k: Tensor,
                      causal: bool = True, return_weights: bool = False):
    """Attention on RMS-normalized queries and keys (per head, gains shared across heads)."""
    d_h = q.shape[-1]
    q_hat = rms_norm(q, g_q)                          # (B, n_h, T, d_h): unit RMS times gain
    k_hat = rms_norm(k, g_k)
    scores = q_hat @ k_hat.transpose(-1, -2) / d_h**0.5   # |score| <= max|g_q g_k| * sqrt(d_h)
    if causal:
        T = q.shape[-2]
        mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = scores.softmax(-1)
    out = weights @ v
    return (out, weights) if return_weights else out


def z_loss(logits: Tensor, coef: float = 1e-4, mask: Tensor | None = None) -> Tensor:
    """coef * mean over positions of (logsumexp of the logits)^2, computed in float32."""
    log_z = torch.logsumexp(logits.float(), dim=-1)   # (...,)
    sq = log_z.pow(2)
    if mask is not None:
        return coef * sq[mask].mean()
    return coef * sq.mean()


def softcap(x: Tensor, cap: float) -> Tensor:
    """cap * tanh(x / cap); torch.tanh saturates cleanly, so huge inputs give exactly +-cap."""
    return cap * torch.tanh(x / cap)


def attention_entropy(attn: Tensor) -> Tensor:
    """-sum p log p with 0 log 0 = 0, safe for exact zeros and for backprop through a masked softmax."""
    # Clamp inside the log: the value is unchanged (p * log(tiny) = 0 when p = 0), and the gradient stays
    # finite. torch.special.entr has an infinite derivative at 0, and inf * 0 (softmax Jacobian) is nan.
    logp = torch.log(attn.clamp_min(torch.finfo(attn.dtype).tiny))
    return -(attn * logp).sum(-1)


class MTPDepth(nn.Module):
    """One sequential MTP depth: proj([RMSNorm(h_prev); RMSNorm(emb_next)]) then a block."""

    def __init__(self, d: int, block: nn.Module, eps: float = 1e-6):
        super().__init__()
        self.proj = nn.Linear(2 * d, d, bias=False)
        self.norm_h = nn.Parameter(torch.ones(d))
        self.norm_e = nn.Parameter(torch.ones(d))
        self.block = block
        self.eps = eps

    def forward(self, h_prev: Tensor, emb_next: Tensor) -> Tensor:
        both = torch.cat([rms_norm(h_prev, self.norm_h, self.eps),
                          rms_norm(emb_next, self.norm_e, self.eps)], dim=-1)   # (B, n, 2d)
        return self.block(self.proj(both))                                       # (B, n, d)


def mtp_loss(hidden: Tensor, targets: Tensor, depth_modules, embed: nn.Embedding,
             head: nn.Linear, lam: float = 0.3) -> Tensor:
    """lam / D * sum_k CE(head(h^k), x_{i+k+1}), where h^k chains through the depth modules."""
    B, T, _ = hidden.shape
    D = len(depth_modules)
    total = hidden.new_zeros(())
    h_prev = hidden
    for k, module in enumerate(depth_modules, start=1):
        n = T - k                                        # the last k positions have no k-ahead label
        emb_next = embed(targets[:, k - 1:k - 1 + n])    # Emb(x_{i+k}) = Emb(targets[i + k - 1])
        labels = targets[:, k:k + n]                     # x_{i+k+1}   = targets[i + k]
        h_k = module(h_prev[:, :n], emb_next)            # chain: input is the previous depth's output
        logits = head(h_k)                               # (B, n, V), shared head
        total = total + F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        h_prev = h_k
    return lam / D * total
