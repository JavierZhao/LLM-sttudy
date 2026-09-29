"""Reference solutions for page 09 (beyond full attention)."""
import math

import torch
import torch.nn.functional as F
from torch import Tensor


def sliding_window_mask(T: int, window: int) -> Tensor:
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    # causal (j <= i) and within the last `window` positions (j > i - window)
    return (j <= i) & (j > i - window)


def sliding_window_attention(q: Tensor, k: Tensor, v: Tensor, window: int) -> Tensor:
    T = q.shape[-2]
    mask = sliding_window_mask(T, window).to(q.device)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # (B, H, T, T)
    scores = scores.masked_fill(~mask, float("-inf"))  # row i always keeps j = i, so no all -inf rows
    return scores.softmax(-1) @ v


def _phi(x: Tensor) -> Tensor:
    return F.elu(x) + 1.0  # strictly positive feature map


def linear_attention_parallel(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    T = q.shape[-2]
    fq, fk = _phi(q), _phi(k)
    scores = fq @ fk.transpose(-2, -1)  # (B, H, T, T), all positive, no softmax
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
    scores = scores.masked_fill(~causal, 0.0)
    num = scores @ v
    den = scores.sum(-1, keepdim=True)  # normalizer: sum of the same weights
    return num / den


def linear_attention_recurrent(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    B, H, T, d_k = q.shape
    d_v = v.shape[-1]
    fq, fk = _phi(q), _phi(k)
    S = q.new_zeros(B, H, d_k, d_v)  # state: sum of phi(k_s)^T v_s
    z = q.new_zeros(B, H, d_k)  # normalizer state: sum of phi(k_s)
    out = []
    for t in range(T):
        S = S + fk[:, :, t, :, None] * v[:, :, t, None, :]  # outer product (d_k, d_v)
        z = z + fk[:, :, t]
        num = (fq[:, :, t, :, None] * S).sum(-2)  # phi(q_t) S_t: (B, H, d_v)
        den = (fq[:, :, t] * z).sum(-1, keepdim=True)  # phi(q_t) . z_t: (B, H, 1)
        out.append(num / den)
    return torch.stack(out, dim=2)


def delta_rule_recurrent(q: Tensor, k: Tensor, v: Tensor, beta: Tensor) -> Tensor:
    B, H, T, d_k = q.shape
    d_v = v.shape[-1]
    S = q.new_zeros(B, H, d_k, d_v)
    out = []
    for t in range(T):
        kt = k[:, :, t]  # (B, H, d_k)
        pred = (kt[:, :, :, None] * S).sum(-2)  # k_t S_{t-1}: what the memory currently returns for k_t
        err = v[:, :, t] - pred  # (B, H, d_v): the correction (the "delta")
        S = S + beta[:, :, t, None, None] * kt[:, :, :, None] * err[:, :, None, :]
        out.append((q[:, :, t, :, None] * S).sum(-2))
    return torch.stack(out, dim=2)


def topk_sparse_attention(q: Tensor, k: Tensor, v: Tensor, index_scores: Tensor, k_top: int) -> Tensor:
    B, H, T, d = q.shape
    pos = torch.arange(T, device=q.device)
    future = pos[None, :] > pos[:, None]  # (T, T): True where key s > query t
    masked = index_scores.masked_fill(future, float("-inf"))
    kk = min(k_top, T)
    idx = masked.topk(kk, dim=-1).indices  # (B, T, kk) selected key positions per query
    valid = idx <= pos[None, :, None]  # early queries have < kk real keys; the rest are future picks
    # gather selected keys/values: (B, H, T, kk, d)
    gidx = idx[:, None, :, :, None].expand(B, H, T, kk, d)
    kg = k[:, :, None].expand(B, H, T, T, d).gather(3, gidx)
    vg = v[:, :, None].expand(B, H, T, T, d).gather(3, gidx)
    scores = (q[:, :, :, None, :] * kg).sum(-1) / math.sqrt(d)  # (B, H, T, kk)
    scores = scores.masked_fill(~valid[:, None], float("-inf"))
    return (scores.softmax(-1)[..., None] * vg).sum(-2)
