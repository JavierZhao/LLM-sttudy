"""Reference solutions: grouped-query attention (page 07)."""
import torch
from torch import Tensor


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """(B, n_kv, S, d) -> (B, n_kv * n_rep, S, d); output head h uses KV head h // n_rep."""
    if n_rep == 1:
        return x
    B, n_kv, S, d = x.shape
    # expand is free (a view); reshape then copies once. Fused kernels skip the copy
    # entirely by indexing KV head h // n_rep inside the kernel.
    return x[:, :, None].expand(B, n_kv, n_rep, S, d).reshape(B, n_kv * n_rep, S, d)


def gqa_attention(q: Tensor, k: Tensor, v: Tensor, causal: bool = True) -> Tensor:
    """Grouped-query attention with an end-aligned causal mask.

    q: (B, n_h, T, d_h); k, v: (B, n_kv, S, d_h) with S >= T. The T queries are the
    LAST T positions of the S-long sequence, so query i sits at absolute position S - T + i.
    """
    B, n_h, T, d_h = q.shape
    n_kv, S = k.shape[1], k.shape[2]
    assert n_h % n_kv == 0 and S >= T
    k = repeat_kv(k, n_h // n_kv)
    v = repeat_kv(v, n_h // n_kv)
    scores = (q @ k.transpose(-1, -2)) / d_h**0.5                # (B, n_h, T, S)
    if causal:
        qpos = torch.arange(T, device=q.device)[:, None] + (S - T)
        kpos = torch.arange(S, device=q.device)[None, :]
        scores = scores.masked_fill(kpos > qpos, float("-inf"))
    return scores.softmax(dim=-1) @ v                             # (B, n_h, T, d_h)


def mha_to_gqa_kv_weight(W: Tensor, n_h: int, n_kv: int) -> Tensor:
    """Mean-pool an MHA key (or value) projection into a GQA one.

    W: (d, n_h * d_h), column block h belongs to head h. Returns (d, n_kv * d_h) where
    KV head j is the mean of original heads j*g .. j*g + g - 1 (g = n_h // n_kv).
    """
    d, total = W.shape
    d_h = total // n_h
    g = n_h // n_kv
    return W.view(d, n_kv, g, d_h).mean(dim=2).reshape(d, n_kv * d_h)
