"""Reference solutions: FlashAttention from scratch (page 16)."""
from typing import Optional, Sequence

import torch
from torch import Tensor

_NEG_INF = float("-inf")


def _finite(m: Tensor) -> Tensor:
    """Replace -inf by 0 so that exp(x - m) is exp(-inf) = 0 instead of exp(-inf - -inf) = NaN."""
    return torch.where(torch.isinf(m), torch.zeros_like(m), m)


def merge_softmax_stats(m_a: Tensor, l_a: Tensor, m_b: Tensor, l_b: Tensor) -> tuple[Tensor, Tensor]:
    """Merge (max, sum-of-exp) statistics of two disjoint score sets: the associative operator of
    Milakov and Gimelshein (2018)."""
    m = torch.maximum(m_a, m_b)
    m_safe = _finite(m)
    # both factors are <= 1, so nothing overflows; an empty set (m = -inf, l = 0) contributes 0
    l = l_a * torch.exp(m_a - m_safe) + l_b * torch.exp(m_b - m_safe)
    return m, l


def online_softmax(x_blocks: Sequence[Tensor]) -> Tensor:
    """One pass over the blocks for (m, l), then normalize each block."""
    m = l = None
    for xb in x_blocks:
        mb = xb.amax(dim=-1)                                        # block max
        lb = torch.exp(xb - _finite(mb)[..., None]).sum(dim=-1)     # block sum of exp(x - block max)
        m, l = (mb, lb) if m is None else merge_softmax_stats(m, l, mb, lb)
    m_safe = _finite(m)[..., None]
    return torch.cat([torch.exp(xb - m_safe) / l[..., None] for xb in x_blocks], dim=-1)


def flash_attention_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    block_q: int = 32,
    block_k: int = 32,
    causal: bool = False,
    scale: Optional[float] = None,
) -> tuple[Tensor, Tensor]:
    """Tiled attention: outer loop over query tiles, inner loop over key tiles (FlashAttention-2 order)."""
    B, H, Tq, d = q.shape
    Tk = k.shape[2]
    assert not causal or Tq <= Tk
    scale = d ** -0.5 if scale is None else scale
    offset = Tk - Tq                                   # query i sits at absolute position offset + i
    acc_dtype = torch.promote_types(q.dtype, torch.float32)
    out = torch.empty_like(q)
    lse = torch.empty(B, H, Tq, dtype=acc_dtype, device=q.device)

    for q0 in range(0, Tq, block_q):
        q1 = min(q0 + block_q, Tq)
        qi = q[:, :, q0:q1].to(acc_dtype) * scale       # (B, H, bq, d), scale folded into q
        qpos = torch.arange(q0, q1, device=q.device) + offset
        m = torch.full((B, H, q1 - q0), _NEG_INF, dtype=acc_dtype, device=q.device)   # running max
        l = torch.zeros(B, H, q1 - q0, dtype=acc_dtype, device=q.device)              # running sum
        acc = torch.zeros(B, H, q1 - q0, d, dtype=acc_dtype, device=q.device)         # running P @ V (unnormalized)

        for k0 in range(0, Tk, block_k):
            if causal and k0 > int(qpos[-1]):
                break                                    # this and all later tiles are fully masked
            k1 = min(k0 + block_k, Tk)
            s = qi @ k[:, :, k0:k1].to(acc_dtype).transpose(-1, -2)   # (B, H, bq, bk): the only score tile
            if causal and k1 - 1 > int(qpos[0]):         # tile straddles the diagonal: mask it
                kpos = torch.arange(k0, k1, device=q.device)
                s = s.masked_fill(kpos[None, :] > qpos[:, None], _NEG_INF)
            m_new = torch.maximum(m, s.amax(dim=-1))
            m_safe = _finite(m_new)
            alpha = torch.exp(m - m_safe)                # rescales everything accumulated so far (<= 1)
            p = torch.exp(s - m_safe[..., None])         # unnormalized probabilities of this tile
            l = l * alpha + p.sum(dim=-1)
            acc = acc * alpha[..., None] + p @ v[:, :, k0:k1].to(acc_dtype)
            m = m_new

        out[:, :, q0:q1] = (acc / l[..., None]).to(q.dtype)      # normalize once, at the end
        lse[:, :, q0:q1] = m + torch.log(l)                      # all the backward pass needs
    return out, lse


def flash_attention_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    lse: Tensor,
    dout: Tensor,
    block_q: int = 32,
    block_k: int = 32,
    causal: bool = False,
    scale: Optional[float] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Backward with recomputation: P = exp(S - lse) is rebuilt tile by tile (FlashAttention-2, Alg. 2)."""
    B, H, Tq, d = q.shape
    Tk = k.shape[2]
    scale = d ** -0.5 if scale is None else scale
    offset = Tk - Tq
    acc_dtype = torch.promote_types(q.dtype, torch.float32)
    dq = torch.zeros(B, H, Tq, d, dtype=acc_dtype, device=q.device)
    dk = torch.zeros(B, H, Tk, d, dtype=acc_dtype, device=q.device)
    dv = torch.zeros(B, H, Tk, d, dtype=acc_dtype, device=q.device)
    # D_i = sum_j P_ij dP_ij = dO_i . O_i: a length-d dot product instead of a length-T reduction
    D = (dout.to(acc_dtype) * out.to(acc_dtype)).sum(dim=-1)              # (B, H, Tq)

    for k0 in range(0, Tk, block_k):                                       # outer loop over key tiles
        k1 = min(k0 + block_k, Tk)
        kj = k[:, :, k0:k1].to(acc_dtype)
        vj = v[:, :, k0:k1].to(acc_dtype)
        for q0 in range(0, Tq, block_q):
            q1 = min(q0 + block_q, Tq)
            qpos = torch.arange(q0, q1, device=q.device) + offset
            if causal and k0 > int(qpos[-1]):
                continue
            qi = q[:, :, q0:q1].to(acc_dtype)
            doi = dout[:, :, q0:q1].to(acc_dtype)
            s = (qi @ kj.transpose(-1, -2)) * scale                        # recompute the score tile
            if causal and k1 - 1 > int(qpos[0]):
                kpos = torch.arange(k0, k1, device=q.device)
                s = s.masked_fill(kpos[None, :] > qpos[:, None], _NEG_INF)
            p = torch.exp(s - lse[:, :, q0:q1, None].to(acc_dtype))       # exact probabilities, no rescaling
            dv[:, :, k0:k1] += p.transpose(-1, -2) @ doi
            dp = doi @ vj.transpose(-1, -2)
            ds = p * (dp - D[:, :, q0:q1, None])                          # softmax Jacobian, row-wise
            dq[:, :, q0:q1] += (ds @ kj) * scale
            dk[:, :, k0:k1] += (ds.transpose(-1, -2) @ qi) * scale
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


def combine_partial_attention(outs: Sequence[Tensor], lses: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
    """FlashDecoding reduction: weight each chunk's output by exp(lse_s - lse)."""
    lse_all = torch.stack(list(lses), dim=0)                       # (S, ..., T)
    m = lse_all.amax(dim=0)
    lse = m + torch.log(torch.exp(lse_all - m).sum(dim=0))         # logsumexp over chunks
    w = torch.exp(lse_all - lse)                                   # (S, ..., T), sums to 1 over S
    out = (torch.stack(list(outs), dim=0) * w[..., None]).sum(dim=0)
    return out, lse


def split_kv_attention(
    q: Tensor, k: Tensor, v: Tensor, num_splits: int, scale: Optional[float] = None
) -> Tensor:
    """Each chunk of K/V is an independent job (on a GPU: its own thread block); then one small reduction."""
    ks = torch.tensor_split(k, num_splits, dim=2)
    vs = torch.tensor_split(v, num_splits, dim=2)
    parts = [flash_attention_forward(q, kc, vc, causal=False, scale=scale) for kc, vc in zip(ks, vs)]
    out, _ = combine_partial_attention([p[0] for p in parts], [p[1] for p in parts])
    return out.to(q.dtype)
