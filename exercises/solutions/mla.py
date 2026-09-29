"""Reference solutions: Multi-head Latent Attention (page 07).

Row-vector convention: activations are (..., d) and linear maps are `x @ W` with W: (in, out).
DeepSeek-V2/V3 also apply RMSNorm to the two latents (c^Q and c^KV); it is omitted here
because it does not change the algebra of caching and absorption.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


# ----------------------------------------------------------------------------- provided
@dataclass
class MLAWeights:
    W_dq: Tensor   # (d, d_cq)             query down-projection
    W_uq: Tensor   # (d_cq, n_h * d_nope)  query up-projection (content part)
    W_qr: Tensor   # (d_cq, n_h * d_r)     decoupled RoPE queries (one per head)
    W_dkv: Tensor  # (d, d_c)              KV down-projection -> the cached latent
    W_kr: Tensor   # (d, d_r)              decoupled RoPE key (ONE, shared by all heads)
    W_uk: Tensor   # (d_c, n_h * d_nope)   key up-projection
    W_uv: Tensor   # (d_c, n_h * d_v)      value up-projection
    W_o: Tensor    # (n_h * d_v, d)        output projection
    n_h: int
    d_nope: int
    d_r: int
    d_v: int


def init_mla(d=64, n_h=4, d_nope=16, d_r=8, d_v=16, d_c=24, d_cq=32, dtype=torch.float64, seed=0) -> MLAWeights:
    g = torch.Generator().manual_seed(seed)
    def w(i, o):
        return torch.randn(i, o, generator=g, dtype=dtype) / i**0.5
    return MLAWeights(w(d, d_cq), w(d_cq, n_h * d_nope), w(d_cq, n_h * d_r), w(d, d_c), w(d, d_r),
                      w(d_c, n_h * d_nope), w(d_c, n_h * d_v), w(n_h * d_v, d), n_h, d_nope, d_r, d_v)


def rope(x: Tensor, pos: Tensor, base: float = 10000.0) -> Tensor:
    """Rotate consecutive pairs (x[..., 2i], x[..., 2i+1]) by angle pos * base**(-2i/d).

    x: (B, T, d) or (B, T, H, d); pos: (T,) absolute positions.
    """
    d = x.shape[-1]
    inv = base ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)       # (d/2,)
    ang = pos.to(torch.float64)[:, None] * inv[None, :]                  # (T, d/2)
    if x.dim() == 4:
        ang = ang[:, None, :]                                            # broadcast over heads
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


# ----------------------------------------------------------------------------- drills
def mla_forward(h: Tensor, w: MLAWeights, start_pos: int = 0) -> tuple[Tensor, Tensor, Tensor]:
    """MLA in 'MHA mode' (training / prefill): decompress per-head keys and values.

    h: (B, T, d). Returns (out (B, T, d), c_kv (B, T, d_c), k_r (B, T, d_r)).
    """
    B, T, _ = h.shape
    n_h, d_nope, d_r, d_v = w.n_h, w.d_nope, w.d_r, w.d_v
    pos = torch.arange(start_pos, start_pos + T)

    c_kv = h @ w.W_dkv                                        # (B, T, d_c)   cached
    k_r = rope(h @ w.W_kr, pos)                               # (B, T, d_r)   cached
    c_q = h @ w.W_dq                                          # (B, T, d_cq)
    q_c = (c_q @ w.W_uq).view(B, T, n_h, d_nope)
    q_r = rope((c_q @ w.W_qr).view(B, T, n_h, d_r), pos)
    k_c = (c_kv @ w.W_uk).view(B, T, n_h, d_nope)             # decompressed keys
    v = (c_kv @ w.W_uv).view(B, T, n_h, d_v)                  # decompressed values

    q = torch.cat([q_c, q_r], dim=-1).transpose(1, 2)                            # (B, n_h, T, d_nope+d_r)
    k = torch.cat([k_c, k_r[:, :, None, :].expand(B, T, n_h, d_r)], dim=-1).transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v.transpose(1, 2), is_causal=True)   # scale 1/sqrt(d_nope+d_r)
    out = o.transpose(1, 2).reshape(B, T, n_h * d_v) @ w.W_o
    return out, c_kv, k_r


def mla_decode_step(h_t: Tensor, w: MLAWeights, cache_c: Tensor, cache_kr: Tensor, pos: int
                    ) -> tuple[Tensor, Tensor, Tensor]:
    """One decoding step in absorbed 'MQA mode': never decompress cached keys/values.

    h_t: (B, 1, d); cache_c: (B, S, d_c); cache_kr: (B, S, d_r); pos: absolute position of h_t.
    Returns (out (B, 1, d), new cache_c (B, S+1, d_c), new cache_kr (B, S+1, d_r)).
    """
    B = h_t.shape[0]
    n_h, d_nope, d_r, d_v = w.n_h, w.d_nope, w.d_r, w.d_v
    d_c = w.W_dkv.shape[1]
    p = torch.tensor([pos])

    # 1) append this token's cache entries: the only per-token state MLA keeps
    cache_c = torch.cat([cache_c, h_t @ w.W_dkv], dim=1)                 # (B, S+1, d_c)
    cache_kr = torch.cat([cache_kr, rope(h_t @ w.W_kr, p)], dim=1)       # (B, S+1, d_r)

    # 2) queries for the new token
    c_q = h_t @ w.W_dq                                                   # (B, 1, d_cq)
    q_c = (c_q @ w.W_uq).view(B, n_h, d_nope)                            # (B, n_h, d_nope)
    q_r = rope((c_q @ w.W_qr).view(B, 1, n_h, d_r), p).view(B, n_h, d_r)

    # 3) absorb W_uk into the query: q_lat[h] = q_c[h] @ W_uk[h]^T lives in latent space
    W_uk = w.W_uk.view(d_c, n_h, d_nope)
    q_lat = torch.einsum("bhn,chn->bhc", q_c, W_uk)                      # (B, n_h, d_c)

    # 4) MQA over the latent cache: shared key [c; k_r] (d_c+d_r), shared value c (d_c)
    scores = (torch.einsum("bhc,bsc->bhs", q_lat, cache_c)
              + torch.einsum("bhr,bsr->bhs", q_r, cache_kr)) / (d_nope + d_r) ** 0.5
    a = scores.softmax(dim=-1)                                           # (B, n_h, S+1), no mask needed
    o_lat = torch.einsum("bhs,bsc->bhc", a, cache_c)                     # (B, n_h, d_c)

    # 5) absorb W_uv on the way out, then the output projection
    W_uv = w.W_uv.view(d_c, n_h, d_v)
    o = torch.einsum("bhc,chv->bhv", o_lat, W_uv)                        # (B, n_h, d_v)
    out = o.reshape(B, 1, n_h * d_v) @ w.W_o
    return out, cache_c, cache_kr


def kv_cache_bytes_per_token(kind: str, n_layers: int, n_h: int = 0, n_kv: int = 0, d_h: int = 0,
                             d_c: int = 0, d_r: int = 0, bytes_per_el: int = 2) -> int:
    """KV-cache bytes per token summed over layers for kind in {'mha','gqa','mqa','mla'}."""
    per_layer = {
        "mha": 2 * n_h * d_h,
        "gqa": 2 * n_kv * d_h,
        "mqa": 2 * d_h,
        "mla": d_c + d_r,
    }[kind]
    return per_layer * n_layers * bytes_per_el
