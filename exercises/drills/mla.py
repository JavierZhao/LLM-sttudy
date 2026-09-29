"""Drill (page 07): Multi-head Latent Attention.

Implement mla_forward, mla_decode_step and kv_cache_bytes_per_token.
MLAWeights, init_mla and rope are provided. Run:
    pytest exercises/tests/test_mla.py


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
    """MLA in 'MHA mode' (training / prefill), causal over the T input tokens.

    Compute c_kv = h W_dkv and k_r = rope(h W_kr); queries via c_q = h W_dq, then
    q_c = c_q W_uq (per head d_nope) and q_r = rope(c_q W_qr) (per head d_r). Decompress
    k_c = c_kv W_uk and v = c_kv W_uv per head. Each head's query is [q_c; q_r] and its key is
    [k_c; k_r] (k_r is shared by all heads). Scale scores by 1/sqrt(d_nope + d_r).
    Positions are start_pos, ..., start_pos + T - 1.

    h: (B, T, d). Returns (out (B, T, d), c_kv (B, T, d_c), k_r (B, T, d_r)),
    where the last two are exactly what an inference engine would cache.
    """
    raise NotImplementedError


def mla_decode_step(h_t: Tensor, w: MLAWeights, cache_c: Tensor, cache_kr: Tensor, pos: int
                    ) -> tuple[Tensor, Tensor, Tensor]:
    """One decoding step in absorbed 'MQA mode'.

    Append this token's latent and RoPE key to the caches, then attend WITHOUT
    decompressing any cached keys or values: fold W_uk into the query (per head,
    q_lat = q_c @ W_uk_h^T, shape d_c), score against the latent cache (plus the RoPE term
    q_r . k_r), take the attention-weighted sum of latents, and only then apply W_uv (per
    head) and W_o. Must match mla_forward's output for the same position.

    h_t: (B, 1, d); cache_c: (B, S, d_c); cache_kr: (B, S, d_r); pos: absolute position of h_t (= S).
    Returns (out (B, 1, d), new cache_c (B, S+1, d_c), new cache_kr (B, S+1, d_r)).
    """
    raise NotImplementedError


def kv_cache_bytes_per_token(kind: str, n_layers: int, n_h: int = 0, n_kv: int = 0, d_h: int = 0,
                             d_c: int = 0, d_r: int = 0, bytes_per_el: int = 2) -> int:
    """KV-cache bytes per token, summed over all layers.

    kind: 'mha' (n_h KV heads), 'gqa' (n_kv KV heads), 'mqa' (1 KV head), or
    'mla' (latent d_c plus decoupled RoPE key d_r). Example: Llama-3-8B
    ('gqa', 32 layers, n_kv=8, d_h=128, bf16) -> 131072.
    """
    raise NotImplementedError
