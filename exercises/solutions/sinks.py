"""Reference solutions: attention sinks, banded masks, hybrid schedules and KV bytes, MXFP4 (page 31)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class HybridCfg:
    n_layers: int
    pattern: str
    window: int
    local_kv_elems: int
    global_kv_elems: int
    bytes_per_elem: int = 2


def softmax_with_sink(scores: Tensor, sink_logit: Tensor | float) -> Tensor:
    """p_j = exp(s_j) / (sum_k exp(s_k) + exp(z_sink)); rows sum to 1 - p_sink."""
    sink = torch.as_tensor(sink_logit, dtype=scores.dtype, device=scores.device)
    sink = sink.expand(*scores.shape[:-1], 1)                    # one extra column per row
    z = torch.cat([scores, sink], dim=-1)
    # torch.softmax subtracts the row max (stable). With a finite sink an all -inf score row
    # has max = sink, so real columns get exp(-inf) = 0 and no NaN appears.
    return torch.softmax(z, dim=-1)[..., :-1]


def banded_causal_mask(T: int, window: int | None, S: int | None = None) -> Tensor:
    S = T if S is None else S
    assert S >= T
    q = torch.arange(T)[:, None] + (S - T)          # absolute query positions (end-aligned)
    j = torch.arange(S)[None, :]
    mask = j <= q
    if window is not None:
        mask &= (q - j) < window                     # `window` keys, current token included
    return mask


def layer_schedule(n_layers: int, pattern: str) -> list[str]:
    if n_layers < 1 or not pattern or set(pattern) - {"L", "G"}:
        raise ValueError("need n_layers >= 1 and a non-empty pattern over {'L', 'G'}")
    return [pattern[i % len(pattern)] for i in range(n_layers)]


def kv_bytes_hybrid(cfg: HybridCfg, T: int) -> int:
    total = 0
    for kind in layer_schedule(cfg.n_layers, cfg.pattern):
        if kind == "G":
            total += T * cfg.global_kv_elems
        else:
            total += min(T, cfg.window) * cfg.local_kv_elems
    return total * cfg.bytes_per_elem


def sink_attention(q: Tensor, k: Tensor, v: Tensor, sinks: Tensor, window: int | None = None) -> Tensor:
    B, n_h, T, d = q.shape
    n_kv, S = k.shape[1], k.shape[2]
    g = n_h // n_kv
    k = k.repeat_interleave(g, dim=1)                # (B, n_h, S, d)
    v = v.repeat_interleave(g, dim=1)
    scores = (q @ k.transpose(-1, -2)) / d**0.5      # (B, n_h, T, S)
    mask = banded_causal_mask(T, window, S).to(q.device)
    scores = scores.masked_fill(~mask, float("-inf"))
    p = softmax_with_sink(scores, sinks.view(1, n_h, 1, 1))
    return p @ v                                      # sink column has a zero value vector


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def mxfp4_quantize(x: Tensor, block: int = 32) -> Tensor:
    assert x.shape[-1] % block == 0
    shape = x.shape
    xb = x.reshape(-1, block).to(torch.float32)
    amax = xb.abs().amax(dim=-1, keepdim=True)
    safe = torch.where(amax > 0, amax, torch.ones_like(amax))
    shared_exp = torch.floor(torch.log2(safe)) - 2                     # emax of E2M1 is 2 (6 = 1.5 * 2^2)
    shared_exp = shared_exp.clamp(-127, 127)                            # E8M0 range
    scale = torch.pow(2.0, shared_exp)
    grid = torch.tensor(_E2M1, dtype=torch.float32)
    y = (xb / scale).abs().clamp(max=6.0)                               # saturate, do not round up past 6
    idx = (y[..., None] - grid).abs().argmin(dim=-1)                    # nearest grid magnitude (ties: lower)
    q = grid[idx] * torch.sign(xb)
    out = q * scale
    out = torch.where(amax > 0, out, torch.zeros_like(out))
    return out.reshape(shape).to(x.dtype)


def weight_bytes_mxfp4(n_mxfp4_params: float, n_bf16_params: float, block: int = 32,
                       scale_bits: int = 8, elem_bits: int = 4) -> float:
    bits_per_mx = elem_bits + scale_bits / block
    return n_mxfp4_params * bits_per_mx / 8 + n_bf16_params * 2
