"""Reference solutions: positional encodings, RoPE in both layouts, sinusoidal and ALiBi (page 05)."""
import math

import torch
from torch import Tensor


def rope_frequencies(d_head: int, base: float = 10000.0) -> Tensor:
    """theta_i = base ** (-2 i / d_head), float64, shape (d_head // 2,)."""
    assert d_head % 2 == 0
    return base ** (-torch.arange(0, d_head, 2, dtype=torch.float64) / d_head)


def _cos_sin(x: Tensor, pos: Tensor, base: float) -> tuple[Tensor, Tensor]:
    """cos/sin tables broadcastable against the (..., d/2) halves or pairs of x."""
    inv = rope_frequencies(x.shape[-1], base)                       # (d/2,)
    ang = pos.to(torch.float64)[:, None] * inv[None, :]             # (T, d/2), computed in float64
    if x.dim() == 4:
        ang = ang[:, None, :]                                       # (T, 1, d/2): shared by all heads
    return ang.cos().to(x.dtype), ang.sin().to(x.dtype)             # cast only AFTER the trig


def apply_rope_interleaved(x: Tensor, pos: Tensor, base: float = 10000.0) -> Tensor:
    """Pairs (x[2i], x[2i+1]); rotation (a, b) -> (a cos - b sin, a sin + b cos)."""
    cos, sin = _cos_sin(x, pos, base)
    x1, x2 = x[..., 0::2], x[..., 1::2]                             # (..., d/2) each
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)   # (..., d/2, 2)
    return out.flatten(-2)                                          # re-interleave to (..., d)


def apply_rope_half(x: Tensor, pos: Tensor, base: float = 10000.0) -> Tensor:
    """Pairs (x[i], x[i + d/2]). Equivalent to x * cat(cos, cos) + rotate_half(x) * cat(sin, sin)."""
    cos, sin = _cos_sin(x, pos, base)
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


def interleaved_to_half_permutation(d_head: int) -> Tensor:
    """perm[s * (d/2) + i] = 2 i + s: all real parts first, then all imaginary parts."""
    return torch.cat((torch.arange(0, d_head, 2), torch.arange(1, d_head, 2)))


def apply_partial_rope(x: Tensor, pos: Tensor, rotary_dim: int, base: float = 10000.0,
                       interleaved: bool = False) -> Tensor:
    """Rotate x[..., :rotary_dim] as a rotary_dim-dimensional head, pass the rest through."""
    fn = apply_rope_interleaved if interleaved else apply_rope_half
    rot, rest = x[..., :rotary_dim], x[..., rotary_dim:]
    return torch.cat((fn(rot, pos, base), rest), dim=-1)            # frequencies use rotary_dim, not d_head


def sinusoidal_embedding(T: int, d: int, base: float = 10000.0) -> Tensor:
    """PE[pos, 2i] = sin(pos * theta_i), PE[pos, 2i+1] = cos(pos * theta_i), theta_i = base ** (-2i/d)."""
    ang = torch.arange(T, dtype=torch.float64)[:, None] * rope_frequencies(d, base)[None, :]   # (T, d/2)
    pe = torch.zeros(T, d, dtype=torch.float64)
    pe[:, 0::2] = ang.sin()
    pe[:, 1::2] = ang.cos()
    return pe.float()


def _slopes(n: int) -> list[float]:
    if math.log2(n).is_integer():
        start = 2.0 ** (-8.0 / n)                                   # 2^-8/n; the ratio equals the start
        return [start ** (i + 1) for i in range(n)]
    closest = 2 ** math.floor(math.log2(n))                         # largest power of two below n
    # borrow every other slope from the sequence for 2 * closest heads, as in the reference code
    return _slopes(closest) + _slopes(2 * closest)[0::2][: n - closest]


def alibi_slopes(n_heads: int) -> Tensor:
    return torch.tensor(_slopes(n_heads), dtype=torch.float32)


def alibi_bias(n_heads: int, T: int) -> Tensor:
    m = alibi_slopes(n_heads)                                       # (H,)
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    dist = (i - j).to(torch.float32)                                # (T, T): query index minus key index
    bias = -m[:, None, None] * dist                                 # (H, T, T)
    return bias.masked_fill(j > i, float("-inf"))                   # causal mask lives in the same tensor
