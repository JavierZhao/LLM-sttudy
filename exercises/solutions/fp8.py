"""Reference solution (page 27): DeepSeek-V3-style FP8 quantization and GEMM, simulated on CPU."""
from __future__ import annotations

import torch
from torch import Tensor

E4M3_MAX = 448.0     # largest finite E4M3 magnitude (S.1111.110 = 1.75 * 2**8)
E4M3_MIN_EXP = -6    # unbiased exponent of the smallest normal number; below it, spacing is fixed at 2**-9


def round_to_bits(x: Tensor, bits: int) -> Tensor:
    m, e = torch.frexp(x)                       # x = m * 2**e with |m| in [0.5, 1) (m = 0 for x = 0)
    m = torch.round(m * 2.0**bits) / 2.0**bits  # `bits` significant bits; torch.round ties to even
    return torch.ldexp(m, e)                    # a carry (m -> 1.0) is fine: 1.0 * 2**e is exact


def round_to_e4m3(x: Tensor) -> Tensor:
    ax = x.abs().clamp(max=E4M3_MAX)                                  # saturate instead of NaN/inf
    e = torch.floor(torch.log2(ax)).clamp(min=E4M3_MIN_EXP)           # binade; clamp gives subnormal spacing
    step = torch.exp2(e - 3)                                          # 3 mantissa bits: 8 steps per binade
    return torch.sign(x) * torch.round(ax / step) * step              # round half to even; log2(0) = -inf -> clamped


def _scale_from_amax(amax: Tensor) -> Tensor:
    return torch.where(amax > 0, amax / E4M3_MAX, torch.ones_like(amax))   # all-zero group: scale 1


def quantize_per_tensor(x: Tensor) -> tuple[Tensor, Tensor]:
    scale = _scale_from_amax(x.abs().max())
    return round_to_e4m3(x / scale), scale


def quantize_tiles(x: Tensor, tile: tuple[int, int] = (1, 128)) -> tuple[Tensor, Tensor]:
    R, C = x.shape
    gr, gc = tile
    assert R % gr == 0 and C % gc == 0, "shape must be divisible by the group shape"
    amax = x.abs().reshape(R // gr, gr, C // gc, gc).amax(dim=(1, 3))     # (R/gr, C/gc): max |x| per group
    scales = _scale_from_amax(amax)
    return round_to_e4m3(x / _expand(scales, tile)), scales


def quantize_blocks(w: Tensor, block: tuple[int, int] = (128, 128)) -> tuple[Tensor, Tensor]:
    return quantize_tiles(w, block)


def _expand(scales: Tensor, group: tuple[int, int]) -> Tensor:
    # broadcast each group's scale over the elements of the group
    return scales.repeat_interleave(group[0], dim=0).repeat_interleave(group[1], dim=1)


def dequantize(q: Tensor, scales: Tensor, group: tuple[int, int] | None = None) -> Tensor:
    if group is None:
        return q * scales
    return q * _expand(scales, group)


def tensor_core_mma(a: Tensor, b: Tensor, acc_bits: int | None = None) -> Tensor:
    if acc_bits is None:
        return a @ b
    acc = torch.zeros(a.shape[0], b.shape[1], dtype=torch.float32)
    for k in range(a.shape[1]):
        # every add is rounded to the accumulator width, so small terms vanish once the sum is large
        acc = round_to_bits(acc + torch.outer(a[:, k], b[k, :]), acc_bits)
    return acc


def fp8_matmul(a_q: Tensor, a_scales: Tensor, w_q: Tensor, w_scales: Tensor,
               promote_every: int = 128, acc_bits: int | None = None) -> Tensor:
    if 128 % promote_every != 0:
        raise ValueError("promote_every must divide 128 so each chunk sits inside one scale group")
    M, K = a_q.shape
    N = w_q.shape[1]
    out = torch.zeros(M, N, dtype=torch.float32)                           # the FP32 registers on CUDA cores
    for c0 in range(0, K, promote_every):
        g = c0 // 128                                                      # scale group along K
        partial = tensor_core_mma(a_q[:, c0:c0 + promote_every], w_q[c0:c0 + promote_every], acc_bits)
        col_scale = w_scales[g].repeat_interleave(128)                     # (N,) one scale per 128 output columns
        out = out + partial * a_scales[:, g:g + 1] * col_scale[None, :]    # promote: dequantize and add in FP32
    return out


def fp8_linear(a: Tensor, w: Tensor, promote_every: int = 128, acc_bits: int | None = None) -> Tensor:
    a_q, a_s = quantize_tiles(a, (1, 128))          # activations: per token, per 128 channels
    w_q, w_s = quantize_blocks(w, (128, 128))       # weights: per 128 x 128 block
    return fp8_matmul(a_q, a_s, w_q, w_s, promote_every, acc_bits)
