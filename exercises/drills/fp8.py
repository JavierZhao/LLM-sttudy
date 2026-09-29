"""Drill (page 27): DeepSeek-V3-style FP8 quantization and GEMM, simulated on CPU.

Implement the functions below. Run:
    pytest exercises/tests/test_fp8.py

Design choice: FP8 is *simulated*. A "quantized" tensor is a float32 tensor whose entries are
exactly representable in E4M3 (1 sign, 4 exponent, 3 mantissa bits, bias 7, maximum 448, smallest
normal 2**-6, smallest subnormal 2**-9, no infinities), plus the scales needed to undo the
scaling. torch.float8_e4m3fn exists on CPU in recent PyTorch and the tests use it as an oracle,
but writing the rounding yourself is the point of the drill, so do not call it in your solution.

Conventions (row vectors, as in the course): activations A are (M, K) and weights W are (K, N),
so the layer computes A @ W. The scale of a group is amax / 448, so q = round_to_e4m3(x / scale)
and x is approximated by q * scale (an all-zero group gets scale 1.0).
"""
from __future__ import annotations

import torch
from torch import Tensor

E4M3_MAX = 448.0     # largest finite E4M3 magnitude (S.1111.110 = 1.75 * 2**8)
E4M3_MIN_EXP = -6    # unbiased exponent of the smallest normal number; below it, spacing is fixed at 2**-9


def round_to_bits(x: Tensor, bits: int) -> Tensor:
    """Round x to `bits` significant bits (including the implicit leading 1), ties to even.

    There is no exponent limit: only the significand is rounded. Zero stays zero and the sign is
    preserved. round_to_bits(x, 24) is the identity on float32, and round_to_bits(x, 8) equals
    x.to(torch.bfloat16).float() for values in the normal range.

    Args:
        x: float32 tensor of any shape.
        bits: number of significant bits kept (>= 1).
    Returns:
        float32 tensor, same shape.
    """
    raise NotImplementedError


def round_to_e4m3(x: Tensor) -> Tensor:
    """Round a float32 tensor to the nearest E4M3 value (ties to even), saturating at +-448.

    Values beyond +-448 clamp to +-448 (they do not become NaN or infinity). Below the smallest
    normal 2**-6 the format has subnormals spaced 2**-9 apart, so tiny values round to a multiple
    of 2**-9 and values under 2**-10 flush to zero.

    Args:
        x: float32 tensor of any shape.
    Returns:
        float32 tensor, same shape, holding exactly representable E4M3 values.
    """
    raise NotImplementedError


def quantize_per_tensor(x: Tensor) -> tuple[Tensor, Tensor]:
    """The baseline: one scale for the whole tensor, chosen so that amax maps to 448.

    Args:
        x: float32 tensor of any shape.
    Returns:
        (q, scale): q has x's shape and holds E4M3 values; scale is a 0-dim float32 tensor.
    """
    raise NotImplementedError


def quantize_tiles(x: Tensor, tile: tuple[int, int] = (1, 128)) -> tuple[Tensor, Tensor]:
    """Group-wise quantization with one scale per (tile[0] x tile[1]) group of a 2-D tensor.

    DeepSeek-V3 quantizes activations in 1x128 tiles: one scale per token per 128 channels.

    Args:
        x: float32 tensor of shape (R, C), with R % tile[0] == 0 and C % tile[1] == 0.
        tile: (rows, cols) of one scaling group.
    Returns:
        (q, scales): q has shape (R, C); scales has shape (R // tile[0], C // tile[1]).
    """
    raise NotImplementedError


def quantize_blocks(w: Tensor, block: tuple[int, int] = (128, 128)) -> tuple[Tensor, Tensor]:
    """Block-wise quantization for weights: one scale per 128 x 128 block.

    Same contract as quantize_tiles (same shapes for q and scales), with weights' default block.
    """
    raise NotImplementedError


def dequantize(q: Tensor, scales: Tensor, group: tuple[int, int] | None = None) -> Tensor:
    """Undo quantization: q times the scale of the group each element belongs to.

    Args:
        q: (R, C) E4M3 values (or any shape when group is None).
        scales: 0-dim tensor when group is None (per-tensor), else (R // group[0], C // group[1]).
        group: (rows, cols) of one scaling group, or None for per-tensor scaling.
    Returns:
        float32 tensor with q's shape.
    """
    raise NotImplementedError


def tensor_core_mma(a: Tensor, b: Tensor, acc_bits: int | None = None) -> Tensor:
    """Model of a tensor-core accumulation chain: sum_k a[:, k] * b[k, :] in a narrow accumulator.

    With acc_bits=None return the exact float32 product a @ b. Otherwise process k = 0, 1, ...
    in order, and after adding each rank-1 term a[:, k:k+1] * b[k:k+1, :] to the running sum, round
    the running sum to `acc_bits` significant bits with round_to_bits. This mimics the Hopper FP8
    tensor core, which the V3 report says keeps only about 14 bits when accumulating (a model,
    not a bit-exact emulation: real hardware aligns and truncates).

    Args:
        a: (M, k) float32, b: (k, N) float32. acc_bits: significand bits of the accumulator.
    Returns:
        (M, N) float32.
    """
    raise NotImplementedError


def fp8_matmul(a_q: Tensor, a_scales: Tensor, w_q: Tensor, w_scales: Tensor,
               promote_every: int = 128, acc_bits: int | None = None) -> Tensor:
    """FP8 GEMM with fine-grained scales and periodic promotion to FP32 (DeepSeek-V3, Fig. 7b).

    The K dimension is cut into chunks of `promote_every` elements. For each chunk, the "tensor
    core" computes the partial product of the raw E4M3 values (tensor_core_mma with acc_bits).
    The partial sum is then promoted: multiplied by the activation-tile scale of its row and the
    weight-block scale of its column, and added into an FP32 accumulator (ordinary float32 adds).
    Scales are constant within a 128-wide group along K, so promote_every must divide 128.

    Args:
        a_q: (M, K) E4M3 values with a_scales (M, K // 128) from quantize_tiles(a, (1, 128)).
        w_q: (K, N) E4M3 values with w_scales (K // 128, N // 128) from quantize_blocks(w).
             K and N are multiples of 128.
        promote_every: chunk length; raise ValueError unless 128 % promote_every == 0.
        acc_bits: significand bits of the tensor-core accumulator (None = exact within a chunk).
    Returns:
        (M, N) float32 approximation of dequantize(a) @ dequantize(w).
    """
    raise NotImplementedError
