"""Drill (page 20): a LoRA linear layer, NF4 block quantization, and a QLoRA forward pass.

Implement everything marked `raise NotImplementedError`. Run:
    pytest exercises/tests/test_lora.py

Keep this file self-contained: do not import other drills.

Conventions. PyTorch stores a Linear weight as W with shape (d_out, d_in) and computes
y = x @ W.T + bias for row-vector inputs x. LoRA is written in the same layout:
A has shape (r, d_in), B has shape (d_out, r), and the adapted weight is W + (alpha / r) * B @ A.
"""
import math
from typing import List, Tuple

import torch
from torch import Tensor, nn

# The 16 NormalFloat-4 levels, sorted, normalized to [-1, 1].
# Source: QLoRA paper (Dettmers et al. 2023), Appendix E; identical to the bitsandbytes NF4 table.
NF4_LEVELS = torch.tensor(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ]
)


class LoRALinear(nn.Module):
    """A frozen nn.Linear plus a trainable low-rank update, with merge / unmerge.

    Args:
        base: the pretrained layer, W of shape (d_out, d_in) and an optional bias. Keep a reference
              to it as `self.base` (do not copy) and freeze all of its parameters
              (requires_grad False).
        r: LoRA rank, r >= 1.
        alpha: scaling numerator. The update is scaled by `self.scaling = alpha / r`.

    Attributes the tests read:
        self.base, self.r, self.alpha, self.scaling (a float), self.merged (bool, initially False).
        self.lora_A: nn.Parameter (r, d_in), same dtype and device as base.weight, initialized
                     uniform on [-1/sqrt(d_in), 1/sqrt(d_in)] (the PEFT default).
        self.lora_B: nn.Parameter (d_out, r), same dtype and device, initialized to zeros.
    The only parameters with requires_grad=True must be lora_A and lora_B.

    forward(x): x has shape (..., d_in) with any number of leading dimensions.
        Not merged: base(x) + scaling * (x @ A.T) @ B.T, computed as two skinny matmuls.
        Merged: just base(x), because the update already lives in base.weight.

    merge(): add scaling * B @ A to base.weight in place (no autograd) and set merged = True.
             Calling merge() on an already merged layer does nothing.
    unmerge(): subtract the same quantity and set merged = False. No-op if not merged.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def merge(self) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def unmerge(self) -> None:
        raise NotImplementedError


def lora_param_count(shapes: List[Tuple[int, int]], r: int) -> int:
    """Total trainable parameters when a rank-r LoRA is attached to every matrix in `shapes`.

    Args:
        shapes: one (d_in, d_out) pair per adapted weight matrix in the whole model
                (repeat a pair once per layer if a layer type appears in many layers).
        r: LoRA rank.
    Returns:
        the number of parameters in all A and B matrices (no biases, no magnitude vectors).
    """
    raise NotImplementedError


def nf4_quantize(w: Tensor, block: int = 64) -> Tuple[Tensor, Tensor]:
    """Block-wise absmax quantization to the 16 NF4 levels.

    Flatten w in row-major order and split it into consecutive blocks of `block` values (pad the
    last block with zeros if numel is not a multiple of block; padding never changes a block's
    absmax). For each block, absmax = max |w|; normalize w / absmax into [-1, 1]; replace each value
    by the index of the nearest entry of NF4_LEVELS (ties go to the lower index).
    A block with absmax == 0 gets scale 1 in the division (so its codes are the index of 0.0),
    but its stored absmax must still be 0.

    Args:
        w: float tensor of any shape.
        block: block size.
    Returns:
        codes: uint8 tensor of shape (w.numel(),) with values in [0, 15] (padding removed;
               no bit-packing needed).
        absmax: float32 tensor of shape (ceil(w.numel() / block),).
    """
    raise NotImplementedError


def nf4_dequantize(codes: Tensor, absmax: Tensor, shape: Tuple[int, ...], block: int = 64) -> Tensor:
    """Inverse of nf4_quantize: look up the level for each code and multiply by its block's absmax.

    Args:
        codes: uint8 (numel,), absmax: float32 (n_blocks,), shape: the original tensor shape,
        block: the block size used to quantize.
    Returns:
        float32 tensor of the given shape.
    """
    raise NotImplementedError


def nf4_bits_per_param(block: int = 64, block2: int = 256, double_quant: bool = True) -> float:
    """Average storage cost in bits per weight, including quantization constants.

    Each weight costs 4 bits. Each block of `block` weights has one scale (absmax).
    Without double quantization the scale is FP32 (32 bits per block).
    With double quantization the scales are stored as 8-bit values in groups of `block2`, and
    each group has one FP32 second-level constant.
    Returns the exact average as a float (for example 4.5 without double quantization at block 64).
    """
    raise NotImplementedError


def qlora_linear(
    x: Tensor,
    codes: Tensor,
    absmax: Tensor,
    weight_shape: Tuple[int, int],
    lora_A: Tensor,
    lora_B: Tensor,
    scaling: float,
    block: int = 64,
    bias: Tensor = None,
) -> Tensor:
    """Forward pass of one QLoRA linear layer: dequantize the frozen base, add the LoRA branch.

    Args:
        x: (..., d_in) activations (any float dtype; do all arithmetic in x's dtype).
        codes, absmax: NF4 storage of the frozen weight W with shape weight_shape = (d_out, d_in).
        lora_A: (r, d_in), lora_B: (d_out, r), scaling = alpha / r.
        bias: optional (d_out,) tensor added to the output.
    Returns:
        (..., d_out): x @ W_deq.T + bias + scaling * (x @ A.T) @ B.T, where W_deq is the dequantized
        weight cast to x's dtype. Gradients must flow to x, lora_A and lora_B (codes are integers,
        so they cannot receive gradients).
    """
    raise NotImplementedError
