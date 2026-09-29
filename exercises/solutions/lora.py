"""Reference solutions: LoRA linear layer, NF4 block quantization, QLoRA forward (page 20)."""
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
    """Frozen base layer plus a trainable low-rank update: y = base(x) + (alpha / r) * x A^T B^T."""

    def __init__(self, base: nn.Linear, r: int, alpha: float) -> None:
        super().__init__()
        assert r >= 1
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)                       # the pretrained weights never train
        self.r, self.alpha = r, alpha
        self.scaling = alpha / r
        w = base.weight
        d_out, d_in = w.shape
        bound = 1.0 / math.sqrt(d_in)                     # Kaiming-uniform(a=sqrt(5)) bound = 1/sqrt(fan_in)
        # A random, B zero: the update starts at exactly zero, and the first gradient step reaches B only.
        self.lora_A = nn.Parameter(torch.empty(r, d_in, dtype=w.dtype, device=w.device).uniform_(-bound, bound))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r, dtype=w.dtype, device=w.device))
        self.merged = False

    def forward(self, x: Tensor) -> Tensor:
        out = self.base(x)
        if self.merged:
            return out                                    # the update is already inside base.weight
        # Two skinny matmuls: (..., d_in) -> (..., r) -> (..., d_out). Never form B @ A here.
        return out + self.scaling * ((x @ self.lora_A.T) @ self.lora_B.T)

    @torch.no_grad()
    def merge(self) -> None:
        if self.merged:
            return
        self.base.weight += self.scaling * (self.lora_B @ self.lora_A)   # (d_out, r) @ (r, d_in)
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self.merged:
            return
        self.base.weight -= self.scaling * (self.lora_B @ self.lora_A)
        self.merged = False


def lora_param_count(shapes: List[Tuple[int, int]], r: int) -> int:
    """Each adapted (d_in, d_out) matrix gets A (r x d_in) and B (d_out x r): r * (d_in + d_out)."""
    return sum(r * (d_in + d_out) for d_in, d_out in shapes)


def nf4_quantize(w: Tensor, block: int = 64) -> Tuple[Tensor, Tensor]:
    flat = w.detach().reshape(-1).float()
    n = flat.numel()
    pad = (-n) % block
    blocks = torch.cat([flat, flat.new_zeros(pad)]).view(-1, block)      # (n_blocks, block)
    absmax = blocks.abs().amax(dim=1)                                    # (n_blocks,)
    scale = torch.where(absmax == 0, torch.ones_like(absmax), absmax)    # avoid 0/0 on all-zero blocks
    normed = blocks / scale[:, None]                                     # in [-1, 1]
    levels = NF4_LEVELS.to(flat.device)
    # Nearest level by brute force: (n_blocks, block, 1) - (16,) -> (n_blocks, block, 16). argmin picks the first tie.
    codes = (normed[..., None] - levels).abs().argmin(dim=-1)
    return codes.reshape(-1)[:n].to(torch.uint8), absmax


def nf4_dequantize(codes: Tensor, absmax: Tensor, shape: Tuple[int, ...], block: int = 64) -> Tensor:
    levels = NF4_LEVELS.to(codes.device)
    vals = levels[codes.long()]                                          # (numel,)
    scale = absmax.float().repeat_interleave(block)[: vals.numel()]      # one scale per element
    return (vals * scale).reshape(shape)


def nf4_bits_per_param(block: int = 64, block2: int = 256, double_quant: bool = True) -> float:
    if double_quant:
        # 8-bit scale per block, plus one FP32 constant per block2 scales
        return 4 + 8 / block + 32 / (block * block2)
    return 4 + 32 / block


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
    # Dequantize on the fly to the compute dtype. The result is a temporary: it is not stored for later.
    w = nf4_dequantize(codes, absmax, weight_shape, block).to(x.dtype)
    out = x @ w.T
    if bias is not None:
        out = out + bias
    return out + scaling * ((x @ lora_A.T) @ lora_B.T)
