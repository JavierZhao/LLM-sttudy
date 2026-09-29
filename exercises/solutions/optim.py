"""Reference solutions: AdamW, LR schedules, Newton-Schulz and Muon (page 14)."""
import math
from typing import Optional, Tuple

import torch
from torch import Tensor


def adamw_step(
    param: Tensor,
    grad: Tensor,
    m: Tensor,
    v: Tensor,
    t: int,
    lr: float,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    wd: float = 0.0,
) -> Tuple[Tensor, Tensor, Tensor]:
    b1, b2 = betas
    m = b1 * m + (1 - b1) * grad                     # first moment (EMA of g)
    v = b2 * v + (1 - b2) * grad * grad              # second moment (EMA of g^2)
    m_hat = m / (1 - b1 ** t)                        # bias correction: the EMAs start at 0
    v_hat = v / (1 - b2 ** t)
    # decoupled decay: shrink the weights directly, never through the gradient
    param = param * (1 - lr * wd) - lr * m_hat / (v_hat.sqrt() + eps)
    return param, m, v


def lr_schedule(
    step: int,
    kind: str,
    peak_lr: float,
    total_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
    decay_steps: Optional[int] = None,
    milestones: Tuple[float, ...] = (0.8, 0.9),
    factors: Tuple[float, ...] = (0.316, 0.1),
) -> float:
    if kind not in ("cosine", "wsd", "step"):
        raise ValueError(f"unknown schedule kind {kind!r}")
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    min_lr = peak_lr * min_lr_ratio
    if kind == "cosine":
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        p = min(max(p, 0.0), 1.0)
        return min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * p))
    if kind == "wsd":
        if decay_steps is None:
            decay_steps = total_steps // 10
        start = total_steps - decay_steps
        if step < start:
            return peak_lr
        p = min((step - start) / max(1, decay_steps), 1.0)
        return peak_lr + (min_lr - peak_lr) * p
    lr = peak_lr                                     # kind == "step"
    for frac, factor in zip(milestones, factors):
        if step >= frac * total_steps:
            lr = peak_lr * factor
    return lr


def newton_schulz(
    G: Tensor,
    steps: int = 5,
    coeffs: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    eps: float = 1e-7,
) -> Tensor:
    a, b, c = coeffs
    X = G / (G.norm() + eps)                         # sigma_max <= ||G||_F, so all sigma <= 1
    transposed = G.shape[0] > G.shape[1]
    if transposed:
        X = X.T                                      # keep X X^T the small (min(n, m))^2 matrix
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)                      # polynomial in A, applied to X's singular values
        X = a * X + B @ X                            # sigma <- a sigma + b sigma^3 + c sigma^5
    return X.T if transposed else X


def muon_step(
    param: Tensor,
    grad: Tensor,
    momentum_buf: Tensor,
    lr: float,
    momentum: float = 0.95,
    wd: float = 0.0,
    ns_steps: int = 5,
    nesterov: bool = True,
    update_rms: Optional[float] = 0.2,
) -> Tuple[Tensor, Tensor]:
    buf = momentum * momentum_buf + grad
    x = momentum * buf + grad if nesterov else buf   # Nesterov look-ahead
    O = newton_schulz(x, steps=ns_steps)
    n, m = param.shape
    if update_rms is not None:
        # orthogonal-ish O has RMS about 1/sqrt(max(n, m)); rescale to update_rms so that
        # AdamW's tuned lr and wd can be reused (Moonlight)
        scale = update_rms * math.sqrt(max(n, m))
    else:
        scale = math.sqrt(max(1.0, n / m))
    return param * (1 - lr * wd) - lr * scale * O, buf
