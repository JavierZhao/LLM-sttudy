"""Drill (page 14): AdamW, learning-rate schedules, Newton-Schulz orthogonalization and a Muon step.

Implement the four functions below. Run:
    pytest exercises/tests/test_optim.py

Keep this file self-contained: do not import other drills. All functions are functional
(they must not modify their inputs) and work for any floating dtype and any shape.
"""
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
    """One AdamW step with decoupled weight decay, matching torch.optim.AdamW.

    Args:
        param, grad, m, v: tensors of identical shape. m and v are the running first and
            second moment estimates BEFORE this step (zeros at the first step).
        t: the 1-based step number of THIS step (t = 1 on the first call), used for bias correction.
        lr: learning rate for this step (already includes any schedule).
        betas: (beta1, beta2) exponential decay rates.
        eps: added to sqrt(v_hat) (outside the square root).
        wd: decoupled weight-decay coefficient. The decay is applied to the parameter itself,
            scaled by lr, and never enters the gradient or the moment estimates.
    Returns:
        (new_param, new_m, new_v). Inputs must not be modified.
    """
    raise NotImplementedError


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
    """Learning rate at optimizer step `step` (0-indexed: step 0 is the first update).

    Warmup (all kinds): for step < warmup_steps the rate is peak_lr * (step + 1) / warmup_steps.
    After warmup, with min_lr = peak_lr * min_lr_ratio:
      kind == "cosine": p = (step - warmup_steps) / (total_steps - warmup_steps) clipped to [0, 1];
          lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + cos(pi * p)).
      kind == "wsd" (warmup-stable-decay): constant peak_lr until step == total_steps - decay_steps,
          then a LINEAR decay from peak_lr to min_lr, reaching min_lr at step == total_steps.
          decay_steps defaults to total_steps // 10. After total_steps the rate stays at min_lr.
      kind == "step" (multi-step decay): peak_lr multiplied by factors[i] once
          step >= milestones[i] * total_steps (milestones are fractions of total_steps,
          increasing; each factor is relative to peak_lr, not cumulative).
    Raises ValueError for an unknown kind.
    """
    raise NotImplementedError


def newton_schulz(
    G: Tensor,
    steps: int = 5,
    coeffs: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    eps: float = 1e-7,
) -> Tensor:
    """Approximately orthogonalize a matrix with the quintic Newton-Schulz iteration (Muon).

    Args:
        G: a 2-D tensor of shape (n, m), either orientation (tall or wide).
        steps: number of iterations.
        coeffs: (a, b, c) of the update X <- a X + b (X X^T) X + c (X X^T)^2 X.
        eps: added to the Frobenius norm in the initial normalization X_0 = G / (||G||_F + eps).
    Returns:
        A tensor of the same shape and dtype as G whose singular values are pushed toward 1
        (roughly U V^T for G = U S V^T). Do the arithmetic in G's own dtype, no casting.
        Run the iteration on the orientation with fewer rows (transpose a tall matrix first,
        transpose back at the end) so that X X^T is the smaller Gram matrix.
    """
    raise NotImplementedError


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
    """One Muon step for a single 2-D weight matrix of shape (n, m).

    Algorithm:
        buf_new = momentum * momentum_buf + grad
        x       = momentum * buf_new + grad   if nesterov else buf_new
        O       = newton_schulz(x, steps=ns_steps)
        scale   = update_rms * sqrt(max(n, m))   if update_rms is not None
                  else sqrt(max(1, n / m))       (the original Muon post's shape rule)
        param_new = param * (1 - lr * wd) - lr * scale * O
    Args:
        param, grad, momentum_buf: tensors of shape (n, m).
    Returns:
        (param_new, buf_new). Inputs must not be modified.
    """
    raise NotImplementedError
