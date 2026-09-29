"""Drill (page 35): LayerNorm and RMSNorm, forward and backward by hand.

Write the forward pass, cache what the backward pass needs, and derive the gradients on paper
before you code them. No autograd inside your functions: the tests compare you against
torch.autograd in float64 (including torch.autograd.gradcheck). Run:
    pytest exercises/tests/test_layernorm_backward.py

Conventions (the tests rely on all of them):
  * x is (..., d) with any number of leading axes (a 1-D x of shape (d,) is allowed). Both norms
    act on the last axis. g and b are (d,).
  * LayerNorm: y = (x - mean) / sqrt(var + eps) * g + b, with the BIASED variance (divide by d).
    RMSNorm:   y = x / sqrt(mean(x^2) + eps) * g, with no centering and no bias.
    eps sits inside the square root.
  * Statistics and all intermediate tensors are computed in at least float32
    (torch.promote_types(x.dtype, torch.float32)), so float64 inputs stay float64 and bfloat16
    or float16 inputs are upcast. y and dx are returned in x.dtype. dg and db are returned in
    g.dtype.
  * `cache` is opaque: forward returns it and backward consumes it, and the tests only pass it
    from one to the other. It must not hold an autograd graph, backward may be called twice with
    the same cache, and neither function may modify its inputs.
  * dg and db are summed over ALL leading axes of dy, so they have shape (d,).
"""
from __future__ import annotations

import torch
from torch import Tensor


def layernorm_forward(x: Tensor, g: Tensor, b: Tensor, eps: float = 1e-5) -> tuple[Tensor, tuple]:
    """LayerNorm over the last axis.

    Args:
        x: (..., d) floating point.
        g: (d,) gain (gamma).
        b: (d,) bias (beta).
        eps: added to the variance inside the square root. eps = 0 must work for inputs whose
            rows are not constant.
    Returns:
        y: same shape and dtype as x.
        cache: whatever layernorm_backward needs (a tuple).
    """
    raise NotImplementedError


def layernorm_backward(dy: Tensor, cache: tuple) -> tuple[Tensor, Tensor, Tensor]:
    """Gradients of LayerNorm.

    Args:
        dy: (..., d), the gradient of the loss with respect to y.
        cache: the second value returned by layernorm_forward.
    Returns:
        dx: same shape and dtype as x.
        dg: (d,), summed over all leading axes, dtype of g.
        db: (d,), summed over all leading axes, dtype of g.
    Notes:
        Use only elementwise operations and reductions over the last axis (no Jacobian matrix,
        no autograd, no .backward()).
    """
    raise NotImplementedError


def rmsnorm_forward(x: Tensor, g: Tensor, eps: float = 1e-6) -> tuple[Tensor, tuple]:
    """RMSNorm over the last axis (no re-centering, no bias).

    Args:
        x: (..., d) floating point.
        g: (d,) gain.
        eps: added to the mean square inside the square root.
    Returns:
        y: same shape and dtype as x.
        cache: whatever rmsnorm_backward needs (a tuple).
    """
    raise NotImplementedError


def rmsnorm_backward(dy: Tensor, cache: tuple) -> tuple[Tensor, Tensor]:
    """Gradients of RMSNorm.

    Args:
        dy: (..., d), the gradient of the loss with respect to y.
        cache: the second value returned by rmsnorm_forward.
    Returns:
        dx: same shape and dtype as x.
        dg: (d,), summed over all leading axes, dtype of g.
    """
    raise NotImplementedError
