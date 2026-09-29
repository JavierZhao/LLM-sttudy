"""Drill (page 35): LayerNorm and RMSNorm, forward and backward by hand (reference solution)."""
from __future__ import annotations

import torch
from torch import Tensor


def _acc_dtype(dtype: torch.dtype) -> torch.dtype:
    # statistics in at least float32; float64 inputs stay float64 (needed for gradcheck)
    return torch.promote_types(dtype, torch.float32)


def layernorm_forward(x: Tensor, g: Tensor, b: Tensor, eps: float = 1e-5) -> tuple[Tensor, tuple]:
    xf = x.to(_acc_dtype(x.dtype))
    mu = xf.mean(-1, keepdim=True)
    xc = xf - mu
    var = (xc * xc).mean(-1, keepdim=True)          # biased variance: divide by d, not d - 1
    rstd = torch.rsqrt(var + eps)                    # eps inside the square root
    xhat = xc * rstd
    y = xhat * g.to(xf.dtype) + b.to(xf.dtype)
    return y.to(x.dtype), (xhat, rstd, g, x.dtype)


def layernorm_backward(dy: Tensor, cache: tuple) -> tuple[Tensor, Tensor, Tensor]:
    xhat, rstd, g, x_dtype = cache
    d = xhat.shape[-1]
    dyf = dy.to(xhat.dtype)
    dxhat = dyf * g.to(xhat.dtype)                   # gradient w.r.t. the normalized values
    # remove the mean (shift invariance) and the component along xhat (scale invariance)
    dx = rstd * (dxhat
                 - dxhat.mean(-1, keepdim=True)
                 - xhat * (dxhat * xhat).mean(-1, keepdim=True))
    # reshape(-1, d).sum(0) also works for 1-D input, where sum(dim=()) would reduce everything
    dg = (dyf * xhat).reshape(-1, d).sum(0)
    db = dyf.reshape(-1, d).sum(0)
    return dx.to(x_dtype), dg.to(g.dtype), db.to(g.dtype)


def rmsnorm_forward(x: Tensor, g: Tensor, eps: float = 1e-6) -> tuple[Tensor, tuple]:
    xf = x.to(_acc_dtype(x.dtype))
    rstd = torch.rsqrt((xf * xf).mean(-1, keepdim=True) + eps)
    xhat = xf * rstd
    y = xhat * g.to(xf.dtype)
    return y.to(x.dtype), (xhat, rstd, g, x.dtype)


def rmsnorm_backward(dy: Tensor, cache: tuple) -> tuple[Tensor, Tensor]:
    xhat, rstd, g, x_dtype = cache
    d = xhat.shape[-1]
    dyf = dy.to(xhat.dtype)
    dxhat = dyf * g.to(xhat.dtype)
    dx = rstd * (dxhat - xhat * (dxhat * xhat).mean(-1, keepdim=True))   # no mean term: no re-centering
    dg = (dyf * xhat).reshape(-1, d).sum(0)
    return dx.to(x_dtype), dg.to(g.dtype)
