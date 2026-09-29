"""Drill (page 03): attention from scratch.

Implement `softmax`, `causal_mask`, `scaled_dot_product_attention` and `MultiHeadAttention`.
Run:
    pytest exercises/tests/test_attention.py

Conventions: activations are row vectors (`x @ W`); boolean masks use True = "may attend".
Do not call torch.softmax, F.softmax, F.scaled_dot_product_attention or nn.MultiheadAttention
in your implementations; the point is to write them yourself.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Numerically stable softmax along `dim`, written by hand.

    Args:
        x: float tensor of any shape (float32, float64, float16 or bfloat16).
        dim: axis to normalize over (may be negative).
    Returns:
        Tensor with the shape and dtype of `x`. Entries along `dim` are >= 0 and sum to 1.
    Requirements:
        - No overflow for large finite logits, e.g. x = [1000., 1001., 1002.] in float32.
        - Entries equal to -inf get weight exactly 0.
        - A slice along `dim` that is entirely -inf returns all zeros (not NaN).
        - Differentiable with autograd by composing torch ops (no custom backward).
    """
    raise NotImplementedError


def causal_mask(T: int, S: Optional[int] = None, device=None) -> Tensor:
    """End-aligned causal mask for T queries over S keys.

    The T queries are the LAST T positions of an S-long sequence, so query i sits at absolute
    position i + (S - T) and may attend to key j iff j <= i + (S - T).

    Args:
        T: number of queries.
        S: number of keys (cache + new tokens). Defaults to T.
        device: torch device for the result.
    Returns:
        Boolean tensor of shape (T, S), True = may attend.
    Raises:
        ValueError: if S < T.
    """
    raise NotImplementedError


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """softmax(q k^T / sqrt(d)) v with an optional boolean mask.

    Args:
        q: (..., T, d) queries.
        k: (..., S, d) keys. v: (..., S, d_v) values. Leading dimensions (batch, heads) match.
        mask: optional boolean tensor broadcastable to (..., T, S); True = keep, False = block.
    Returns:
        (..., T, d_v). Scale by 1/sqrt(d) with d = q.shape[-1].
    Edge case:
        A query whose mask row has no True entry must produce an all-zero output row (no NaN),
        and gradients must stay finite.
    """
    raise NotImplementedError


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with an optional causal mask.

    Parameters (create exactly these attributes so the tests can load weights into them):
        q_proj, k_proj, v_proj, o_proj: nn.Linear(d_model, d_model, bias=False).
    Head h uses columns [h * d_head : (h + 1) * d_head] of the q/k/v projection outputs, with
    d_head = d_model // n_heads. Head outputs are concatenated in head order before o_proj.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        """Raises ValueError if d_model is not divisible by n_heads."""
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor, causal: bool = True) -> Tensor:
        """x: (B, T, d_model) -> (B, T, d_model).

        causal=True: position t may attend only to positions <= t. causal=False: full attention.
        Use your own scaled_dot_product_attention and causal_mask from this module.
        """
        raise NotImplementedError
