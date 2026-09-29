"""Drill (page 05): positional encodings, RoPE in both layouts, sinusoidal and ALiBi.

Implement every function below (replace `raise NotImplementedError`). Run:
    pytest exercises/tests/test_rope.py

Conventions
-----------
* Row vectors: a head's query is q_t = x_t W^Q with shape (d_head,); rotations act as q @ R_t.
* "pos" is a 1-D tensor of absolute position ids, shape (T,). It need not be arange(T):
  during decoding with a KV cache it is e.g. tensor([n_cached]).
* x has shape (B, T, H, d_head) or (B, T, d_head); pos indexes the T axis, and every head and
  batch element shares the same angles.
* d_head is even. Pair i (i = 0 .. d_head/2 - 1) rotates by angle pos * theta_i with
  theta_i = base ** (-2 i / d_head).
"""
import math

import torch
from torch import Tensor


def rope_frequencies(d_head: int, base: float = 10000.0) -> Tensor:
    """Inverse frequencies theta_i = base ** (-2 i / d_head), i = 0 .. d_head/2 - 1.

    Returns:
        (d_head // 2,) float64 tensor. theta_0 = 1 rad/token; the last entry is
        base ** (-(d_head - 2) / d_head).
    """
    raise NotImplementedError


def apply_rope_interleaved(x: Tensor, pos: Tensor, base: float = 10000.0) -> Tensor:
    """RoPE with INTERLEAVED pairs: the pairs are (x[..., 0], x[..., 1]), (x[..., 2], x[..., 3]), ...

    Pair i is treated as the point (x[..., 2i], x[..., 2i+1]) in the plane and rotated
    counter-clockwise by pos * theta_i. This is the layout of the original RoFormer paper,
    GPT-J and Meta's Llama reference code.

    Args:
        x: (B, T, H, d_head) or (B, T, d_head), any floating dtype.
        pos: (T,) integer position ids.
        base: RoPE base b.
    Returns:
        Same shape and dtype as x. Compute the angles pos * theta_i in float64 and cast cos/sin
        to x.dtype only after the trig: angles like 100_000 rad are unrepresentable in bfloat16,
        and the float64 tests compare against a float64 reference to about 1e-7, which float32
        angles miss.
    """
    raise NotImplementedError


def apply_rope_half(x: Tensor, pos: Tensor, base: float = 10000.0) -> Tensor:
    """RoPE with SPLIT HALVES ("rotate_half"): the pairs are (x[..., i], x[..., i + d_head/2]).

    Pair i is the point (x[..., i], x[..., i + d_head/2]) in the plane, rotated counter-clockwise
    by pos * theta_i. This is the layout of GPT-NeoX and Hugging Face Llama/Qwen/Mistral.

    Args / Returns: as apply_rope_interleaved.
    """
    raise NotImplementedError


def interleaved_to_half_permutation(d_head: int) -> Tensor:
    """Index tensor `perm` (LongTensor, shape (d_head,)) mapping the interleaved layout to the half layout.

    With x_half = x_interleaved[..., perm], the two RoPE implementations agree:
        apply_rope_half(x_interleaved[..., perm], pos) == apply_rope_interleaved(x_interleaved, pos)[..., perm]
    (each pair keeps its own frequency theta_i). Applying the same permutation to the output
    columns of W^Q and W^K (per head) converts a checkpoint from one layout to the other.
    """
    raise NotImplementedError


def apply_partial_rope(x: Tensor, pos: Tensor, rotary_dim: int, base: float = 10000.0,
                       interleaved: bool = False) -> Tensor:
    """Partial RoPE (GPT-NeoX `rotary_pct`): rotate only the first `rotary_dim` dims of the last axis.

    The remaining d_head - rotary_dim dims pass through unchanged. The frequencies are those of a
    `rotary_dim`-dimensional head: theta_i = base ** (-2 i / rotary_dim).

    Args:
        x: (B, T, H, d_head) or (B, T, d_head).
        pos: (T,) position ids.
        rotary_dim: even, 0 < rotary_dim <= d_head.
        interleaved: which pair layout to use inside the rotated slice.
    Returns:
        Same shape and dtype as x.
    """
    raise NotImplementedError


def sinusoidal_embedding(T: int, d: int, base: float = 10000.0) -> Tensor:
    """Sinusoidal absolute positional encoding of Vaswani et al. (2017).

    PE[pos, 2i] = sin(pos / base ** (2i / d)),  PE[pos, 2i + 1] = cos(pos / base ** (2i / d)).

    Args:
        T: number of positions (pos = 0 .. T-1). d: even embedding width. base: the 10000 of Vaswani et al.
    Returns:
        (T, d) float32 tensor.
    """
    raise NotImplementedError


def alibi_slopes(n_heads: int) -> Tensor:
    """Per-head ALiBi slopes m_h (Press et al. 2021), as in the authors' reference code.

    For n_heads a power of two the slopes are the geometric sequence with start and ratio
    2 ** (-8 / n_heads): 8 heads give 2^-1, 2^-2, ..., 2^-8. For other n_heads, take the slopes for
    the closest power of two below n_heads, then append every other slope of the sequence for
    twice that power, until n_heads slopes exist (e.g. 12 heads: the 8-head sequence followed
    by 2^-0.5, 2^-1.5, 2^-2.5, 2^-3.5).

    Returns:
        (n_heads,) float32 tensor. Decreasing when n_heads is a power of two; otherwise the borrowed
        slopes restart at a larger value (12 heads: ..., 2^-8, then 2^-0.5, ...), so it is not monotone.
    """
    raise NotImplementedError


def alibi_bias(n_heads: int, T: int) -> Tensor:
    """Causal ALiBi bias to add to the attention logits (before softmax).

    Entry [h, i, j] is -m_h * (i - j) for j <= i and -inf for j > i (query i, key j), where
    m_h = alibi_slopes(n_heads)[h]. No 1/sqrt(d) scaling is applied to this bias.

    Returns:
        (n_heads, T, T) float32 tensor.
    """
    raise NotImplementedError
