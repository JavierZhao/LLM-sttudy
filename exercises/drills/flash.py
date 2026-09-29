"""Drill (page 16): FlashAttention from scratch (CPU, PyTorch).

Implement the functions below. The point is the *structure* of the computation: the
T x T score matrix must never exist, only (block_q, block_k) tiles of it, and the softmax is
assembled from running statistics. Run:
    pytest exercises/tests/test_flash.py

Keep this file self-contained: do not import other drills.
The tests forbid torch.softmax / log_softmax / logsumexp / scaled_dot_product_attention inside
`online_softmax` and `flash_attention_forward`, and they watch every intermediate tensor's size.
"""
from typing import Optional, Sequence

import torch
from torch import Tensor


def merge_softmax_stats(m_a: Tensor, l_a: Tensor, m_b: Tensor, l_b: Tensor) -> tuple[Tensor, Tensor]:
    """Merge the softmax statistics of two disjoint sets of scores (Milakov and Gimelshein 2018).

    A pair (m, l) summarizes a set of scores s (one set per row):
        m = max(s),   l = sum(exp(s - m)).
    Args:
        m_a, l_a, m_b, l_b: tensors of identical shape (...,). float32 or float64.
            An empty or fully masked set is represented by m = -inf and l = 0.
    Returns:
        (m, l) for the union of the two sets: the same statistics you would get from all scores at once.
        With one empty input the result equals the other input. With two empty inputs l is exactly 0
        (not NaN) and m stays -inf.
    """
    raise NotImplementedError


def online_softmax(x_blocks: Sequence[Tensor]) -> Tensor:
    """Softmax over the last dimension of x, where x arrives as a list of blocks.

    Args:
        x_blocks: tensors with identical leading shape (..., n_i). Their concatenation along the
            last dimension is the logits vector x. Entries may be -inf (but every row of x has at
            least one finite entry) and may be large in magnitude (|x| up to about 1e3).
    Returns:
        softmax(x, dim=-1), shape (..., sum_i n_i), same dtype as the inputs.
    Rules:
        The normalizer statistics (m, l) of every row must be obtained in ONE pass over the
        blocks, updating running values block by block. Do not concatenate the blocks before the
        statistics are known, and do not call torch.softmax, log_softmax or logsumexp.
    """
    raise NotImplementedError


def flash_attention_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    block_q: int = 32,
    block_k: int = 32,
    causal: bool = False,
    scale: Optional[float] = None,
) -> tuple[Tensor, Tensor]:
    """Exact attention with explicit tiling and an online softmax (the FlashAttention forward pass).

    Args:
        q: (B, H, Tq, d) queries.
        k, v: (B, H, Tk, d) keys and values (same number of heads as q).
        block_q, block_k: tile sizes along the query and key axes. They need not divide Tq or Tk.
        causal: if True, query i (absolute position Tk - Tq + i) attends only to keys j with
            j <= Tk - Tq + i (end-aligned mask, as in decoding with a KV cache). Requires Tq <= Tk.
        scale: multiplier on q k^T. Default 1 / sqrt(d).
    Returns:
        out: (B, H, Tq, d), same dtype as q.
        lse: (B, H, Tq), float32 or float64: the natural log of sum_j exp(scale * q_i . k_j) over the
            keys visible to query i.
    Rules:
        Loop over query tiles (outer) and key tiles (inner), keeping per-row running max m, running
        sum l and an output accumulator in at least float32. Never build a tensor with more than
        about block_q * block_k score entries per (batch, head). With causal=True, skip key tiles that
        lie entirely above the diagonal. Do not call torch.softmax, log_softmax, logsumexp or
        F.scaled_dot_product_attention.
    """
    raise NotImplementedError


def flash_attention_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    lse: Tensor,
    dout: Tensor,
    block_q: int = 32,
    block_k: int = 32,
    causal: bool = False,
    scale: Optional[float] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Gradients of attention w.r.t. q, k, v, tiled, using only q, k, v, out, lse and dout.

    Args:
        q, k, v, causal, scale: as in `flash_attention_forward`.
        out, lse: the outputs of `flash_attention_forward` for the same inputs.
        dout: (B, H, Tq, d) gradient of the loss w.r.t. out.
    Returns:
        (dq, dk, dv) with the shapes and dtypes of q, k, v.
    Rules:
        The attention probabilities P must be recomputed tile by tile from q, k and lse (they are
        not an input). Never build a tensor with more than about block_q * block_k score entries
        per (batch, head).
    """
    raise NotImplementedError


def combine_partial_attention(outs: Sequence[Tensor], lses: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
    """Merge attention results computed over disjoint chunks of the keys (the FlashDecoding reduction).

    Args:
        outs: list of S tensors (..., T, d); outs[s] is the attention output of every query over
            chunk s only, softmax-normalized within that chunk.
        lses: list of S tensors (..., T); lses[s] is the logsumexp of the scores over chunk s.
    Returns:
        (out, lse) exactly equal to attention over the union of the chunks, shapes (..., T, d) and
        (..., T). Must stay finite when the lse values are large (hundreds), so do not exponentiate them
        without subtracting a maximum.
    """
    raise NotImplementedError


def split_kv_attention(
    q: Tensor, k: Tensor, v: Tensor, num_splits: int, scale: Optional[float] = None
) -> Tensor:
    """Decode-style attention with the keys split into `num_splits` chunks processed independently.

    Args:
        q: (B, H, Tq, d) with small Tq (for example 1). Every cached key is visible to every query
            (no mask).
        k, v: (B, H, Tk, d), with 1 <= num_splits <= Tk. Split along Tk into num_splits chunks of
            (nearly) equal length.
    Returns:
        (B, H, Tq, d), equal to dense softmax(q k^T * scale) v. Each chunk must be processed with
        `flash_attention_forward` and the results merged with `combine_partial_attention`.
    """
    raise NotImplementedError
