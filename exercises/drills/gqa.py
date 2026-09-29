"""Drill (page 07): grouped-query attention.

Implement the three functions below. Run:
    pytest exercises/tests/test_gqa.py
"""
import torch
from torch import Tensor


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Expand KV heads so each query head has its own copy.

    Args:
        x: (B, n_kv, S, d) keys or values.
        n_rep: query heads per KV head (g = n_h // n_kv).
    Returns:
        (B, n_kv * n_rep, S, d) where output head h equals input head h // n_rep
        (heads of one group are contiguous: 0,0,0,0,1,1,1,1,... for n_rep=4).
    """
    raise NotImplementedError


def gqa_attention(q: Tensor, k: Tensor, v: Tensor, causal: bool = True) -> Tensor:
    """Grouped-query scaled dot-product attention.

    Args:
        q: (B, n_h, T, d_h) queries for the LAST T positions of the sequence.
        k, v: (B, n_kv, S, d_h) keys/values for all S positions (cache + new), S >= T,
              n_h divisible by n_kv.
        causal: if True, query i (absolute position S - T + i) may attend only to key
                positions <= S - T + i. This end-aligned mask is what you need when
                decoding with a KV cache (T = 1) or prefilling a chunk after a cache.
    Returns:
        (B, n_h, T, d_h). Use scale 1/sqrt(d_h).
    """
    raise NotImplementedError


def mha_to_gqa_kv_weight(W: Tensor, n_h: int, n_kv: int) -> Tensor:
    """Convert an MHA key (or value) projection to GQA by mean-pooling heads (Ainslie et al. 2023).

    Args:
        W: (d, n_h * d_h). Columns [h*d_h:(h+1)*d_h] are head h's projection.
        n_h: number of original heads. n_kv: number of KV heads after conversion.
    Returns:
        (d, n_kv * d_h): new KV head j is the mean of original heads j*g ... j*g+g-1, g = n_h // n_kv.
    """
    raise NotImplementedError
