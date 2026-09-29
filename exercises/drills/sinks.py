"""Drill (page 31): attention sinks, banded masks, local/global schedules, hybrid KV bytes, MXFP4.

The first four functions are the 20 to 30 minute core (the shapes an interviewer asks for on
a whiteboard). `sink_attention` and `mxfp4_quantize` are stretch goals. Run:
    pytest exercises/tests/test_sinks.py
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class HybridCfg:
    """Cache-relevant shape of a decoder with local (windowed) and global attention layers.

    n_layers: number of transformer layers.
    pattern: string over {"L", "G"} repeated cyclically from layer 0, e.g. "LLLLLG".
    window: number of keys a local layer keeps (the current token included).
    local_kv_elems: cached elements per token per LOCAL layer (for plain GQA: 2 * n_kv * d_h).
    global_kv_elems: cached elements per token per GLOBAL layer.
    bytes_per_elem: 2 for bf16.
    """
    n_layers: int
    pattern: str
    window: int
    local_kv_elems: int
    global_kv_elems: int
    bytes_per_elem: int = 2


def softmax_with_sink(scores: Tensor, sink_logit: Tensor | float) -> Tensor:
    """Softmax whose denominator also contains one extra learned logit (a sink).

    p_j = exp(s_j) / (sum_k exp(s_k) + exp(z_sink))

    The sink adds mass to the denominator only: it has no output column, so the returned
    probabilities over the real keys sum to LESS than 1 (they sum to 1 - p_sink).

    Args:
        scores: (..., Tq, Tk) pre-softmax logits. Masked positions are already -inf.
        sink_logit: python float, or a tensor broadcastable to scores[..., :1]
                    (for scores (B, n_h, Tq, Tk) and one sink per head: shape (n_h, 1, 1)).
                    -inf must recover the ordinary softmax.
    Returns:
        (..., Tq, Tk), same dtype as scores. Must not overflow for logits of magnitude 1e4,
        and a row whose scores are all -inf must return zeros (all mass goes to the sink).
    """
    raise NotImplementedError


def banded_causal_mask(T: int, window: int | None, S: int | None = None) -> Tensor:
    """Boolean causal mask, optionally banded (sliding window), end-aligned like a KV cache.

    The T queries are the LAST T of S positions (S defaults to T), so query i has absolute
    position q = S - T + i. It may attend to key j iff j <= q and (window is None or q - j < window):
    that is `window` keys including itself, which matches a rolling cache of `window` entries.

    Args:
        T: number of queries. window: band width in keys (>= 1), or None for plain causal.
        S: number of keys (>= T).
    Returns:
        (T, S) bool, True where attention is allowed.
    """
    raise NotImplementedError


def layer_schedule(n_layers: int, pattern: str) -> list[str]:
    """Layer types for a local/global hybrid: `pattern` repeated cyclically, cut to n_layers.

    "LLLLLG" with 62 layers puts global layers at indices 5, 11, ..., 59 (Gemma 3 27B);
    "LG" with 36 layers puts global layers at the odd indices (gpt-oss-120b); "L" is
    Mistral 7B. Raise ValueError if the pattern is empty or contains a character other
    than "L" and "G", or if n_layers < 1.

    Returns:
        list of length n_layers with entries "L" or "G".
    """
    raise NotImplementedError


def kv_bytes_hybrid(cfg: HybridCfg, T: int) -> int:
    """KV-cache bytes for ONE sequence of T tokens.

    A global layer caches T tokens. A local layer caches min(T, cfg.window) tokens.
    Each cached token costs local_kv_elems (or global_kv_elems) * bytes_per_elem bytes per layer.
    Sinks are per-head scalars and add no per-token cache.

    Returns:
        total bytes as a python int.
    """
    raise NotImplementedError


def sink_attention(q: Tensor, k: Tensor, v: Tensor, sinks: Tensor, window: int | None = None) -> Tensor:
    """gpt-oss style attention: GQA, end-aligned causal mask, optional band, per-head sink logit.

    Args:
        q: (B, n_h, T, d) queries for the LAST T positions. k, v: (B, n_kv, S, d), S >= T,
           n_h divisible by n_kv (query head h uses KV head h // (n_h // n_kv)).
        sinks: (n_h,) learned sink logits, one per query head.
        window: band width as in banded_causal_mask, or None for a dense layer.
    Returns:
        (B, n_h, T, d). Scale 1/sqrt(d). Probabilities come from softmax_with_sink;
        the sink contributes a zero value vector, so heads may output (nearly) zero.
    """
    raise NotImplementedError


def mxfp4_quantize(x: Tensor, block: int = 32) -> Tensor:
    """Round-trip x through MXFP4 (OCP Microscaling): quantize, then dequantize.

    Blocks of `block` consecutive values along the last dimension share one power-of-two scale
    X = 2 ** (floor(log2(max|block|)) - 2); the exponent 2 is emax of the E2M1 element format.
    Each element v is stored as the nearest E2M1 value to v / X (magnitudes
    0, 0.5, 1, 1.5, 2, 3, 4, 6, with sign; values beyond 6 are clamped to 6, not rounded up).
    A value exactly midway between two grid points may go to either neighbor (the tests avoid
    ties; the microscaling paper's experiments used round-half-to-nearest-even).
    An all-zero block returns zeros. Return the dequantized tensor X * P, same shape and dtype.

    Args:
        x: (..., n) float tensor, n divisible by block.
    Returns:
        Tensor, same shape as x.
    """
    raise NotImplementedError


def weight_bytes_mxfp4(n_mxfp4_params: float, n_bf16_params: float, block: int = 32,
                       scale_bits: int = 8, elem_bits: int = 4) -> float:
    """Checkpoint bytes when some parameters are stored as MXFP4 and the rest in bf16.

    An MXFP4 parameter costs elem_bits + scale_bits / block bits (4.25 with the defaults).
    A bf16 parameter costs 16 bits.

    Returns:
        bytes as a float.
    """
    raise NotImplementedError
