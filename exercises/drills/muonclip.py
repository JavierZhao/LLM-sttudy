"""Drill (page 32): QK-Clip, the attention-logit controller in Kimi K2's MuonClip.

Implement the three functions below. Run:
    pytest exercises/tests/test_muonclip.py

Setting (Kimi K2 report, arXiv 2507.20534, section 2.1 and Algorithm 1). For attention head h,
with q_i = x_i W_q^h and k_j = x_j W_k^h, the max logit of the batch is
    S_max^h = (1 / sqrt(d)) * max over sequences, positions i, j (j <= i if causal) of q_i . k_j
QK-Clip runs after each optimizer update. If S_max^h exceeds a threshold tau, that head's query
and key projection weights are rescaled so that the head's logits shrink by gamma_h = tau / S_max^h.
The rule acts on weights only: the forward and backward passes of the step are not changed.

Conventions: row vectors (x W), float dtype preserved, no in-place modification of the inputs.
"""
from __future__ import annotations

import torch
from torch import Tensor


def max_attention_logit(q: Tensor, k: Tensor, causal: bool = True) -> Tensor:
    """Per-head maximum pre-softmax attention logit over a batch.

    Args:
        q: (B, H, T, d) queries. k: (B, H, T, d) keys, same T (no KV cache here).
        causal: if True only pairs j <= i count, because those are the only logits that reach
            the softmax; if False all pairs count.
    Returns:
        (H,) tensor with the dtype of q, S_max^h = max over batch entries, i and j of (q_i . k_j) / sqrt(d),
        where d is the last dimension. This is the signed maximum, not the maximum of the absolute value.
    """
    raise NotImplementedError


def qk_clip(
    W_q_heads: Tensor,
    W_k_heads: Tensor,
    max_logits: Tensor,
    tau: float,
    alpha: float = 0.5,
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-head QK-Clip for multi-head attention.

    Args:
        W_q_heads: (H, D, d) query projection of every head (x @ W_q_heads[h] gives head h's queries).
        W_k_heads: (H, D, d) key projection of every head.
        max_logits: (H,) the S_max^h observed in the forward pass of this step.
        tau: threshold, tau > 0.
        alpha: how the shrink is split between queries and keys, 0 <= alpha <= 1.
    Returns:
        (W_q_new, W_k_new, gamma), with gamma (H,) the per-head factor in the dtype of W_q_heads.
        Heads whose max logit does not exceed tau (including zero or negative max logits) must come
        back exactly unchanged (gamma = 1). For the others W_q is multiplied by gamma**alpha and W_k by
        gamma**(1 - alpha), so the logits of that head scale by gamma = tau / S_max^h and the head's max
        logit on the same inputs becomes exactly tau. New tensors are returned; the inputs are not modified.
    Raises:
        ValueError: if tau <= 0 or alpha is outside [0, 1].
    """
    raise NotImplementedError


def qk_clip_mla(
    W_qc: Tensor,
    W_kc: Tensor,
    W_qr: Tensor,
    W_kr: Tensor,
    max_logits: Tensor,
    tau: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Per-head QK-Clip for multi-head latent attention (MLA), as in Kimi K2.

    An MLA head scores with the sum of a content term and a rotary term:
        logit_h(i, j) = (q^C_{i,h} . k^C_{j,h} + q^R_{i,h} . k^R_j) / sqrt(d_nope + d_rope)
    The content query and key are head-specific, the rotary query is head-specific, and the
    rotary key k^R is ONE tensor shared by all heads.

    Args:
        W_qc: (H, D, d_nope) content-query weights (per head). W_kc: (H, D, d_nope) content-key weights.
        W_qr: (H, D, d_rope) rotary-query weights (per head).
        W_kr: (D, d_rope) the shared rotary-key weight.
        max_logits: (H,) S_max^h. tau: threshold.
    Returns:
        (W_qc_new, W_kc_new, W_qr_new, W_kr_new, gamma). For a head with S_max^h > tau,
        gamma_h = tau / S_max^h and the whole logit of that head scales by gamma_h; other heads
        are unchanged (gamma = 1, including for zero or negative max logits). W_qc and W_kc are each
        multiplied by sqrt(gamma_h) and W_qr by gamma_h. The shared W_kr must come back unchanged
        (rescaling it would shrink every head's rotary term). Inputs are not modified.
    Raises:
        ValueError: if tau <= 0.
    """
    raise NotImplementedError
