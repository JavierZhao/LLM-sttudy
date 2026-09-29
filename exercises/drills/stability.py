"""Drill (page 11): stability primitives and multi-token prediction.

Implement the functions and the `MTPDepth.forward` method below. Run:
    pytest exercises/tests/test_stability.py

Conventions: q, k, v are (B, n_h, T, d_h). Row vectors, softmax over the last axis,
natural log. All of these run on CPU with tiny shapes in the tests.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def rms_norm(x: Tensor, gain: Tensor | None = None, eps: float = 1e-6) -> Tensor:
    """RMSNorm over the last axis: x / sqrt(mean(x^2) + eps) * gain.

    Args:
        x: (..., d), any floating dtype.
        gain: optional (d,) learned gain, broadcast over the leading axes. None means 1.
        eps: added inside the square root.
    Returns:
        Same shape and dtype as x. The statistics are computed in float32 and the result
        is cast back to x.dtype, as production kernels do, so bf16 inputs do not lose
        the mean-square to rounding.
    """
    raise NotImplementedError


def qk_norm_attention(q: Tensor, k: Tensor, v: Tensor, g_q: Tensor, g_k: Tensor,
                      causal: bool = True, return_weights: bool = False):
    """Causal multi-head attention with QK-norm (Dehghani et al. 2023 style, RMS variant).

    Each query and each key vector (one per head and position) is RMS-normalized over
    d_h with a learned gain, THEN the usual scaled dot product is taken:
        q_hat = rms_norm(q, g_q), k_hat = rms_norm(k, g_k)
        weights = softmax(q_hat @ k_hat^T / sqrt(d_h)) (with the causal mask), out = weights @ v

    Args:
        q, k, v: (B, n_h, T, d_h), same T for queries and keys (no KV cache here).
        g_q, g_k: (d_h,) gains shared by all heads, as in Gemma 3 and Qwen3. Use eps = 1e-6.
        causal: if True, position i attends to positions <= i.
        return_weights: if True also return the attention weights.
    Returns:
        out: (B, n_h, T, d_h), or (out, weights) with weights (B, n_h, T, T).
    """
    raise NotImplementedError


def z_loss(logits: Tensor, coef: float = 1e-4, mask: Tensor | None = None) -> Tensor:
    """PaLM's auxiliary loss on the softmax normalizer: coef * mean_positions (log Z)^2.

    Args:
        logits: (..., V) unnormalized output logits, possibly bf16.
        coef: loss weight (1e-4 in PaLM).
        mask: optional bool tensor with the shape of logits without the last axis;
              only True positions are averaged.
    Returns:
        A float32 scalar. log Z = logsumexp over the vocabulary, computed in float32
        (large bf16 logits have coarse spacing, so do not exponentiate in bf16).
    """
    raise NotImplementedError


def softcap(x: Tensor, cap: float) -> Tensor:
    """Logit soft-capping (Gemma 2): cap * tanh(x / cap).

    Args:
        x: any shape and floating dtype.
        cap: positive float (Gemma 2 uses 50.0 for attention logits and 30.0 for final logits).
    Returns:
        Same shape and dtype as x, strictly inside (-cap, cap), identity-like near 0.
        Must not overflow to nan for |x| / cap in the thousands.
    """
    raise NotImplementedError


def attention_entropy(attn: Tensor) -> Tensor:
    """Shannon entropy in nats of each attention row.

    Args:
        attn: (..., S) probabilities along the last axis; rows sum to 1, exact zeros
              are allowed (causal mask) and count as 0 * log 0 = 0.
    Returns:
        (...,) entropy per row. Uniform over m keys gives log m, one-hot gives 0.
        Values and gradients must stay finite at exact zeros.
    """
    raise NotImplementedError


class MTPDepth(nn.Module):
    """One DeepSeek-V3 style multi-token-prediction module (a single depth k).

    For position i it combines the previous-depth hidden state with the embedding of the
    token k steps ahead, then runs one block (V3 eq. 21-22):
        h'_i = M [ RMSNorm(h_prev_i) ; RMSNorm(Emb(t_{i+k})) ],   h_i = block(h'_{1:n})
    where M = self.proj is a (2d -> d) linear map without bias and the two RMSNorms have
    learned gains self.norm_h and self.norm_e. The embedding table and the output head are
    NOT owned by this module: they are shared with the main model and passed to mtp_loss.
    """

    def __init__(self, d: int, block: nn.Module, eps: float = 1e-6):
        super().__init__()
        self.proj = nn.Linear(2 * d, d, bias=False)
        self.norm_h = nn.Parameter(torch.ones(d))
        self.norm_e = nn.Parameter(torch.ones(d))
        self.block = block          # (B, n, d) -> (B, n, d)
        self.eps = eps

    def forward(self, h_prev: Tensor, emb_next: Tensor) -> Tensor:
        """h_prev, emb_next: (B, n, d). Returns (B, n, d) = block(proj(concat of the two normed inputs))."""
        raise NotImplementedError


def mtp_loss(hidden: Tensor, targets: Tensor, depth_modules, embed: nn.Embedding,
             head: nn.Linear, lam: float = 0.3) -> Tensor:
    """Sequential multi-token-prediction loss (DeepSeek-V3, eq. 23-25).

    Args:
        hidden: (B, T, d) final hidden states of the main model. hidden[:, i] has seen
                inputs x_0..x_i and predicts x_{i+1}.
        targets: (B, T) long, targets[:, i] = x_{i+1} (the main next-token labels). There is
                 no padding and no ignore index.
        depth_modules: sequence of D MTPDepth modules; module k (1-based) predicts x_{i+k+1}.
        embed: shared nn.Embedding(V, d). head: shared nn.Linear(d, V, bias=False).
        lam: weight lambda.
    Depth k works on the first n = T - k positions. Its embedding input is Emb(x_{i+k}), its label
    is x_{i+k+1}, and its previous-depth state is the first n positions of the previous depth's output
    (the main model's hidden for k = 1). This keeps the causal chain, unlike parallel heads.
    Returns:
        lam / D * sum_k L_k as a scalar, where L_k is the mean cross entropy over the B * n
        positions of depth k.
    """
    raise NotImplementedError
