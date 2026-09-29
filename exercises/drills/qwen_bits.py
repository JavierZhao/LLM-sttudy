"""Drill (page 30): Qwen building blocks.

Five short pieces, each a common interview question about Qwen3 and Qwen3-Next.
The two central ones are `gated_attention_output` (about 10 minutes) and the pair of
load-balancing losses (about 20 minutes). The last two are back-of-envelope calculators that
you should be able to write from a config in a few minutes. Run:
    pytest exercises/tests/test_qwen_bits.py

Conventions
- Hidden states are row vectors: linear maps are `x @ W`.
- H = number of query heads, D = head dimension, E = number of routed experts, k = experts per token.
- Balance losses follow page 08: f_i = (number of (token, slot) assignments to expert i) / (tokens * k),
  so f sums to 1 and is NOT differentiable; P_i = mean over tokens of the router probability of
  expert i. A perfectly uniform router gives a loss of exactly 1.
"""
from __future__ import annotations

import torch
from torch import Tensor


def gated_attention_output(attn_out: Tensor, x: Tensor, W_gate: Tensor) -> Tensor:
    """Sigmoid output gate of gated attention (Qiu et al. 2025, position G1).

    Args:
        attn_out: (B, T, H, D) scaled-dot-product attention output of every head, BEFORE the
            output projection W_O.
        x: (B, T, d) the (pre-normalized) hidden states that entered the attention layer.
        W_gate: (d, H * D) for an elementwise, head-specific gate, or (d, H) for a headwise gate
            (one scalar per head, broadcast over D). In the elementwise case column h * D + c
            gates channel c of head h (head-major, i.e. the gate is viewed as (B, T, H, D)).
    Returns:
        (B, T, H, D): attn_out multiplied by the gate sigmoid(x @ W_gate), where the gate is
        viewed as (B, T, H, D) (elementwise) or (B, T, H, 1) and broadcast over D (headwise).
        The gate depends on the token's own hidden state x only, not on other tokens, and the
        sigmoid is applied to the gate logits, never to attn_out.
    """
    raise NotImplementedError


def micro_batch_balance_loss(router_probs_list: list[Tensor], expert_idx_list: list[Tensor], n_experts: int) -> Tensor:
    """Load-balancing loss computed inside each micro-batch, then averaged (the framework default).

        L_micro = (1 / N_P) * sum_j  E * sum_i f^j_i * P^j_i

    Args:
        router_probs_list: N_P tensors, each (n_j, E): router distribution of the n_j tokens of
            micro-batch (parallel group) j, rows sum to 1, differentiable. n_j may differ.
        expert_idx_list: N_P int64 tensors, each (n_j, k): experts chosen by each token.
        n_experts: E.
    Returns:
        Scalar. Every group counts equally (weight 1 / N_P), whatever its token count.
    """
    raise NotImplementedError


def global_batch_balance_loss(router_probs_list: list[Tensor], expert_idx_list: list[Tensor], n_experts: int) -> Tensor:
    """Global-batch load-balancing loss (Qiu et al. 2025, used by Qwen3).

    The selection frequency is synchronized across all micro-batches before it is multiplied by
    the probabilities:

        L_global = (1 / N_P) * sum_j  E * sum_i fbar_i * P^j_i

    where fbar_i = (assignments to expert i over ALL groups) / (total tokens * k) is the
    token-weighted global frequency (no gradient), and P^j_i is still the mean probability inside
    group j. Gradients flow only through the local P^j (the all-reduce carries counts, not
    gradients).

    Args and returns: as `micro_batch_balance_loss`.
    """
    raise NotImplementedError


def qwen3_moe_param_counts(
    n_layers: int,
    d: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    d_expert: int,
    n_experts: int,
    top_k: int,
    vocab: int,
    tie_embeddings: bool = False,
) -> dict[str, int]:
    """Exact parameter count of a Qwen3-style MoE language model (every layer is MoE, no shared expert).

    Count, per layer: the four attention projections (no biases), the per-head QK-norm gains
    (two vectors of size head_dim), two RMSNorm gains of size d, the router (d x n_experts) and
    n_experts SwiGLU experts (gate, up, down: 3 * d * d_expert each). Add the embedding matrix
    (and a separate LM head unless tied) and the final RMSNorm.

    Returns a dict with:
        "total": all parameters.
        "active": parameters used per token when only top_k experts run, counting the embedding
            matrix and the LM head in full (the convention that reproduces the reported "22B").
        "expert_per_layer": parameters of all routed experts in one layer.
        "attention_per_layer": the four attention projections in one layer.
    """
    raise NotImplementedError


def hybrid_cache_bytes(
    seq_len: int,
    n_layers: int,
    full_attn_interval: int,
    n_kv_heads: int,
    head_dim: int,
    n_state_heads: int,
    d_k: int,
    d_v: int,
    kv_bytes: int = 2,
    state_bytes: int = 4,
) -> dict[str, int]:
    """Inference memory of a Gated DeltaNet hybrid (Qwen3-Next style) for ONE sequence.

    Every `full_attn_interval`-th layer (layers interval, 2*interval, ...) is full attention and
    keeps a KV cache that grows with seq_len; every other layer is a linear-attention layer with a
    fixed recurrent state of shape (n_state_heads, d_k, d_v). Ignore the small convolution state.
    With full_attn_interval = 1 the model is an ordinary Transformer. n_layers need not be a
    multiple of the interval: a trailing partial block has no full-attention layer (layers are
    numbered from 1, so 10 layers with interval 4 have full attention at layers 4 and 8 only).

    Returns a dict with:
        "kv_per_token": bytes of KV cache per token, summed over the full-attention layers.
        "kv_total": kv_per_token * seq_len.
        "state_total": bytes of recurrent state summed over the linear layers (independent of seq_len).
        "total": kv_total + state_total.
    """
    raise NotImplementedError
