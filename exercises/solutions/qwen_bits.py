"""Reference solutions: Qwen building blocks (page 30)."""
from __future__ import annotations

import torch
from torch import Tensor


def gated_attention_output(attn_out: Tensor, x: Tensor, W_gate: Tensor) -> Tensor:
    """Y' = Y * sigmoid(x @ W_gate), gate computed per token from the layer input (Qiu et al. 2025, G1)."""
    B, T, H, D = attn_out.shape
    g = torch.sigmoid(x @ W_gate)                        # (B, T, H*D) elementwise or (B, T, H) headwise
    if W_gate.shape[1] == H * D:
        g = g.view(B, T, H, D)
    elif W_gate.shape[1] == H:
        g = g.unsqueeze(-1)                              # broadcast one scalar over the head dimension
    else:
        raise ValueError("W_gate must have H*D (elementwise) or H (headwise) columns")
    return attn_out * g


def _counts(idx: Tensor, n_experts: int) -> Tensor:
    # (n, k) int64 -> (E,) float assignment counts; integer indices carry no gradient
    return torch.bincount(idx.reshape(-1), minlength=n_experts).to(torch.float32)


def micro_batch_balance_loss(router_probs_list: list[Tensor], expert_idx_list: list[Tensor], n_experts: int) -> Tensor:
    """Mean over groups of E * sum_i f^j_i P^j_i, with f^j from group j only."""
    terms = []
    for p, idx in zip(router_probs_list, expert_idx_list):
        f = _counts(idx, n_experts) / idx.numel()        # assignments / (n_j * k)
        terms.append(n_experts * (f.to(p.dtype) * p.mean(0)).sum())
    return torch.stack(terms).mean()


def global_batch_balance_loss(router_probs_list: list[Tensor], expert_idx_list: list[Tensor], n_experts: int) -> Tensor:
    """Mean over groups of E * sum_i fbar_i P^j_i, with fbar the token-weighted frequency of the whole batch."""
    total = sum(_counts(idx, n_experts) for idx in expert_idx_list)      # what the all-reduce would carry
    fbar = total / sum(idx.numel() for idx in expert_idx_list)           # sums to 1, no gradient
    terms = [n_experts * (fbar.to(p.dtype) * p.mean(0)).sum() for p in router_probs_list]
    return torch.stack(terms).mean()


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
    attn = d * n_heads * head_dim + 2 * d * n_kv_heads * head_dim + n_heads * head_dim * d   # q, k, v, o
    expert = 3 * d * d_expert                                                                  # SwiGLU: gate, up, down
    per_layer = attn + 2 * head_dim + 2 * d + d * n_experts + n_experts * expert               # + qk-norm, 2 norms, router
    embed = vocab * d * (1 if tie_embeddings else 2)                                           # embedding (+ LM head)
    total = n_layers * per_layer + embed + d                                                   # + final RMSNorm
    active = total - n_layers * (n_experts - top_k) * expert                                   # skip the idle experts
    return {
        "total": total,
        "active": active,
        "expert_per_layer": n_experts * expert,
        "attention_per_layer": attn,
    }


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
    n_full = n_layers // full_attn_interval                    # layers interval, 2*interval, ... (floor: a partial trailing block has none)
    n_linear = n_layers - n_full
    kv_per_token = n_full * 2 * n_kv_heads * head_dim * kv_bytes          # K and V of the full layers only
    kv_total = kv_per_token * seq_len
    state_total = n_linear * n_state_heads * d_k * d_v * state_bytes      # fixed size, independent of seq_len
    return {
        "kv_per_token": kv_per_token,
        "kv_total": kv_total,
        "state_total": state_total,
        "total": kv_total + state_total,
    }
