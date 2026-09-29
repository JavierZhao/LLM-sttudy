"""Drill (page 08): Mixture of Experts.

Implement the eight items below. The interview core is `topk_router` plus `MoELayer`
(about 30 minutes); the losses, the bias update and the capacity mask are 5-minute pieces.
Run:
    pytest exercises/tests/test_moe.py

Conventions
- Tokens are flattened: x is (T, d), a row-vector convention, linear maps are `x @ W`.
- E = number of routed experts, k = experts per token, "slot" = position 0..k-1 among a
  token's chosen experts (slot 0 = best).
- Router scores ("affinities") are either softmax over all E experts or an independent
  sigmoid per expert. A per-expert `bias` (shape (E,)) is used ONLY to pick the top-k;
  gate values always come from the un-biased scores (DeepSeek-V3, auxiliary-loss-free balancing).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


def topk_router(
    x: Tensor,
    W_r: Tensor,
    k: int,
    score: str = "softmax",
    bias: Tensor | None = None,
    norm_topk: bool = True,
    route_scale: float = 1.0,
    n_group: int = 1,
    topk_group: int = 1,
) -> tuple[Tensor, Tensor]:
    """Select k experts per token and compute their gate values.

    Args:
        x: (T, d) token representations.
        W_r: (d, E) router weights. Logits are `x @ W_r`.
        k: experts per token.
        score: "softmax" (softmax over all E logits) or "sigmoid" (independent per expert).
        bias: optional (E,) selection bias. Selection uses `scores + bias`; gates use `scores`.
        norm_topk: if True, divide the k selected scores by their sum so gates sum to 1
            (Mixtral, Qwen3, DeepSeek-V3). If False, keep the raw scores (Switch, DeepSeek-V2).
        route_scale: multiply the final gates by this constant (DeepSeek-V3 uses 2.5).
        n_group, topk_group: group-limited routing (DeepSeek-V2/V3 device- or node-limited
            routing). If n_group > 1, the E experts are split into n_group contiguous groups
            of E // n_group. Each group gets a group score equal to the sum of its top
            max(1, k // topk_group) (biased) scores; only the topk_group best groups are
            eligible; the top-k is then taken among the experts of those groups.
    Returns:
        indices: (T, k) int64, sorted by descending selection score (biased, if bias is given).
        gates: (T, k) same dtype as x, differentiable with respect to x and W_r.
    """
    raise NotImplementedError


class SwiGLUExpert(nn.Module):
    """One expert: a bias-free SwiGLU FFN, y = W_down( silu(x W_gate) * (x W_up) ).

    Must expose bias-free `nn.Linear` submodules named `w_gate` (d_model -> d_hidden),
    `w_up` (d_model -> d_hidden) and `w_down` (d_hidden -> d_model). Maps (n, d_model) to
    (n, d_model), including n = 0.
    """

    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class MoELayer(nn.Module):
    """A DeepSeekMoE-style layer: n_shared always-on experts plus top_k of n_routed experts.

        y_t = sum_s Shared_s(x_t) + sum_{i in topk(t)} g_{t,i} * Expert_i(x_t)

    Attributes the tests read:
        W_r: nn.Parameter (d_model, n_routed), router weights (any reasonable init).
        bias: buffer (n_routed,), zeros at init, the selection bias (not a Parameter).
        experts: nn.ModuleList of n_routed SwiGLUExpert(d_model, d_expert).
        shared: nn.ModuleList of n_shared SwiGLUExpert(d_model, d_expert) (may be empty).
    Set by every forward call:
        last_indices: (T, top_k) int64, the chosen experts.
        last_probs: (T, n_routed), the router distribution used by the balance loss, with
            gradient: softmax over all experts, or sigmoid scores divided by their sum.
        last_load: (n_routed,) int64, tokens per expert (`bincount` of last_indices).

    forward(x): x is (T, d_model) or (B, T, d_model); returns the same shape. Route with
    `topk_router`, then dispatch tokens to experts. A correct implementation loops over
    experts and gathers only that expert's tokens (index_select / index_add_, or a sort by
    expert id); it must never run an expert on all tokens, must handle experts that receive
    zero tokens, and must keep gradients flowing through the gates into W_r.
    """

    def __init__(
        self,
        d_model: int,
        d_expert: int,
        n_routed: int,
        n_shared: int,
        top_k: int,
        score: str = "sigmoid",
        norm_topk: bool = True,
        route_scale: float = 1.0,
    ):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


def switch_aux_loss(router_probs: Tensor, expert_indices: Tensor, n_experts: int) -> Tensor:
    """Switch-style load-balancing loss, N * sum_i f_i * P_i (without the coefficient alpha).

    Args:
        router_probs: (T, N) router distribution per token (rows sum to 1), differentiable.
        expert_indices: (T, k) int64 experts chosen per token (k >= 1).
        n_experts: N.
    Returns:
        Scalar. f_i = (number of (token, slot) assignments to expert i) / (T * k), so f sums
        to 1 and is NOT differentiable; P_i = mean over tokens of router_probs[:, i].
        Equals 1 for perfectly uniform f and P.
    """
    raise NotImplementedError


def router_z_loss(router_logits: Tensor) -> Tensor:
    """ST-MoE router z-loss: the mean over tokens of (logsumexp of the token's logits)^2.

    Args:
        router_logits: (T, N) raw router logits.
    Returns:
        Scalar.
    """
    raise NotImplementedError


def update_balance_bias(bias: Tensor, expert_load: Tensor, gamma: float) -> Tensor:
    """Auxiliary-loss-free balancing update (Wang et al. 2024, DeepSeek-V3).

    After a step, lower the bias of every overloaded expert by gamma and raise the bias of every
    underloaded expert by gamma. "Overloaded" means load above the mean load over experts.
    An expert exactly at the mean is unchanged. The size of the violation does not matter.

    Args:
        bias: (N,) float tensor.
        expert_load: (N,) tokens routed to each expert during the step (int or float).
        gamma: bias update speed.
    Returns:
        A new (N,) tensor with bias's dtype. Do not modify `bias` in place.
    """
    raise NotImplementedError


def expert_capacity(n_tokens: int, n_experts: int, top_k: int, capacity_factor: float) -> int:
    """Tokens each expert can accept: ceil(capacity_factor * n_tokens * top_k / n_experts)."""
    raise NotImplementedError


def capacity_dispatch_mask(expert_indices: Tensor, n_experts: int, capacity: int) -> Tensor:
    """Which (token, slot) assignments survive a per-expert capacity limit.

    Assignments are processed in this priority order (as in GShard's Algorithm 1): every
    token's slot 0 in token order, then every token's slot 1 in token order, and so on. An
    assignment is kept if its expert has accepted fewer than `capacity` earlier assignments
    in that order; otherwise it is dropped.

    Args:
        expert_indices: (T, k) int64.
        n_experts: number of experts.
        capacity: per-expert limit (int >= 0).
    Returns:
        (T, k) bool tensor, True where the assignment is kept.
    """
    raise NotImplementedError
