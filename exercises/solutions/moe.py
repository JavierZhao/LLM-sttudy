"""Reference solutions: Mixture of Experts (page 08)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    """Top-k routing with optional selection bias, gate normalization and group limits."""
    logits = x @ W_r                                              # (T, E)
    if score == "softmax":
        scores = logits.softmax(dim=-1)
    elif score == "sigmoid":
        scores = logits.sigmoid()
    else:
        raise ValueError(score)
    T, E = scores.shape
    # selection scores: the bias steers WHICH experts are picked but never enters the gate values
    sel = scores if bias is None else scores + bias.detach()
    if n_group > 1:
        per_group = max(1, k // topk_group)
        grouped = sel.view(T, n_group, E // n_group)
        group_score = grouped.topk(per_group, dim=-1).values.sum(-1)      # (T, n_group)
        keep = group_score.topk(topk_group, dim=-1).indices               # (T, topk_group)
        group_mask = torch.zeros(T, n_group, dtype=torch.bool, device=x.device)
        group_mask.scatter_(1, keep, True)
        expert_mask = group_mask.unsqueeze(-1).expand(T, n_group, E // n_group).reshape(T, E)
        sel = sel.masked_fill(~expert_mask, float("-inf"))
    indices = sel.detach().topk(k, dim=-1).indices                # (T, k), descending selection score
    gates = scores.gather(1, indices)                             # un-biased scores of the chosen experts
    if norm_topk:
        gates = gates / gates.sum(dim=-1, keepdim=True)
    return indices, gates * route_scale


class SwiGLUExpert(nn.Module):
    """One bias-free SwiGLU expert."""

    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_hidden, bias=False)
        self.w_up = nn.Linear(d_model, d_hidden, bias=False)
        self.w_down = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MoELayer(nn.Module):
    """Shared experts (always on) plus top_k of n_routed routed experts."""

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
        self.n_routed, self.top_k = n_routed, top_k
        self.score, self.norm_topk, self.route_scale = score, norm_topk, route_scale
        self.W_r = nn.Parameter(torch.randn(d_model, n_routed) * d_model ** -0.5)
        self.register_buffer("bias", torch.zeros(n_routed))       # selection bias: a buffer, not trained by SGD
        self.experts = nn.ModuleList(SwiGLUExpert(d_model, d_expert) for _ in range(n_routed))
        self.shared = nn.ModuleList(SwiGLUExpert(d_model, d_expert) for _ in range(n_shared))

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        xf = x.reshape(-1, shape[-1])                             # (T, d)
        indices, gates = topk_router(xf, self.W_r, self.top_k, self.score, self.bias,
                                     self.norm_topk, self.route_scale)
        # router distribution for the balance loss (differentiable)
        logits = xf @ self.W_r
        if self.score == "softmax":
            probs = logits.softmax(-1)
        else:
            s = logits.sigmoid()
            probs = s / s.sum(-1, keepdim=True)
        self.last_indices, self.last_probs = indices, probs
        self.last_load = torch.bincount(indices.reshape(-1), minlength=self.n_routed)

        y = torch.zeros_like(xf)
        for e in range(self.n_routed):
            tok, slot = torch.where(indices == e)                 # tokens that chose expert e, and in which slot
            if tok.numel() == 0:
                continue                                          # zero-token experts are skipped entirely
            out = self.experts[e](xf.index_select(0, tok))        # (n_e, d): only this expert's tokens
            y.index_add_(0, tok, out * gates[tok, slot].unsqueeze(-1))
        for s_exp in self.shared:                                 # shared experts see every token, gate 1
            y = y + s_exp(xf)
        return y.reshape(shape)


def switch_aux_loss(router_probs: Tensor, expert_indices: Tensor, n_experts: int) -> Tensor:
    """N * sum_i f_i P_i with f_i the (non-differentiable) dispatch fraction."""
    T, k = expert_indices.shape
    counts = torch.bincount(expert_indices.reshape(-1), minlength=n_experts).to(router_probs.dtype)
    f = counts / (T * k)                                          # constant w.r.t. autograd
    P = router_probs.mean(dim=0)                                  # gradient flows only through P
    return n_experts * (f * P).sum()


def router_z_loss(router_logits: Tensor) -> Tensor:
    """Mean over tokens of logsumexp(logits)^2 (penalizes large router logits)."""
    return torch.logsumexp(router_logits, dim=-1).pow(2).mean()


def update_balance_bias(bias: Tensor, expert_load: Tensor, gamma: float) -> Tensor:
    """b_i <- b_i + gamma * sign(mean_load - load_i)."""
    load = expert_load.to(bias.dtype)
    return bias + gamma * torch.sign(load.mean() - load)


def expert_capacity(n_tokens: int, n_experts: int, top_k: int, capacity_factor: float) -> int:
    return math.ceil(capacity_factor * n_tokens * top_k / n_experts)


def capacity_dispatch_mask(expert_indices: Tensor, n_experts: int, capacity: int) -> Tensor:
    """Slot-major first-come capacity: vectorized with a stable sort by expert id."""
    T, k = expert_indices.shape
    flat = expert_indices.t().reshape(-1)                         # slot-major priority order, length k*T
    order = torch.argsort(flat, stable=True)                      # group by expert, keep priority order inside a group
    sorted_e = flat[order]
    first = torch.searchsorted(sorted_e, sorted_e, right=False)   # index of the first entry of each expert's run
    pos_sorted = torch.arange(flat.numel(), device=flat.device) - first   # rank of each assignment within its expert
    pos = torch.empty_like(pos_sorted)
    pos[order] = pos_sorted                                       # back to priority order
    return (pos < capacity).view(k, T).t()
