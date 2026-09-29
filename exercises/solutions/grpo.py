"""Reference solution for the page-24 drill (GRPO and successors)."""
from typing import Optional

import torch
from torch import Tensor


def group_advantages(rewards: Tensor, group_size: int, kind: str = "grpo", eps: float = 1e-6) -> Tensor:
    if kind not in ("grpo", "dr_grpo", "rloo"):
        raise ValueError(f"unknown kind {kind!r}")
    if group_size < 2 or rewards.numel() % group_size != 0:
        raise ValueError("rewards must be divisible into groups of size >= 2")
    r = rewards.view(-1, group_size)                       # (P, G)
    mean = r.mean(dim=1, keepdim=True)
    if kind == "grpo":
        return ((r - mean) / (r.std(dim=1, keepdim=True) + eps)).reshape(-1)
    if kind == "dr_grpo":
        return (r - mean).reshape(-1)
    # RLOO: baseline of response i is the mean of the other G - 1 rewards
    g = group_size
    baseline = (r.sum(dim=1, keepdim=True) - r) / (g - 1)
    return (r - baseline).reshape(-1)


def grpo_loss(logp_new: Tensor, logp_old: Tensor, logp_ref: Tensor, adv: Tensor, mask: Tensor,
              eps_low: float = 0.2, eps_high: float = 0.2, beta: float = 0.0,
              agg: str = "seq_mean", max_len: Optional[int] = None) -> Tensor:
    if agg not in ("seq_mean", "token_mean", "const"):
        raise ValueError(f"unknown agg {agg!r}")
    if agg == "const" and max_len is None:
        raise ValueError("agg='const' needs max_len")
    m = mask.to(logp_new.dtype)
    # zero the junk at masked positions BEFORE exp so that inf/NaN cannot leak into the gradient
    d_old = torch.where(m > 0, logp_new - logp_old, torch.zeros_like(logp_new))
    d_ref = torch.where(m > 0, logp_ref - logp_new, torch.zeros_like(logp_new))
    ratio = d_old.exp()
    a = adv.unsqueeze(-1)                                   # (B, 1), shared by all tokens of a response
    surrogate = torch.minimum(ratio * a, ratio.clamp(1 - eps_low, 1 + eps_high) * a)
    k3 = d_ref.exp() - d_ref - 1                            # >= 0, and 0 exactly when logp_new == logp_ref
    obj = (surrogate - beta * k3) * m                       # (B, T), zero at padding
    if agg == "seq_mean":
        per_seq = obj.sum(-1) / m.sum(-1).clamp(min=1)      # 1/|o_i| sum_t
        return -per_seq.mean()                              # 1/G sum_i
    if agg == "token_mean":
        return -obj.sum() / m.sum().clamp(min=1)            # 1 / sum_i |o_i|
    return -obj.sum() / (obj.shape[0] * max_len)            # constant normalizer, no length bias


def gspo_ratio(logp_new: Tensor, logp_old: Tensor, mask: Tensor) -> Tensor:
    m = mask.to(logp_new.dtype)
    diff = torch.where(m > 0, logp_new - logp_old, torch.zeros_like(logp_new))
    mean_diff = diff.sum(-1) / m.sum(-1).clamp(min=1)       # empty row: 0 -> ratio 1
    return mean_diff.exp()


def gspo_loss(logp_new: Tensor, logp_old: Tensor, adv: Tensor, mask: Tensor,
              eps_low: float = 3e-4, eps_high: float = 4e-4) -> Tensor:
    s = gspo_ratio(logp_new, logp_old, mask)                # (B,)
    obj = torch.minimum(s * adv, s.clamp(1 - eps_low, 1 + eps_high) * adv)
    return -obj.mean()


def dynamic_sampling_filter(rewards: Tensor, group_size: int) -> Tensor:
    r = rewards.view(-1, group_size)
    return r.max(dim=1).values > r.min(dim=1).values        # not all equal
