"""Reference solutions: RL systems primitives (page 25)."""
from __future__ import annotations

import math

import torch
from torch import Tensor

_ROLES = {"system", "user", "assistant", "tool"}


@torch.no_grad()  # the weights are constants in the loss
def truncated_is_weights(logp_trainer: Tensor, logp_rollout: Tensor, cap: float, mask: Tensor) -> Tensor:
    """w = min(exp(logp_trainer - logp_rollout), cap) on masked-in tokens, 0 elsewhere."""
    # clamp in log space BEFORE exp so a +100 log-ratio cannot overflow float32 to inf
    log_w = (logp_trainer - logp_rollout).clamp(max=math.log(cap))
    w = torch.exp(log_w)
    # torch.where (not multiplication): garbage at masked positions may be nan/inf, and nan * 0 = nan
    return torch.where(mask.bool(), w, torch.zeros_like(w))


def agent_loss_mask(
    segments: list[tuple[str, list[int]]], train_roles: tuple[str, ...] = ("assistant",)
) -> tuple[list[int], list[int]]:
    """Token-provenance mask: 1 only on tokens of segments the policy sampled."""
    ids: list[int] = []
    mask: list[int] = []
    for role, toks in segments:
        if role not in _ROLES:
            raise ValueError(f"unknown role {role!r}")
        ids.extend(toks)
        mask.extend([1 if role in train_roles else 0] * len(toks))
    return ids, mask


def sequence_ratio_drift(per_token_logp_diff: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Sequence log-ratio, ratio and length-normalized (geometric-mean) ratio."""
    d = per_token_logp_diff.to(torch.float64)
    T = d.shape[1]
    valid = torch.arange(T, device=d.device)[None, :] < lengths[:, None]
    # where, not multiply: padded entries may be nan or inf
    seq_log_ratio = torch.where(valid, d, torch.zeros_like(d)).sum(dim=1)
    n = lengths.clamp(min=1).to(torch.float64)
    return seq_log_ratio, torch.exp(seq_log_ratio), torch.exp(seq_log_ratio / n)


def staleness_admits(n_submitted: int, batch_size: int, policy_version: int, eta: int) -> bool:
    """floor((N_r - 1) / B) <= i + eta with N_r = n_submitted + 1."""
    return (n_submitted // batch_size) <= policy_version + eta
