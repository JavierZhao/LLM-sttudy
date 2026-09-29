"""Reference solution for the page-23 drill: the numerical core of a PPO-RLHF loop."""
import torch
from torch import Tensor


def whiten(x: Tensor, mask: Tensor, shift_mean: bool = True, eps: float = 1e-8) -> Tensor:
    m = mask.to(x.dtype)
    n = m.sum()
    mean = (x * m).sum() / n
    var = (((x - mean) ** 2) * m).sum() / n            # population variance over valid entries only
    out = (x - mean) * torch.rsqrt(var + eps)
    if not shift_mean:
        out = out + mean                                # keep the mean, fix only the scale
    return out * m                                      # padding stays exactly 0


def shaped_rewards(rm_score: Tensor, logp_policy: Tensor, logp_ref: Tensor, beta: float,
                   mask: Tensor) -> Tensor:
    m = mask.to(logp_policy.dtype)
    kl = (logp_policy - logp_ref).detach()               # sampled-token log-ratio (k1 estimate of KL)
    rewards = -beta * kl * m                             # dense per-token penalty
    B, T = m.shape
    last = (m * torch.arange(T, device=m.device, dtype=m.dtype)).argmax(dim=1)   # index of last real token
    rewards[torch.arange(B, device=m.device), last] += rm_score.detach().to(rewards.dtype)
    return rewards


def apply_eos_penalty(rm_score: Tensor, ended_with_eos: Tensor, penalty: float = -1.0) -> Tensor:
    return torch.where(ended_with_eos.bool(), rm_score, torch.full_like(rm_score, penalty))


class AdaptiveKLController:
    def __init__(self, init_beta: float, target_kl: float, horizon: int):
        self.beta = float(init_beta)
        self.target_kl = float(target_kl)
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int) -> float:
        error = min(max(current_kl / self.target_kl - 1.0, -0.2), 0.2)   # saturating proportional error
        self.beta *= 1.0 + error * n_steps / self.horizon                # multiplicative, so beta stays > 0
        return self.beta
