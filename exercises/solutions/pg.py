"""Reference solutions: policy-gradient building blocks for LLM RL (page 22)."""
import torch
from torch import Tensor


def _mask(mask: Tensor | None, like: Tensor) -> Tensor:
    return torch.ones_like(like) if mask is None else mask.to(like.dtype)


def _keep(x: Tensor, m: Tensor) -> Tensor:
    # x where m == 1, exactly 0 elsewhere. Unlike `x * m` this also removes NaN/inf at padding (0 * nan = nan),
    # and it gives padded entries an exactly zero gradient.
    return torch.where(m > 0, x, torch.zeros_like(x))


def reinforce_loss(logprobs: Tensor, rewards: Tensor, baseline: Tensor,
                   mask: Tensor | None = None, gamma: float = 1.0) -> Tensor:
    """-(1/B) sum_i sum_t mask * (G - b) * logprobs, with G the masked reward-to-go."""
    B, T = logprobs.shape
    m = _mask(mask, logprobs)
    r = _keep(rewards, m)
    # reward-to-go by a backward recursion G_t = r_t + gamma * G_{t+1}
    G = torch.zeros_like(r)
    running = torch.zeros(B, dtype=r.dtype, device=r.device)
    for t in reversed(range(T)):
        running = r[:, t] + gamma * running
        G[:, t] = running
    b = baseline if baseline.dim() == 2 else baseline[:, None]
    adv = _keep((G - b).detach(), m)             # the advantage is a constant weight, never differentiated
    return -(adv * _keep(logprobs, m)).sum() / B  # sum over tokens, mean over responses: no 1/length


def gae(rewards: Tensor, values: Tensor, gamma: float, lam: float,
        mask: Tensor | None = None, last_value: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """delta_t = r_t + gamma V_{t+1} - V_t ; A_t = delta_t + gamma lam A_{t+1} ; returns = A + V."""
    B, T = rewards.shape
    m = _mask(mask, rewards)
    values = _keep(values.detach(), m)           # garbage (even NaN) at padding must not leak into any delta
    rewards = _keep(rewards.detach(), m)
    # V(s_{t+1}) for each t: the next column, which is already 0 if that position is padding (episode terminated)
    nxt = torch.zeros_like(values)
    nxt[:, :-1] = values[:, 1:]
    if last_value is not None:
        nxt[:, -1] = _keep(last_value.detach().to(values.dtype), m[:, -1])   # only rows that reach the end are bootstrapped
    delta = (rewards + gamma * nxt - values) * m
    adv = torch.zeros_like(delta)
    acc = torch.zeros(B, dtype=delta.dtype, device=delta.device)
    for t in reversed(range(T)):
        acc = (delta[:, t] + gamma * lam * acc) * m[:, t]   # mask kills the recursion at padding
        adv[:, t] = acc
    return adv, adv + values                     # values is already 0 at padding


def ppo_clip_loss(logp_new: Tensor, logp_old: Tensor, adv: Tensor, eps: float = 0.2,
                  mask: Tensor | None = None, eps_high: float | None = None,
                  agg: str = "token") -> Tensor:
    """-min(rho A, clip(rho, 1-eps, 1+eps_high) A), reduced over tokens or sequences."""
    m = _mask(mask, logp_new)
    eps_high = eps if eps_high is None else eps_high
    a = _keep((adv if adv.dim() == 2 else adv[:, None]).detach(), m)
    rho = torch.exp(_keep(logp_new, m) - _keep(logp_old.detach(), m))   # padded entries: rho = 1, zero gradient
    obj = torch.minimum(rho * a, torch.clamp(rho, 1 - eps, 1 + eps_high) * a)
    per_token = -obj * m
    if agg == "token":
        return per_token.sum() / m.sum().clamp(min=1)
    if agg == "seq":
        return (per_token.sum(1) / m.sum(1).clamp(min=1)).mean()
    raise ValueError(f"unknown agg {agg!r}")


def kl_estimators(logp: Tensor, logp_ref: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """k1 = -log r, k2 = (log r)^2 / 2, k3 = r - 1 - log r, with r = pi_ref / pi."""
    logr = logp_ref - logp
    k1 = -logr
    k2 = 0.5 * logr**2
    k3 = torch.expm1(logr) - logr                # expm1 avoids catastrophic cancellation for small |log r|
    return k1, k2, k3


def masked_whiten(x: Tensor, mask: Tensor | None = None, shift_mean: bool = True,
                  eps: float = 1e-8) -> Tensor:
    """(x - mean) / sqrt(var + eps) over real entries; optional mean shift back; zero on padding."""
    m = _mask(mask, x)
    x = _keep(x, m)                              # padding may hold NaN or inf
    n = m.sum().clamp(min=1)
    mean = (x * m).sum() / n
    var = (((x - mean) ** 2) * m).sum() / n      # biased variance, pooled over the whole batch
    out = (x - mean) * torch.rsqrt(var + eps)
    if not shift_mean:
        out = out + mean
    return out * m
