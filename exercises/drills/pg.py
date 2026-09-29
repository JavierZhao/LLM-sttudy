"""Drill (page 22): policy-gradient building blocks for LLM reinforcement learning.

Implement the five functions below. Run:
    pytest exercises/tests/test_pg.py

Conventions shared by all functions
    B = number of sampled responses, T = padded response length (right padding).
    Every tensor is per *response token*: entry (i, t) belongs to the token a_t that the
    policy emitted in state s_t = (prompt, response tokens before t) of response i.
    `mask` is a (B, T) tensor of 0/1 (or bool): 1 on real response tokens, 0 on padding.
    Real tokens of a row are contiguous from position 0. Under `mask` the episode of row i
    ends after its last real token (the EOS token), and all padded positions must
    contribute exactly zero to every loss and gradient.
"""
import torch
from torch import Tensor


def reinforce_loss(logprobs: Tensor, rewards: Tensor, baseline: Tensor,
                   mask: Tensor | None = None, gamma: float = 1.0) -> Tensor:
    """Surrogate loss whose gradient is the REINFORCE policy gradient with reward-to-go and a baseline.

    Args:
        logprobs: (B, T) log pi_theta(a_t | s_t) of the sampled tokens. Carries the gradient.
        rewards: (B, T) per-token rewards r_t (for an outcome reward only the last real token
            is nonzero). No gradient.
        baseline: (B, T) baseline b(s_t) for each token, or (B,) for a baseline that is constant
            along a response (for example a leave-one-out mean). Treated as a constant.
        mask: (B, T) 0/1, see the module docstring. Default: all ones.
        gamma: discount factor.
    Returns:
        A scalar, minimized by gradient descent:
            -(1/B) * sum_i sum_t mask[i,t] * (G[i,t] - b[i,t]) * logprobs[i,t]
        where G[i,t] = sum_{t' >= t} gamma^(t'-t) * r[i,t'] is the masked reward-to-go.
        The advantage G - b must not receive a gradient. Sum over tokens, mean over responses:
        do NOT divide by the response length (that would bias the gradient).
    """
    raise NotImplementedError


def gae(rewards: Tensor, values: Tensor, gamma: float, lam: float,
        mask: Tensor | None = None, last_value: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Generalized advantage estimation (Schulman et al. 2016) for a batch of padded responses.

    Args:
        rewards: (B, T) r_t.
        values: (B, T) V(s_t) for every token position, i.e. the critic's value of the state
            *before* token t is emitted.
        gamma, lam: discount and GAE parameter.
        mask: (B, T) 0/1, see the module docstring. Default: all ones.
        last_value: optional (B,) value V(s_T) of the state after the last *padded* position.
            It is used only by rows whose mask is 1 at position T-1 (a response that was cut off
            at the length limit, so it is bootstrapped). Default: zeros.
    Terminal convention: after the last real token of a row the episode is over and the
    value of the next state is 0, unless the row is bootstrapped by `last_value` as above.
    Values stored at padded positions are garbage and must not influence the result.
    Returns:
        (advantages, returns), both (B, T), zero at padded positions, where
            delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            A_t     = delta_t + gamma * lam * A_{t+1}         (A after the last real token is 0)
            returns = A + V                                    (the lambda-return, the critic's target)
        Both outputs are constants (no gradient is needed).
    """
    raise NotImplementedError


def ppo_clip_loss(logp_new: Tensor, logp_old: Tensor, adv: Tensor, eps: float = 0.2,
                  mask: Tensor | None = None, eps_high: float | None = None,
                  agg: str = "token") -> Tensor:
    """PPO clipped surrogate loss (to be minimized).

    Args:
        logp_new: (B, T) log pi_theta(a_t | s_t), carries the gradient.
        logp_old: (B, T) log pi_theta_old(a_t | s_t) of the same tokens at sampling time. Constant.
        adv: (B, T) advantages, or (B,) for one advantage per response. Constant.
        eps: lower clip half-width: the ratio is clipped to [1 - eps, 1 + eps_high].
        mask: (B, T) 0/1, see the module docstring. Default: all ones.
        eps_high: upper clip half-width; None means the same as eps.
        agg: how the per-token losses are reduced.
            "token": sum of masked losses / number of real tokens in the whole batch.
            "seq": mean over each response's real tokens first, then mean over the B responses.
    Returns:
        A scalar: -mean of min(rho * A, clip(rho, 1 - eps, 1 + eps_high) * A) with rho = exp(logp_new - logp_old),
        reduced as described by `agg`. Padded positions contribute nothing.
    """
    raise NotImplementedError


def kl_estimators(logp: Tensor, logp_ref: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Per-token Monte-Carlo estimators k1, k2, k3 of KL(pi || pi_ref) (Schulman 2020).

    Args:
        logp: log pi(a_t | s_t) of tokens sampled from pi (the policy being trained).
        logp_ref: log pi_ref(a_t | s_t) of the same tokens under the reference policy. Same shape.
    With r = pi_ref / pi and log r = logp_ref - logp, return (k1, k2, k3) with the same shape as the inputs:
        k1 = -log r,   k2 = (log r)^2 / 2,   k3 = (r - 1) - log r.
    k3 must stay accurate when |log r| is tiny (float32): avoid computing exp(x) - 1 as a
    difference of two numbers close to 1. Everything must be differentiable in `logp`.
    """
    raise NotImplementedError


def masked_whiten(x: Tensor, mask: Tensor | None = None, shift_mean: bool = True,
                  eps: float = 1e-8) -> Tensor:
    """Normalize x over its real entries to zero mean and unit variance (advantage whitening).

    Args:
        x: (B, T) values, for example advantages.
        mask: (B, T) 0/1. The mean and the (biased, divide by N) variance are computed over
            entries with mask 1 only, pooled across the whole batch. Default: all ones.
        shift_mean: if True the output has mean 0. If False the mean is added back after
            whitening, so the output keeps the original mean and has unit variance
            (this is how reward whitening is done in some RLHF code bases).
        eps: added to the variance inside the square root.
    Returns:
        (B, T) tensor equal to (x - mean) / sqrt(var + eps) on real entries (plus mean if
        shift_mean is False) and exactly 0 on padded entries.
    """
    raise NotImplementedError
