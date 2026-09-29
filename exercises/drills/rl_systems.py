"""Drill (page 25): RL systems primitives.

Implement the four functions below. Run:
    pytest exercises/tests/test_rl_systems.py
"""
from __future__ import annotations

import torch
from torch import Tensor


def truncated_is_weights(logp_trainer: Tensor, logp_rollout: Tensor, cap: float, mask: Tensor) -> Tensor:
    """Truncated importance-sampling weights for training-inference mismatch.

    Args:
        logp_trainer: (B, T) log-probs of the sampled tokens under the TRAINER (pi_train).
        logp_rollout: (B, T) log-probs of the same tokens as reported by the INFERENCE ENGINE (mu).
        cap: truncation threshold C > 0. The cap is one-sided: weights are limited from above only.
        mask: (B, T) bool or {0, 1}. 1 marks real, trainable tokens; 0 marks padding or non-policy
            tokens, whose log-prob entries may hold garbage (-inf, nan).
    Returns:
        (B, T) float tensor with w = min(exp(logp_trainer - logp_rollout), cap) where mask is 1 and
        exactly 0 elsewhere. The weights are constants in the loss: no gradient may flow through
        them. The function must not overflow for large log-ratios (for example +100 in float32)
        and must not return nan at masked positions.
    """
    raise NotImplementedError


def agent_loss_mask(
    segments: list[tuple[str, list[int]]], train_roles: tuple[str, ...] = ("assistant",)
) -> tuple[list[int], list[int]]:
    """Flatten a multi-turn trajectory into (input_ids, loss_mask).

    Args:
        segments: list of (role, token_ids) in trajectory order. role is one of "system", "user",
            "assistant", "tool". token_ids is a list of ints and may be empty.
        train_roles: roles whose tokens were sampled by the policy and are therefore trained.
    Returns:
        input_ids: the concatenation of all token_ids, in order.
        loss_mask: list of 0/1 ints of the same length; 1 for every token of a segment whose role
            is in train_roles, 0 otherwise.
    Raises:
        ValueError: if a segment has a role outside the four above.
    """
    raise NotImplementedError


def sequence_ratio_drift(per_token_logp_diff: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """How per-token log-probability gaps compound into sequence-level importance ratios.

    Args:
        per_token_logp_diff: (B, T) d_t = log pi_train(y_t) - log mu_rollout(y_t). Entries at padded
            positions (index >= length) are arbitrary and must be ignored (they may be nan or inf).
        lengths: (B,) integer tensor, the number of real tokens per sequence (0 <= length <= T).
    Returns:
        A tuple of three float64 tensors of shape (B,):
        seq_log_ratio: sum of d_t over the first lengths[b] positions.
        seq_ratio: exp(seq_log_ratio). Computing in float64 keeps e^160 finite.
        geo_mean_ratio: exp(seq_log_ratio / max(length, 1)), the length-normalized ratio.
        A sequence of length 0 gives log-ratio 0 and both ratios equal to 1.
    """
    raise NotImplementedError


def staleness_admits(n_submitted: int, batch_size: int, policy_version: int, eta: int) -> bool:
    """AReaL's admission rule for a new generation request.

    A request is admitted iff floor((N_r - 1) / B) <= i + eta, where N_r counts the new request
    itself, B is the trainer batch size, i the current policy version and eta the maximum staleness.

    Args:
        n_submitted: number of requests already submitted before this one (so N_r = n_submitted + 1).
        batch_size: B > 0, trajectories consumed per training step.
        policy_version: i >= 0, number of completed training steps.
        eta: maximum permitted staleness, eta >= 0 (eta = 0 is fully synchronous).
    Returns:
        True if the new request may be submitted, else False.
    """
    raise NotImplementedError
