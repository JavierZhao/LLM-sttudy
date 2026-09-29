"""Drill (page 23): the numerical core of a PPO-RLHF loop.

Implement the functions and the class below. Run:
    pytest exercises/tests/test_rlhf.py

Conventions used throughout:
  * A batch holds B sampled responses. Everything per token is right-padded to length T, and
    `mask` is (B, T) with 1 on real response tokens and 0 on padding. Prompt tokens are NOT part of
    these tensors: they never receive a reward.
  * `logp_policy` and `logp_ref` are the log-probabilities (in nats) of the SAMPLED tokens under
    the current policy and under the frozen reference (SFT) model, shape (B, T). Their difference
    is the single-sample estimate of the per-token KL that InstructGPT and Ziegler et al. use.
  * Rewards and advantages are constants for the policy update: results must not carry gradients.
"""
import torch
from torch import Tensor


def whiten(x: Tensor, mask: Tensor, shift_mean: bool = True, eps: float = 1e-8) -> Tensor:
    """Normalize the entries of x that lie under the mask (Ziegler et al. 2019 style).

    Args:
        x: (B, T) float values (advantages or rewards).
        mask: (B, T) bool or float, 1 where x is a real value, 0 on padding.
        shift_mean: if True, return (x - mean) / sqrt(var + eps). If False, return
            (x - mean) / sqrt(var + eps) + mean, i.e. rescale to unit variance but keep the mean
            (this is how the reference implementation whitens rewards).
        eps: added to the variance inside the square root.
    Returns:
        (B, T) tensor. Mean and variance are computed over ALL entries where mask == 1, pooled
        across the whole batch (not per row), and the variance is the population variance
        (divide by the count, not count - 1). Entries where mask == 0 are returned as exactly 0.
    """
    raise NotImplementedError


def shaped_rewards(rm_score: Tensor, logp_policy: Tensor, logp_ref: Tensor, beta: float,
                   mask: Tensor) -> Tensor:
    """Per-token reward vector for one PPO-RLHF batch.

    Every real response token t gets the KL penalty  -beta * (logp_policy[t] - logp_ref[t]).
    The scalar reward-model score is then ADDED to the last real response token of each row.
    Padding positions get exactly 0.

    Args:
        rm_score: (B,) scalar reward-model score of each full response.
        logp_policy, logp_ref: (B, T) log-probs of the sampled tokens (see module docstring).
        beta: KL coefficient (float).
        mask: (B, T) with 1 on response tokens. The ones of each row are contiguous and start at
            column 0 (right padding), and every row has at least one 1.
    Returns:
        (B, T) float tensor of rewards, detached from the autograd graph.
    """
    raise NotImplementedError


def apply_eos_penalty(rm_score: Tensor, ended_with_eos: Tensor, penalty: float = -1.0) -> Tensor:
    """Truncation handling ("EOS trick"): a response that hit the length limit without emitting
    EOS gets a fixed penalty INSTEAD of its reward-model score.

    Args:
        rm_score: (B,) reward-model scores.
        ended_with_eos: (B,) bool, True if the response ended with the EOS token.
        penalty: constant used for the rows without EOS (replaces the score, is not added to it).
    Returns:
        (B,) tensor with rm_score where ended_with_eos, and `penalty` elsewhere.
    """
    raise NotImplementedError


class AdaptiveKLController:
    """Proportional controller for the KL coefficient beta (Ziegler et al. 2019; the released
    lm-human-preferences code). After every PPO batch, call update() with the measured KL.

        error = clip(current_kl / target_kl - 1, -0.2, 0.2)
        beta  = beta * (1 + error * n_steps / horizon)

    Attributes:
        beta: the current coefficient (float), initialized to init_beta.
    """

    def __init__(self, init_beta: float, target_kl: float, horizon: int):
        """
        Args:
            init_beta: starting coefficient.
            target_kl: KL (nats per sequence, summed over tokens) the controller steers towards.
            horizon: number of episodes over which a saturated error would change beta by
                (a factor of) about 1 + error.
        """
        raise NotImplementedError

    def update(self, current_kl: float, n_steps: int) -> float:
        """Apply one controller step and return the new beta.

        Args:
            current_kl: mean over the batch of the per-sequence KL (sum over response tokens of
                logp_policy - logp_ref).
            n_steps: number of episodes (sequences) in the batch that produced current_kl.
        """
        raise NotImplementedError
