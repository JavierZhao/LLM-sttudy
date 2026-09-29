"""Drill (page 17): speculative decoding with the lossless accept/resample rule.

Implement the four functions below. Run:
    pytest exercises/tests/test_speculative.py

Notation (Leviathan et al. 2023): p is the TARGET distribution we must sample from, q is the DRAFT
distribution the guesses came from. Both are probability vectors over the vocabulary, already
adjusted for temperature, top-k or top-p (the correctness proof holds for whatever distributions
you pass in, so the caller applies those adjustments before calling). One speculative step drafts
gamma tokens and then scores them with the target in a single pass, which gives gamma + 1
target distributions: one for each drafted position and one more after the last drafted token.
"""
from typing import List

import torch
from torch import Tensor


def acceptance_rate(p: Tensor, q: Tensor) -> float:
    """Probability that a token sampled from q is accepted by the rule min(1, p(x)/q(x)).

    Args:
        p: (V,) target probabilities, non-negative, sums to 1.
        q: (V,) draft probabilities, non-negative, sums to 1.
    Returns:
        beta = sum_x min(p(x), q(x)) as a Python float. It equals 1 - TV(p, q), so it is 1 exactly
        when p == q and 0 when the supports are disjoint.
    """
    raise NotImplementedError


def expected_tokens_per_step(alpha: float, gamma: int) -> float:
    """Expected number of tokens emitted by one speculative step.

    Assume every draft token is accepted independently with the same probability alpha. A step
    emits the accepted prefix of the draft plus exactly one more token (the correction after a
    rejection, or the bonus token after gamma acceptances).

    Args:
        alpha: per-token acceptance probability in [0, 1].
        gamma: number of drafted tokens, an integer >= 0.
    Returns:
        (1 - alpha**(gamma + 1)) / (1 - alpha), which equals gamma + 1 when alpha == 1 (do not divide
        by zero) and 1 when gamma == 0 or alpha == 0.
    Raises:
        ValueError if alpha is outside [0, 1] or gamma < 0.
    """
    raise NotImplementedError


def speedup(alpha: float, gamma: int, c: float) -> float:
    """Expected wall-clock speedup over plain decoding (Leviathan et al., Theorem 3.8).

    One speculative step costs gamma draft forward passes plus one target pass. With c the ratio
    (time of one draft pass) / (time of one target pass), the step takes (1 + gamma * c) target-pass
    units, and plain decoding produces one token per unit.

    Args:
        alpha: per-token acceptance probability in [0, 1].
        gamma: number of drafted tokens (>= 0).
        c: draft-to-target cost ratio (>= 0).
    Returns:
        expected_tokens_per_step(alpha, gamma) / (1 + gamma * c).
    """
    raise NotImplementedError


def speculative_accept(p_target: Tensor, q_draft: Tensor, draft_tokens: Tensor,
                       generator: torch.Generator) -> List[int]:
    """Run the accept/resample rule for ONE draft sequence and return the tokens to emit.

    Args:
        p_target: (gamma + 1, V) float tensor. Row i is the target distribution for the token at
            draft position i, conditioned on the context plus draft_tokens[:i]. The last row
            (index gamma) is the target distribution after all drafted tokens.
        q_draft: (gamma, V) float tensor. Row i is the draft distribution that draft_tokens[i] was
            sampled from. Every drafted token has q_draft[i, token] > 0.
        draft_tokens: (gamma,) int64 tensor of drafted token ids. gamma may be 0.
        generator: torch.Generator that supplies ALL randomness (use it for torch.rand and
            torch.multinomial, so that the same seed reproduces the same output).
    Returns:
        A list of Python ints of length between 1 and gamma + 1:
          - the longest accepted prefix of draft_tokens, where token i is accepted independently
            with probability min(1, p_target[i, x] / q_draft[i, x]) and acceptance stops at the
            first rejection, followed by exactly one more token:
          - at the first rejection at position i, a token sampled from the residual distribution
            norm(max(0, p_target[i] - q_draft[i])); if that residual has no mass (p == q up to
            rounding), sample from p_target[i] instead;
          - if all gamma tokens were accepted, a bonus token sampled from p_target[gamma].
        The emitted sequence has exactly the distribution of sampling from the target model alone.
    """
    raise NotImplementedError
