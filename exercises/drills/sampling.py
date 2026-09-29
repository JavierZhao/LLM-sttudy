"""Drill (page 06): sampling the next token from logits.

Implement temperature scaling, the three truncation filters (top-k, top-p, min-p) and a
`sample` function that chains them in the order Hugging Face `generate` uses:
temperature -> top-k -> top-p -> min-p. Run:
    pytest exercises/tests/test_sampling.py

Conventions shared by all functions:
  * `logits` has shape (..., V); every operation acts on the last axis and treats the
    leading axes as independent rows.
  * "Removed" means the logit is replaced by -inf. Kept logits are returned unchanged.
  * A filter must never remove the most probable token, so at least one token survives.
  * Inputs may already contain -inf (from an earlier filter); those tokens have probability 0.
"""
from typing import Optional

import torch
from torch import Tensor

NEG_INF = float("-inf")


def apply_temperature(logits: Tensor, tau: float) -> Tensor:
    """Scale logits by 1 / tau.

    Args:
        logits: (..., V) floating-point logits.
        tau: temperature. Must be > 0, otherwise raise ValueError.
    Returns:
        (..., V) tensor equal to logits / tau, same dtype. tau < 1 sharpens the softmax,
        tau > 1 flattens it.
    """
    raise NotImplementedError


def top_k_filter(logits: Tensor, k: int) -> Tensor:
    """Keep only the k largest logits in each row.

    Args:
        logits: (..., V).
        k: number of tokens to keep. Must be >= 1, otherwise raise ValueError.
           If k >= V, nothing is removed.
    Returns:
        (..., V) with every other entry set to -inf.
    Ties: every token whose logit equals the k-th largest logit is kept, so more than k
    tokens can survive (this is what Hugging Face's TopKLogitsWarper does).
    """
    raise NotImplementedError


def top_p_filter(logits: Tensor, p: float) -> Tensor:
    """Nucleus (top-p) filtering, Holtzman et al. 2019.

    Args:
        logits: (..., V).
        p: cumulative-probability threshold, must satisfy 0 < p <= 1, otherwise raise ValueError.
    Returns:
        (..., V). Let probs = softmax(logits), sorted in descending order. Keep the smallest
        prefix of that order whose cumulative probability is >= p and set all other logits to
        -inf. The most probable token is always kept. With p == 1 every token is kept.
    Ties: sort stably in descending probability, so among equal probabilities the token with
    the lower index comes first. The cut may fall inside a group of tied tokens, in which case
    only the lower-index part of the group is kept.
    """
    raise NotImplementedError


def min_p_filter(logits: Tensor, p_min: float) -> Tensor:
    """Min-p filtering, Nguyen et al. 2024.

    Args:
        logits: (..., V).
        p_min: relative threshold, must satisfy 0 <= p_min <= 1, otherwise raise ValueError.
    Returns:
        (..., V). Let probs = softmax(logits). Keep the tokens with probs >= p_min * probs.max()
        (per row) and set the rest to -inf. Tokens tied with the maximum are all kept.
    """
    raise NotImplementedError


def sample(
    logits: Tensor,
    generator: torch.Generator,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    min_p: Optional[float] = None,
) -> Tensor:
    """Draw one token id per row from the (filtered) next-token distribution.

    Args:
        logits: (..., V), for example (V,) for one sequence or (B, V) for a batch.
        generator: a torch.Generator. All randomness must come from it (pass it to
            torch.multinomial), so that a fixed seed reproduces the draw exactly.
        temperature: 0 means greedy decoding (argmax, lowest index on ties, no randomness).
            Values > 0 scale the logits. Negative values raise ValueError.
        top_k, top_p, min_p: filters, disabled when None.
    Order of operations: temperature, then top-k, then top-p, then min-p. Each filter sees the
    output of the previous one, so its softmax is renormalized over the surviving tokens
    (this matters for top-p). Finally take softmax and sample.
    Returns:
        LongTensor of shape logits.shape[:-1].
    """
    raise NotImplementedError
