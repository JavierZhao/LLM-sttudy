"""Drill (page 35): beam search with length normalization and EOS handling.

Implement `normalized_score`, `beam_search` and `exhaustive_search` (the brute-force oracle you
test the beam search against). Run:
    pytest exercises/tests/test_beam_search.py

The model is a black box, `step_logprobs_fn`, called once per decoding step on all live
hypotheses at once (the interface of a batched decoder):
    step_logprobs_fn(prefixes: LongTensor (n, t)) -> Tensor (n, V)
Row i holds log P(next token | prefixes[i]) over the V tokens. Every prefix starts with `bos`
and all n prefixes have the same length t >= 1. Entries may be -inf (a forbidden token).
The function is never called with n = 0.

Conventions (the tests rely on all of them):
  * A hypothesis is the generated tokens after `bos`. `max_len` is the largest number of generated
    tokens, and the EOS token counts as a generated token. Its log-probability is added to the
    score, and its length is included in |Y|.
  * A hypothesis is FINISHED when it emits `eos` (the EOS token is kept as its last token), or when
    it reaches `max_len` tokens without EOS (a truncated hypothesis, no EOS appended).
  * Score of a finished hypothesis with total log-probability L and length n (GNMT, Wu et al. 2016,
    Eq. 14, without the coverage term):   L / (((5 + n) / 6) ** alpha).
    alpha = length_penalty >= 0, and alpha = 0 means no normalization. Cumulative sums in the
    search are ranked by L alone: within one step all candidates have the same length.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor


def normalized_score(sum_logprob: float, length: int, alpha: float) -> float:
    """Length-normalized score of a finished hypothesis.

    Args:
        sum_logprob: total log-probability of the generated tokens (EOS included if emitted).
        length: number of generated tokens |Y| (EOS included if emitted), at least 1.
        alpha: length penalty exponent, alpha >= 0.
    Returns:
        sum_logprob / (((5 + length) / 6) ** alpha) as a Python float.
    """
    raise NotImplementedError


def beam_search(
    step_logprobs_fn: Callable[[Tensor], Tensor],
    bos: int,
    eos: int,
    beam: int,
    max_len: int,
    length_penalty: float = 0.0,
    early_stop: bool = False,
) -> list[tuple[list[int], float]]:
    """Beam search that keeps `beam` live hypotheses and collects finished ones.

    Step rule (repeat for step = 1 .. max_len):
      1. Call step_logprobs_fn once on the live hypotheses (initially the single prefix [bos]).
      2. Every live hypothesis has exactly one EOS extension. Each one with a finite score is a
         finished hypothesis, and finished hypotheses are never extended again.
      3. The new live set is the `beam` best NON-EOS extensions overall, ranked by cumulative
         log-probability. Candidates with score -inf never enter the live set, so it may hold
         fewer than `beam` hypotheses, or none (then the search ends).
      4. After step max_len, the surviving live hypotheses are finished as truncated hypotheses
         (no EOS), with length max_len.
    Keep only the best `beam` finished hypotheses by normalized score.

    early_stop=True adds one exact stopping rule and must return exactly the same list as
    early_stop=False (compare with strict inequality): after a step (before max_len) with
    `beam` finished hypotheses kept, stop if the optimistic score of every live hypothesis is
    strictly below the worst kept finished score. The optimistic score of a live hypothesis with
    log-probability L is L / (((5 + max_len) / 6) ** alpha): any completion has a
    log-probability <= L <= 0 and a length <= max_len.

    Args:
        step_logprobs_fn: see the module docstring.
        bos, eos: token ids. Generated tokens may include any id except that `eos` ends a hypothesis.
        beam: beam width, at least 1. It may exceed the vocabulary size or the number of candidates.
        max_len: maximum number of generated tokens (EOS included), at least 1.
        length_penalty: alpha >= 0. Raise ValueError if negative. Also raise it for beam < 1 or
            max_len < 1.
        early_stop: see above.
    Returns:
        Up to `beam` pairs (tokens, score), best first. `tokens` are Python ints without `bos`,
        ending with `eos` if the hypothesis ended by EOS. `score` is the normalized score as a
        Python float. Empty if nothing finishes.
    """
    raise NotImplementedError


def exhaustive_search(
    step_logprobs_fn: Callable[[Tensor], Tensor],
    bos: int,
    eos: int,
    max_len: int,
    length_penalty: float = 0.0,
    top_n: int = 1,
) -> list[tuple[list[int], float]]:
    """Brute-force oracle: score EVERY finished hypothesis and return the best `top_n`.

    Same hypotheses, lengths and scores as beam_search: every sequence that ends with `eos` at
    some length <= max_len, and every sequence of exactly max_len tokens that contains no EOS.
    Tokens with log-probability -inf are never used. Call step_logprobs_fn with one prefix at a
    time (a batch of size 1).

    Returns:
        Up to `top_n` pairs (tokens, score), best first, in the format of beam_search.
    """
    raise NotImplementedError
