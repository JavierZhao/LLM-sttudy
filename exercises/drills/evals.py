"""Drill (page 18): evaluation metrics and statistics.

Implement the functions below. Run:
    pytest exercises/tests/test_evals.py

Conventions: a "correct" indicator is 1 (right) or 0 (wrong) per question. Accuracy is a fraction in
[0, 1], not a percentage. Log-probabilities are natural logs. Everything is pure Python or NumPy.
"""
from typing import Optional, Sequence, Tuple

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimate for ONE problem (Chen et al. 2021).

    You drew n samples for the problem and c of them were correct. The estimate is the probability
    that a uniformly random subset of k of the n samples (without replacement) contains at least
    one correct sample.

    Args:
        n: total samples drawn, n >= 1.
        c: number of correct samples, 0 <= c <= n.
        k: the k in pass@k, 1 <= k <= n.
    Returns:
        A float in [0, 1]. It must be exact (for example n=20, c=3, k=5 gives 0.600877...)
        and numerically stable: it must work for n up to 10**6 without overflow or the
        catastrophic cancellation you get from dividing two huge binomial coefficients.
    Raises:
        ValueError: if k > n, c > n, c < 0, or k < 1.
    """
    raise NotImplementedError


def mean_pass_at_k(correct_counts: Sequence[int], n: int, k: int) -> float:
    """Benchmark-level pass@k: the mean of pass_at_k(n, c_i, k) over problems.

    Args:
        correct_counts: for each problem i, the number c_i of correct samples out of n.
        n: samples drawn per problem (the same for every problem).
        k: the k in pass@k.
    Returns:
        The average per-problem estimate, a float in [0, 1]. Note that this is NOT the same as
        1 - (1 - mean pass@1)**k, because problems differ in difficulty.
    """
    raise NotImplementedError


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Reliability metric pass^k for ONE problem (tau-bench): P(all k samples are correct).

    Unbiased estimate C(c, k) / C(n, k): the probability that a random subset of k of the n
    samples contains only correct ones.

    Args:
        n: total samples, n >= 1. c: correct samples, 0 <= c <= n. k: 1 <= k <= n.
    Returns:
        A float in [0, 1]. It is 0 whenever c < k. Must not overflow for large n.
    Raises:
        ValueError: on invalid arguments (same rules as pass_at_k).
    """
    raise NotImplementedError


def accuracy_ci(correct: int, n: int, z: float = 1.96, method: str = "wald") -> Tuple[float, float, float]:
    """Confidence interval for an accuracy measured on n independent questions.

    Args:
        correct: number of correct answers, 0 <= correct <= n.
        n: number of questions, n >= 1.
        z: normal quantile (1.96 for a two-sided 95% interval).
        method: "wald" for p_hat +/- z * sqrt(p_hat * (1 - p_hat) / n), or "wilson" for the Wilson
            score interval. The Wilson interval is centered at (p_hat + z^2/(2n)) / (1 + z^2/n)
            and has half-width z * sqrt(p_hat*(1-p_hat)/n + z^2/(4n^2)) / (1 + z^2/n).
    Returns:
        (p_hat, lo, hi) with p_hat = correct / n. Clip lo and hi to [0, 1].
    Raises:
        ValueError: for an unknown method or invalid counts.
    """
    raise NotImplementedError


def paired_bootstrap(a_correct: Sequence[int], b_correct: Sequence[int], n_boot: int = 2000,
                     seed: int = 0, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Paired bootstrap confidence interval for the accuracy difference mean(a) - mean(b).

    Both models were run on the SAME questions, so resample question indices (with replacement)
    and apply the same indices to both models.

    Args:
        a_correct, b_correct: equal-length 0/1 sequences, one entry per question.
        n_boot: number of bootstrap resamples.
        seed: seed for numpy.random.default_rng; the same seed must give the same result.
        alpha: two-sided level; use the percentile interval [alpha/2, 1 - alpha/2].
    Returns:
        (observed_difference, lo, hi), differences as fractions (not percentages).
    Raises:
        ValueError: if the inputs have different lengths or are empty.
    """
    raise NotImplementedError


def mc_loglik_choice(logprobs_per_choice: Sequence[float], lengths: Sequence[int],
                     normalize: bool = False,
                     unconditional_logprobs: Optional[Sequence[float]] = None) -> int:
    """Pick the answer of a multiple-choice question by cloze-style likelihood scoring.

    Args:
        logprobs_per_choice: for each choice i, the TOTAL log-probability log P(choice_i | question)
            of the continuation, summed over its tokens (all values <= 0).
        lengths: for each choice, its length in whatever unit you normalize by (tokens or characters),
            all >= 1.
        normalize: if True, divide each total log-probability by its length (a length-normalized
            score, as in lm-evaluation-harness acc_norm when lengths are characters).
        unconditional_logprobs: if given, score choice i by the pointwise-mutual-information style
            score log P(choice_i | question) - log P(choice_i | generic prompt) and ignore
            normalize and lengths. Same length as logprobs_per_choice.
    Returns:
        The index of the highest-scoring choice. Ties go to the lowest index.
    Raises:
        ValueError: if the sequences differ in length, are empty, or normalize is True with a
            non-positive length.
    """
    raise NotImplementedError
