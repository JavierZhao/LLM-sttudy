"""Reference solutions: evaluation metrics and statistics (page 18)."""
from typing import Optional, Sequence, Tuple

import numpy as np


def _check(n: int, c: int, k: int) -> None:
    if n < 1 or k < 1 or k > n:
        raise ValueError(f"need 1 <= k <= n, got n={n}, k={k}")
    if c < 0 or c > n:
        raise ValueError(f"need 0 <= c <= n, got n={n}, c={c}")


def pass_at_k(n: int, c: int, k: int) -> float:
    """1 - C(n-c, k) / C(n, k), evaluated as 1 - prod_{i=n-c+1}^{n} (1 - k/i)."""
    _check(n, c, k)
    if n - c < k:
        return 1.0  # every k-subset must contain a correct sample
    # C(n-c,k)/C(n,k) = prod_{i=n-c+1}^{n} (i-k)/i. Each factor is in (0, 1], so nothing overflows,
    # and log1p keeps the product accurate when c is large.
    i = np.arange(n - c + 1, n + 1, dtype=np.float64)
    log_fail = np.sum(np.log1p(-k / i))
    return float(-np.expm1(log_fail))  # 1 - exp(log_fail), accurate when the failure prob is near 1


def mean_pass_at_k(correct_counts: Sequence[int], n: int, k: int) -> float:
    """Average the per-problem estimates (problems differ in difficulty, so average AFTER the estimator)."""
    return float(np.mean([pass_at_k(n, int(c), k) for c in correct_counts]))


def pass_hat_k(n: int, c: int, k: int) -> float:
    """C(c, k) / C(n, k) = prod_{j=0}^{k-1} (c - j) / (n - j)."""
    _check(n, c, k)
    if c < k:
        return 0.0
    j = np.arange(k, dtype=np.float64)
    return float(np.exp(np.sum(np.log((c - j) / (n - j)))))


def accuracy_ci(correct: int, n: int, z: float = 1.96, method: str = "wald") -> Tuple[float, float, float]:
    if n < 1 or correct < 0 or correct > n:
        raise ValueError("need n >= 1 and 0 <= correct <= n")
    p = correct / n
    if method == "wald":
        half = z * np.sqrt(p * (1 - p) / n)
        centre = p
    elif method == "wilson":
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    else:
        raise ValueError(f"unknown method {method!r}")
    return p, float(max(0.0, centre - half)), float(min(1.0, centre + half))


def paired_bootstrap(a_correct: Sequence[int], b_correct: Sequence[int], n_boot: int = 2000,
                     seed: int = 0, alpha: float = 0.05) -> Tuple[float, float, float]:
    a = np.asarray(a_correct, dtype=np.float64)
    b = np.asarray(b_correct, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("a_correct and b_correct must be equal-length, non-empty 1-D sequences")
    d = a - b                      # per-question difference in {-1, 0, +1}; the pairing lives here
    n = d.size
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    chunk = max(1, min(n_boot, 4_000_000 // n))  # bound memory: chunk * n indices at a time
    for s in range(0, n_boot, chunk):
        m = min(chunk, n_boot - s)
        idx = rng.integers(0, n, size=(m, n))     # same indices for both models: resample d, not a and b
        means[s:s + m] = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)


def mc_loglik_choice(logprobs_per_choice: Sequence[float], lengths: Sequence[int],
                     normalize: bool = False,
                     unconditional_logprobs: Optional[Sequence[float]] = None) -> int:
    lp = np.asarray(logprobs_per_choice, dtype=np.float64)
    ln = np.asarray(lengths, dtype=np.float64)
    if lp.size == 0 or lp.shape != ln.shape:
        raise ValueError("logprobs_per_choice and lengths must be non-empty and the same length")
    if unconditional_logprobs is not None:
        un = np.asarray(unconditional_logprobs, dtype=np.float64)
        if un.shape != lp.shape:
            raise ValueError("unconditional_logprobs must match logprobs_per_choice")
        score = lp - un            # log P(a|q) - log P(a|generic prompt): removes the answer's prior plausibility
    elif normalize:
        if np.any(ln <= 0):
            raise ValueError("lengths must be positive when normalize=True")
        score = lp / ln            # long answers have more factors < 1, so divide by length
    else:
        score = lp
    return int(np.argmax(score))   # argmax returns the first maximum: ties go to the lowest index
