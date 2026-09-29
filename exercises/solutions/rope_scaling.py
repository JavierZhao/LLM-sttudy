"""Reference solutions: RoPE context-extension methods and a needle-in-a-haystack grid (page 10)."""
import math
from typing import NamedTuple, Sequence

import torch
from torch import Tensor


def inv_freq(d_head: int, base: float = 10000.0) -> Tensor:
    """theta_i = base ** (-2 i / d_head), float64, shape (d_head // 2,)."""
    assert d_head % 2 == 0
    return base ** (-torch.arange(0, d_head, 2, dtype=torch.float64) / d_head)


def wavelengths(freqs: Tensor) -> Tensor:
    """Tokens per full turn: 2 pi / theta_i."""
    return 2 * math.pi / freqs


def pi_scale(freqs: Tensor, s: float) -> Tensor:
    """Scaling positions by 1/s equals scaling every frequency by 1/s; a new tensor is returned."""
    return freqs / s


def ntk_base(base: float, s: float, d_head: int) -> float:
    """Solve b'^(-(d-2)/d) = b^(-(d-2)/d) / s: the last pair (index d/2 - 1, exponent (d-2)/d) is slowed
    by exactly s, while pair 0 (exponent 0) is untouched.  =>  b' = b * s^(d / (d - 2))."""
    return base * s ** (d_head / (d_head - 2))


def _ramp_weight(d_head: int, base: float, orig_ctx: int, alpha: float, beta: float, ramp: str) -> Tensor:
    """g_i in [0, 1]: weight of the ORIGINAL frequency (1 = untouched, 0 = fully interpolated)."""
    n = d_head // 2
    if ramp == "wavelength":
        r = orig_ctx / wavelengths(inv_freq(d_head, base))                 # turns inside the original context
        return ((r - alpha) / (beta - alpha)).clamp(0.0, 1.0)              # linear in r (paper Eq. 11)
    if ramp == "index":
        def find(rot: float) -> float:                                     # pair index that makes `rot` turns in orig_ctx
            return d_head * math.log(orig_ctx / (rot * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(find(beta)), 0)
        high = min(math.ceil(find(alpha)), d_head - 1)
        if low == high:
            high += 0.001                                                  # avoid dividing by zero, as the reference code does
        i = torch.arange(n, dtype=torch.float64)
        return 1.0 - ((i - low) / (high - low)).clamp(0.0, 1.0)            # linear in the pair index
    raise ValueError(f"unknown ramp {ramp!r}")


def yarn_inv_freq(d_head: int, base: float, s: float, orig_ctx: int,
                  alpha: float = 1.0, beta: float = 32.0, ramp: str = "wavelength") -> Tensor:
    """NTK-by-parts: blend interpolated (theta/s) and original (theta) frequencies per pair."""
    theta = inv_freq(d_head, base)
    g = _ramp_weight(d_head, base, orig_ctx, alpha, beta, ramp)
    return (1.0 - g) * theta / s + g * theta                               # g = 1: untouched, g = 0: divided by s


def yarn_mscale(s: float) -> float:
    """sqrt(1/t) = 0.1 ln s + 1, and no change when there is no extension."""
    return 0.1 * math.log(s) + 1.0 if s > 1 else 1.0


def llama3_inv_freq(d_head: int, base: float, factor: float, low_freq_factor: float,
                    high_freq_factor: float, orig_ctx: int) -> Tensor:
    """Llama 3.1 rule. Same blend as YaRN's paper ramp with alpha = low_freq_factor, beta = high_freq_factor."""
    theta = inv_freq(d_head, base)
    lam = wavelengths(theta)
    low_wl = orig_ctx / low_freq_factor
    high_wl = orig_ctx / high_freq_factor
    w = (orig_ctx / lam - low_freq_factor) / (high_freq_factor - low_freq_factor)   # only used in the middle band
    blended = (1.0 - w) * theta / factor + w * theta
    out = torch.where(lam > low_wl, theta / factor, theta)                 # long wavelengths: interpolate
    middle = (lam >= high_wl) & (lam <= low_wl)                            # same band as HF's ~(<high) * ~(>low)
    return torch.where(middle, blended, out)


def effective_length(lengths: Sequence[int], scores: Sequence[float], threshold: float,
                     strict: bool = False) -> int:
    """Largest passing length (RULER), or the largest length whose whole prefix of lengths passes (strict)."""
    best = 0
    for length, score in zip(lengths, scores):
        if score >= threshold:
            best = length
        elif strict:
            break
    return best


class NiahCase(NamedTuple):
    prompt: str
    answer: str
    depth: float
    length: int


def make_niah_prompts(haystack_tokens: Sequence[str], needle: str, depths: Sequence[float],
                      lengths: Sequence[int], question: str = "", answer: str = "") -> list[NiahCase]:
    """One case per (length, depth): exactly `length` words, needle after floor(depth * n_hay) haystack words."""
    if len(haystack_tokens) == 0:
        raise ValueError("haystack_tokens is empty")
    if any(t == "" or len(t.split()) != 1 for t in haystack_tokens):
        raise ValueError("haystack tokens must be non-empty and contain no whitespace")
    if any(not (0.0 <= d <= 1.0) for d in depths):
        raise ValueError("depths must lie in [0, 1]")
    needle_tokens, question_tokens = needle.split(), question.split()
    cases = []
    for length in lengths:
        n_hay = length - len(needle_tokens) - len(question_tokens)
        if n_hay < 1:
            raise ValueError(f"length {length} leaves no room for haystack tokens")
        hay = [haystack_tokens[j % len(haystack_tokens)] for j in range(n_hay)]   # cycle a short haystack
        for depth in depths:
            pos = int(depth * n_hay)                                       # floor for depth >= 0
            tokens = hay[:pos] + needle_tokens + hay[pos:] + question_tokens
            cases.append(NiahCase(" ".join(tokens), answer, depth, length))
    return cases
