"""Drill (page 10): RoPE context-extension methods and a needle-in-a-haystack grid.

Implement every function below (replace `raise NotImplementedError`). Run:
    pytest exercises/tests/test_rope_scaling.py

Conventions
-----------
* A head of dimension d_head has d_head // 2 rotation pairs. Pair i (i = 0 .. d_head/2 - 1) turns by
  theta_i radians per token, theta_i = base ** (-2 i / d_head), so pair 0 is the fastest (1 rad/token)
  and the last pair is the slowest. Its wavelength is lambda_i = 2 pi / theta_i tokens.
* Every method here is a rule "new frequencies from old frequencies". Inputs and outputs are
  float64 tensors of shape (d_head // 2,); no positions and no model are involved.
* s = L' / L is the scale factor: target context over the context the model was trained on
  (`orig_ctx`). Every method must return the vanilla frequencies exactly when s = 1.
"""
import math
from typing import NamedTuple, Sequence

import torch
from torch import Tensor


def inv_freq(d_head: int, base: float = 10000.0) -> Tensor:
    """Vanilla RoPE frequencies theta_i = base ** (-2 i / d_head), i = 0 .. d_head/2 - 1.

    Returns:
        (d_head // 2,) float64 tensor. theta_0 = 1; the last entry is base ** (-(d_head - 2) / d_head).
        d_head must be even.
    """
    raise NotImplementedError


def wavelengths(freqs: Tensor) -> Tensor:
    """Tokens per full turn of each pair: 2 pi / theta_i. Same shape and dtype as `freqs`."""
    raise NotImplementedError


def pi_scale(freqs: Tensor, s: float) -> Tensor:
    """Position Interpolation (Chen et al. 2023) written as a frequency rule.

    Scaling positions m -> m / s is the same as scaling every frequency theta_i -> theta_i / s.
    Returns a new tensor and leaves `freqs` unchanged.
    """
    raise NotImplementedError


def ntk_base(base: float, s: float, d_head: int) -> float:
    """The "NTK-aware" base change b' such that inv_freq(d_head, b') keeps the fastest pair
    unchanged and slows the slowest pair by exactly s (YaRN paper, Appendix A.2). Returns a float.
    """
    raise NotImplementedError


def yarn_inv_freq(d_head: int, base: float, s: float, orig_ctx: int,
                  alpha: float = 1.0, beta: float = 32.0, ramp: str = "wavelength") -> Tensor:
    """YaRN's per-pair frequencies ("NTK-by-parts"; Peng et al. 2023, Eq. 10-13). No attention scaling here.

    Let r_i = orig_ctx / lambda_i be the number of turns pair i makes inside the original context.
        r_i <= alpha  -> the pair is fully interpolated: theta_i / s      (wavelength longer than the context)
        r_i >= beta   -> the pair is untouched:          theta_i          (many turns inside the context)
        in between    -> blend (1 - g) * theta_i / s + g * theta_i, where g in [0, 1] rises from 0 to 1.

    `ramp` chooses how g rises:
        "wavelength": g_i = clip((r_i - alpha) / (beta - alpha), 0, 1). The formula printed in the paper;
                      linear in r_i, so linear in frequency.
        "index":      the convention of the reference code (Hugging Face `rope_type="yarn"`, DeepSeek-V3's
                      `precompute_freqs_cis`). With dim = d_head, define
                          find(rot) = dim * ln(orig_ctx / (rot * 2 pi)) / (2 ln base)
                          low  = max(floor(find(beta)), 0)        high = min(ceil(find(alpha)), dim - 1)
                      and g_i = 1 - clip((i - low) / (high - low), 0, 1). Linear in the pair INDEX i between
                      the two boundary pairs (so in log-wavelength). If low == high use high + 0.001.
                      The two conventions agree far from the ramp and differ inside it.

    Args:
        d_head: head dimension actually rotated (even). base: RoPE base b. s: scale factor L'/L.
        orig_ctx: context length the model was trained with (L). alpha, beta: ramp boundaries in turns.
    Returns:
        (d_head // 2,) float64 tensor. Equals inv_freq(d_head, base) when s == 1.
    """
    raise NotImplementedError


def yarn_mscale(s: float) -> float:
    """YaRN's attention temperature, as the amplitude sqrt(1/t) = 0.1 ln(s) + 1 (Peng et al. 2023, Eq. 15).

    The amplitude multiplies both q and k (for instance by scaling cos and sin), so the attention logits
    are multiplied by yarn_mscale(s) ** 2. Return 1.0 when s <= 1 (no extension, no change).
    """
    raise NotImplementedError


def llama3_inv_freq(d_head: int, base: float, factor: float, low_freq_factor: float,
                    high_freq_factor: float, orig_ctx: int) -> Tensor:
    """Llama 3.1's `rope_type="llama3"` rule (Hugging Face `_compute_llama3_parameters`).

    With wavelength lambda_i = 2 pi / theta_i and
        low_freq_wavelen  = orig_ctx / low_freq_factor,
        high_freq_wavelen = orig_ctx / high_freq_factor:
        lambda_i <  high_freq_wavelen -> theta_i                      (unchanged)
        lambda_i >  low_freq_wavelen  -> theta_i / factor             (fully interpolated)
        otherwise                     -> (1 - w) * theta_i / factor + w * theta_i,
                                         w = (orig_ctx / lambda_i - low_freq_factor) / (high_freq_factor - low_freq_factor)
    Llama 3.1 ships base 500000, factor 8, low_freq_factor 1, high_freq_factor 4, orig_ctx 8192.

    Returns:
        (d_head // 2,) float64 tensor. Equals inv_freq(d_head, base) when factor == 1.
    """
    raise NotImplementedError


def effective_length(lengths: Sequence[int], scores: Sequence[float], threshold: float,
                     strict: bool = False) -> int:
    """RULER-style effective context length: which tested length still "passes"?

    Args:
        lengths: tested context lengths in ascending order. scores: score at each length.
        threshold: pass mark (RULER uses Llama-2-7B's score at 4K, 85.6; NoLiMa uses 85% of the base score).
        strict: if False, return the largest tested length whose score is >= threshold (RULER's
            definition: a non-monotone curve can pass at a length above a failing one). If True, return the
            largest length L such that every tested length <= L passes.
    Returns:
        The length, or 0 if no length passes.
    """
    raise NotImplementedError


class NiahCase(NamedTuple):
    prompt: str    # the full prompt: haystack with the needle inserted, then the question
    answer: str    # the string a correct response must contain
    depth: float   # requested needle depth in [0, 1]
    length: int    # number of whitespace-separated tokens in `prompt`


def make_niah_prompts(haystack_tokens: Sequence[str], needle: str, depths: Sequence[float],
                      lengths: Sequence[int], question: str = "", answer: str = "") -> list[NiahCase]:
    """Build a needle-in-a-haystack grid (pure Python, no model).

    A "token" here is one whitespace-separated word, so lengths are exact and testable. For every
    (length, depth) pair build ONE case whose prompt has exactly `length` tokens:
        n_hay = length - len(needle.split()) - len(question.split())      (haystack tokens needed)
        haystack token j is haystack_tokens[j % len(haystack_tokens)]      (the haystack cycles if too short)
        pos   = floor(depth * n_hay)                                       (haystack tokens BEFORE the needle)
        prompt = " ".join(haystack[:pos] + needle.split() + haystack[pos:n_hay] + question.split())
    So depth 0.0 puts the needle first and depth 1.0 puts it right before the question.

    Order: lengths in the outer loop, depths in the inner loop, both in the order given
    (len(lengths) * len(depths) cases). `answer` is copied into each case unchanged.

    Raises:
        ValueError: if a depth is outside [0, 1], or a length leaves no room for n_hay >= 1.
        ValueError: if a haystack token is empty or contains whitespace, or `haystack_tokens` is empty.
    """
    raise NotImplementedError
