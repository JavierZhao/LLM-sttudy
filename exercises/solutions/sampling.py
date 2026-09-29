"""Reference solutions: sampling from logits (page 06)."""
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

NEG_INF = float("-inf")


def _probs(logits: Tensor) -> Tensor:
    """softmax in at least float32 (bf16/fp16 logits are upcast; float64 is kept)."""
    x = logits if logits.dtype in (torch.float32, torch.float64) else logits.float()
    return torch.softmax(x, dim=-1)


def apply_temperature(logits: Tensor, tau: float) -> Tensor:
    """logits / tau; tau must be positive."""
    if tau <= 0:
        raise ValueError(f"temperature must be > 0, got {tau}")
    return logits / tau


def top_k_filter(logits: Tensor, k: int) -> Tensor:
    """Keep the k largest logits per row (all ties with the k-th value are kept)."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    k = min(k, logits.shape[-1])
    kth = torch.topk(logits, k, dim=-1).values[..., -1:]        # (..., 1) value of the k-th largest
    return logits.masked_fill(logits < kth, NEG_INF)


def top_p_filter(logits: Tensor, p: float) -> Tensor:
    """Keep the smallest descending-probability prefix whose mass is >= p."""
    if not 0.0 < p <= 1.0:
        raise ValueError(f"p must be in (0, 1], got {p}")
    if p == 1.0:
        return logits
    probs = _probs(logits)
    sorted_probs, order = torch.sort(probs, dim=-1, descending=True, stable=True)
    cum = sorted_probs.cumsum(dim=-1)
    # mass strictly BEFORE each token: shift the cumulative sum right by one (exact, no subtraction)
    before = F.pad(cum[..., :-1], (1, 0), value=0.0)
    drop_sorted = before >= p          # a token is needed only while the mass so far is still < p
    drop = torch.zeros_like(drop_sorted).scatter(-1, order, drop_sorted)   # back to vocabulary order
    return logits.masked_fill(drop, NEG_INF)


def min_p_filter(logits: Tensor, p_min: float) -> Tensor:
    """Keep tokens with probability >= p_min * (largest probability in the row)."""
    if not 0.0 <= p_min <= 1.0:
        raise ValueError(f"p_min must be in [0, 1], got {p_min}")
    probs = _probs(logits)
    threshold = p_min * probs.amax(dim=-1, keepdim=True)        # relative to the row's most likely token
    return logits.masked_fill(probs < threshold, NEG_INF)


def sample(
    logits: Tensor,
    generator: torch.Generator,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    min_p: Optional[float] = None,
) -> Tensor:
    """temperature -> top-k -> top-p -> min-p -> softmax -> multinomial (Hugging Face order)."""
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if temperature == 0:
        return logits.argmax(dim=-1)      # greedy: no randomness; the filters never remove the argmax
    x = apply_temperature(logits, temperature)
    if top_k is not None:
        x = top_k_filter(x, top_k)
    if top_p is not None:
        x = top_p_filter(x, top_p)
    if min_p is not None:
        x = min_p_filter(x, min_p)
    probs = _probs(x)                                           # float32 even for bf16 logits
    flat = probs.reshape(-1, probs.shape[-1])                   # multinomial wants (N, V)
    ids = torch.multinomial(flat, num_samples=1, generator=generator).squeeze(-1)
    return ids.reshape(probs.shape[:-1])
