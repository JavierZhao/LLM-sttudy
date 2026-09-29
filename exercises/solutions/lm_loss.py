"""Reference solutions: the language-modeling loss and its metrics (page 01)."""
import math
from typing import Tuple

import torch
from torch import Tensor

IGNORE_INDEX = -100


def shift_for_lm(tokens: Tensor, pad_id: int) -> Tuple[Tensor, Tensor]:
    """inputs = tokens; labels[:, t] = tokens[:, t + 1], with padding and the last slot set to -100."""
    inputs = tokens.clone()
    labels = torch.full_like(tokens, IGNORE_INDEX)
    labels[:, :-1] = tokens[:, 1:]                 # shift left by one: position t predicts token t + 1
    labels[labels == pad_id] = IGNORE_INDEX        # padding is never a target (masks by value)
    return inputs, labels


def lm_cross_entropy(logits: Tensor, labels: Tensor, ignore_index: int = IGNORE_INDEX) -> Tensor:
    """Token-mean NLL with a hand-written, numerically stable log-softmax."""
    valid = labels != ignore_index                                   # (B, T) bool
    safe_labels = labels.masked_fill(~valid, 0)                      # any legal index for gather; masked out below
    # log softmax(z)_i = z_i - logsumexp(z). Subtracting the row max keeps exp() in [0, 1].
    m = logits.max(dim=-1, keepdim=True).values.detach()             # (B, T, 1); constant w.r.t. autograd is fine
    shifted = logits - m                                             # (B, T, V), max entry is 0
    lse = torch.log(torch.exp(shifted).sum(dim=-1))                  # (B, T); sum >= 1 so log is safe
    target_logit = shifted.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)   # (B, T)
    nll = lse - target_logit                                         # -log p(label) = lse(z - m) - (z_y - m)
    nll = nll * valid                                                # zero out ignored positions
    n_valid = valid.sum()
    if n_valid == 0:
        return nll.sum()                                             # 0-dim zero, still attached to the graph
    return nll.sum() / n_valid


def perplexity(mean_nll):
    """exp of the mean NLL; keeps the input type."""
    if isinstance(mean_nll, Tensor):
        return torch.exp(mean_nll)
    return math.exp(mean_nll)


def bits_per_byte(total_nll_nats: float, n_bytes: int) -> float:
    """Total NLL in nats -> bits (divide by ln 2) -> per raw byte."""
    if n_bytes <= 0:
        raise ValueError("n_bytes must be positive")
    return total_nll_nats / (n_bytes * math.log(2.0))
