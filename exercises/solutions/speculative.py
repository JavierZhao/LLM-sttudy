"""Reference solutions: speculative decoding accept/resample rule (page 17)."""
from typing import List

import torch
from torch import Tensor


def acceptance_rate(p: Tensor, q: Tensor) -> float:
    """beta = sum_x min(p(x), q(x)) = 1 - TV(p, q)."""
    return float(torch.minimum(p, q).sum())


def expected_tokens_per_step(alpha: float, gamma: int) -> float:
    """Truncated geometric mean: sum_{k=0}^{gamma} alpha**k = (1 - alpha**(gamma+1)) / (1 - alpha)."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if gamma < 0:
        raise ValueError("gamma must be >= 0")
    if alpha == 1.0:
        return float(gamma + 1)  # the closed form is 0/0 here; the sum is gamma + 1 ones
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def speedup(alpha: float, gamma: int, c: float) -> float:
    """Leviathan et al., Theorem 3.8: tokens per step over the cost of gamma draft passes + 1 target pass."""
    return expected_tokens_per_step(alpha, gamma) / (1.0 + gamma * c)


def speculative_accept(p_target: Tensor, q_draft: Tensor, draft_tokens: Tensor,
                       generator: torch.Generator) -> List[int]:
    gamma = int(draft_tokens.shape[0])
    out: List[int] = []
    # One uniform per drafted position, drawn up front from the caller's generator.
    u = torch.rand(gamma, generator=generator)
    for i in range(gamma):
        x = int(draft_tokens[i])
        p_x, q_x = float(p_target[i, x]), float(q_draft[i, x])
        # Accept with probability min(1, p/q). u < p/q is the same event as u * q < p; q_x > 0 by contract.
        if float(u[i]) < min(1.0, p_x / q_x):
            out.append(x)
            continue
        # First rejection: resample from the residual norm(max(0, p - q)), whose mass is 1 - beta.
        residual = (p_target[i] - q_draft[i]).clamp_min(0.0)
        if float(residual.sum()) <= 0.0:
            residual = p_target[i]  # p == q (up to rounding): a rejection is essentially impossible
        out.append(int(torch.multinomial(residual, 1, generator=generator)))
        return out  # everything after the rejected position is discarded
    # All gamma drafts accepted: the pass that scored them also produced p for the next position (free bonus token).
    out.append(int(torch.multinomial(p_target[gamma], 1, generator=generator)))
    return out
