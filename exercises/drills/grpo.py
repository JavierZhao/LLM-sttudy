"""Drill (page 24): GRPO and its successors, the loss-level details interviewers ask about.

Implement the functions below. Run:
    pytest exercises/tests/test_grpo.py

Conventions used throughout:
  * A batch holds N = P * G responses: P prompts with G sampled responses each. The G responses of
    one prompt are CONTIGUOUS (rows 0..G-1 belong to prompt 0, rows G..2G-1 to prompt 1, ...).
  * "logp_*" are per-token log-probabilities of the SAMPLED tokens, shape (B, T), where B is the
    number of responses and T the padded length. logp_old and logp_ref are constants (no gradient);
    only logp_new carries gradient.
  * "mask" is (B, T), 1 on response tokens and 0 on padding (bool or float). Values of every logp
    tensor at masked positions are arbitrary (they may be huge or NaN-free junk) and must never
    influence the result or its gradient.
  * "adv" is one scalar per response, shape (B,): every token of response i shares adv[i].
  * Losses are returned as a scalar to MINIMIZE (that is, the negative of the objective).
"""
from typing import Optional

import torch
from torch import Tensor


def group_advantages(rewards: Tensor, group_size: int, kind: str = "grpo", eps: float = 1e-6) -> Tensor:
    """Critic-free advantages computed inside each group of `group_size` responses.

    Args:
        rewards: (N,) scalar outcome rewards, N divisible by group_size, groups contiguous.
        group_size: G >= 2 responses per prompt.
        kind:
            "grpo":    (r - mean) / (std + eps), with std the SAMPLE standard deviation of the group
                       (divide by G - 1, the torch.std default), as in verl (DeepSeekMath does not say
                       which). eps is added to the std, not to the variance.
            "dr_grpo": r - mean (Dr. GRPO: no std normalization).
            "rloo":    r_i minus the mean of the OTHER G - 1 rewards of the group (leave-one-out).
        eps: added to the std in the denominator ("grpo" only).
    Returns:
        (N,) advantages with the dtype of `rewards`. A group whose rewards are all equal gets
        advantage exactly 0 for every kind (no NaN or inf), also for rewards such as 0.1 that are
        not exact in floating point and also when eps=0. Raise ValueError for an unknown kind
        or if N is not divisible by group_size.
    """
    raise NotImplementedError


def grpo_loss(logp_new: Tensor, logp_old: Tensor, logp_ref: Tensor, adv: Tensor, mask: Tensor,
              eps_low: float = 0.2, eps_high: float = 0.2, beta: float = 0.0,
              agg: str = "seq_mean", max_len: Optional[int] = None) -> Tensor:
    """GRPO / DAPO / Dr. GRPO surrogate loss with a decoupled clip range and a k3 KL term.

    Per token t of response i, with ratio = exp(logp_new - logp_old) and A = adv[i]:
        surrogate = min(ratio * A, clip(ratio, 1 - eps_low, 1 + eps_high) * A)
        kl        = k3 = exp(logp_ref - logp_new) - (logp_ref - logp_new) - 1      (>= 0)
        objective = surrogate - beta * kl          (the KL sits in the loss, not in the reward)
    The loss is minus the aggregated objective. Aggregation over the B responses and their tokens:
        "seq_mean":   sum_t objective / |o_i| per response, then the mean over the B responses
                      (the DeepSeekMath form, 1/G sum_i 1/|o_i| sum_t).
        "token_mean": sum of objective over ALL valid tokens divided by the total number of valid
                      tokens (the DAPO form, 1 / sum_i |o_i|).
        "const":      sum of objective over all valid tokens divided by (B * max_len)
                      (the Dr. GRPO form; max_len is required for this mode).
    Returns:
        Scalar tensor. Raise ValueError for an unknown agg, or for agg="const" with max_len=None.
        Gradient flows only through logp_new; it must be exactly 0 at masked positions.
    """
    raise NotImplementedError


def gspo_ratio(logp_new: Tensor, logp_old: Tensor, mask: Tensor) -> Tensor:
    """Length-normalized sequence-level importance ratio of GSPO.

    s_i = (pi_new(o_i) / pi_old(o_i)) ** (1 / |o_i|) = exp( mean over valid t of (logp_new - logp_old) ).

    Returns:
        (B,) tensor. A response with no valid token returns 1.0. Must not depend on the values at
        masked positions.
    """
    raise NotImplementedError


def gspo_loss(logp_new: Tensor, logp_old: Tensor, adv: Tensor, mask: Tensor,
              eps_low: float = 3e-4, eps_high: float = 4e-4) -> Tensor:
    """GSPO objective: clip the sequence ratio s_i, not the token ratios.

    loss = - mean over the B responses of min(s_i * A_i, clip(s_i, 1 - eps_low, 1 + eps_high) * A_i),
    where s_i comes from gspo_ratio. No KL term. Returns a scalar to minimize. Every token of a
    response gets the same clipping decision.
    """
    raise NotImplementedError


def dynamic_sampling_filter(rewards: Tensor, group_size: int) -> Tensor:
    """DAPO dynamic sampling: which groups carry a learning signal?

    A group is kept iff its rewards are not all equal (for binary rewards: the group accuracy is
    strictly between 0 and 1), because an all-correct or all-wrong group has zero advantage for
    every response.

    Args:
        rewards: (N,) rewards, groups contiguous, N divisible by group_size.
    Returns:
        (P,) bool tensor, P = N // group_size, True for groups to keep.
    """
    raise NotImplementedError
