"""Drill (page 21): reward-model and preference-optimization losses.

Implement the functions below. Run:
    pytest exercises/tests/test_dpo.py

Conventions used throughout:
  * "pi_w", "pi_l", "ref_w", "ref_l" are per-sequence log-probabilities, shape (B,), each the SUM of
    token log-probabilities over the response tokens only (prompt and padding excluded), for the
    chosen (w = winner) and rejected (l = loser) response, under the trainable policy (pi_*) and the
    frozen reference model (ref_*).
  * All losses are returned per example, shape (B,), so callers choose the reduction (.mean()).
  * Everything must be differentiable with respect to the policy inputs and numerically stable
    (use log-sigmoid / softplus style functions, never log(sigmoid(x)) on large |x|).
"""
from typing import Optional, Tuple

import torch
from torch import Tensor


def sequence_logprob(logits: Tensor, labels: Tensor, mask: Tensor) -> Tensor:
    """Sum of token log-probabilities of `labels` over the positions where `mask` is 1.

    Args:
        logits: (B, T, V) unnormalized scores. The caller has already aligned them so that
                logits[:, t] is the model's distribution over labels[:, t] (i.e. shifted by one
                relative to the input ids).
        labels: (B, T) int64 target token ids. At positions where mask == 0 the label may be ANY
                value, including -100 (the usual ignore index); it must neither affect the result
                nor raise an error.
        mask:   (B, T) bool or float, 1 on response tokens, 0 on prompt and padding tokens.
    Returns:
        (B,) float tensor: sum over t of mask[b, t] * log softmax(logits[b, t])[labels[b, t]].
        A row whose mask is all zeros returns 0. Must stay finite for logits of magnitude 1e4.
    """
    raise NotImplementedError


def bt_reward_loss(r_chosen: Tensor, r_rejected: Tensor, margin: Optional[Tensor] = None) -> Tensor:
    """Bradley-Terry pairwise loss for a reward model, averaged over the batch.

    P(chosen beats rejected) = sigmoid(r_chosen - r_rejected - margin), loss = -log of that.

    Args:
        r_chosen, r_rejected: (B,) scalar rewards from the reward model.
        margin: optional (B,) or scalar non-negative margin m (Llama 2 style: larger for pairs the
                annotators rated "significantly better"). None means 0.
    Returns:
        Scalar tensor: mean over the batch of -log sigmoid(r_chosen - r_rejected - margin).
    """
    raise NotImplementedError


def pairwise_accuracy(r_chosen: Tensor, r_rejected: Tensor) -> Tensor:
    """RewardBench-style accuracy: fraction of pairs with r_chosen > r_rejected.

    Exact ties count as 0.5 (a constant reward model must score 0.5, not 0 or 1).
    Returns a scalar tensor in [0, 1].
    """
    raise NotImplementedError


def dpo_loss(pi_w: Tensor, pi_l: Tensor, ref_w: Tensor, ref_l: Tensor, beta: float,
             label_smoothing: float = 0.0) -> Tuple[Tensor, Tensor, Tensor]:
    """Direct Preference Optimization loss (Rafailov et al. 2023, Eq. 7), per example.

    implicit reward  r_hat(y) = beta * (log pi(y|x) - log pi_ref(y|x))
    loss = -log sigmoid(r_hat(y_w) - r_hat(y_l))

    With label_smoothing = eps in [0, 0.5) return the conservative-DPO loss
    (1 - eps) * L(w, l) + eps * L(l, w), where L(a, b) = -log sigmoid(r_hat(a) - r_hat(b)).

    Returns:
        loss:       (B,) per-example loss.
        reward_w:   (B,) implicit reward of the chosen response, DETACHED (for logging only).
        reward_l:   (B,) implicit reward of the rejected response, DETACHED.
    """
    raise NotImplementedError


def dpo_grad_weight(pi_w: Tensor, pi_l: Tensor, ref_w: Tensor, ref_l: Tensor, beta: float) -> Tensor:
    """Per-example weight in the DPO gradient: sigma(r_hat(y_l) - r_hat(y_w)), shape (B,).

    The gradient of the per-example loss is
        -beta * weight * (grad log pi(y_w) - grad log pi(y_l)).
    """
    raise NotImplementedError


def ipo_loss(pi_w: Tensor, pi_l: Tensor, ref_w: Tensor, ref_l: Tensor, tau: float) -> Tensor:
    """IPO loss (Azar et al. 2023, Eq. 17), per example: a squared regression to a target margin.

    h = (log pi(y_w) - log pi_ref(y_w)) - (log pi(y_l) - log pi_ref(y_l))
    loss = (h - 1 / (2 * tau)) ** 2

    Returns (B,).
    """
    raise NotImplementedError


def simpo_loss(pi_w: Tensor, pi_l: Tensor, len_w: Tensor, len_l: Tensor, beta: float, gamma: float) -> Tensor:
    """SimPO loss (Meng et al. 2024, Eq. 6), per example. No reference model.

    reward(y) = beta / |y| * log pi(y|x)                  (length-normalized log-probability)
    loss = -log sigmoid(reward(y_w) - reward(y_l) - gamma)

    Args:
        pi_w, pi_l: (B,) summed response log-probs under the policy.
        len_w, len_l: (B,) number of response tokens (positive).
    Returns (B,).
    """
    raise NotImplementedError


def orpo_odds_ratio_loss(avg_logp_w: Tensor, avg_logp_l: Tensor) -> Tensor:
    """ORPO odds-ratio term L_OR (Hong et al. 2024, Eq. 7), per example. No reference model.

    P(y) = exp(average token log-prob of y), odds(y) = P / (1 - P)
    L_OR = -log sigmoid( log odds(y_w) - log odds(y_l) )

    Args:
        avg_logp_w, avg_logp_l: (B,) MEAN token log-probs (strictly negative). They can be very
        close to 0 (P close to 1), where log(1 - P) must not underflow to -inf.
    Returns (B,). (The full ORPO objective adds the SFT loss on y_w; that is not part of this drill.)
    """
    raise NotImplementedError
