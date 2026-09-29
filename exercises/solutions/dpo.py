"""Reference solutions: reward-model and preference-optimization losses (page 21)."""
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def sequence_logprob(logits: Tensor, labels: Tensor, mask: Tensor) -> Tensor:
    """Sum of token log-probs of `labels` over mask == 1 positions. logits (B, T, V) -> (B,)."""
    logp = torch.log_softmax(logits.float(), dim=-1)                  # stable: subtracts the max inside
    safe = labels.clamp(min=0)                                        # -100 (ignore index) would crash gather
    tok = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)             # (B, T) log p(label_t | context)
    return (tok * mask.to(tok.dtype)).sum(-1)                         # masked positions contribute exactly 0


def bt_reward_loss(r_chosen: Tensor, r_rejected: Tensor, margin: Optional[Tensor] = None) -> Tensor:
    """Mean of -log sigmoid(r_w - r_l - margin)."""
    delta = r_chosen - r_rejected
    if margin is not None:
        delta = delta - margin
    return -F.logsigmoid(delta).mean()                                # logsigmoid is stable for large |delta|


def pairwise_accuracy(r_chosen: Tensor, r_rejected: Tensor) -> Tensor:
    """Fraction of pairs with r_w > r_l; ties count 0.5."""
    win = (r_chosen > r_rejected).float()
    tie = (r_chosen == r_rejected).float()
    return (win + 0.5 * tie).mean()


def dpo_loss(pi_w: Tensor, pi_l: Tensor, ref_w: Tensor, ref_l: Tensor, beta: float,
             label_smoothing: float = 0.0) -> Tuple[Tensor, Tensor, Tensor]:
    """DPO loss per example plus detached implicit rewards. cDPO when label_smoothing > 0."""
    reward_w = beta * (pi_w - ref_w)                                  # implicit rewards (log Z cancels below)
    reward_l = beta * (pi_l - ref_l)
    margin = reward_w - reward_l
    loss = -(1 - label_smoothing) * F.logsigmoid(margin) - label_smoothing * F.logsigmoid(-margin)
    return loss, reward_w.detach(), reward_l.detach()


def dpo_grad_weight(pi_w: Tensor, pi_l: Tensor, ref_w: Tensor, ref_l: Tensor, beta: float) -> Tensor:
    """sigma(r_hat_l - r_hat_w): large when the implicit reward ranks the pair the wrong way round."""
    margin = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    return torch.sigmoid(-margin)


def ipo_loss(pi_w: Tensor, pi_l: Tensor, ref_w: Tensor, ref_l: Tensor, tau: float) -> Tensor:
    """(h - 1/(2 tau))^2 with h the difference of log-ratios (no beta: tau is the regularization)."""
    h = (pi_w - ref_w) - (pi_l - ref_l)
    return (h - 1.0 / (2.0 * tau)) ** 2


def simpo_loss(pi_w: Tensor, pi_l: Tensor, len_w: Tensor, len_l: Tensor, beta: float, gamma: float) -> Tensor:
    """-log sigmoid(beta * (avg logp_w - avg logp_l) - gamma); needs no reference model."""
    margin = beta * (pi_w / len_w - pi_l / len_l) - gamma
    return -F.logsigmoid(margin)


def orpo_odds_ratio_loss(avg_logp_w: Tensor, avg_logp_l: Tensor) -> Tensor:
    """-log sigmoid(logit(P_w) - logit(P_l)) with P = exp(mean token log-prob)."""
    def log_odds(lp: Tensor) -> Tensor:
        # log(P / (1 - P)) = lp - log(1 - exp(lp)), and 1 - exp(lp) = -expm1(lp) stays accurate as lp -> 0-
        # (writing torch.log(1 - lp.exp()) would lose all precision, then hit -inf, for lp ~ -1e-9)
        return lp - torch.log(-torch.expm1(lp))
    return -F.logsigmoid(log_odds(avg_logp_w) - log_odds(avg_logp_l))
