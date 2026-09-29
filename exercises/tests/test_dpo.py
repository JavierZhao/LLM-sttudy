import math

import torch
import torch.nn.functional as F

MODULE = "dpo"

# Hand-computed numbers from page 21 (beta = 0.1). Per-sequence log-probs in nats.
#   A: policy has moved the right way   pi_w=-41 pi_l=-57 | ref_w=-44 ref_l=-52  -> margin 0.8
#   B: policy ranks the pair wrongly    pi_w=-50 pi_l=-48 | ref_w=-47 ref_l=-52  -> margin -0.7
A = dict(pi_w=-41.0, pi_l=-57.0, ref_w=-44.0, ref_l=-52.0)
B = dict(pi_w=-50.0, pi_l=-48.0, ref_w=-47.0, ref_l=-52.0)


def t(*xs):
    return torch.tensor(xs, dtype=torch.float64)


def args(d):
    return t(d["pi_w"]), t(d["pi_l"]), t(d["ref_w"]), t(d["ref_l"])


# ----------------------------------------------------------------------------- sequence_logprob
def test_sequence_logprob_matches_cross_entropy(impl):
    B_, T, V = 3, 7, 11
    logits = torch.randn(B_, T, V)
    labels = torch.randint(0, V, (B_, T))
    mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1],       # prompt of 2 tokens
                         [0, 0, 0, 0, 1, 1, 0],       # prompt of 4, then padding
                         [1, 1, 1, 1, 1, 1, 1]], dtype=torch.float32)
    ce = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1), reduction="none").view(B_, T)
    expected = -(ce * mask).sum(-1)
    torch.testing.assert_close(impl.sequence_logprob(logits, labels, mask), expected, atol=1e-5, rtol=1e-5)
    # bool masks work too
    torch.testing.assert_close(impl.sequence_logprob(logits, labels, mask.bool()), expected, atol=1e-5, rtol=1e-5)


def test_sequence_logprob_ignores_masked_labels(impl):
    B_, T, V = 2, 6, 9
    logits = torch.randn(B_, T, V)
    labels = torch.randint(0, V, (B_, T))
    mask = torch.tensor([[0, 0, 1, 1, 1, 0], [0, 1, 1, 1, 0, 0]], dtype=torch.float32)
    ref = impl.sequence_logprob(logits, labels, mask)
    labels2 = labels.clone()
    labels2[mask == 0] = -100                          # ignore index must not crash or leak in
    torch.testing.assert_close(impl.sequence_logprob(logits, labels2, mask), ref)
    # changing the LOGITS at masked positions cannot matter either
    logits2 = logits.clone()
    logits2[mask == 0] = torch.randn(int((mask == 0).sum()), V) * 50
    torch.testing.assert_close(impl.sequence_logprob(logits2, labels, mask), ref)


def test_sequence_logprob_is_stable_and_handles_empty_rows(impl):
    logits = torch.zeros(2, 3, 4)
    logits[:, :, 0] = 1e4                              # naive softmax-then-log overflows / underflows
    labels = torch.tensor([[0, 1, 0], [0, 0, 0]])
    mask = torch.tensor([[1, 1, 0], [0, 0, 0]], dtype=torch.float32)
    out = impl.sequence_logprob(logits, labels, mask)
    assert torch.isfinite(out).all()
    assert abs(out[0].item() - (0.0 - 1e4)) < 1e-2     # log p(0)=0, log p(1)=-1e4
    assert out[1].item() == 0.0                        # all-masked row


def test_sequence_logprob_is_differentiable(impl):
    logits = torch.randn(2, 4, 5, requires_grad=True)
    labels = torch.randint(0, 5, (2, 4))
    mask = torch.tensor([[0, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.float32)
    impl.sequence_logprob(logits, labels, mask).sum().backward()
    g = logits.grad
    assert (g[mask == 0] == 0).all()                   # no gradient into prompt/padding positions
    # gradient of a summed log-prob wrt logits is onehot(label) - softmax at each response position
    expected = (F.one_hot(labels, 5).float() - torch.softmax(logits.detach(), -1)) * mask.unsqueeze(-1)
    torch.testing.assert_close(g, expected, atol=1e-6, rtol=1e-6)


# ----------------------------------------------------------------------------- Bradley-Terry
def test_bt_values(impl):
    torch.testing.assert_close(impl.bt_reward_loss(t(1.2), t(0.4)), torch.tensor(0.3711006659, dtype=torch.float64))
    torch.testing.assert_close(impl.bt_reward_loss(t(0.3, -2.0), t(0.3, -2.0)), torch.tensor(math.log(2), dtype=torch.float64))
    # mean over the batch
    two = impl.bt_reward_loss(t(1.2, 0.0), t(0.4, 0.0))
    torch.testing.assert_close(two, torch.tensor((0.3711006659 + math.log(2)) / 2, dtype=torch.float64))


def test_bt_shift_invariance_and_margin(impl):
    rw, rl = t(0.7, -1.0, 2.0), t(0.1, -0.5, 2.5)
    base = impl.bt_reward_loss(rw, rl)
    torch.testing.assert_close(impl.bt_reward_loss(rw + 10.0, rl + 10.0), base)   # rewards live up to a shift
    m = t(0.5, 1.0, 2.0)
    torch.testing.assert_close(impl.bt_reward_loss(rw, rl, margin=m), impl.bt_reward_loss(rw - m, rl))
    assert impl.bt_reward_loss(rw, rl, margin=m) > base                         # margin makes the task harder


def test_bt_gradients_and_stability(impl):
    rw = t(0.3, 1.0).requires_grad_()
    rl = t(0.9, -1.0).requires_grad_()
    impl.bt_reward_loss(rw, rl).backward()
    d = (rw - rl).detach()
    expected = -torch.sigmoid(-d) / 2                  # d loss / d r_w for a batch mean over 2 pairs
    torch.testing.assert_close(rw.grad, expected)
    torch.testing.assert_close(rl.grad, -expected)
    assert (rw.grad < 0).all() and (rl.grad > 0).all()
    huge = impl.bt_reward_loss(t(-500.0), t(500.0))    # margin -1000: loss ~ 1000, not inf/nan
    assert torch.isfinite(huge) and abs(huge.item() - 1000.0) < 1e-6


def test_pairwise_accuracy(impl):
    acc = impl.pairwise_accuracy(t(1.0, 0.0, 2.0, 5.0), t(0.0, 1.0, 2.0, 4.0))
    assert abs(acc.item() - (1 + 0 + 0.5 + 1) / 4) < 1e-9
    assert abs(impl.pairwise_accuracy(t(3.0, 3.0), t(3.0, 3.0)).item() - 0.5) < 1e-9   # constant RM = chance


# ----------------------------------------------------------------------------- DPO
def test_dpo_hand_numbers(impl):
    loss, rw, rl = impl.dpo_loss(*args(A), beta=0.1)
    assert abs(loss.item() - 0.3711006659) < 1e-6
    assert abs(rw.item() - 0.3) < 1e-9 and abs(rl.item() - (-0.5)) < 1e-9
    loss, rw, rl = impl.dpo_loss(*args(B), beta=0.1)
    assert abs(loss.item() - 1.1031860489) < 1e-6
    assert abs(rw.item() - (-0.3)) < 1e-9 and abs(rl.item() - 0.4) < 1e-9
    # per-example output for a batch
    both = impl.dpo_loss(t(A["pi_w"], B["pi_w"]), t(A["pi_l"], B["pi_l"]),
                         t(A["ref_w"], B["ref_w"]), t(A["ref_l"], B["ref_l"]), beta=0.1)[0]
    assert both.shape == (2,)


def test_dpo_identical_policy_and_reference_gives_log2(impl):
    pi = torch.randn(6, dtype=torch.float64) * 30 - 60
    pj = torch.randn(6, dtype=torch.float64) * 30 - 60
    for beta in (0.01, 0.1, 1.0, 5.0):
        loss, rw, rl = impl.dpo_loss(pi, pj, pi.clone(), pj.clone(), beta)
        torch.testing.assert_close(loss, torch.full_like(loss, math.log(2)))
        assert torch.allclose(rw, torch.zeros_like(rw)) and torch.allclose(rl, torch.zeros_like(rl))


def test_dpo_gradient_signs_and_weight(impl):
    beta = 0.1
    pw, pl, rw_, rl_ = (x.clone() for x in args(A))
    pw.requires_grad_(), pl.requires_grad_()
    loss, _, _ = impl.dpo_loss(pw, pl, rw_, rl_, beta)
    loss.sum().backward()
    assert pw.grad.item() < 0 and pl.grad.item() > 0               # descent raises chosen, lowers rejected
    w = math.exp(-0.8) / (1 + math.exp(-0.8))                      # sigma(-0.8) = 0.3100
    assert abs(pw.grad.item() + beta * w) < 1e-9
    assert abs(pl.grad.item() - beta * w) < 1e-9
    # the closed-form weight helper agrees, and the wrongly ordered pair gets the bigger weight
    wa = impl.dpo_grad_weight(*args(A), beta=beta)
    wb = impl.dpo_grad_weight(*args(B), beta=beta)
    assert abs(wa.item() - 0.3100255189) < 1e-8 and abs(wb.item() - 0.6681877722) < 1e-8
    assert wb > wa
    assert abs(impl.dpo_grad_weight(t(-3.0), t(-3.0), t(-3.0), t(-3.0), beta).item() - 0.5) < 1e-12


def test_dpo_rewards_are_detached(impl):
    pw = t(-1.0).requires_grad_()
    _, rw, rl = impl.dpo_loss(pw, t(-2.0), t(-1.5), t(-1.5), beta=0.5)
    assert not rw.requires_grad and not rl.requires_grad


def test_dpo_partition_function_cancels(impl):
    # Adding any prompt-dependent constant f(x) to BOTH policy log-probs (or to both reference
    # log-probs) leaves the loss unchanged: this is exactly the log Z(x) shift cancelling.
    pw, pl, rw, rl = args(A)
    base = impl.dpo_loss(pw, pl, rw, rl, 0.1)[0]
    shifted = impl.dpo_loss(pw + 7.3, pl + 7.3, rw - 2.0, rl - 2.0, 0.1)[0]
    torch.testing.assert_close(base, shifted)
    # but shifting only the chosen policy log-prob does change it
    assert not torch.allclose(base, impl.dpo_loss(pw + 7.3, pl, rw, rl, 0.1)[0])


def test_dpo_is_bt_loss_on_implicit_rewards(impl):
    torch.manual_seed(1)
    pw, pl, rw, rl = (torch.randn(8, dtype=torch.float64) * 10 - 50 for _ in range(4))
    beta = 0.3
    loss, r_hat_w, r_hat_l = impl.dpo_loss(pw, pl, rw, rl, beta)
    torch.testing.assert_close(loss.mean(), impl.bt_reward_loss(r_hat_w, r_hat_l))


def test_dpo_chosen_likelihood_can_fall_while_loss_falls(impl):
    ref = (t(-44.0), t(-52.0))
    before = impl.dpo_loss(ref[0], ref[1], *ref, beta=0.1)[0]
    # chosen log-prob DROPS by 1 nat, rejected drops by 20 nats: the loss still improves
    after = impl.dpo_loss(t(-45.0), t(-72.0), *ref, beta=0.1)[0]
    assert after < before


def test_dpo_loss_of_0p1_needs_a_22_nat_gap(impl):
    # beta = 0.1: loss 0.1 needs implicit reward margin logit(exp(-0.1)) = 2.2522, i.e. 22.52 nats
    gap = 22.5216846
    loss = impl.dpo_loss(t(gap), t(0.0), t(0.0), t(0.0), beta=0.1)[0]
    assert abs(loss.item() - 0.1) < 1e-6
    assert impl.dpo_loss(t(gap / 2), t(0.0), t(0.0), t(0.0), beta=0.1)[0] > 0.1


def test_dpo_label_smoothing_is_cdpo(impl):
    beta, eps = 0.1, 0.2
    m = 0.8
    loss = impl.dpo_loss(*args(A), beta=beta, label_smoothing=eps)[0]
    expected = (1 - eps) * math.log(1 + math.exp(-m)) + eps * math.log(1 + math.exp(m))
    assert abs(loss.item() - expected) < 1e-9
    # eps = 0.5 makes the loss even in the margin, so its gradient at margin 0 vanishes
    a, b = impl.dpo_loss(*args(A), beta=beta, label_smoothing=0.5)[0], impl.dpo_loss(*args(B), beta=beta, label_smoothing=0.5)[0]
    assert a.item() > math.log(2) and b.item() > math.log(2)


# ----------------------------------------------------------------------------- IPO
def test_ipo_values_and_two_sided_penalty(impl):
    # h = 8 for pair A; tau = 0.1 -> target 5 -> loss 9
    assert abs(impl.ipo_loss(*args(A), tau=0.1).item() - 9.0) < 1e-9
    zero = impl.ipo_loss(t(5.0), t(0.0), t(0.0), t(0.0), tau=0.1)
    assert abs(zero.item()) < 1e-12
    # symmetric: undershooting and overshooting the target by the same amount cost the same
    under = impl.ipo_loss(t(2.0), t(0.0), t(0.0), t(0.0), tau=0.1)
    over = impl.ipo_loss(t(8.0), t(0.0), t(0.0), t(0.0), tau=0.1)
    torch.testing.assert_close(under, over)


def test_ipo_gradient_changes_sign_at_the_target(impl):
    def grads(gap):
        pw = t(gap).requires_grad_()
        pl = t(0.0).requires_grad_()
        impl.ipo_loss(pw, pl, t(0.0), t(0.0), tau=0.1).sum().backward()
        return pw.grad.item(), pl.grad.item()
    gw, gl = grads(2.0)                                # below the target 5: raise chosen, lower rejected
    assert gw < 0 and gl > 0
    gw, gl = grads(8.0)                                # above the target: IPO pulls the gap back
    assert gw > 0 and gl < 0
    # DPO never changes sign: its gradient keeps pushing the gap up however large it is
    pw = t(8.0).requires_grad_()
    impl.dpo_loss(pw, t(0.0), t(0.0), t(0.0), 0.1)[0].sum().backward()
    assert pw.grad.item() < 0


# ----------------------------------------------------------------------------- SimPO
def test_simpo_values(impl):
    pw, pl = t(-78.0), t(-190.0)
    lw, ll = t(120.0), t(300.0)
    loss = impl.simpo_loss(pw, pl, lw, ll, beta=2.0, gamma=0.5)
    assert abs(loss.item() - 0.9949558260) < 1e-8            # margin = 2 * (-0.65 + 0.63333) - 0.5 = -0.5333
    assert impl.simpo_loss(pw, pl, lw, ll, beta=2.0, gamma=0.0) < loss   # a larger target margin costs more


def test_simpo_is_length_normalized_and_reference_free(impl):
    pw, pl, lw, ll = t(-30.0), t(-40.0), t(20.0), t(25.0)
    base = impl.simpo_loss(pw, pl, lw, ll, 2.0, 0.3)
    # doubling every length and every summed log-prob keeps the per-token averages, hence the loss
    torch.testing.assert_close(base, impl.simpo_loss(2 * pw, 2 * pl, 2 * lw, 2 * ll, 2.0, 0.3))
    # gradient wrt the summed log-prob is beta / length times the sigmoid weight
    pw_ = pw.clone().requires_grad_()
    impl.simpo_loss(pw_, pl, lw, ll, 2.0, 0.3).sum().backward()
    margin = 2.0 * (pw / lw - pl / ll) - 0.3
    expected = -2.0 / lw * torch.sigmoid(-margin)
    torch.testing.assert_close(pw_.grad, expected)


# ----------------------------------------------------------------------------- ORPO
def test_orpo_values_and_stability(impl):
    loss = impl.orpo_odds_ratio_loss(t(-0.65), t(-1.10))
    assert abs(loss.item() - 0.3762551923) < 1e-8
    # a model that is nearly certain of both responses: average log-prob ~ -1e-9
    tiny = impl.orpo_odds_ratio_loss(t(-1e-9), t(-2e-9))
    assert torch.isfinite(tiny)
    # equal probabilities -> odds ratio 1 -> loss log 2
    torch.testing.assert_close(impl.orpo_odds_ratio_loss(t(-0.4), t(-0.4)), torch.tensor([math.log(2)], dtype=torch.float64)[0])


def test_orpo_gradient_signs(impl):
    a = t(-0.65).requires_grad_()
    b = t(-1.10).requires_grad_()
    impl.orpo_odds_ratio_loss(a, b).sum().backward()
    assert a.grad.item() < 0 and b.grad.item() > 0


# ----------------------------------------------------------------------------- end to end
def test_dpo_training_loop_on_a_toy_policy(impl):
    """Optimize a tiny 'language model' (free logits) with DPO for a few steps."""
    torch.manual_seed(0)
    B_, Tp, T, V = 4, 3, 6, 12
    base = torch.randn(B_, 2, T, V)                    # [chosen, rejected] logits from the reference
    labels = torch.randint(0, V, (B_, 2, T))
    mask = torch.zeros(B_, 2, T)
    mask[:, :, Tp:] = 1.0                              # first 3 positions are the prompt
    delta = torch.zeros_like(base, requires_grad=True)
    opt = torch.optim.SGD([delta], lr=0.5)

    def seq_lp(logits):
        return impl.sequence_logprob(logits.reshape(-1, T, V), labels.reshape(-1, T), mask.reshape(-1, T)).view(B_, 2)

    ref_lp = seq_lp(base).detach()
    first = None
    for step in range(30):
        pol = seq_lp(base + delta)
        loss, rw, rl = impl.dpo_loss(pol[:, 0], pol[:, 1], ref_lp[:, 0], ref_lp[:, 1], beta=0.5)
        if first is None:
            first = loss.mean().item()
            assert abs(first - math.log(2)) < 1e-5     # policy == reference at step 0
        opt.zero_grad()
        loss.mean().backward()
        opt.step()
    assert loss.mean().item() < 0.5 * first
    assert (rw > rl).all()                             # implicit reward now ranks every pair correctly
    assert (delta.grad[:, :, :Tp] == 0).all()          # prompt positions never receive gradient
