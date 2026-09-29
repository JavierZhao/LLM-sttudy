import math

import pytest
import torch

MODULE = "rlhf"

# Hand example from page 23: a 6-token response, beta = 0.1, reward-model score 1.5.
LP = [-0.9, -1.4, -0.3, -2.1, -0.6, -0.2]      # log pi_theta(y_t)
LR = [-1.1, -1.2, -0.9, -2.0, -1.3, -0.9]      # log pi_ref(y_t)
# log-ratios:  0.2, -0.2, 0.6, -0.1, 0.7, 0.7  (sum 1.9)
# KL rewards: -0.02, 0.02, -0.06, 0.01, -0.07, -0.07 ; last token gets +1.5
EXPECTED = [-0.02, 0.02, -0.06, 0.01, -0.07, 1.43]


def T(*rows, dtype=torch.float64):
    return torch.tensor(rows, dtype=dtype)


# ------------------------------------------------------------------------------ shaped_rewards
def test_shaped_rewards_hand_example(impl):
    out = impl.shaped_rewards(torch.tensor([1.5], dtype=torch.float64), T(LP), T(LR), 0.1,
                              torch.ones(1, 6, dtype=torch.float64))
    assert out.shape == (1, 6)
    torch.testing.assert_close(out, T(EXPECTED), atol=1e-9, rtol=0)
    assert abs(out.sum().item() - (1.5 - 0.1 * 1.9)) < 1e-9        # return = score - beta * sequence KL


def test_shaped_rewards_padding_and_last_token(impl):
    lp = T([-1.0, -2.0, -3.0, -4.0], [-0.5, -0.5, -9.0, -9.0])
    lr = T([-1.5, -1.0, -3.0, -6.0], [-0.5, -1.5, -0.1, -0.2])
    mask = T([1, 1, 1, 1], [1, 1, 0, 0])                # second response has only 2 real tokens
    score = torch.tensor([2.0, -1.0], dtype=torch.float64)
    out = impl.shaped_rewards(score, lp, lr, 0.5, mask)
    # row 0: log-ratios 0.5, -1.0, 0.0, 2.0 -> KL rewards -0.25, 0.5, 0.0, -1.0, then +2.0 on the last token
    # row 1: log-ratios 0.0, 1.0 (2 real tokens) -> KL rewards 0.0, -0.5, then -1.0 on token 2
    want = T([-0.25, 0.5, 0.0, -1.0 + 2.0], [0.0, -0.5 + -1.0, 0.0, 0.0])
    torch.testing.assert_close(out, want, atol=1e-9, rtol=0)
    assert out[1, 2:].abs().sum() == 0                  # nothing leaks onto padding


def test_shaped_rewards_single_token_row(impl):
    out = impl.shaped_rewards(torch.tensor([3.0], dtype=torch.float64), T([-1.0, 0.0]), T([-2.0, 0.0]),
                              0.1, T([1, 0]))
    torch.testing.assert_close(out, T([3.0 - 0.1, 0.0]), atol=1e-9, rtol=0)


def test_shaped_rewards_beta_zero_is_pure_score(impl):
    out = impl.shaped_rewards(torch.tensor([0.7]), torch.randn(1, 5), torch.randn(1, 5), 0.0, torch.ones(1, 5))
    torch.testing.assert_close(out, torch.tensor([[0, 0, 0, 0, 0.7]]), atol=1e-6, rtol=0)


def test_shaped_rewards_is_a_constant(impl):
    lp = torch.randn(2, 4, requires_grad=True)
    lr = torch.randn(2, 4, requires_grad=True)
    sc = torch.randn(2, requires_grad=True)
    out = impl.shaped_rewards(sc, lp, lr, 0.1, torch.ones(2, 4))
    assert not out.requires_grad                        # rewards must be detached from the policy graph


# ------------------------------------------------------------------------------ apply_eos_penalty
def test_eos_penalty_replaces_score(impl):
    s = torch.tensor([2.5, -0.5, 4.0])
    ended = torch.tensor([True, False, False])
    out = impl.apply_eos_penalty(s, ended, penalty=-1.0)
    torch.testing.assert_close(out, torch.tensor([2.5, -1.0, -1.0]))
    out10 = impl.apply_eos_penalty(s, ended, penalty=-10.0)
    torch.testing.assert_close(out10, torch.tensor([2.5, -10.0, -10.0]))


# ------------------------------------------------------------------------------ whiten
def test_whiten_masked_statistics(impl):
    x = T([1.0, 2.0, 3.0, 99.0], [4.0, 5.0, 99.0, 99.0])   # 99 sits under the padding: must be ignored
    mask = T([1, 1, 1, 0], [1, 1, 0, 0])
    out = impl.whiten(x, mask)
    vals = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    want = (vals - vals.mean()) / torch.sqrt(vals.var(unbiased=False) + 1e-8)
    torch.testing.assert_close(out[mask.bool()], want, atol=1e-9, rtol=0)
    assert out[~mask.bool()].abs().sum() == 0
    valid = out[mask.bool()]
    assert abs(valid.mean().item()) < 1e-9 and abs(valid.var(unbiased=False).item() - 1.0) < 1e-6


def test_whiten_shift_mean_false_keeps_the_mean(impl):
    x = T([1.0, 2.0, 3.0, 4.0])
    mask = torch.ones(1, 4, dtype=torch.float64)
    out = impl.whiten(x, mask, shift_mean=False)
    assert abs(out.mean().item() - 2.5) < 1e-9
    assert abs(out.var(unbiased=False).item() - 1.0) < 1e-6
    torch.testing.assert_close(out - 2.5, impl.whiten(x, mask), atol=1e-9, rtol=0)


def test_whiten_constant_input_is_finite(impl):
    out = impl.whiten(torch.full((2, 3), 7.0), torch.ones(2, 3))
    assert torch.isfinite(out).all() and out.abs().max() < 1e-3


# ------------------------------------------------------------------------------ AdaptiveKLController
def test_controller_exact_update(impl):
    c = impl.AdaptiveKLController(init_beta=0.1, target_kl=6.0, horizon=10_000)
    assert abs(c.beta - 0.1) < 1e-15
    new = c.update(current_kl=9.0, n_steps=512)          # error clipped from +0.5 to +0.2
    assert abs(new - 0.1 * (1 + 0.2 * 512 / 10_000)) < 1e-12
    assert abs(c.beta - new) < 1e-15


def test_controller_moves_beta_the_right_way(impl):
    up = impl.AdaptiveKLController(0.05, 6.0, 10_000)
    down = impl.AdaptiveKLController(0.05, 6.0, 10_000)
    still = impl.AdaptiveKLController(0.05, 6.0, 10_000)
    assert up.update(8.0, 512) > 0.05                     # KL above target: penalize more
    assert down.update(3.0, 512) < 0.05                   # KL below target: relax
    assert abs(still.update(6.0, 512) - 0.05) < 1e-15     # on target: unchanged


def test_controller_error_is_clipped_and_scales_with_batch(impl):
    a = impl.AdaptiveKLController(0.1, 6.0, 10_000)
    b = impl.AdaptiveKLController(0.1, 6.0, 10_000)
    assert abs(a.update(7.2, 512) - b.update(600.0, 512)) < 1e-12          # 7.2 = 1.2 x target: already saturated
    c = impl.AdaptiveKLController(0.1, 6.0, 10_000)
    d = impl.AdaptiveKLController(0.1, 6.0, 10_000)
    step1 = c.update(6.3, 256) / 0.1 - 1
    step2 = d.update(6.3, 512) / 0.1 - 1
    assert abs(step2 - 2 * step1) < 1e-12                                  # gain is proportional to n_steps


def test_controller_converges_on_a_toy_plant(impl):
    # Toy plant: the measured KL falls as beta rises, KL = 12 / (1 + 20 beta). Target 4 nats -> beta* = 0.1.
    c = impl.AdaptiveKLController(init_beta=0.5, target_kl=4.0, horizon=2_000)
    for _ in range(3_000):
        c.update(12.0 / (1.0 + 20.0 * c.beta), n_steps=64)
    assert abs(c.beta - 0.1) < 5e-3
    assert c.beta > 0
