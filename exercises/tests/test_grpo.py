import math

import pytest
import torch

MODULE = "grpo"

# Hand-computed numbers from page 24: G = 8 responses, 3 correct (reward 1), 5 wrong (reward 0).
#   mean = 3/8 = 0.375
#   sample std = sqrt(8/7 * 0.375 * 0.625) = 0.517549
#   GRPO    : correct 0.625 / 0.517549 = +1.207615, wrong -0.375 / 0.517549 = -0.724569
#   Dr.GRPO : correct +0.625, wrong -0.375
#   RLOO    : correct 1 - 2/7 = +5/7,   wrong 0 - 3/7 = -3/7
R8 = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def f64(*xs):
    return torch.tensor(xs, dtype=torch.float64)


def as_rows(*rows, dtype=torch.float64):
    """Pad a ragged list of per-token lists to (B, T) and return (tensor, mask)."""
    t_max = max(len(r) for r in rows)
    x = torch.zeros(len(rows), t_max, dtype=dtype)
    m = torch.zeros(len(rows), t_max, dtype=dtype)
    for i, r in enumerate(rows):
        x[i, : len(r)] = torch.tensor(r, dtype=dtype)
        m[i, : len(r)] = 1
    return x, m


# ----------------------------------------------------------------------------- group_advantages
def test_grpo_advantages_hand_computed(impl):
    adv = impl.group_advantages(f64(*R8), 8, "grpo")
    torch.testing.assert_close(adv[:3], torch.full((3,), 1.207615, dtype=torch.float64), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(adv[3:], torch.full((5,), -0.724569, dtype=torch.float64), atol=1e-4, rtol=1e-4)


def test_dr_grpo_advantages_hand_computed(impl):
    adv = impl.group_advantages(f64(*R8), 8, "dr_grpo")
    torch.testing.assert_close(adv, f64(*([0.625] * 3 + [-0.375] * 5)))


def test_rloo_advantages_hand_computed_and_brute_force(impl):
    adv = impl.group_advantages(f64(*R8), 8, "rloo")
    torch.testing.assert_close(adv, f64(*([5 / 7] * 3 + [-3 / 7] * 5)))
    r = torch.randn(12, dtype=torch.float64)
    brute = torch.stack([r[i] - torch.cat([r[:i], r[i + 1:]]).mean() for i in range(12)])
    torch.testing.assert_close(impl.group_advantages(r, 12, "rloo"), brute)


def test_rloo_is_rescaled_dr_grpo(impl):
    # RLOO advantage = G / (G - 1) * (r - mean): same direction as Dr. GRPO, slightly larger
    r = torch.randn(3 * 5, dtype=torch.float64)
    a_rloo = impl.group_advantages(r, 5, "rloo")
    a_dr = impl.group_advantages(r, 5, "dr_grpo")
    torch.testing.assert_close(a_rloo, a_dr * 5 / 4)


@pytest.mark.parametrize("kind", ["grpo", "dr_grpo", "rloo"])
def test_advantages_are_centered_per_group_and_groups_are_independent(impl, kind):
    r = torch.tensor([1., 0., 0., 1.,  0., 0., 0., 1.,  1., 1., 0., 0.], dtype=torch.float64)
    adv = impl.group_advantages(r, 4, kind)
    assert adv.shape == r.shape
    torch.testing.assert_close(adv.view(3, 4).sum(1), torch.zeros(3, dtype=torch.float64), atol=1e-9, rtol=0)
    # changing group 0 must not change groups 1 and 2
    r2 = r.clone()
    r2[:4] = torch.tensor([0., 0., 0., 0.], dtype=torch.float64)
    adv2 = impl.group_advantages(r2, 4, kind)
    torch.testing.assert_close(adv2[4:], adv[4:])


@pytest.mark.parametrize("kind", ["grpo", "dr_grpo", "rloo"])
def test_uniform_group_has_exactly_zero_advantage(impl, kind):
    r = torch.tensor([1., 1., 1., 1.,  0., 0., 0., 0.,  1., 0., 1., 0.], dtype=torch.float32)
    adv = impl.group_advantages(r, 4, kind)
    assert torch.isfinite(adv).all()
    assert torch.equal(adv[:8], torch.zeros(8))
    assert adv[8:].abs().sum() > 0


def test_grpo_epsilon_is_added_to_the_std(impl):
    # (r - mean) / (std + eps): with a large eps the placement is visible (not sqrt(var + eps), not a clamp)
    std = math.sqrt(8 / 7 * 0.375 * 0.625)
    adv = impl.group_advantages(f64(*R8), 8, "grpo", eps=0.5)
    torch.testing.assert_close(adv[:3], torch.full((3,), 0.625 / (std + 0.5), dtype=torch.float64))
    torch.testing.assert_close(adv[3:], torch.full((5,), -0.375 / (std + 0.5), dtype=torch.float64))


@pytest.mark.parametrize("kind", ["grpo", "dr_grpo", "rloo"])
def test_uniform_group_is_exactly_zero_even_for_inexact_rewards_and_zero_eps(impl, kind):
    # 0.1 has no exact float representation: the mean of equal values can differ from them by one ulp, and
    # dividing that residual by a tiny std + eps would turn round-off into a real advantage
    for g in (3, 7, 8):
        for v in (0.1, 0.3, 1 / 3):
            r = torch.cat([torch.full((g,), v), torch.arange(g, dtype=torch.float32)])   # uniform group, mixed group
            adv = impl.group_advantages(r, g, kind, eps=1e-6)
            assert torch.equal(adv[:g], torch.zeros(g)), (kind, g, v)
            assert adv[g:].abs().sum() > 0
    adv = impl.group_advantages(torch.ones(4), 4, kind, eps=0.0)             # 0 / 0 must not leak a NaN
    assert torch.equal(adv, torch.zeros(4))


def test_group_advantages_validation_and_dtype(impl):
    with pytest.raises(ValueError):
        impl.group_advantages(torch.zeros(8), 4, "ppo")
    with pytest.raises(ValueError):
        impl.group_advantages(torch.zeros(7), 4, "grpo")
    assert impl.group_advantages(torch.rand(8, dtype=torch.float32), 4).dtype == torch.float32


def test_grpo_advantage_scale_invariance_but_dr_grpo_is_not(impl):
    # std normalization removes the reward scale; the Dr. GRPO advantage keeps it
    r = torch.tensor([1., 0., 0., 1., 0., 0., 0., 0.], dtype=torch.float64)
    a1 = impl.group_advantages(r, 8, "grpo", eps=0.0)
    a10 = impl.group_advantages(10 * r, 8, "grpo", eps=0.0)
    torch.testing.assert_close(a1, a10)
    torch.testing.assert_close(impl.group_advantages(10 * r, 8, "dr_grpo"), 10 * impl.group_advantages(r, 8, "dr_grpo"))


# ----------------------------------------------------------------------------- grpo_loss: clipping
def _single_token_grad(impl, ratio, adv, eps_low=0.2, eps_high=0.2):
    lp_old = torch.zeros(1, 1, dtype=torch.float64)
    lp_new = torch.full((1, 1), math.log(ratio), dtype=torch.float64, requires_grad=True)
    loss = impl.grpo_loss(lp_new, lp_old, lp_old, f64(adv), torch.ones(1, 1, dtype=torch.float64),
                          eps_low=eps_low, eps_high=eps_high, beta=0.0)
    loss.backward()
    return loss.item(), lp_new.grad.item()


def test_clip_behavior_four_quadrants(impl):
    # positive advantage, ratio above 1 + eps: clipped, objective = (1 + eps) * A, zero gradient
    loss, g = _single_token_grad(impl, 1.5, +2.0)
    assert loss == pytest.approx(-1.2 * 2.0) and g == 0.0
    # positive advantage, ratio below 1 - eps: NOT clipped (pessimistic min), gradient = -A * ratio
    loss, g = _single_token_grad(impl, 0.5, +2.0)
    assert loss == pytest.approx(-0.5 * 2.0) and g == pytest.approx(-2.0 * 0.5)
    # negative advantage, ratio above 1 + eps: NOT clipped, gradient = -A * ratio
    loss, g = _single_token_grad(impl, 1.5, -2.0)
    assert loss == pytest.approx(1.5 * 2.0) and g == pytest.approx(2.0 * 1.5)
    # negative advantage, ratio below 1 - eps: clipped, objective = (1 - eps) * A, zero gradient
    loss, g = _single_token_grad(impl, 0.5, -2.0)
    assert loss == pytest.approx(0.8 * 2.0) and g == 0.0
    # inside the trust region: plain policy gradient, d(-ratio * A)/d logp = -ratio * A
    loss, g = _single_token_grad(impl, 1.1, 3.0)
    assert loss == pytest.approx(-1.1 * 3.0) and g == pytest.approx(-1.1 * 3.0)


def test_clip_higher_is_asymmetric(impl):
    # ratio 1.25 with A > 0: clipped at eps_high = 0.2, but alive with DAPO's eps_high = 0.28
    _, g_sym = _single_token_grad(impl, 1.25, 1.0, eps_low=0.2, eps_high=0.2)
    loss, g_high = _single_token_grad(impl, 1.25, 1.0, eps_low=0.2, eps_high=0.28)
    assert g_sym == 0.0
    assert g_high == pytest.approx(-1.25) and loss == pytest.approx(-1.25)
    # and it is capped at 1.28 once the ratio exceeds it
    loss, g = _single_token_grad(impl, 1.4, 1.0, eps_low=0.2, eps_high=0.28)
    assert loss == pytest.approx(-1.28) and g == 0.0
    # the lower bound is unchanged: A < 0, ratio 0.75 is clipped at 0.8 under eps_low = 0.2
    loss, g = _single_token_grad(impl, 0.75, -1.0, eps_low=0.2, eps_high=0.28)
    assert loss == pytest.approx(0.8) and g == 0.0


# ----------------------------------------------------------------------------- grpo_loss: aggregation
def _token_weights(impl, lens, agg, max_len=None):
    """Gradient of the loss w.r.t. logp_new at ratio = 1, adv = 1, beta = 0 is -(per-token weight)."""
    rows = [[0.0] * n for n in lens]
    lp, m = as_rows(*rows)
    lp.requires_grad_(True)
    old = torch.zeros_like(lp)
    loss = impl.grpo_loss(lp, old, old, torch.ones(len(lens), dtype=torch.float64), m,
                          beta=0.0, agg=agg, max_len=max_len)
    loss.backward()
    return -lp.grad, m


def test_seq_mean_weights_shorter_responses_more_per_token(impl):
    w, m = _token_weights(impl, [2, 4], "seq_mean")
    # weight = 1 / (B * |o_i|): 1/(2*2) = 0.25 per token of the short response, 1/(2*4) = 0.125 of the long one
    torch.testing.assert_close(w[0, :2], torch.full((2,), 0.25, dtype=torch.float64))
    torch.testing.assert_close(w[1, :4], torch.full((4,), 0.125, dtype=torch.float64))
    assert (w * (1 - m)).abs().sum() == 0


def test_token_mean_weights_every_token_equally(impl):
    w, m = _token_weights(impl, [2, 4], "token_mean")
    torch.testing.assert_close(w[m > 0], torch.full((6,), 1 / 6, dtype=torch.float64))
    assert (w * (1 - m)).abs().sum() == 0


def test_const_normalizer_matches_dr_grpo(impl):
    w, m = _token_weights(impl, [2, 4], "const", max_len=8)
    torch.testing.assert_close(w[m > 0], torch.full((6,), 1 / (2 * 8), dtype=torch.float64))
    with pytest.raises(ValueError):
        impl.grpo_loss(torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 1), torch.ones(1),
                       torch.ones(1, 1), agg="const")


def test_page_example_100_vs_1000_tokens(impl):
    # the length-bias example of page 24: same advantage, 100- vs 1000-token responses
    w, m = _token_weights(impl, [100, 1000], "seq_mean")
    assert w[0, 0].item() == pytest.approx(1 / (2 * 100))
    assert w[1, 0].item() == pytest.approx(1 / (2 * 1000))
    assert w[0, 0].item() / w[1, 0].item() == pytest.approx(10.0)
    w, m = _token_weights(impl, [100, 1000], "token_mean")
    assert w[0, 0].item() == pytest.approx(1 / 1100) and w[1, 0].item() == pytest.approx(1 / 1100)


def test_unknown_agg_raises(impl):
    z = torch.zeros(1, 1)
    with pytest.raises(ValueError):
        impl.grpo_loss(z, z, z, torch.ones(1), torch.ones(1, 1), agg="batch_mean")


# ----------------------------------------------------------------------------- grpo_loss: KL and masking
def test_k3_kl_term_hand_computed(impl):
    # advantage 0 isolates the KL term. pi_new = 0.5, pi_ref = 0.25 -> ref/new = 0.5,
    # k3 = 0.5 - log(0.5) - 1 = 0.193147
    lp_new = torch.full((1, 1), math.log(0.5), dtype=torch.float64)
    lp_ref = torch.full((1, 1), math.log(0.25), dtype=torch.float64)
    m = torch.ones(1, 1, dtype=torch.float64)
    loss = impl.grpo_loss(lp_new, lp_new, lp_ref, f64(0.0), m, beta=0.5)
    assert loss.item() == pytest.approx(0.5 * (0.5 - math.log(0.5) - 1))
    # the estimator is asymmetric: swap the roles and it changes (0.306853)
    loss = impl.grpo_loss(lp_ref, lp_ref, lp_new, f64(0.0), m, beta=1.0)
    assert loss.item() == pytest.approx(2.0 - math.log(2.0) - 1)


def test_kl_is_nonnegative_and_has_zero_gradient_at_the_reference(impl):
    lp = torch.randn(4, 6, dtype=torch.float64)
    m = torch.ones(4, 6, dtype=torch.float64)
    x = lp.clone().requires_grad_(True)
    loss = impl.grpo_loss(x, lp, lp, torch.zeros(4, dtype=torch.float64), m, beta=1.0)
    loss.backward()
    assert loss.item() == pytest.approx(0.0, abs=1e-12)
    assert x.grad.abs().max().item() < 1e-12
    y = (lp + 0.3 * torch.randn_like(lp)).requires_grad_(True)
    assert impl.grpo_loss(y, y.detach(), lp, torch.zeros(4, dtype=torch.float64), m, beta=1.0).item() > 0


@pytest.mark.parametrize("agg", ["seq_mean", "token_mean", "const"])
def test_padding_never_matters(impl, agg):
    lp_new = torch.randn(3, 5, dtype=torch.float64)
    lp_old = lp_new + 0.05 * torch.randn(3, 5, dtype=torch.float64)
    lp_ref = lp_new + 0.05 * torch.randn(3, 5, dtype=torch.float64)
    m = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.float64)
    adv = f64(0.7, -1.1, 0.4)

    def run(junk):
        a = lp_new.clone()
        a[m == 0] = junk
        a.requires_grad_(True)
        o = lp_old.clone()
        o[m == 0] = -junk
        r = lp_ref.clone()
        r[m == 0] = junk / 2
        loss = impl.grpo_loss(a, o, r, adv, m, eps_low=0.2, eps_high=0.28, beta=0.04, agg=agg, max_len=5)
        loss.backward()
        return loss.item(), a.grad.clone()

    l0, g0 = run(0.0)
    l1, g1 = run(30.0)
    assert l0 == pytest.approx(l1)
    torch.testing.assert_close(g0, g1)
    assert g1[m == 0].abs().sum() == 0


@pytest.mark.parametrize("agg", ["seq_mean", "token_mean", "const"])
def test_padding_junk_that_overflows_exp_cannot_reach_the_loss(impl, agg):
    # junk of +-800 makes exp() overflow to inf at padded positions, and inf * 0 is NaN: the junk must be
    # neutralized BEFORE the exponential, not just multiplied by the mask afterwards
    lp = torch.randn(2, 4, dtype=torch.float64)
    m = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.float64)
    # (junk in logp_new, logp_old, logp_ref): the first overflows the ratio, the second overflows the k3 exponent
    for j_new, j_old, j_ref in ((800.0, -800.0, -800.0), (-800.0, -800.0, 800.0)):
        new, old, ref = lp.clone(), lp.clone(), lp.clone()
        new[1, 2:], old[1, 2:], ref[1, 2:] = j_new, j_old, j_ref
        new.requires_grad_(True)
        for a in (f64(1.0, -1.0), f64(-1.0, 1.0)):
            new.grad = None
            loss = impl.grpo_loss(new, old, ref, a, m, beta=0.04, agg=agg, max_len=4)
            loss.backward()
            assert torch.isfinite(loss) and torch.isfinite(new.grad).all()
            assert new.grad[1, 2:].abs().sum() == 0


def test_seq_mean_equals_token_mean_when_lengths_are_equal(impl):
    lp_new = torch.randn(4, 6, dtype=torch.float64)
    lp_old = lp_new + 0.1 * torch.randn(4, 6, dtype=torch.float64)
    m = torch.ones(4, 6, dtype=torch.float64)
    adv = torch.randn(4, dtype=torch.float64)
    a = impl.grpo_loss(lp_new, lp_old, lp_new, adv, m, agg="seq_mean")
    b = impl.grpo_loss(lp_new, lp_old, lp_new, adv, m, agg="token_mean")
    c = impl.grpo_loss(lp_new, lp_old, lp_new, adv, m, agg="const", max_len=6)
    assert a.item() == pytest.approx(b.item()) and b.item() == pytest.approx(c.item())


# ----------------------------------------------------------------------------- the whole pipeline
def test_uniform_reward_group_gives_zero_gradient(impl):
    # two prompts, G = 4. Group 0: all correct (no signal). Group 1: mixed.
    G = 4
    rewards = torch.tensor([1., 1., 1., 1.,  1., 0., 0., 0.], dtype=torch.float64)
    adv = impl.group_advantages(rewards, G, "grpo")
    lp_old = torch.randn(8, 5, dtype=torch.float64)
    lp_new = (lp_old + 0.02 * torch.randn(8, 5, dtype=torch.float64)).requires_grad_(True)
    m = torch.ones(8, 5, dtype=torch.float64)
    for agg in ("seq_mean", "token_mean"):
        lp_new.grad = None
        impl.grpo_loss(lp_new, lp_old, lp_old, adv, m, beta=0.0, agg=agg).backward()
        assert lp_new.grad[:4].abs().sum().item() == 0.0          # uniform group: exactly zero
        assert lp_new.grad[4:].abs().sum().item() > 0.0
    # dynamic sampling keeps exactly the informative group
    keep = impl.dynamic_sampling_filter(rewards, G)
    assert keep.tolist() == [False, True]


# ----------------------------------------------------------------------------- GSPO
def test_gspo_ratio_hand_computed(impl):
    lp_new, m = as_rows([0.1, 0.3, 9.0], [-0.2, 9.0, 9.0], [0.05, 0.05, 0.05])
    lp_old = torch.zeros_like(lp_new)
    m[0, 2] = 0
    m[1, 1:] = 0
    s = impl.gspo_ratio(lp_new, lp_old, m)
    # masked positions hold junk (9.0) in row 0 and 1 and must be ignored
    torch.testing.assert_close(s, f64(math.exp(0.2), math.exp(-0.2), math.exp(0.05)))


def test_gspo_ratio_is_geometric_mean_of_token_ratios(impl):
    lp_new = torch.randn(3, 7, dtype=torch.float64)
    lp_old = lp_new + 0.1 * torch.randn(3, 7, dtype=torch.float64)
    m = torch.tensor([[1] * 7, [1] * 4 + [0] * 3, [1] * 2 + [0] * 5], dtype=torch.float64)
    s = impl.gspo_ratio(lp_new, lp_old, m)
    for i in range(3):
        n = int(m[i].sum())
        ratios = (lp_new[i, :n] - lp_old[i, :n]).exp()
        assert s[i].item() == pytest.approx(ratios.prod().item() ** (1 / n))
    # a response whose tokens are all masked returns ratio 1
    assert impl.gspo_ratio(lp_new[:1], lp_old[:1], torch.zeros(1, 7, dtype=torch.float64)).item() == pytest.approx(1.0)


def test_gspo_length_normalization_keeps_long_sequences_in_range(impl):
    # every token drifts by 0.01 nats: token-level ratio 1.01 for all lengths; the un-normalized
    # sequence ratio would be exp(0.01 * T), the normalized one stays exp(0.01)
    for T in (10, 1000):
        lp_new = torch.full((1, T), 0.01, dtype=torch.float64)
        s = impl.gspo_ratio(lp_new, torch.zeros_like(lp_new), torch.ones(1, T, dtype=torch.float64))
        assert s.item() == pytest.approx(math.exp(0.01))


def test_gspo_loss_clips_whole_sequences(impl):
    # sequence ratio exp(0.2) = 1.2214 is far outside [1 - 3e-4, 1 + 4e-4]
    lp_new = torch.full((2, 3), 0.2, dtype=torch.float64, requires_grad=True)
    lp_old = torch.zeros(2, 3, dtype=torch.float64)
    m = torch.ones(2, 3, dtype=torch.float64)
    adv = f64(1.0, -1.0)
    loss = impl.gspo_loss(lp_new, lp_old, adv, m)
    loss.backward()
    # response 0 (A > 0, ratio above the band): clipped -> value 1.0004, zero gradient on ALL its tokens
    # response 1 (A < 0, ratio above the band): not clipped -> value -1.2214, live gradient on all tokens
    expected = -(1.0004 * 1.0 + math.exp(0.2) * -1.0) / 2
    assert loss.item() == pytest.approx(expected)
    assert lp_new.grad[0].abs().sum().item() == 0.0
    g1 = lp_new.grad[1]
    assert (g1 != 0).all()
    # d(-(s * A)/2)/d logp_t = -A * s / (2 * |o|), the same for every token of the response
    assert torch.allclose(g1, torch.full((3,), 1.0 * math.exp(0.2) / (2 * 3), dtype=torch.float64))


def test_gspo_loss_clips_sequences_not_tokens(impl):
    # token log-ratios +0.01, -0.01, 0: two token ratios are far outside [1 - 3e-4, 1 + 4e-4], but their geometric
    # mean is exactly 1, so the response is NOT clipped and every token keeps the same live gradient
    lp_new = torch.tensor([[0.01, -0.01, 0.0]], dtype=torch.float64, requires_grad=True)
    loss = impl.gspo_loss(lp_new, torch.zeros_like(lp_new), f64(1.0), torch.ones(1, 3, dtype=torch.float64))
    loss.backward()
    assert loss.item() == pytest.approx(-1.0)
    torch.testing.assert_close(lp_new.grad, torch.full((1, 3), -1 / 3, dtype=torch.float64))


# ----------------------------------------------------------------------------- dynamic sampling
def test_dynamic_sampling_filter(impl):
    r = torch.tensor([1., 1., 1., 1.,   0., 0., 0., 0.,   1., 0., 1., 0.,   0., 0., 0., 1.], dtype=torch.float64)
    keep = impl.dynamic_sampling_filter(r, 4)
    assert keep.dtype == torch.bool and keep.tolist() == [False, False, True, True]
    # continuous rewards: identical values are uninformative, any spread is informative
    r2 = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.1, 0.2, 0.2, 0.2], dtype=torch.float64)
    assert impl.dynamic_sampling_filter(r2, 4).tolist() == [False, True]
    # -1 / +1 rewards (DAPO's convention) behave the same
    r3 = torch.tensor([-1., -1., -1., -1., 1., -1., -1., 1.], dtype=torch.float64)
    assert impl.dynamic_sampling_filter(r3, 4).tolist() == [False, True]
