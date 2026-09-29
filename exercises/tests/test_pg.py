import itertools
import math

import numpy as np
import torch

MODULE = "pg"

# ---------------------------------------------------------------------------------------------
# A two-token toy "language model" with exact enumeration (the worked example of page 22).
# Token a1 ~ Bernoulli(sigmoid(z0)); token a2 ~ Bernoulli(sigmoid(zA)) if a1 = 0 else sigmoid(zB).
# Terminal reward R(a1, a2). Parameters z = (z0, zA, zB) are logits.
# ---------------------------------------------------------------------------------------------
SEQS = list(itertools.product([0, 1], [0, 1]))
REWARD = {(0, 0): 0.0, (0, 1): 0.0, (1, 0): 0.0, (1, 1): 1.0}


def _seq_logprobs(z, a1, a2):
    z0, zA, zB = z
    lp1 = torch.nn.functional.logsigmoid(z0 if a1 else -z0)
    z2 = zB if a1 else zA
    lp2 = torch.nn.functional.logsigmoid(z2 if a2 else -z2)
    return torch.stack([lp1, lp2])                     # (2,)


def _probs(z):
    return torch.stack([_seq_logprobs(z, a1, a2).sum().exp() for a1, a2 in SEQS])


def _exact_grad(z, reward=REWARD):
    """grad of J(z) = sum_y pi(y) R(y) by autograd through the enumeration."""
    z = z.clone().requires_grad_(True)
    J = (_probs(z) * torch.tensor([reward[s] for s in SEQS], dtype=z.dtype)).sum()
    g, = torch.autograd.grad(J, z)
    return J.item(), g


def _batch(z, idx, reward=REWARD):
    lp = torch.stack([_seq_logprobs(z, *SEQS[i]) for i in idx])          # (len(idx), 2)
    rew = torch.zeros(len(idx), 2, dtype=z.dtype)
    for row, i in enumerate(idx):
        rew[row, 1] = reward[SEQS[i]]                                     # terminal reward only
    return lp, rew


def _baselines(z, J, which):
    """Baselines as (4, 2) arrays: b(s_0), b(s_1) per sequence."""
    p = _probs(z).detach()
    R = torch.tensor([reward for reward in REWARD.values()], dtype=z.dtype)
    if which == "zero":
        return torch.zeros(4, 2, dtype=z.dtype)
    if which == "const":
        return torch.full((4, 2), J, dtype=z.dtype)
    # state value baseline V(s_t): V(s_0) = J, V(s_1 | a1) = E[R | a1]
    V1 = {a1: (p[[2 * a1, 2 * a1 + 1]] * R[[2 * a1, 2 * a1 + 1]]).sum() / p[[2 * a1, 2 * a1 + 1]].sum() for a1 in (0, 1)}
    return torch.tensor([[J, V1[a1].item()] for a1, _ in SEQS], dtype=z.dtype)


def _weighted_grad(impl, z, which):
    """sum_y pi(y) * grad_z loss(y) computed with one response per call (B = 1)."""
    J, _ = _exact_grad(z)
    B = _baselines(z, J, which)
    p = _probs(z).detach()
    total = torch.zeros(3, dtype=z.dtype)
    per_sample = []
    for i in range(4):
        zz = z.clone().requires_grad_(True)
        lp, rew = _batch(zz, [i])
        loss = impl.reinforce_loss(lp, rew, B[i:i + 1])
        g, = torch.autograd.grad(loss, zz)
        per_sample.append(-g)                          # loss = -(estimator), so estimator = -grad
        total += p[i] * (-g)
    return total, torch.stack(per_sample), p


# ------------------------------------------------------------------------------ REINFORCE ----
def test_reinforce_is_unbiased_for_every_baseline(impl):
    for z in (torch.zeros(3, dtype=torch.float64), torch.tensor([0.4, -0.3, 0.8], dtype=torch.float64)):
        _, g_true = _exact_grad(z)
        for which in ("zero", "const", "state"):
            est, _, _ = _weighted_grad(impl, z, which)
            torch.testing.assert_close(est, g_true, atol=1e-10, rtol=1e-8)


def test_reinforce_exact_gradient_at_uniform_policy(impl):
    z = torch.zeros(3, dtype=torch.float64)
    _, g_true = _exact_grad(z)
    torch.testing.assert_close(g_true, torch.tensor([0.125, 0.0, 0.125], dtype=torch.float64))
    # one batch holding all four equally likely sequences: batch mean = exact expectation
    zz = z.clone().requires_grad_(True)
    lp, rew = _batch(zz, [0, 1, 2, 3])
    loss = impl.reinforce_loss(lp, rew, torch.zeros(4, 2, dtype=torch.float64))
    g, = torch.autograd.grad(loss, zz)
    torch.testing.assert_close(-g, g_true)


def test_reinforce_baselines_reduce_variance(impl):
    z = torch.zeros(3, dtype=torch.float64)
    var = {}
    for which in ("zero", "const", "state"):
        est, per_sample, p = _weighted_grad(impl, z, which)
        var[which] = ((p[:, None] * per_sample ** 2).sum() - (est ** 2).sum()).item()
    assert math.isclose(var["zero"], 0.09375, rel_tol=1e-9)
    assert math.isclose(var["const"], 0.0625, rel_tol=1e-9)
    assert math.isclose(var["state"], 0.046875, rel_tol=1e-9)


def test_reinforce_shifted_rewards_only_hurt_without_baseline(impl):
    z = torch.zeros(3, dtype=torch.float64)
    shifted = {k: v + 10.0 for k, v in REWARD.items()}
    J, _ = _exact_grad(z, shifted)
    p = _probs(z).detach()
    out = {}
    for name, b in (("none", torch.zeros(4, 2, dtype=torch.float64)),
                    ("const", torch.full((4, 2), J, dtype=torch.float64))):
        gs = []
        for i in range(4):
            zz = z.clone().requires_grad_(True)
            lp, rew = _batch(zz, [i], shifted)
            g, = torch.autograd.grad(impl.reinforce_loss(lp, rew, b[i:i + 1]), zz)
            gs.append(-g)
        gs = torch.stack(gs)
        mean = (p[:, None] * gs).sum(0)
        out[name] = ((p[:, None] * gs ** 2).sum() - (mean ** 2).sum()).item()
    assert out["none"] > 50 and math.isclose(out["const"], 0.0625, rel_tol=1e-9)


def test_reinforce_reward_to_go_gamma_and_mask(impl):
    lp = torch.tensor([[-1.0, -2.0, -3.0], [-0.5, -1.5, 9.0]], requires_grad=True)
    rew = torch.tensor([[1.0, 0.0, 2.0], [0.0, 4.0, 7.0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    b = torch.tensor([0.5, 1.0])                       # sequence-level baseline, shape (B,)
    gamma = 0.5
    loss = impl.reinforce_loss(lp, rew, b, mask=mask, gamma=gamma)
    G0 = [1 + 0.5 * 0 + 0.25 * 2, 0 + 0.5 * 2, 2]
    G1 = [0 + 0.5 * 4, 4, 0]
    manual = -(sum((g - 0.5) * l for g, l in zip(G0, [-1.0, -2.0, -3.0]))
               + sum((g - 1.0) * l for g, l in zip(G1[:2], [-0.5, -1.5]))) / 2
    assert math.isclose(loss.item(), manual, rel_tol=1e-6)
    loss.backward()
    assert lp.grad[1, 2].item() == 0.0                 # padded token gets no gradient
    torch.testing.assert_close(lp.grad[0], -torch.tensor([g - 0.5 for g in G0]) / 2)


def test_reinforce_no_gradient_through_advantage(impl):
    lp = torch.randn(3, 4, requires_grad=True)
    rew = torch.randn(3, 4, requires_grad=True)
    base = torch.randn(3, 4, requires_grad=True)
    impl.reinforce_loss(lp, rew, base).backward()
    assert rew.grad is None or torch.count_nonzero(rew.grad) == 0
    assert base.grad is None or torch.count_nonzero(base.grad) == 0


# ------------------------------------------------------------------------------------ GAE ----
def _gae_brute(r, v, gamma, lam, length, last_value=0.0):
    """A_t = sum_l (gamma lam)^l delta_{t+l}, python floats, one row of real length `length`."""
    T = len(r)
    vv = list(v[:length]) + [last_value if length == T else 0.0]
    delta = [r[t] + gamma * vv[t + 1] - vv[t] for t in range(length)]
    return [sum((gamma * lam) ** l * delta[t + l] for l in range(length - t)) for t in range(length)] + [0.0] * (T - length)


def test_gae_matches_brute_force(impl):
    torch.manual_seed(1)
    B, T = 5, 7
    r = torch.randn(B, T, dtype=torch.float64)
    v = torch.randn(B, T, dtype=torch.float64)
    lengths = [7, 3, 5, 1, 6]
    mask = torch.tensor([[1] * L + [0] * (T - L) for L in lengths])
    lv = torch.randn(B, dtype=torch.float64)
    for gamma, lam in [(1.0, 0.95), (1.0, 1.0), (0.99, 0.9), (0.9, 0.0), (0.97, 0.5)]:
        adv, ret = impl.gae(r, v, gamma, lam, mask=mask, last_value=lv)
        for i in range(B):
            ref = _gae_brute(r[i].tolist(), v[i].tolist(), gamma, lam, lengths[i], lv[i].item())
            torch.testing.assert_close(adv[i], torch.tensor(ref, dtype=torch.float64), atol=1e-10, rtol=1e-9)
        torch.testing.assert_close(ret, (adv + v) * mask)
        assert torch.all(adv[mask == 0] == 0) and torch.all(ret[mask == 0] == 0)


def test_gae_ignores_values_at_padding(impl):
    r = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 1, 0, 0]])
    v1 = torch.tensor([[0.2, 0.4, 0.6, 5.0, -7.0]])
    v2 = torch.tensor([[0.2, 0.4, 0.6, -3.0, 11.0]])
    a1, _ = impl.gae(r, v1, 1.0, 0.95, mask=mask)
    a2, _ = impl.gae(r, v2, 1.0, 0.95, mask=mask)
    torch.testing.assert_close(a1, a2)


def test_gae_worked_example_of_the_page(impl):
    r = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
    v = torch.tensor([[0.30, 0.35, 0.40, 0.55, 0.80]], dtype=torch.float64)
    a95, ret95 = impl.gae(r, v, 1.0, 0.95)
    torch.testing.assert_close(a95, torch.tensor([[0.61012, 0.5896, 0.568, 0.44, 0.20]], dtype=torch.float64))
    torch.testing.assert_close(ret95, torch.tensor([[0.91012, 0.9396, 0.968, 0.99, 1.00]], dtype=torch.float64))
    a1, ret1 = impl.gae(r, v, 1.0, 1.0)                # lambda = 1: Monte-Carlo return minus V
    torch.testing.assert_close(a1, 1.0 - v)
    torch.testing.assert_close(ret1, torch.ones_like(v))
    a0, _ = impl.gae(r, v, 1.0, 0.0)                   # lambda = 0: the TD residual
    torch.testing.assert_close(a0, torch.tensor([[0.05, 0.05, 0.15, 0.25, 0.20]], dtype=torch.float64))


def test_gae_lambda_one_is_return_minus_value(impl):
    torch.manual_seed(2)
    r = torch.randn(4, 6, dtype=torch.float64)
    v = torch.randn(4, 6, dtype=torch.float64)
    gamma = 0.9
    adv, _ = impl.gae(r, v, gamma, 1.0)
    G = torch.zeros_like(r)
    run = torch.zeros(4, dtype=torch.float64)
    for t in reversed(range(6)):
        run = r[:, t] + gamma * run
        G[:, t] = run
    torch.testing.assert_close(adv, G - v)


def test_gae_zero_critic_terminal_reward_decays_as_lambda_power(impl):
    T = 6
    r = torch.zeros(1, T, dtype=torch.float64); r[0, -1] = 1.0
    adv, _ = impl.gae(r, torch.zeros_like(r), 1.0, 0.95)
    expected = torch.tensor([[0.95 ** (T - 1 - t) for t in range(T)]], dtype=torch.float64)
    torch.testing.assert_close(adv, expected)


# ------------------------------------------------------------------------------------ PPO ----
def _grad_at_ratio(impl, ratio, A, **kw):
    lp_old = torch.zeros(1, 1)
    lp_new = torch.full((1, 1), math.log(ratio), requires_grad=True)
    loss = impl.ppo_clip_loss(lp_new, lp_old, torch.tensor([[A]]), **kw)
    loss.backward()
    return loss.item(), lp_new.grad.item()


def test_ppo_matches_naive_formula(impl):
    torch.manual_seed(3)
    B, T, eps = 4, 6, 0.2
    lp_old = torch.randn(B, T) * 0.5 - 2
    lp_new = lp_old + torch.randn(B, T) * 0.4
    adv = torch.randn(B, T)
    mask = torch.tensor([[1] * 6, [1] * 4 + [0] * 2, [1] * 2 + [0] * 4, [1] * 5 + [0]])
    rho = (lp_new - lp_old).exp()
    naive = torch.where(adv >= 0, torch.minimum(rho, torch.tensor(1 + eps)), torch.maximum(rho, torch.tensor(1 - eps))) * adv
    # (for A >= 0: min(rho A, clip A) = A * min(rho, 1+eps); for A < 0: A * max(rho, 1-eps))
    want = -(naive * mask).sum() / mask.sum()
    got = impl.ppo_clip_loss(lp_new, lp_old, adv, eps, mask)
    torch.testing.assert_close(got, want)


def test_ppo_gradient_in_the_four_regions(impl):
    eps = 0.2
    # A > 0
    _, g = _grad_at_ratio(impl, 1.5, +1.0, eps=eps); assert g == 0.0            # clipped: no incentive to go higher
    _, g = _grad_at_ratio(impl, 0.5, +1.0, eps=eps); assert math.isclose(g, -0.5, rel_tol=1e-6)   # still pushed up
    _, g = _grad_at_ratio(impl, 1.1, +1.0, eps=eps); assert math.isclose(g, -1.1, rel_tol=1e-6)
    # A < 0
    _, g = _grad_at_ratio(impl, 0.5, -1.0, eps=eps); assert g == 0.0            # clipped: no incentive to go lower
    _, g = _grad_at_ratio(impl, 1.5, -1.0, eps=eps); assert math.isclose(g, +1.5, rel_tol=1e-6)   # still pushed down
    _, g = _grad_at_ratio(impl, 0.9, -1.0, eps=eps); assert math.isclose(g, +0.9, rel_tol=1e-6)


def test_ppo_value_at_the_boundaries(impl):
    l, _ = _grad_at_ratio(impl, 1.5, +2.0, eps=0.2); assert math.isclose(l, -1.2 * 2.0, rel_tol=1e-6)
    l, _ = _grad_at_ratio(impl, 0.5, -2.0, eps=0.2); assert math.isclose(l, +0.8 * 2.0, rel_tol=1e-6)
    l, _ = _grad_at_ratio(impl, 0.5, +2.0, eps=0.2); assert math.isclose(l, -0.5 * 2.0, rel_tol=1e-6)
    l, _ = _grad_at_ratio(impl, 1.5, -2.0, eps=0.2); assert math.isclose(l, +1.5 * 2.0, rel_tol=1e-6)


def test_ppo_on_policy_gradient_is_vanilla_pg(impl):
    torch.manual_seed(4)
    lp = torch.randn(3, 5, requires_grad=True)
    adv = torch.randn(3, 5)
    mask = torch.ones(3, 5)
    loss = impl.ppo_clip_loss(lp, lp.detach().clone(), adv, 0.2, mask)
    loss.backward()
    torch.testing.assert_close(lp.grad, -adv / 15)      # rho = 1 exactly: grad = -A / N


def test_ppo_clip_higher(impl):
    _, g = _grad_at_ratio(impl, 1.25, +1.0, eps=0.2); assert g == 0.0
    _, g = _grad_at_ratio(impl, 1.25, +1.0, eps=0.2, eps_high=0.28); assert math.isclose(g, -1.25, rel_tol=1e-6)
    _, g = _grad_at_ratio(impl, 0.75, -1.0, eps=0.2, eps_high=0.28); assert g == 0.0   # lower clip unchanged


def test_ppo_aggregation(impl):
    lp_old = torch.zeros(2, 4)
    lp_new = torch.zeros(2, 4)                           # rho = 1, so the loss is -A per token
    adv = torch.tensor([1.0, -1.0])                      # one advantage per response
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])    # lengths 2 and 4
    tok = impl.ppo_clip_loss(lp_new, lp_old, adv, mask=mask, agg="token")
    seq = impl.ppo_clip_loss(lp_new, lp_old, adv, mask=mask, agg="seq")
    assert math.isclose(tok.item(), (-2 * 1.0 + 4 * 1.0) / 6, rel_tol=1e-6)     # long response weighs more
    assert math.isclose(seq.item(), (-1.0 + 1.0) / 2, abs_tol=1e-6)             # each response weighs the same


def test_ppo_old_logprobs_are_constants(impl):
    lp_new = torch.randn(2, 3, requires_grad=True)
    lp_old = torch.randn(2, 3, requires_grad=True)
    impl.ppo_clip_loss(lp_new, lp_old, torch.randn(2, 3)).backward()
    assert lp_old.grad is None or torch.count_nonzero(lp_old.grad) == 0


# ------------------------------------------------------------------------------------- KL ----
Q = np.array([0.5, 0.3, 0.2])        # pi (sampling distribution)
P = np.array([0.4, 0.4, 0.2])        # pi_ref


def _true_kl():
    return float((Q * np.log(Q / P)).sum())


def test_kl_estimator_formulas_and_signs(impl):
    lp = torch.log(torch.tensor(Q)); lr_ = torch.log(torch.tensor(P))
    k1, k2, k3 = impl.kl_estimators(lp, lr_)
    r = torch.tensor(P / Q)
    torch.testing.assert_close(k1, -torch.log(r))
    torch.testing.assert_close(k2, 0.5 * torch.log(r) ** 2)
    torch.testing.assert_close(k3, (r - 1) - torch.log(r))
    assert torch.all(k2 >= 0) and torch.all(k3 >= 0)
    assert (k1 < 0).any()                                  # k1 is negative for some samples
    z = torch.randn(50)
    k1, k2, k3 = impl.kl_estimators(z, z)
    assert torch.all(k1 == 0) and torch.all(k2 == 0) and torch.all(k3 == 0)


def test_kl_expectations_under_the_sampling_distribution(impl):
    k1, k2, k3 = impl.kl_estimators(torch.log(torch.tensor(Q)), torch.log(torch.tensor(P)))
    q = torch.tensor(Q)
    kl = _true_kl()
    assert math.isclose((q * k1).sum().item(), kl, rel_tol=1e-9)        # unbiased
    assert math.isclose((q * k3).sum().item(), kl, rel_tol=1e-9)        # unbiased
    e2 = (q * k2).sum().item()
    assert not math.isclose(e2, kl, rel_tol=1e-4) and math.isclose(e2, kl, rel_tol=0.05)   # biased, but only slightly here
    var = lambda k: ((q * k ** 2).sum() - (q * k).sum() ** 2).item()
    assert var(k3) < var(k1) / 100 and var(k2) < var(k1) / 100          # both far less noisy than k1 here


def test_kl_monte_carlo_on_sampled_tokens(impl):
    g = torch.Generator().manual_seed(0)
    idx = torch.multinomial(torch.tensor(Q), 400_000, replacement=True, generator=g)
    lp = torch.log(torch.tensor(Q))[idx]; lr_ = torch.log(torch.tensor(P))[idx]
    k1, k2, k3 = impl.kl_estimators(lp, lr_)
    kl = _true_kl()
    assert abs(k3.mean().item() - kl) / kl < 0.01
    assert abs(k1.mean().item() - kl) / kl < 0.05
    assert k3.min().item() >= 0.0


def test_kl_k3_is_stable_for_tiny_log_ratios(impl):
    x = torch.tensor([1e-3, -1e-3, 3e-4], dtype=torch.float32)
    _, _, k3 = impl.kl_estimators(torch.zeros(3, dtype=torch.float32), x)      # log r = x
    exact = (torch.expm1(x.double()) - x.double())
    torch.testing.assert_close(k3.double(), exact, rtol=1e-3, atol=0)


def test_kl_gradients_of_the_estimators_used_as_losses(impl):
    """Tang and Munos (2025): grad of E_pi[k] with the sampling weights frozen."""
    logits = torch.tensor([0.3, -0.2, 0.5], dtype=torch.float64, requires_grad=True)
    pref = torch.tensor([0.4, 0.4, 0.2], dtype=torch.float64)

    def loss_grad(which):
        pi = torch.softmax(logits, 0)
        ks = impl.kl_estimators(torch.log(pi), torch.log(pref))
        g, = torch.autograd.grad((pi.detach() * ks[which]).sum(), logits)
        return g

    def kl_grad(forward):
        pi = torch.softmax(logits, 0)
        kl = (pi * (pi.log() - pref.log())).sum() if forward else (pref * (pref.log() - pi.log())).sum()
        g, = torch.autograd.grad(kl, logits)
        return g

    torch.testing.assert_close(loss_grad(0), torch.zeros(3, dtype=torch.float64), atol=1e-12, rtol=0)   # k1: zero in expectation
    torch.testing.assert_close(loss_grad(1), kl_grad(True))     # k2: gradient of KL(pi || pi_ref)
    torch.testing.assert_close(loss_grad(2), kl_grad(False))    # k3: gradient of the REVERSE KL(pi_ref || pi)


# ------------------------------------------------------------------------------ whitening ----
def test_masked_whiten(impl):
    torch.manual_seed(5)
    x = torch.randn(3, 6) * 4 + 7
    mask = torch.tensor([[1] * 6, [1] * 3 + [0] * 3, [1] * 5 + [0]])
    x[mask == 0] = 1e6                                    # garbage at padding must not matter
    w = impl.masked_whiten(x, mask)
    real = w[mask == 1]
    assert abs(real.mean().item()) < 1e-5 and abs(real.var(unbiased=False).item() - 1) < 1e-4
    assert torch.all(w[mask == 0] == 0)
    w2 = impl.masked_whiten(x, mask, shift_mean=False)
    real2 = w2[mask == 1]
    assert abs(real2.mean().item() - x[mask == 1].mean().item()) < 1e-4
    assert abs(real2.var(unbiased=False).item() - 1) < 1e-4
