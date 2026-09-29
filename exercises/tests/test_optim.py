import math

import pytest
import torch

MODULE = "optim"


# ---------------------------------------------------------------- AdamW

def _run_ours(impl, p0, grads, lr, betas, eps, wd):
    p, m, v = p0.clone(), torch.zeros_like(p0), torch.zeros_like(p0)
    for t, g in enumerate(grads, start=1):
        p, m, v = impl.adamw_step(p, g, m, v, t, lr, betas, eps, wd)
    return p, m, v


def _run_torch(p0, grads, lr, betas, eps, wd):
    p = torch.nn.Parameter(p0.clone())
    opt = torch.optim.AdamW([p], lr=lr, betas=betas, eps=eps, weight_decay=wd, foreach=False)
    for g in grads:
        p.grad = g.clone()
        opt.step()
    st = opt.state[p]
    return p.detach(), st["exp_avg"], st["exp_avg_sq"]


@pytest.mark.parametrize("betas,eps,wd", [((0.9, 0.999), 1e-8, 0.01), ((0.9, 0.95), 1e-5, 0.1), ((0.8, 0.9), 1e-3, 0.0)])
def test_adamw_matches_torch(impl, betas, eps, wd):
    p0 = torch.randn(7, 5, dtype=torch.float64)
    grads = [torch.randn(7, 5, dtype=torch.float64) * (1 + 0.3 * i) for i in range(6)]
    ours = _run_ours(impl, p0, grads, 3e-3, betas, eps, wd)
    ref = _run_torch(p0, grads, 3e-3, betas, eps, wd)
    for a, b in zip(ours, ref):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-12)


def test_adamw_first_step_is_lr_times_sign(impl):
    # with bias correction, m_hat = g and v_hat = g^2 at t = 1, so the step is lr * sign(g)
    p = torch.zeros(6, dtype=torch.float64)
    g = torch.tensor([3.0, -0.2, 50.0, -7.0, 1e-3, -1e3], dtype=torch.float64)
    z = torch.zeros_like(p)
    new_p, m, v = impl.adamw_step(p, g, z, z, 1, 0.01, (0.9, 0.999), 1e-12, 0.0)
    torch.testing.assert_close(new_p, -0.01 * g.sign(), rtol=1e-6, atol=1e-9)
    torch.testing.assert_close(m, 0.1 * g)
    torch.testing.assert_close(v, 0.001 * g * g)


def test_adamw_weight_decay_is_decoupled(impl):
    p = torch.randn(4, 3, dtype=torch.float64)
    g = torch.randn(4, 3, dtype=torch.float64)
    z = torch.zeros_like(p)
    lr, wd = 0.02, 0.3
    with_wd, m1, v1 = impl.adamw_step(p, g, z, z, 1, lr, (0.9, 0.95), 1e-8, wd)
    no_wd, m2, v2 = impl.adamw_step(p, g, z, z, 1, lr, (0.9, 0.95), 1e-8, 0.0)
    # the decay term is exactly -lr * wd * param, whatever the gradient is ...
    torch.testing.assert_close(with_wd - no_wd, -lr * wd * p)
    # ... and it never leaks into the moment estimates (that is what L2 regularization would do)
    torch.testing.assert_close(m1, m2)
    torch.testing.assert_close(v1, v2)
    # zero gradient, zero state: pure multiplicative decay
    out, _, _ = impl.adamw_step(p, torch.zeros_like(p), z, z, 1, lr, (0.9, 0.95), 1e-8, wd)
    torch.testing.assert_close(out, p * (1 - lr * wd))


def test_adamw_does_not_mutate_inputs(impl):
    p, g = torch.randn(5), torch.randn(5)
    m, v = torch.randn(5), torch.rand(5)
    saved = [x.clone() for x in (p, g, m, v)]
    impl.adamw_step(p, g, m, v, 3, 1e-3, (0.9, 0.95), 1e-8, 0.1)
    for x, y in zip((p, g, m, v), saved):
        assert torch.equal(x, y)


# ---------------------------------------------------------------- schedules

def test_cosine_values(impl):
    kw = dict(kind="cosine", peak_lr=1.0, total_steps=1000, warmup_steps=100, min_lr_ratio=0.1)
    f = lambda s: impl.lr_schedule(s, **kw)
    assert f(0) == pytest.approx(1 / 100)
    assert f(49) == pytest.approx(0.5)
    assert f(99) == pytest.approx(1.0)
    assert f(100) == pytest.approx(1.0)                                  # cosine starts at the peak
    assert f(550) == pytest.approx(0.55)                                 # halfway: (peak + min) / 2
    assert f(1000) == pytest.approx(0.1)
    assert f(5000) == pytest.approx(0.1)                                 # stays at the floor
    vals = [f(s) for s in range(100, 1001)]
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:]))           # monotone non-increasing


def test_cosine_matches_torch_scheduler(impl):
    W, T, peak, ratio = 20, 200, 0.5, 0.1
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=peak)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T - W, eta_min=peak * ratio)
    for k in range(T - W):
        assert impl.lr_schedule(W + k, "cosine", peak, T, W, ratio) == pytest.approx(opt.param_groups[0]["lr"], rel=1e-9)
        opt.step()
        sched.step()


def test_wsd_values(impl):
    kw = dict(kind="wsd", peak_lr=2.0, total_steps=1000, warmup_steps=50, min_lr_ratio=0.1, decay_steps=200)
    f = lambda s: impl.lr_schedule(s, **kw)
    assert f(0) == pytest.approx(2.0 / 50)
    assert f(49) == pytest.approx(2.0)
    assert f(50) == pytest.approx(2.0)
    assert f(799) == pytest.approx(2.0)                                  # stable phase
    assert f(800) == pytest.approx(2.0)                                  # decay starts at total - decay_steps
    assert f(900) == pytest.approx(1.1)                                  # linear midpoint of 2.0 -> 0.2
    assert f(1000) == pytest.approx(0.2)
    assert f(2000) == pytest.approx(0.2)
    # default decay length is 10% of training
    g = lambda s: impl.lr_schedule(s, "wsd", 1.0, 1000, 0, 0.0)
    assert g(899) == pytest.approx(1.0) and g(950) == pytest.approx(0.5) and g(1000) == pytest.approx(0.0)


def test_wsd_and_cosine_share_the_warmup_but_not_the_tail(impl):
    for s in range(0, 10):
        assert impl.lr_schedule(s, "wsd", 1.0, 100, 10) == pytest.approx(impl.lr_schedule(s, "cosine", 1.0, 100, 10))
    assert impl.lr_schedule(60, "wsd", 1.0, 100, 10) > impl.lr_schedule(60, "cosine", 1.0, 100, 10)


def test_step_schedule_is_deepseek_style(impl):
    f = lambda s: impl.lr_schedule(s, "step", 2.4e-4, 1000, 2)
    assert f(0) == pytest.approx(2.4e-4 / 2) and f(1) == pytest.approx(2.4e-4)
    assert f(799) == pytest.approx(2.4e-4)
    assert f(800) == pytest.approx(2.4e-4 * 0.316)
    assert f(899) == pytest.approx(2.4e-4 * 0.316)
    assert f(900) == pytest.approx(2.4e-4 * 0.1)
    # DeepSeek-V2 variant: x0.316 at 60% and again (relative to peak: x0.1) at 90%
    g = lambda s: impl.lr_schedule(s, "step", 1.0, 100, 0, milestones=(0.6, 0.9), factors=(0.316, 0.1))
    assert g(59) == pytest.approx(1.0) and g(60) == pytest.approx(0.316) and g(90) == pytest.approx(0.1)


def test_schedule_edge_cases(impl):
    assert impl.lr_schedule(0, "cosine", 1.0, 10, 0) == pytest.approx(1.0)     # no warmup: no division by zero
    with pytest.raises(ValueError):
        impl.lr_schedule(0, "linear", 1.0, 10)


# ---------------------------------------------------------------- Newton-Schulz

def _svals(x):
    return torch.linalg.svdvals(x.double())


@pytest.mark.parametrize("shape", [(64, 256), (256, 64), (32, 96)])
def test_newton_schulz_pushes_singular_values_to_one(impl, shape):
    G = torch.randn(*shape, dtype=torch.float64)
    before = _svals(G / G.norm())
    out = impl.newton_schulz(G, steps=5)
    assert out.shape == G.shape and out.dtype == G.dtype
    s = _svals(out)
    assert before.max() < 0.3                                            # normalized input: tiny singular values
    assert 0.5 < s.min() and s.max() < 1.3                               # Muon's quintic lands in roughly [0.7, 1.15]
    assert s.max() / s.min() < 2.0                                       # condition number: ~3+ before, < 2 after


def test_newton_schulz_close_to_polar_factor(impl):
    G = torch.randn(32, 96, dtype=torch.float64)
    U, S, Vh = torch.linalg.svd(G, full_matrices=False)
    ref = U @ Vh
    out = impl.newton_schulz(G, steps=5)
    cos = (out * ref).sum() / (out.norm() * ref.norm())
    assert cos > 0.95                                                    # not exact: S' is in [0.7, 1.15], not 1
    assert (out - ref).norm() / ref.norm() < 0.4


def test_newton_schulz_exact_coefficients_converge_to_polar_factor(impl):
    # f(x) = 2x - 1.5x^3 + 0.5x^5 has f(1) = 1 and f'(1) = 0: converges to U V^T
    G = torch.randn(16, 40, dtype=torch.float64)
    U, S, Vh = torch.linalg.svd(G, full_matrices=False)
    out = impl.newton_schulz(G, steps=40, coeffs=(2.0, -1.5, 0.5))
    torch.testing.assert_close(out, U @ Vh, rtol=1e-6, atol=1e-6)


def test_newton_schulz_equivariance_and_scale_invariance(impl):
    G = torch.randn(20, 50, dtype=torch.float64)
    torch.testing.assert_close(impl.newton_schulz(G.T), impl.newton_schulz(G).T)
    torch.testing.assert_close(impl.newton_schulz(7.0 * G), impl.newton_schulz(G), rtol=1e-5, atol=1e-8)
    assert impl.newton_schulz(torch.zeros(5, 8)).abs().max() == 0        # zero in, zero out (eps guards the norm)


def test_newton_schulz_amplifies_small_directions(impl):
    # a matrix with one dominant direction: after NS the small directions are not small any more
    n, m = 32, 64
    U, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))
    V, _ = torch.linalg.qr(torch.randn(m, n, dtype=torch.float64))
    sig = torch.logspace(0, -1.5, n, dtype=torch.float64)      # condition number about 32
    G = U @ torch.diag(sig) @ V.T
    out = impl.newton_schulz(G, steps=5)
    assert _svals(out).min() > 0.5
    assert _svals(G / G.norm()).min() < 0.02


# ---------------------------------------------------------------- Muon step

def _ref_muon(param, grad, buf, lr, mu, wd, nesterov, update_rms):
    buf = mu * buf + grad
    x = mu * buf + grad if nesterov else buf
    U, S, Vh = torch.linalg.svd(x, full_matrices=False)
    n, m = x.shape
    O = (U @ Vh) * update_rms * math.sqrt(max(n, m))
    return param * (1 - lr * wd) - lr * O, buf


@pytest.mark.parametrize("shape", [(40, 40), (24, 80), (80, 24)])
def test_muon_step_agrees_with_svd_reference(impl, shape):
    p = torch.randn(*shape, dtype=torch.float64)
    g = torch.randn(*shape, dtype=torch.float64)
    buf = torch.randn(*shape, dtype=torch.float64)
    new_p, new_buf = impl.muon_step(p, g, buf, lr=0.02, momentum=0.95, wd=0.1)
    ref_p, ref_buf = _ref_muon(p, g, buf, 0.02, 0.95, 0.1, True, 0.2)
    torch.testing.assert_close(new_buf, ref_buf)                        # the momentum buffer is exact
    d, dref = new_p - p * (1 - 0.02 * 0.1), ref_p - p * (1 - 0.02 * 0.1)
    cos = (d * dref).sum() / (d.norm() * dref.norm())
    assert cos > 0.95                                                    # NS is an approximation of U V^T
    assert 0.7 < d.norm() / dref.norm() < 1.3


def test_muon_update_rms_is_shape_independent(impl):
    # the whole point of the 0.2 * sqrt(max(n, m)) rescale: same update RMS for every shape
    lr = 1.0
    rms = []
    for shape in [(64, 64), (64, 256), (512, 64), (32, 1024)]:
        g = torch.randn(*shape, dtype=torch.float64)
        z = torch.zeros(*shape, dtype=torch.float64)
        p, _ = impl.muon_step(z, g, z, lr=lr, update_rms=0.2)
        rms.append(p.pow(2).mean().sqrt().item())
    assert all(0.15 < r < 0.25 for r in rms), rms
    # without it (original post's rule) the RMS shrinks with the larger dimension
    g = torch.randn(64, 1024, dtype=torch.float64)
    z = torch.zeros_like(g)
    p, _ = impl.muon_step(z, g, z, lr=1.0, update_rms=None)
    assert p.pow(2).mean().sqrt().item() < 0.05


def test_muon_weight_decay_and_zero_gradient(impl):
    p = torch.randn(10, 12, dtype=torch.float64)
    z = torch.zeros_like(p)
    out, buf = impl.muon_step(p, z, z, lr=0.1, wd=0.5)
    torch.testing.assert_close(out, p * (1 - 0.1 * 0.5))
    assert buf.abs().max() == 0


def test_muon_nesterov_and_buffer(impl):
    p = torch.randn(16, 24, dtype=torch.float64)
    g1, g2 = torch.randn_like(p), torch.randn_like(p)
    z = torch.zeros_like(p)
    # first step: Nesterov input is (1 + mu) * g, and NS is scale-invariant, so both variants agree
    a, buf_a = impl.muon_step(p, g1, z, lr=0.05, nesterov=True)
    b, buf_b = impl.muon_step(p, g1, z, lr=0.05, nesterov=False)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(buf_a, g1)
    # second step: they differ, and the buffer follows buf <- mu * buf + g
    a2, buf_a2 = impl.muon_step(p, g2, buf_a, lr=0.05, nesterov=True)
    b2, buf_b2 = impl.muon_step(p, g2, buf_b, lr=0.05, nesterov=False)
    torch.testing.assert_close(buf_a2, 0.95 * g1 + g2)
    torch.testing.assert_close(buf_a2, buf_b2)
    assert (a2 - b2).abs().max() > 1e-4


def test_muon_does_not_mutate_inputs(impl):
    p, g, b = torch.randn(6, 9), torch.randn(6, 9), torch.randn(6, 9)
    saved = [x.clone() for x in (p, g, b)]
    impl.muon_step(p, g, b, lr=0.02, wd=0.1)
    for x, y in zip((p, g, b), saved):
        assert torch.equal(x, y)
