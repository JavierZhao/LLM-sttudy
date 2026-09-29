import math

import numpy as np
import pytest

MODULE = "scaling"

HOFF = dict(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28)                 # Hoffmann et al., as printed
EPOCH = dict(E=1.8172, A=482.01, B=2085.43, alpha=0.3478, beta=0.3658)       # Besiroglu et al. replication
SARDANA = dict(E=1.69, A=406.4, B=410.7, alpha=0.336, beta=0.283)            # constants in Sardana & Frankle, App. A


# ------------------------------------------------------------ independent helpers (not the drill)
def loss_ref(N, D, p):
    return p["E"] + p["A"] / N ** p["alpha"] + p["B"] / D ** p["beta"]


def optimum_ref(C, p):
    """Minimize the loss on the IsoFLOP curve by brute force over a fine grid in ln N."""
    lo, hi = math.log(1e5), math.log(1e15)
    lnN = np.linspace(lo, hi, 800_001)
    N = np.exp(lnN)
    L = loss_ref(N, C / (6 * N), p)
    i = int(L.argmin())
    assert 0 < i < len(N) - 1
    return float(N[i]), float(C / (6 * N[i]))


def closed_form_ref(C, p):
    a = p["beta"] / (p["alpha"] + p["beta"])
    G = (p["alpha"] * p["A"] / (p["beta"] * p["B"])) ** (1 / (p["alpha"] + p["beta"]))
    N = G * (C / 6) ** a
    return N, C / (6 * N)


# ------------------------------------------------------------------ chinchilla_loss
def test_loss_scalar_value(impl):
    N, D = 1e9, 2e10
    got = impl.chinchilla_loss(N, D, **HOFF)
    want = 1.69 + 406.4 / N**0.34 + 410.7 / D**0.28
    assert isinstance(got, float)
    assert got == pytest.approx(want, rel=1e-12)
    assert 2.5 < got < 2.7


def test_loss_broadcasting_and_no_mutation(impl):
    N = np.array([[1e8], [1e9], [1e10]])           # (3, 1)
    D = np.array([[1e10, 1e11, 1e12, 1e13]])       # (1, 4)
    N0, D0 = N.copy(), D.copy()
    got = impl.chinchilla_loss(N, D, **EPOCH)
    assert got.shape == (3, 4)
    for i in range(3):
        for j in range(4):
            assert got[i, j] == pytest.approx(loss_ref(N[i, 0], D[0, j], EPOCH), rel=1e-12)
    assert np.array_equal(N, N0) and np.array_equal(D, D0)


def test_loss_limits_and_monotonicity(impl):
    p = HOFF
    # infinite data leaves the model-size term; infinite model and data leave E
    assert impl.chinchilla_loss(1e9, 1e200, **p) == pytest.approx(p["E"] + p["A"] / 1e9**p["alpha"], rel=1e-12)
    assert impl.chinchilla_loss(1e200, 1e200, **p) == pytest.approx(p["E"], abs=1e-12)
    Ns = np.logspace(7, 12, 20)
    assert np.all(np.diff(impl.chinchilla_loss(Ns, 1e11, **p)) < 0)
    Ds = np.logspace(9, 14, 20)
    assert np.all(np.diff(impl.chinchilla_loss(1e9, Ds, **p)) < 0)


# ------------------------------------------------------------------ compute_optimal
@pytest.mark.parametrize("p", [HOFF, EPOCH, SARDANA], ids=["hoffmann", "epoch", "sardana"])
@pytest.mark.parametrize("C", [1e19, 1e22, 1e25])
def test_compute_optimal_uses_the_budget_and_first_order_condition(impl, p, C):
    N, D = impl.compute_optimal(C, **p)
    assert 6 * N * D == pytest.approx(C, rel=1e-9)
    # at the optimum the marginal loss reductions per e-fold of N and of D are equal
    assert p["alpha"] * p["A"] / N ** p["alpha"] == pytest.approx(p["beta"] * p["B"] / D ** p["beta"], rel=1e-9)


@pytest.mark.parametrize("p", [HOFF, EPOCH], ids=["hoffmann", "epoch"])
@pytest.mark.parametrize("C", [1e20, 1e23, 1e26])
def test_compute_optimal_matches_numeric_minimization(impl, p, C):
    N, D = impl.compute_optimal(C, **p)
    Nn, Dn = optimum_ref(C, p)
    assert N == pytest.approx(Nn, rel=2e-4)
    assert D == pytest.approx(Dn, rel=2e-4)
    # and no size on the IsoFLOP curve does better
    for f in (0.8, 0.95, 1.05, 1.25):
        assert loss_ref(N * f, C / (6 * N * f), p) > loss_ref(N, D, p)


def test_compute_optimal_regression_values(impl):
    # values computed once from the closed form, at C = 1e24 FLOPs
    N, D = impl.compute_optimal(1e24, **EPOCH)
    assert N == pytest.approx(9.586e10, rel=1e-2) and D == pytest.approx(1.739e12, rel=1e-2)
    N, D = impl.compute_optimal(1e24, **HOFF)
    assert N == pytest.approx(4.13e10, rel=1e-2) and D == pytest.approx(4.036e12, rel=1e-2)


def test_compute_optimal_scaling_exponents(impl):
    p = EPOCH
    a = p["beta"] / (p["alpha"] + p["beta"])
    N1, D1 = impl.compute_optimal(1e21, **p)
    N2, D2 = impl.compute_optimal(1e24, **p)
    assert N2 / N1 == pytest.approx(1000 ** a, rel=1e-9)
    assert D2 / D1 == pytest.approx(1000 ** (1 - a), rel=1e-9)


def test_compute_optimal_equal_exponents_gives_constant_ratio(impl):
    p = dict(E=1.7, A=500.0, B=250.0, alpha=0.3, beta=0.3)
    ratios = []
    for C in (1e20, 1e23, 1e26):
        N, D = impl.compute_optimal(C, **p)
        ratios.append(D / N)
    assert ratios[0] == pytest.approx(ratios[1], rel=1e-9) == pytest.approx(ratios[2], rel=1e-9)
    assert ratios[0] == pytest.approx((p["B"] / p["A"]) ** (1 / p["alpha"]), rel=1e-9)


def test_compute_optimal_ignores_E_and_validates(impl):
    q = dict(HOFF)
    q["E"] = 0.1
    assert impl.compute_optimal(1e22, **q) == pytest.approx(impl.compute_optimal(1e22, **HOFF), rel=1e-12)
    for bad in (dict(C=0.0), dict(C=-1.0), dict(A=0.0), dict(B=-2.0), dict(alpha=0.0), dict(beta=-0.1)):
        kw = dict(C=1e22, **HOFF)
        kw.update(bad)
        with pytest.raises(ValueError):
            impl.compute_optimal(**kw)


# ------------------------------------------------------------------ fit_power_law
def test_fit_power_law_exact(impl):
    x = np.logspace(6, 11, 12)
    p, k = impl.fit_power_law(x, 5.0 * x**-0.3)
    assert isinstance(p, float) and isinstance(k, float)
    assert p == pytest.approx(-0.3, abs=1e-10)
    assert k == pytest.approx(5.0, rel=1e-8)


def test_fit_power_law_recovers_exponents_from_noisy_data(impl):
    rng = np.random.default_rng(0)
    x = np.logspace(17, 22, 40)
    for true_p, true_k in [(0.5, 1.0e-1), (-0.05, 8.0), (0.27, 2.0e-2)]:
        y = true_k * x**true_p * np.exp(rng.normal(0, 0.02, x.size))   # 2% multiplicative noise
        p, k = impl.fit_power_law(list(x), list(y))                      # plain lists must work
        assert p == pytest.approx(true_p, abs=0.01)
        assert k == pytest.approx(true_k, rel=0.3)   # the coefficient is much less well determined than the exponent


def test_fit_power_law_on_published_chinchilla_frontier(impl):
    # (training FLOPs, compute-optimal parameters), Approach 1, Hoffmann et al. 2022 Table 3
    C = [1.92e19, 1.21e20, 1.23e22, 5.76e23, 3.85e24, 9.90e24, 3.43e25, 1.27e26, 1.30e28]
    N = [4e8, 1e9, 1e10, 67e9, 175e9, 280e9, 520e9, 1e12, 1e13]
    p, _ = impl.fit_power_law(C, N)
    assert p == pytest.approx(0.50, abs=0.02)


def test_fit_power_law_rescaling(impl):
    x = np.array([1e3, 1e4, 1e5, 1e6])
    y = np.array([9.0, 4.4, 2.3, 1.1])
    p1, k1 = impl.fit_power_law(x, y)
    p2, k2 = impl.fit_power_law(10 * x, y)
    assert p2 == pytest.approx(p1, rel=1e-9)
    assert k2 == pytest.approx(k1 * 10 ** (-p1), rel=1e-9)   # y = k1 x^p1 = (k1 10^-p1) (10x)^p1


def test_fit_power_law_validation(impl):
    with pytest.raises(ValueError):
        impl.fit_power_law([1.0], [1.0])
    with pytest.raises(ValueError):
        impl.fit_power_law([1.0, 2.0, 3.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        impl.fit_power_law([1.0, 2.0, 3.0], [1.0, 0.0, 3.0])
    with pytest.raises(ValueError):
        impl.fit_power_law([1.0, -2.0, 3.0], [1.0, 2.0, 3.0])


# ------------------------------------------------------------------ isoflop_optimum
def test_isoflop_optimum_exact_parabola(impl):
    n0 = 3.0e9
    sizes = np.array([2e8, 5e8, 1e9, 2e9, 4e9, 8e9, 2e10])
    losses = 0.3 * (np.log(sizes) - math.log(n0)) ** 2 + 2.05
    n_opt, l_min = impl.isoflop_optimum(sizes, losses)
    assert isinstance(n_opt, float) and isinstance(l_min, float)
    assert n_opt == pytest.approx(n0, rel=1e-9)
    assert l_min == pytest.approx(2.05, rel=1e-9)
    # order must not matter, lists are fine
    perm = [4, 0, 6, 2, 1, 5, 3]
    n2, l2 = impl.isoflop_optimum([sizes[i] for i in perm], [losses[i] for i in perm])
    assert n2 == pytest.approx(n0, rel=1e-9) and l2 == pytest.approx(2.05, rel=1e-9)


@pytest.mark.parametrize("p", [HOFF, EPOCH], ids=["hoffmann", "epoch"])
def test_isoflop_optimum_on_chinchilla_slice(impl, p):
    C = 1e21
    n_true, _ = closed_form_ref(C, p)
    sizes = n_true * np.logspace(-math.log10(4), math.log10(4), 7)        # +-4x around the optimum
    losses = loss_ref(sizes, C / (6 * sizes), p)
    n_opt, l_min = impl.isoflop_optimum(sizes, losses)
    l_true = loss_ref(n_true, C / (6 * n_true), p)
    assert n_opt == pytest.approx(n_true, rel=0.02)     # a parabola only approximates the asymmetric curve
    assert l_min == pytest.approx(l_true, rel=0.005)


def test_isoflop_optimum_validation(impl):
    with pytest.raises(ValueError):
        impl.isoflop_optimum([1e8, 1e9], [2.5, 2.4])                       # too few points
    with pytest.raises(ValueError):
        impl.isoflop_optimum([1e8, 1e9, 1e10], [2.5, 2.4])                 # length mismatch
    with pytest.raises(ValueError):                                         # concave data: no minimum
        impl.isoflop_optimum([1e8, 1e9, 1e10, 1e11], [2.0, 2.4, 2.5, 2.1])


# ------------------------------------------------------------------ effective_tokens
def test_effective_tokens_no_repetition(impl):
    assert impl.effective_tokens(100.0, 100.0) == pytest.approx(100.0)
    assert impl.effective_tokens(100.0, 60.0) == pytest.approx(60.0)          # only a subset of the corpus is used
    assert impl.effective_tokens(100.0, 400.0, r_star=math.inf) == pytest.approx(400.0)


def test_effective_tokens_known_values(impl):
    # U = 1, R* = 15.387756: four epochs (R = 3) are worth 93% of fresh data
    assert impl.effective_tokens(1.0, 4.0) == pytest.approx(3.726, abs=2e-3)
    assert impl.effective_tokens(10e12, 40e12) == pytest.approx(37.26e12, rel=1e-3)
    # a very long run saturates at U (1 + R*)
    assert impl.effective_tokens(1.0, 1e6) == pytest.approx(1 + 15.387756, rel=1e-6)
    # a different decay constant
    assert impl.effective_tokens(1.0, 4.0, r_star=3.0) == pytest.approx(1 + 3 * (1 - math.exp(-1.0)), rel=1e-9)


def test_effective_tokens_shape_and_scaling(impl):
    U = 5.0e11
    totals = U * np.array([1, 2, 3, 5, 9, 17, 33, 65], dtype=float)
    vals = np.array([impl.effective_tokens(U, t) for t in totals])
    assert np.all(np.diff(vals) > 0)                                          # more repeats never hurt in this law
    gains = np.diff(vals) / np.diff(totals)                                    # marginal value of one more token
    assert np.all(np.diff(gains) < 0)                                          # and each repeat is worth less
    assert np.all(vals <= totals + 1e-6) and np.all(vals < U * (1 + 15.387756))
    assert impl.effective_tokens(3 * U, 3 * totals[4]) == pytest.approx(3 * vals[4], rel=1e-12)


# ------------------------------------------------------------------ inference_aware_optimum
def _d_for(N, ell, p):
    r = ell - p["E"] - p["A"] / N ** p["alpha"]
    return (p["B"] / r) ** (1 / p["beta"])


def test_inference_aware_matches_compute_optimal_without_inference(impl):
    for p in (EPOCH, SARDANA):
        N0, D0 = closed_form_ref(1e23, p)
        ell = loss_ref(N0, D0, p)
        N, D = impl.inference_aware_optimum(ell, 0.0, **p)
        assert N == pytest.approx(N0, rel=1e-4)
        assert D == pytest.approx(D0, rel=1e-3)


def test_inference_aware_hits_target_and_is_locally_optimal(impl):
    p = EPOCH
    N0, D0 = closed_form_ref(1e24, p)
    ell = loss_ref(N0, D0, p)
    d_inf = 2e13
    N, D = impl.inference_aware_optimum(ell, d_inf, **p)
    assert loss_ref(N, D, p) == pytest.approx(ell, rel=1e-9)

    def cost(n):
        return 6 * n * _d_for(n, ell, p) + 2 * n * d_inf

    assert 6 * N * D + 2 * N * d_inf == pytest.approx(cost(N), rel=1e-9)
    for f in (0.9, 0.97, 1.03, 1.1):
        assert cost(N) <= cost(N * f) * (1 + 1e-9)


def test_inference_aware_shifts_towards_smaller_models_trained_longer(impl):
    p = EPOCH
    N0, D0 = closed_form_ref(1e24, p)
    ell = loss_ref(N0, D0, p)
    prev_N, prev_D = impl.inference_aware_optimum(ell, 0.0, **p)
    for d_inf in (1e11, 1e12, 1e13, 1e14):
        N, D = impl.inference_aware_optimum(ell, d_inf, **p)
        assert N < prev_N and D > prev_D
        # can never go below the size at which infinite data would be needed
        assert N > (p["A"] / (ell - p["E"])) ** (1 / p["alpha"])
        prev_N, prev_D = N, D


def test_inference_aware_reproduces_sardana_frankle_table(impl):
    # 70B Chinchilla-quality model, 10T tokens of lifetime inference: 41.6B parameters, 7.92T tokens
    p = SARDANA
    N70 = 70e9
    G = (p["alpha"] * p["A"] / (p["beta"] * p["B"])) ** (1 / (p["alpha"] + p["beta"]))
    a = p["beta"] / (p["alpha"] + p["beta"])
    C70 = 6 * (N70 / G) ** (1 / a)                    # compute at which 70B is the optimal size
    D70 = C70 / (6 * N70)
    ell = loss_ref(N70, D70, p)
    N, D = impl.inference_aware_optimum(ell, 1e13, **p)
    assert N == pytest.approx(41.6e9, rel=0.015)
    assert D == pytest.approx(7.92e12, rel=0.015)
    saving = 1 - (6 * N * D + 2 * N * 1e13) / (6 * N70 * D70 + 2 * N70 * 1e13)
    assert saving == pytest.approx(0.12, abs=0.01)


def test_inference_aware_validation(impl):
    with pytest.raises(ValueError):
        impl.inference_aware_optimum(1.5, 1e12, **EPOCH)        # target below the irreducible loss E
    with pytest.raises(ValueError):
        impl.inference_aware_optimum(EPOCH["E"], 1e12, **EPOCH)
