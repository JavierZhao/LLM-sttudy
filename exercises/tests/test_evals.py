import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

MODULE = "evals"


# ------------------------------------------------------------ independent references (not the drill)
def brute_pass_at_k(n, c, k):
    """Fraction of size-k subsets of n samples (c of them correct) that contain a correct one."""
    flags = [True] * c + [False] * (n - c)
    subsets = list(itertools.combinations(range(n), k))
    hits = sum(any(flags[i] for i in s) for s in subsets)
    return hits / len(subsets)


def brute_pass_hat_k(n, c, k):
    flags = [True] * c + [False] * (n - c)
    subsets = list(itertools.combinations(range(n), k))
    return sum(all(flags[i] for i in s) for s in subsets) / len(subsets)


def exact_pass_at_k(n, c, k):
    return 1 - Fraction(math.comb(n - c, k), math.comb(n, k))


# ------------------------------------------------------------------------------------ pass@k
def test_pass_at_k_matches_brute_force(impl):
    for n in range(1, 9):
        for c in range(n + 1):
            for k in range(1, n + 1):
                assert impl.pass_at_k(n, c, k) == pytest.approx(brute_pass_at_k(n, c, k), abs=1e-12), (n, c, k)


def test_pass_at_k_page_example(impl):
    # n = 20 samples, c = 3 correct: the worked numbers on the page
    assert impl.pass_at_k(20, 3, 1) == pytest.approx(0.15, abs=1e-12)
    assert impl.pass_at_k(20, 3, 5) == pytest.approx(1 - 6188 / 15504, abs=1e-12)
    assert impl.pass_at_k(20, 3, 10) == pytest.approx(1 - 19448 / 184756, abs=1e-12)


def test_pass_at_k_edges(impl):
    assert impl.pass_at_k(10, 0, 5) == 0.0
    assert impl.pass_at_k(10, 10, 5) == 1.0
    assert impl.pass_at_k(10, 6, 5) == 1.0          # n - c = 4 < k = 5: every subset has a correct one
    assert impl.pass_at_k(7, 2, 7) == 1.0           # k = n: any correct sample is enough
    assert impl.pass_at_k(7, 0, 7) == 0.0
    assert impl.pass_at_k(50, 13, 1) == pytest.approx(13 / 50, abs=1e-12)   # pass@1 = c / n


def test_pass_at_k_is_stable_for_huge_n(impl):
    # C(10**6, k) overflows float64 for large k; a careless implementation returns nan or inf
    for n, c, k in [(10**6, 3, 10), (10**6, 500_000, 100), (10**5, 50, 5000), (10**6, 1, 1)]:
        val = impl.pass_at_k(n, c, k)
        assert math.isfinite(val) and 0.0 <= val <= 1.0
    assert impl.pass_at_k(10**6, 3, 10) == pytest.approx(1 - (1 - 10 / 10**6) ** 3, rel=1e-3)
    # exact rational check at a size where math.comb is still cheap
    n, c, k = 2000, 37, 60
    assert impl.pass_at_k(n, c, k) == pytest.approx(float(exact_pass_at_k(n, c, k)), rel=1e-9)


def test_pass_at_k_monotone_in_k_and_c(impl):
    n = 40
    for c in (1, 5, 17):
        vals = [impl.pass_at_k(n, c, k) for k in range(1, n + 1)]
        assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))
    for k in (1, 4, 16):
        vals = [impl.pass_at_k(n, c, k) for c in range(n + 1)]
        assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))


def test_pass_at_k_validation(impl):
    for args in [(5, 2, 6), (5, 6, 1), (5, -1, 1), (5, 2, 0)]:
        with pytest.raises(ValueError):
            impl.pass_at_k(*args)


def test_pass_at_k_is_unbiased_and_plugin_is_not(impl):
    # Take the exact expectation over c ~ Binomial(n, p). The estimator averages to 1 - (1 - p)^k;
    # the plug-in 1 - (1 - c/n)^k is biased low (Jensen: (1 - x)^k is convex in x).
    for n, p, k in [(10, 0.3, 5), (20, 0.15, 10), (8, 0.5, 3)]:
        pmf = [math.comb(n, c) * p**c * (1 - p) ** (n - c) for c in range(n + 1)]
        truth = 1 - (1 - p) ** k
        est = sum(w * impl.pass_at_k(n, c, k) for c, w in enumerate(pmf))
        plug = sum(w * (1 - (1 - c / n) ** k) for c, w in enumerate(pmf))
        assert est == pytest.approx(truth, abs=1e-9)
        assert plug < truth - 0.01


def test_mean_pass_at_k_is_not_a_function_of_mean_pass_at_1(impl):
    # half the problems easy (c=9 of 10), half hard (c=1 of 10): mean pass@1 = 0.5
    counts = [9] * 50 + [1] * 50
    got = impl.mean_pass_at_k(counts, 10, 5)
    want = np.mean([brute_pass_at_k(10, c, 5) for c in (9, 1)])
    assert got == pytest.approx(want, abs=1e-12)
    assert got == pytest.approx(0.5 * 1.0 + 0.5 * (1 - math.comb(9, 5) / math.comb(10, 5)), abs=1e-12)
    assert abs(got - (1 - 0.5**5)) > 0.2


# ------------------------------------------------------------------------------------ pass^k
def test_pass_hat_k(impl):
    for n in range(1, 8):
        for c in range(n + 1):
            for k in range(1, n + 1):
                assert impl.pass_hat_k(n, c, k) == pytest.approx(brute_pass_hat_k(n, c, k), abs=1e-12)
    assert impl.pass_hat_k(20, 3, 1) == pytest.approx(0.15)
    assert impl.pass_hat_k(20, 3, 2) == pytest.approx(3 / 190)
    assert impl.pass_hat_k(20, 3, 3) == pytest.approx(1 / 1140)
    assert impl.pass_hat_k(20, 2, 3) == 0.0
    val = impl.pass_hat_k(10**6, 999_000, 500)
    assert math.isfinite(val) and 0.0 < val < 1.0


# ------------------------------------------------------------------------------------ accuracy CI
def test_accuracy_ci_wald_matches_hand_computation(impl):
    p, lo, hi = impl.accuracy_ci(350, 500)
    se = math.sqrt(0.7 * 0.3 / 500)
    assert p == pytest.approx(0.7)
    assert (hi - lo) / 2 == pytest.approx(1.96 * se, rel=1e-9)
    assert (lo + hi) / 2 == pytest.approx(0.7)
    assert 100 * se == pytest.approx(2.049, abs=1e-3)               # the 2.05-point standard error on the page
    p2, lo2, hi2 = impl.accuracy_ci(9800, 14000)
    assert 100 * (hi2 - lo2) / 2 == pytest.approx(0.759, abs=1e-3)  # +/- 0.76 points
    _, lo3, hi3 = impl.accuracy_ci(350, 500, z=2.576)
    assert (hi3 - lo3) / 2 == pytest.approx(2.576 * se, rel=1e-9)


def test_accuracy_ci_wald_degenerates_at_the_edges_but_wilson_does_not(impl):
    _, lo, hi = impl.accuracy_ci(0, 30, method="wald")
    assert (lo, hi) == (0.0, 0.0)                                   # zero-width interval: the Wald failure
    _, lo, hi = impl.accuracy_ci(0, 30, method="wilson")
    assert lo == 0.0 and 0.05 < hi < 0.2
    _, lo, hi = impl.accuracy_ci(30, 30, method="wilson")
    assert hi == 1.0 and 0.8 < lo < 0.95
    p, lo, hi = impl.accuracy_ci(3, 30, method="wilson")
    assert 0.0 <= lo < p < hi <= 1.0


def test_accuracy_ci_wilson_formula(impl):
    p, lo, hi = impl.accuracy_ci(45, 50, method="wilson")
    z, n = 1.96, 50
    centre = (0.9 + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(0.9 * 0.1 / n + z * z / (4 * n * n)) / (1 + z * z / n)
    assert lo == pytest.approx(centre - half) and hi == pytest.approx(centre + half)


def test_accuracy_ci_coverage_on_simulated_data(impl):
    rng = np.random.default_rng(1)
    n, trials = 60, 4000
    for p_true in (0.5, 0.9):
        wald_hit = wil_hit = 0
        for c in rng.binomial(n, p_true, size=trials):
            _, lo, hi = impl.accuracy_ci(int(c), n, method="wald")
            wald_hit += lo <= p_true <= hi
            _, lo, hi = impl.accuracy_ci(int(c), n, method="wilson")
            wil_hit += lo <= p_true <= hi
        assert 0.925 <= wil_hit / trials <= 0.975
        if p_true == 0.5:
            assert 0.92 <= wald_hit / trials <= 0.975
        else:
            assert wald_hit / trials < wil_hit / trials + 0.01      # Wald is no better, and worse near the edge


def test_accuracy_ci_validation(impl):
    with pytest.raises(ValueError):
        impl.accuracy_ci(5, 4)
    with pytest.raises(ValueError):
        impl.accuracy_ci(1, 10, method="bayes")


# ------------------------------------------------------------------------------------ paired bootstrap
def test_paired_bootstrap_degenerate_cases(impl):
    a = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    d, lo, hi = impl.paired_bootstrap(a, a.copy(), n_boot=200, seed=0)
    assert (d, lo, hi) == (0.0, 0.0, 0.0)
    d, lo, hi = impl.paired_bootstrap(np.ones(30, dtype=int), np.zeros(30, dtype=int), n_boot=200, seed=0)
    assert (d, lo, hi) == (1.0, 1.0, 1.0)


def test_paired_bootstrap_reproducible_and_seed_sensitive(impl):
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, 300)
    b = rng.integers(0, 2, 300)
    r1 = impl.paired_bootstrap(a, b, n_boot=500, seed=7)
    r2 = impl.paired_bootstrap(a, b, n_boot=500, seed=7)
    r3 = impl.paired_bootstrap(a, b, n_boot=500, seed=8)
    assert r1 == r2
    assert r1[0] == r3[0] and r1[1:] != r3[1:]      # observed difference does not depend on the seed


def test_paired_bootstrap_matches_analytic_paired_interval(impl):
    # 500 questions; A is right / B wrong on 40, B right / A wrong on 25 (the page example)
    a = np.array([1] * 40 + [0] * 25 + [1] * 300 + [0] * 135)
    b = np.array([0] * 40 + [1] * 25 + [1] * 300 + [0] * 135)
    d, lo, hi = impl.paired_bootstrap(a, b, n_boot=4000, seed=0)
    n = len(a)
    diff = a - b
    se = diff.std(ddof=1) / math.sqrt(n)
    assert d == pytest.approx(0.03)
    assert se == pytest.approx(0.0161, abs=1e-3)
    assert (hi - lo) / 2 == pytest.approx(1.96 * se, rel=0.12)
    assert lo < 0.0 < hi                            # not significant at 5%, as computed on the page


def test_paired_bootstrap_uses_pairing(impl):
    # Highly correlated models: a paired interval is much narrower than an unpaired one would be
    rng = np.random.default_rng(3)
    n = 400
    base = rng.random(n)
    a = (base < 0.7).astype(int)
    b = (base < 0.68).astype(int)                    # differs from a on ~2% of questions
    d, lo, hi = impl.paired_bootstrap(a, b, n_boot=2000, seed=0)
    unpaired_half = 1.96 * math.sqrt(0.7 * 0.3 / n + 0.68 * 0.32 / n)
    assert (hi - lo) / 2 < 0.4 * unpaired_half


def test_paired_bootstrap_coverage(impl):
    # simulate paired data with a known accuracy gap; the 90% interval should cover it about 90% of the time
    rng = np.random.default_rng(5)
    n, gap, sims = 150, 0.04, 200
    hit = 0
    for s in range(sims):
        u = rng.random(n)
        a = (u < 0.65 + gap).astype(int)
        b = (u < 0.65).astype(int)
        flip = rng.random(n) < 0.15                  # independent noise so the models also disagree
        b = np.where(flip, rng.integers(0, 2, n), b)
        a = np.where(flip, rng.integers(0, 2, n), a)
        true_gap = (0.65 + gap) * 0.85 + 0.5 * 0.15 - (0.65 * 0.85 + 0.5 * 0.15)
        d, lo, hi = impl.paired_bootstrap(a, b, n_boot=400, seed=s, alpha=0.10)
        hit += lo <= true_gap <= hi
    assert 0.80 <= hit / sims <= 0.97


def test_paired_bootstrap_validation(impl):
    with pytest.raises(ValueError):
        impl.paired_bootstrap([1, 0, 1], [1, 0], n_boot=10)
    with pytest.raises(ValueError):
        impl.paired_bootstrap([], [], n_boot=10)


# ------------------------------------------------------------------------------------ cloze scoring
# Illustrative numbers from the page: "The capital of Australia is ..."
LP = [-3.0, -4.4, -3.4]            # Sydney, Canberra, Melbourne: total log P(choice | question)
TOK = [2, 4, 3]
CHR = [6, 8, 9]
UNC = [-4.0, -9.0, -5.0]           # log P(choice | generic prompt)


def test_mc_loglik_choice_normalizations_disagree(impl):
    assert impl.mc_loglik_choice(LP, TOK, normalize=False) == 0        # raw total: favors the short "Sydney"
    assert impl.mc_loglik_choice(LP, TOK, normalize=True) == 1         # per-token: Canberra
    assert impl.mc_loglik_choice(LP, CHR, normalize=True) == 2         # per-character: Melbourne
    assert impl.mc_loglik_choice(LP, TOK, unconditional_logprobs=UNC) == 1   # PMI: Canberra


def test_mc_loglik_choice_pmi_ignores_lengths_and_normalize(impl):
    a = impl.mc_loglik_choice(LP, [1, 1, 1], normalize=True, unconditional_logprobs=UNC)
    b = impl.mc_loglik_choice(LP, [100, 1, 1], normalize=False, unconditional_logprobs=UNC)
    assert a == b == 1


def test_mc_loglik_choice_ties_and_basics(impl):
    assert impl.mc_loglik_choice([-1.0, -1.0, -2.0], [1, 1, 1]) == 0    # ties to the lowest index
    assert impl.mc_loglik_choice([-2.0], [5], normalize=True) == 0
    # a longer answer with a better per-token score wins only when normalized
    assert impl.mc_loglik_choice([-4.0, -5.0], [1, 5], normalize=False) == 0
    assert impl.mc_loglik_choice([-4.0, -5.0], [1, 5], normalize=True) == 1


def test_mc_loglik_choice_validation(impl):
    with pytest.raises(ValueError):
        impl.mc_loglik_choice([-1.0, -2.0], [1])
    with pytest.raises(ValueError):
        impl.mc_loglik_choice([], [])
    with pytest.raises(ValueError):
        impl.mc_loglik_choice([-1.0, -2.0], [0, 3], normalize=True)
    with pytest.raises(ValueError):
        impl.mc_loglik_choice([-1.0, -2.0], [1, 1], unconditional_logprobs=[-1.0])
