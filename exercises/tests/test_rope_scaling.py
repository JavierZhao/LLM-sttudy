import math

import pytest
import torch

MODULE = "rope_scaling"


# ------------------------------------------------------------------ naive references (self-contained)
def _theta(d, base):
    return [base ** (-2 * i / d) for i in range(d // 2)]


def _ref_yarn_wavelength(d, base, s, L, alpha, beta):
    out = []
    for th in _theta(d, base):
        r = L / (2 * math.pi / th)                       # turns inside the original context
        g = min(max((r - alpha) / (beta - alpha), 0.0), 1.0)
        out.append((1 - g) * th / s + g * th)
    return out


def _ref_yarn_index(d, base, s, L, alpha, beta):
    def find(rot):
        return d * math.log(L / (rot * 2 * math.pi)) / (2 * math.log(base))
    low, high = max(math.floor(find(beta)), 0), min(math.ceil(find(alpha)), d - 1)
    if low == high:
        high += 0.001
    out = []
    for i, th in enumerate(_theta(d, base)):
        ramp = min(max((i - low) / (high - low), 0.0), 1.0)
        m = 1 - ramp                                     # weight of the original frequency
        out.append(th / s * (1 - m) + th * m)
    return out


def _ref_llama3(d, base, factor, lf, hf, L):
    """Hugging Face _compute_llama3_parameters, written with scalar loops."""
    out = []
    for th in _theta(d, base):
        wl = 2 * math.pi / th
        if wl < L / hf:
            out.append(th)
        elif wl > L / lf:
            out.append(th / factor)
        else:
            smooth = (L / wl - lf) / (hf - lf)
            out.append((1 - smooth) * th / factor + smooth * th)
    return out


def _t(xs):
    return torch.tensor(xs, dtype=torch.float64)


# ------------------------------------------------------------------ vanilla frequencies
def test_inv_freq_and_wavelengths(impl):
    th = impl.inv_freq(128, 10000.0)
    assert th.shape == (64,) and th.dtype == torch.float64
    assert th[0].item() == pytest.approx(1.0)
    assert th[-1].item() == pytest.approx(10000.0 ** (-126 / 128))
    ratios = th[1:] / th[:-1]                                                # geometric ladder
    torch.testing.assert_close(ratios, torch.full_like(ratios, 10000.0 ** (-2 / 128)))
    lam = impl.wavelengths(th)
    assert lam[0].item() == pytest.approx(2 * math.pi)
    assert lam[-1].item() == pytest.approx(54410.1, rel=1e-4)                # page 05: slowest pair at b = 1e4
    slow = impl.wavelengths(impl.inv_freq(128, 500000.0))[-1].item()
    assert slow == pytest.approx(2559196.0, rel=1e-4)                        # ... and at b = 5e5


# ------------------------------------------------------------------ every method reduces to vanilla at s = 1
@pytest.mark.parametrize("d,base,L", [(128, 10000.0, 4096), (64, 500000.0, 8192)])
def test_all_methods_reduce_to_vanilla_at_s1(impl, d, base, L):
    v = impl.inv_freq(d, base)
    torch.testing.assert_close(impl.pi_scale(v, 1.0), v)
    assert impl.ntk_base(base, 1.0, d) == pytest.approx(base)
    torch.testing.assert_close(impl.inv_freq(d, impl.ntk_base(base, 1.0, d)), v)
    for ramp in ("wavelength", "index"):
        torch.testing.assert_close(impl.yarn_inv_freq(d, base, 1.0, L, ramp=ramp), v)
    torch.testing.assert_close(impl.llama3_inv_freq(d, base, 1.0, 1.0, 4.0, L), v)
    assert impl.yarn_mscale(1.0) == 1.0
    assert impl.yarn_mscale(0.5) == 1.0


# ------------------------------------------------------------------ Position Interpolation
def test_pi_divides_frequencies_by_s(impl):
    v = impl.inv_freq(64, 10000.0)
    keep = v.clone()
    out = impl.pi_scale(v, 8.0)
    torch.testing.assert_close(out, keep / 8.0)
    torch.testing.assert_close(v, keep)                                      # input not modified
    m = 1234                                                                 # positions m/s with theta == position m with theta/s
    torch.testing.assert_close((m / 8.0) * v, m * out)


# ------------------------------------------------------------------ NTK-aware
def test_ntk_base_value_and_endpoints(impl):
    d, base, s = 128, 10000.0, 8.0
    nb = impl.ntk_base(base, s, d)
    assert isinstance(nb, float)
    assert nb == pytest.approx(base * 8 ** (128 / 126))
    assert nb == pytest.approx(82684.62, rel=1e-6)
    v, w = impl.inv_freq(d, base), impl.inv_freq(d, nb)
    assert w[0].item() == pytest.approx(v[0].item())                         # highest frequency unchanged
    assert w[-1].item() == pytest.approx(v[-1].item() / s)                   # lowest slowed by exactly s
    factor = v / w                                                           # per-pair slow-down: s^(i / (n - 1))
    expect = s ** (torch.arange(64, dtype=torch.float64) / 63)
    torch.testing.assert_close(factor, expect)


def test_ntk_base_tiny_head(impl):
    # d = 4 has two pairs; b' = b s^2 gives theta = [1, 1 / (s sqrt(b))]
    nb = impl.ntk_base(100.0, 3.0, 4)
    assert nb == pytest.approx(900.0)
    w = impl.inv_freq(4, nb)
    assert w[1].item() == pytest.approx(1 / (3.0 * math.sqrt(100.0)))


# ------------------------------------------------------------------ YaRN
@pytest.mark.parametrize("d,base,s,L", [(128, 10000.0, 8.0, 4096), (128, 500000.0, 8.0, 8192), (64, 10000.0, 40.0, 4096)])
def test_yarn_wavelength_ramp_matches_reference(impl, d, base, s, L):
    got = impl.yarn_inv_freq(d, base, s, L, alpha=1.0, beta=32.0, ramp="wavelength")
    torch.testing.assert_close(got, _t(_ref_yarn_wavelength(d, base, s, L, 1.0, 32.0)))
    got2 = impl.yarn_inv_freq(d, base, s, L, alpha=1.0, beta=4.0)            # default ramp is "wavelength"
    torch.testing.assert_close(got2, _t(_ref_yarn_wavelength(d, base, s, L, 1.0, 4.0)))


@pytest.mark.parametrize("d,base,s,L", [(128, 10000.0, 8.0, 4096), (128, 500000.0, 8.0, 8192), (64, 10000.0, 40.0, 4096)])
def test_yarn_index_ramp_matches_reference(impl, d, base, s, L):
    got = impl.yarn_inv_freq(d, base, s, L, alpha=1.0, beta=32.0, ramp="index")
    torch.testing.assert_close(got, _t(_ref_yarn_index(d, base, s, L, 1.0, 32.0)))


def test_yarn_regimes_wavelength_ramp(impl):
    d, base, s, L = 128, 10000.0, 8.0, 4096
    v = impl.inv_freq(d, base)
    lam = 2 * math.pi / v
    out = impl.yarn_inv_freq(d, base, s, L, 1.0, 32.0)
    factor = v / out                                                         # how much each pair is slowed
    assert torch.all(factor[lam <= L / 32] == 1.0)                           # >= 32 turns in context: untouched
    torch.testing.assert_close(factor[lam >= L], torch.full_like(factor[lam >= L], s))   # <= 1 turn: divided by s
    assert int((lam <= L / 32).sum()) == 21 and int((lam >= L).sum()) == 18  # pairs 0..20 and 46..63
    mid = (lam > L / 32) & (lam < L)
    assert torch.all(factor[mid] > 1.0) and torch.all(factor[mid] < s)       # blended pairs sit strictly between
    assert torch.all(factor[1:] >= factor[:-1] - 1e-12)                      # slow-down never decreases with i


def test_yarn_regimes_index_ramp_deepseek_v3(impl):
    # DeepSeek-V3 rotates 64 dims (32 pairs), base 1e4, s = 40, 4096 original positions, alpha = 1, beta = 32:
    # floor(find(32)) = 10 and ceil(find(1)) = 23.
    d, base, s, L = 64, 10000.0, 40.0, 4096
    v = impl.inv_freq(d, base)
    out = impl.yarn_inv_freq(d, base, s, L, 1.0, 32.0, ramp="index")
    factor = v / out
    assert torch.all(factor[:11] == 1.0)                                     # pairs 0..10 untouched
    torch.testing.assert_close(factor[23:], torch.full_like(factor[23:], s))  # pairs 23..31 divided by 40
    assert torch.all(factor[11:23] > 1.0) and torch.all(factor[11:23] < s)
    w11 = 1 - (11 - 10) / (23 - 10)                                          # weight of the original frequency at pair 11
    assert factor[11].item() == pytest.approx(1 / (w11 + (1 - w11) / s))


def test_yarn_ramps_agree_at_the_ends_but_not_inside(impl):
    d, base, s, L = 128, 10000.0, 8.0, 4096
    a = impl.yarn_inv_freq(d, base, s, L, ramp="wavelength")
    b = impl.yarn_inv_freq(d, base, s, L, ramp="index")
    v = impl.inv_freq(d, base)
    assert torch.equal(a[:20], v[:20]) and torch.equal(b[:20], v[:20])       # fastest pairs untouched by both
    torch.testing.assert_close(a[46:], v[46:] / s)                           # slowest pairs divided by both
    torch.testing.assert_close(b[46:], v[46:] / s)
    assert (v[40] / a[40]).item() == pytest.approx(6.4532, rel=1e-3)         # pair 40 mid-ramp: 6.45 (paper) vs 3.06 (code)
    assert (v[40] / b[40]).item() == pytest.approx(3.0588, rel=1e-3)


def test_yarn_mscale(impl):
    assert impl.yarn_mscale(4.0) == pytest.approx(0.1 * math.log(4) + 1)
    assert impl.yarn_mscale(8.0) == pytest.approx(1.2079, abs=1e-4)
    assert impl.yarn_mscale(40.0) == pytest.approx(1.3689, abs=1e-4)
    assert impl.yarn_mscale(40.0) ** 2 == pytest.approx(1.8739, abs=1e-4)    # the logit multiplier 1/t for s = 40
    assert impl.yarn_mscale(4.0) < impl.yarn_mscale(8.0) < impl.yarn_mscale(40.0)


# ------------------------------------------------------------------ Llama 3.1
def test_llama3_rule_matches_hf_reference_and_counts(impl):
    d, base, f, lf, hf, L = 128, 500000.0, 8.0, 1.0, 4.0, 8192                # Llama-3.1-8B config
    got = impl.llama3_inv_freq(d, base, f, lf, hf, L)
    torch.testing.assert_close(got, _t(_ref_llama3(d, base, f, lf, hf, L)))
    v = impl.inv_freq(d, base)
    factor = v / got
    assert int((factor == 1.0).sum()) == 29                                   # 29 fastest pairs untouched
    assert int(((factor > 1.0) & (factor < f - 1e-9)).sum()) == 6             # 6 blended pairs (indices 29..34)
    torch.testing.assert_close(factor[35:], torch.full_like(factor[35:], f))  # 29 slowest pairs divided by 8
    assert impl.wavelengths(got)[-1].item() == pytest.approx(8 * 2559196.0, rel=1e-4)
    torch.testing.assert_close(got, impl.yarn_inv_freq(d, base, f, L, alpha=lf, beta=hf, ramp="wavelength"))
    # Llama-3.2-1B: head dim 64, factor 32
    got2 = impl.llama3_inv_freq(64, 500000.0, 32.0, 1.0, 4.0, 8192)
    torch.testing.assert_close(got2, _t(_ref_llama3(64, 500000.0, 32.0, 1.0, 4.0, 8192)))


# ------------------------------------------------------------------ effective context length
RULER_LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072]


@pytest.mark.parametrize("scores,expected", [
    ([96.6, 96.3, 95.2, 93.2, 87.0, 81.2], 65536),      # GPT-4
    ([95.5, 93.8, 91.6, 87.4, 84.7, 77.0], 32768),      # Llama 3.1 8B
    ([93.6, 91.2, 87.2, 75.4, 49.0, 13.8], 16384),      # Mistral-v0.2 7B
    ([94.7, 92.8, 92.1, 89.9, 86.7, 83.1], 65536),      # GLM4 9B
    ([82.3, 78.4, 73.7, 69.1, 68.1, 65.0], 0),          # LWM 7B: fails even at 4K
])
def test_effective_length_ruler_table(impl, scores, expected):
    assert impl.effective_length(RULER_LENGTHS, scores, 85.6) == expected
    assert impl.effective_length(RULER_LENGTHS, scores, 85.6, strict=True) == expected


def test_effective_length_non_monotone_curve(impl):
    lengths, scores = [1000, 2000, 3000, 4000], [90.0, 60.0, 88.0, 40.0]
    assert impl.effective_length(lengths, scores, 85.0) == 3000              # RULER: largest passing length
    assert impl.effective_length(lengths, scores, 85.0, strict=True) == 1000  # every shorter length must pass
    assert impl.effective_length(lengths, scores, 95.0) == 0
    assert impl.effective_length(lengths, scores, 88.0) == 3000              # threshold is inclusive


# ------------------------------------------------------------------ needle-in-a-haystack grid
HAY = [f"w{i}" for i in range(50)]
NEEDLE = "The magic number is 42736 ."
QUESTION = "What is the magic number ?"


def test_niah_grid_size_order_and_lengths(impl):
    depths, lengths = [0.0, 0.25, 0.5, 0.75, 1.0], [64, 128, 200]
    cases = impl.make_niah_prompts(HAY, NEEDLE, depths, lengths, QUESTION, "42736")
    assert len(cases) == 15
    assert [(c.length, c.depth) for c in cases] == [(L, d) for L in lengths for d in depths]
    for c in cases:
        assert len(c.prompt.split()) == c.length                              # exact length in tokens
        assert c.answer == "42736"
        assert c.prompt.count("42736") == 1                                   # one needle
        assert c.prompt.endswith(QUESTION)


def test_niah_needle_position(impl):
    n_needle, n_q = len(NEEDLE.split()), len(QUESTION.split())
    for length in (64, 150):
        n_hay = length - n_needle - n_q
        cases = impl.make_niah_prompts(HAY, NEEDLE, [0.0, 0.25, 0.5, 0.75, 1.0, 0.3], [length], QUESTION, "42736")
        for c in cases:
            toks = c.prompt.split()
            start = toks.index("The")                                         # first needle token (haystack words are 'w<i>')
            assert toks[start:start + n_needle] == NEEDLE.split()
            assert start == math.floor(c.depth * n_hay)                       # haystack tokens before the needle
    c0, c1 = impl.make_niah_prompts(HAY, NEEDLE, [0.0, 1.0], [64], QUESTION)
    assert c0.prompt.startswith(NEEDLE) and c0.answer == ""                   # depth 0: needle first
    assert c1.prompt.endswith(NEEDLE + " " + QUESTION)                        # depth 1: right before the question


def test_niah_haystack_cycles_and_is_deterministic(impl):
    small = ["a", "b", "c"]
    a = impl.make_niah_prompts(small, "N1 N2", [0.0], [12], "Q ?", "x")[0]
    assert a.prompt.split() == ["N1", "N2"] + ["a", "b", "c", "a", "b", "c", "a", "b"] + ["Q", "?"]
    assert impl.make_niah_prompts(small, "N1 N2", [0.5], [12], "Q ?", "x") == impl.make_niah_prompts(small, "N1 N2", [0.5], [12], "Q ?", "x")
    no_q = impl.make_niah_prompts(small, "N", [1.0], [5])[0]                  # empty question: the needle ends the prompt
    assert no_q.prompt.split() == ["a", "b", "c", "a", "N"]


def test_niah_errors(impl):
    with pytest.raises(ValueError):
        impl.make_niah_prompts(HAY, NEEDLE, [1.5], [64], QUESTION)
    with pytest.raises(ValueError):
        impl.make_niah_prompts(HAY, NEEDLE, [-0.1], [64], QUESTION)
    with pytest.raises(ValueError):
        impl.make_niah_prompts(HAY, NEEDLE, [0.5], [len(NEEDLE.split()) + len(QUESTION.split())], QUESTION)   # no room for haystack
    with pytest.raises(ValueError):
        impl.make_niah_prompts([], NEEDLE, [0.5], [64], QUESTION)
    with pytest.raises(ValueError):
        impl.make_niah_prompts(["two words"], NEEDLE, [0.5], [64], QUESTION)
