import math

import pytest
import torch

MODULE = "sampling"
NEG_INF = float("-inf")


def logits_of(probs, dtype=torch.float64):
    """Logits whose softmax is exactly `probs` (which must sum to 1)."""
    return torch.log(torch.tensor(probs, dtype=dtype))


def kept(x):
    """Indices that survived a filter (finite logits)."""
    return torch.nonzero(torch.isfinite(x), as_tuple=False).flatten().tolist()


def freqs(ids, vocab):
    return torch.bincount(ids.flatten(), minlength=vocab).double() / ids.numel()


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


# ----------------------------------------------------------------------------- temperature

def test_temperature_scales_logits(impl):
    z = torch.tensor([2.0, 4.0, -2.0])
    torch.testing.assert_close(impl.apply_temperature(z, 2.0), torch.tensor([1.0, 2.0, -1.0]))
    torch.testing.assert_close(impl.apply_temperature(z, 1.0), z)
    z64 = torch.randn(3, 5, dtype=torch.float64)
    out = impl.apply_temperature(z64, 0.5)
    assert out.dtype == torch.float64 and out.shape == z64.shape
    torch.testing.assert_close(out, z64 * 2)


@pytest.mark.parametrize("tau", [0.0, -1.0])
def test_temperature_must_be_positive(impl, tau):
    with pytest.raises(ValueError):
        impl.apply_temperature(torch.zeros(4), tau)


def test_temperature_limits(impl):
    z = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    cold = torch.softmax(impl.apply_temperature(z, 0.01), -1)
    torch.testing.assert_close(cold, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64), atol=1e-6, rtol=0)
    hot = torch.softmax(impl.apply_temperature(z, 1e6), -1)
    torch.testing.assert_close(hot, torch.full((3,), 1 / 3, dtype=torch.float64), atol=1e-5, rtol=0)

    def entropy(tau):
        p = torch.softmax(impl.apply_temperature(z, tau), -1)
        return -(p * p.log()).sum().item()

    assert entropy(0.5) < entropy(1.0) < entropy(2.0)


# ----------------------------------------------------------------------------- top-k

def test_top_k_keeps_k_largest(impl):
    z = torch.tensor([0.1, 3.0, 1.0, 2.0, -1.0])
    out = impl.top_k_filter(z, 2)
    assert kept(out) == [1, 3]
    assert out[1] == 3.0 and out[3] == 2.0                       # kept logits are untouched
    assert torch.all(out[[0, 2, 4]] == NEG_INF)


def test_top_k_ties_are_all_kept(impl):
    z = torch.tensor([1.0, 2.0, 2.0, 2.0, 0.0])
    assert kept(impl.top_k_filter(z, 2)) == [1, 2, 3]
    assert kept(impl.top_k_filter(z, 4)) == [0, 1, 2, 3]


def test_top_k_edge_cases(impl):
    z = torch.tensor([0.5, 1.5, -0.5, 1.0])
    assert kept(impl.top_k_filter(z, 1)) == [1]
    assert torch.equal(impl.top_k_filter(z, 4), z)
    assert torch.equal(impl.top_k_filter(z, 100), z)
    with pytest.raises(ValueError):
        impl.top_k_filter(z, 0)


def test_top_k_rows_are_independent(impl):
    z = torch.tensor([[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]])
    out = impl.top_k_filter(z, 2)
    assert kept(out[0]) == [2, 3] and kept(out[1]) == [0, 1]


# ----------------------------------------------------------------------------- top-p

P5 = [0.5, 0.3, 0.1, 0.06, 0.04]


@pytest.mark.parametrize("p, n_keep", [(0.3, 1), (0.55, 2), (0.75, 2), (0.85, 3), (0.95, 4), (0.99, 5)])
def test_top_p_hand_example(impl, p, n_keep):
    # cumulative mass: 0.5, 0.8, 0.9, 0.96, 1.0 -> smallest prefix with mass >= p
    out = impl.top_p_filter(logits_of(P5), p)
    assert kept(out) == list(range(n_keep))


def test_top_p_respects_vocabulary_order(impl):
    probs = [0.06, 0.5, 0.04, 0.1, 0.3]              # same distribution, shuffled
    out = impl.top_p_filter(logits_of(probs), 0.75)
    assert kept(out) == [1, 4]                       # the 0.5 and 0.3 tokens, at their original indices


def test_top_p_edge_cases(impl):
    z = logits_of(P5)
    assert torch.equal(impl.top_p_filter(z, 1.0), z)
    assert kept(impl.top_p_filter(z, 1e-6)) == [0]   # always keep the most likely token
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            impl.top_p_filter(z, bad)


def test_top_p_is_the_smallest_sufficient_set(impl):
    g = torch.Generator().manual_seed(1)
    z = torch.randn(8, 50, generator=g, dtype=torch.float64) * 2
    probs = torch.softmax(z, -1)
    for p in (0.3, 0.6, 0.9):
        out = impl.top_p_filter(z, p)
        for row in range(z.shape[0]):
            idx = kept(out[row])
            mass = probs[row, idx].sum().item()
            assert mass >= p
            assert mass - probs[row, idx].min().item() < p            # minimal: dropping any kept token breaks it
            worst_kept = probs[row, idx].min()
            dropped = torch.isinf(out[row])
            assert torch.all(probs[row][dropped] <= worst_kept)       # kept tokens are the most probable ones
            assert torch.equal(out[row][~dropped], z[row][~dropped])  # kept logits unchanged


def test_top_p_ignores_tokens_removed_earlier(impl):
    # already -inf entries have probability 0, so top-p works on the renormalized survivors
    z = logits_of([0.4, 0.3, 0.2, 0.1])
    assert kept(impl.top_p_filter(z, 0.5)) == [0, 1]                       # 0.4 < 0.5: need a second token
    after_top_k = impl.top_k_filter(z, 2)                                  # survivors renormalize to 4/7, 3/7
    assert kept(impl.top_p_filter(after_top_k, 0.5)) == [0]                # 4/7 >= 0.5: one token is enough


# ----------------------------------------------------------------------------- min-p

P5M = [0.6, 0.2, 0.1, 0.06, 0.04]


@pytest.mark.parametrize("p_min, n_keep", [(0.0, 5), (0.15, 3), (0.2, 2), (0.5, 1), (1.0, 1)])
def test_min_p_hand_example(impl, p_min, n_keep):
    # thresholds: 0, 0.09, 0.12, 0.30, 0.60
    out = impl.min_p_filter(logits_of(P5M), p_min)
    assert kept(out) == list(range(n_keep))


def test_min_p_threshold_follows_model_confidence(impl):
    peaked = logits_of([0.9, 0.05, 0.03, 0.02])
    flat = logits_of([0.3, 0.3, 0.2, 0.2])
    assert kept(impl.min_p_filter(peaked, 0.1)) == [0]              # threshold 0.09 removes the tail
    assert kept(impl.min_p_filter(flat, 0.1)) == [0, 1, 2, 3]       # threshold 0.03 removes nothing


def test_min_p_ties_shift_invariance_and_errors(impl):
    z = torch.tensor([2.0, 2.0, 0.0, -3.0], dtype=torch.float64)
    assert kept(impl.min_p_filter(z, 1.0)) == [0, 1]                # ties with the maximum are all kept
    base = impl.min_p_filter(z, 0.1)
    shifted = impl.min_p_filter(z + 7.5, 0.1)                       # softmax is shift-invariant
    assert kept(base) == kept(shifted)
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            impl.min_p_filter(z, bad)


def test_min_p_batched(impl):
    z = torch.stack([logits_of(P5M), logits_of([0.28, 0.28, 0.19, 0.19, 0.06])])
    out = impl.min_p_filter(z, 0.5)
    assert kept(out[0]) == [0]
    assert kept(out[1]) == [0, 1, 2, 3]                             # 0.2 >= 0.5 * 0.3


# ----------------------------------------------------------------------------- sample

def test_sample_temperature_zero_is_argmax(impl):
    z = torch.tensor([[0.1, 2.0, 1.0], [3.0, -1.0, 0.0]])
    out = impl.sample(z, gen(), temperature=0.0, top_k=2, top_p=0.5, min_p=0.1)
    assert out.tolist() == [1, 0]
    assert out.dtype == torch.long
    assert impl.sample(torch.tensor([1.0, 5.0, 2.0]), gen(), temperature=0.0).item() == 1


def test_sample_shapes_and_dtype(impl):
    v = 7
    for shape in [(v,), (3, v), (2, 3, v)]:
        z = torch.randn(shape)
        out = impl.sample(z, gen(), temperature=0.8, top_k=5, top_p=0.9, min_p=0.05)
        assert out.shape == shape[:-1]
        assert out.dtype == torch.long
        assert out.min() >= 0 and out.max() < v
    with pytest.raises(ValueError):
        impl.sample(torch.zeros(4), gen(), temperature=-1.0)


def test_sample_is_reproducible_and_uses_only_the_generator(impl):
    z = torch.randn(64, 20)
    torch.manual_seed(1)
    a = impl.sample(z, gen(42), temperature=1.0)
    torch.manual_seed(999)                                        # global RNG must not matter
    b = impl.sample(z, gen(42), temperature=1.0)
    c = impl.sample(z, gen(43), temperature=1.0)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_sample_does_not_modify_its_input(impl):
    z = torch.randn(4, 9)
    before = z.clone()
    impl.sample(z, gen(), temperature=0.7, top_k=4, top_p=0.9, min_p=0.1)
    assert torch.equal(z, before)


def test_sample_frequencies_match_softmax(impl):
    probs = [0.5, 0.3, 0.15, 0.05]
    z = logits_of(probs, torch.float32).repeat(40000, 1)
    f = freqs(impl.sample(z, gen(0), temperature=1.0), 4)
    torch.testing.assert_close(f, torch.tensor(probs, dtype=torch.float64), atol=0.015, rtol=0)


def test_sample_frequencies_with_temperature(impl):
    probs = [0.5, 0.3, 0.15, 0.05]
    tau = 0.5
    target = torch.tensor(probs, dtype=torch.float64) ** (1 / tau)
    target = target / target.sum()                                  # softmax(log p / tau) = p^(1/tau) / Z
    z = logits_of(probs, torch.float32).repeat(40000, 1)
    f = freqs(impl.sample(z, gen(1), temperature=tau), 4)
    torch.testing.assert_close(f, target, atol=0.015, rtol=0)


def test_sample_frequencies_with_top_k(impl):
    probs = [0.5, 0.3, 0.15, 0.05]
    z = logits_of(probs, torch.float32).repeat(40000, 1)
    f = freqs(impl.sample(z, gen(2), top_k=3), 4)
    target = torch.tensor([0.5, 0.3, 0.15, 0.0], dtype=torch.float64) / 0.95
    assert f[3] == 0                                                # filtered tokens are never drawn
    torch.testing.assert_close(f, target, atol=0.015, rtol=0)


def test_sample_frequencies_with_top_p_and_min_p(impl):
    probs = [0.5, 0.3, 0.1, 0.06, 0.04]
    z = logits_of(probs, torch.float32).repeat(40000, 1)
    f = freqs(impl.sample(z, gen(3), top_p=0.75), 5)               # keeps {0.5, 0.3}
    assert torch.all(f[2:] == 0)
    torch.testing.assert_close(f[:2], torch.tensor([0.5, 0.3], dtype=torch.float64) / 0.8, atol=0.015, rtol=0)
    f = freqs(impl.sample(z, gen(4), min_p=0.15), 5)               # threshold 0.075 keeps {0.5, 0.3, 0.1}
    assert torch.all(f[3:] == 0)
    torch.testing.assert_close(f[:3], torch.tensor([0.5, 0.3, 0.1], dtype=torch.float64) / 0.9, atol=0.015, rtol=0)


def test_sample_pipeline_order(impl):
    # (a) top-p sees the distribution renormalized over the top-k survivors: 4/7 >= 0.5 -> only token 0
    z = logits_of([0.4, 0.3, 0.2, 0.1], torch.float32).repeat(3000, 1)
    ids = impl.sample(z, gen(5), top_k=2, top_p=0.5)
    assert torch.all(ids == 0)
    # (b) temperature comes before top-p: at tau = 0.5 the distribution is [0.658, 0.237, 0.105],
    #     so top-p = 0.6 keeps only token 0 (on the untempered [0.5, 0.3, 0.2] it would keep two tokens)
    z = logits_of([0.5, 0.3, 0.2], torch.float32).repeat(3000, 1)
    ids = impl.sample(z, gen(6), temperature=0.5, top_p=0.6)
    assert torch.all(ids == 0)
    assert set(impl.sample(z, gen(7), temperature=1.0, top_p=0.6).unique().tolist()) == {0, 1}
