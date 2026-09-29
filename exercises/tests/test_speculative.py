import math

import pytest
import torch

MODULE = "speculative"


# ---------- closed forms ----------

def test_acceptance_rate_is_one_minus_tv(impl):
    p = torch.tensor([0.5, 0.3, 0.2, 0.0])
    q = torch.tensor([0.25, 0.25, 0.25, 0.25])
    # sum of elementwise minima: 0.25 + 0.25 + 0.2 + 0 = 0.7 = 1 - 0.5 * ||p - q||_1
    assert math.isclose(impl.acceptance_rate(p, q), 0.7, rel_tol=1e-6)
    assert math.isclose(impl.acceptance_rate(p, p), 1.0, rel_tol=1e-6)
    assert math.isclose(impl.acceptance_rate(q, p), impl.acceptance_rate(p, q), rel_tol=1e-6)  # symmetric
    disjoint = torch.tensor([0.0, 0.0, 0.5, 0.5])
    assert impl.acceptance_rate(torch.tensor([0.5, 0.5, 0.0, 0.0]), disjoint) == 0.0


def test_expected_tokens_closed_form(impl):
    for alpha in (0.0, 0.3, 0.6, 0.8, 0.9, 0.99):
        for gamma in (0, 1, 2, 5, 10):
            want = sum(alpha ** k for k in range(gamma + 1))
            assert math.isclose(impl.expected_tokens_per_step(alpha, gamma), want, rel_tol=1e-9)
    assert math.isclose(impl.expected_tokens_per_step(0.8, 4), 3.36160, rel_tol=1e-9)  # (1 - 0.8**5) / 0.2


def test_expected_tokens_edges(impl):
    assert impl.expected_tokens_per_step(1.0, 4) == 5.0       # no division by zero
    assert impl.expected_tokens_per_step(0.0, 4) == 1.0       # always exactly the correction token
    assert impl.expected_tokens_per_step(0.7, 0) == 1.0       # no draft: plain decoding
    with pytest.raises(ValueError):
        impl.expected_tokens_per_step(1.2, 3)
    with pytest.raises(ValueError):
        impl.expected_tokens_per_step(-0.1, 3)
    with pytest.raises(ValueError):
        impl.expected_tokens_per_step(0.5, -1)


def test_expected_tokens_monotone(impl):
    vals = [impl.expected_tokens_per_step(0.7, g) for g in range(8)]
    assert all(b > a for a, b in zip(vals, vals[1:]))
    assert all(v < 1 / (1 - 0.7) for v in vals)               # bounded by 1 / (1 - alpha)
    assert impl.expected_tokens_per_step(0.9, 3) > impl.expected_tokens_per_step(0.6, 3)


def test_speedup_formula(impl):
    # Leviathan et al. Table 1 (c = 0): alpha = 0.8, gamma = 5 gives 3.69x
    assert round(impl.speedup(0.8, 5, 0.0), 2) == 3.69
    assert math.isclose(impl.speedup(0.8, 4, 0.05), 3.3616 / 1.2, rel_tol=1e-6)
    # a drafter that costs as much as the target and accepts 60% of the time is slower than plain decoding
    assert impl.speedup(0.6, 4, 1.0) < 1.0
    # gamma = 0 is plain decoding
    assert impl.speedup(0.9, 0, 0.3) == 1.0


# ---------- the accept/resample rule ----------

def _gen(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _rows(*rows):
    return torch.tensor(rows, dtype=torch.float32)


def test_output_shape_and_prefix(impl):
    V, gamma = 6, 4
    g = _gen(0)
    for _ in range(200):
        p = torch.softmax(torch.randn(gamma + 1, V, generator=g), -1)
        q = torch.softmax(torch.randn(gamma, V, generator=g), -1)
        toks = torch.stack([torch.multinomial(q[i], 1, generator=g)[0] for i in range(gamma)])
        out = impl.speculative_accept(p, q, toks, g)
        assert 1 <= len(out) <= gamma + 1
        assert all(isinstance(t, int) for t in out)
        assert all(0 <= t < V for t in out)
        assert out[:-1] == toks[: len(out) - 1].tolist()      # everything before the last token is an accepted draft


def test_identical_models_accept_everything(impl):
    V, gamma = 5, 3
    g = _gen(1)
    p = torch.softmax(torch.randn(gamma + 1, V, generator=g), -1)
    q = p[:gamma].clone()
    for _ in range(100):
        toks = torch.stack([torch.multinomial(q[i], 1, generator=g)[0] for i in range(gamma)])
        out = impl.speculative_accept(p, q, toks, g)
        assert len(out) == gamma + 1                          # all accepted plus the bonus token
        assert out[:gamma] == toks.tolist()


def test_impossible_draft_is_rejected_and_corrected(impl):
    # the target gives probability 0 to the drafted token: it must be rejected at position 0
    p = _rows([0.0, 0.5, 0.5, 0.0], [0.25, 0.25, 0.25, 0.25])
    q = _rows([1.0, 0.0, 0.0, 0.0])
    toks = torch.tensor([0])
    for seed in range(50):
        out = impl.speculative_accept(p, q, toks, _gen(seed))
        assert len(out) == 1
        assert out[0] in (1, 2)                               # residual norm(max(0, p - q)) = p here


def test_greedy_target_matches_argmax(impl):
    # one-hot target rows (temperature 0): accept iff the draft equals the argmax, else emit the argmax
    p = _rows([0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1])
    q = _rows([0.25] * 4, [0.25] * 4)
    for seed in range(20):
        assert impl.speculative_accept(p, q, torch.tensor([1, 2]), _gen(seed)) == [1, 2, 3]
        assert impl.speculative_accept(p, q, torch.tensor([1, 0]), _gen(seed)) == [1, 2]
        assert impl.speculative_accept(p, q, torch.tensor([3, 2]), _gen(seed)) == [1]


def test_zero_draft_length_is_plain_sampling(impl):
    p = _rows([0.0, 0.0, 1.0, 0.0])
    out = impl.speculative_accept(p, torch.zeros(0, 4), torch.zeros(0, dtype=torch.long), _gen(3))
    assert out == [2]


def test_reproducible_with_seed(impl):
    V, gamma = 7, 3
    g = _gen(4)
    p = torch.softmax(torch.randn(gamma + 1, V, generator=g), -1)
    q = torch.softmax(torch.randn(gamma, V, generator=g), -1)
    toks = torch.tensor([1, 4, 2])
    a = [impl.speculative_accept(p, q, toks, _gen(s)) for s in range(30)]
    b = [impl.speculative_accept(p, q, toks, _gen(s)) for s in range(30)]
    assert a == b


# ---------- exactness: the emitted stream is distributed exactly like the target ----------

def _bigram_models(V=3):
    g = _gen(123)
    Tp = torch.softmax(2.0 * torch.randn(V, V, generator=g), -1)     # target: next-token distribution given previous token
    Tq = torch.softmax(1.5 * torch.randn(V, V, generator=g), -1)     # draft: a different, imperfect bigram model
    p0 = torch.tensor([0.5, 0.3, 0.2])                                # target distribution of the first token
    q0 = torch.tensor([0.2, 0.3, 0.5])                                # draft distribution of the first token
    return Tp, Tq, p0, q0


def _spec_step(impl, prev, gamma, Tp, Tq, p0, q0, g):
    """One speculative step. `prev` is the last emitted token (None at the start of the sequence)."""
    q_rows, toks, last = [], [], prev
    for _ in range(gamma):
        q = q0 if last is None else Tq[last]
        x = int(torch.multinomial(q, 1, generator=g))
        q_rows.append(q)
        toks.append(x)
        last = x
    p_rows = [p0 if prev is None else Tp[prev]]
    for x in toks:
        p_rows.append(Tp[x])                                          # target row after each drafted token
    return impl.speculative_accept(torch.stack(p_rows), torch.stack(q_rows), torch.tensor(toks, dtype=torch.long), g)


def test_marginal_of_first_token_equals_target(impl):
    Tp, Tq, p0, q0 = _bigram_models()
    g = _gen(7)
    n = 8000
    counts = torch.zeros(3)
    for _ in range(n):
        out = _spec_step(impl, None, 3, Tp, Tq, p0, q0, g)
        counts[out[0]] += 1
    freq = counts / n
    # 5 sigma for the worst cell (p = 0.5): 5 * sqrt(0.25 / 8000) = 0.028
    assert (freq - p0).abs().max() < 0.03, (freq, p0)
    # the draft distribution is different, so a broken rule that just trusts the draft would fail this
    assert (q0 - p0).abs().max() > 0.25


def test_joint_distribution_of_first_three_tokens_equals_target(impl):
    Tp, Tq, p0, q0 = _bigram_models()
    g = _gen(11)
    n, L, gamma = 8000, 3, 3
    counts = torch.zeros(3, 3, 3)
    for _ in range(n):
        seq, prev = [], None
        while len(seq) < L:
            seq += _spec_step(impl, prev, gamma, Tp, Tq, p0, q0, g)
            prev = seq[-1]
        a, b, c = seq[:L]
        counts[a, b, c] += 1
    freq = counts / n
    want = torch.zeros(3, 3, 3)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                want[a, b, c] = p0[a] * Tp[a, b] * Tp[b, c]
    assert (freq - want).abs().max() < 0.03, (freq - want).abs().max()


def test_empirical_acceptance_matches_beta(impl):
    Tp, Tq, p0, q0 = _bigram_models()
    beta = impl.acceptance_rate(p0, q0)
    g = _gen(5)
    n, acc = 8000, 0
    p = torch.stack([p0, Tp[0]])
    for _ in range(n):
        x = int(torch.multinomial(q0, 1, generator=g))
        out = impl.speculative_accept(p, q0[None], torch.tensor([x]), g)
        acc += int(len(out) == 2)                                     # both tokens out means the draft was accepted
    assert abs(acc / n - beta) < 0.03, (acc / n, beta)
