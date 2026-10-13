import math

import pytest
import torch

MODULE = "beam_search"
NEG_INF = float("-inf")
BOS, A, B, EOS = 0, 1, 2, 3          # the hand-made toy vocabulary


# ---------------------------------------------------------------- toy models and references
def table_model(rows):
    """Markov model: rows[last_token] = probabilities over the next token (0 means forbidden)."""
    logp = torch.tensor(rows, dtype=torch.float64).log()          # log(0) = -inf

    def fn(prefixes):
        assert prefixes.dtype == torch.long and prefixes.dim() == 2 and prefixes.shape[0] >= 1
        return logp[prefixes[:, -1]]
    return fn


# BOS -> a .45, b .40, EOS .15 ; a -> a .05, b .85, EOS .10 ; b -> a .35, b .10, EOS .55
TOY = [[0, .45, .40, .15],
       [0, .05, .85, .10],
       [0, .35, .10, .55],
       [0, .25, .25, .50]]


def rnn_model(V, seed, H=8):
    """A tiny random RNN in float64: the next-token distribution depends on the whole prefix."""
    g = torch.Generator().manual_seed(seed)
    E = torch.randn(V, H, generator=g, dtype=torch.float64)
    W = torch.randn(H, H, generator=g, dtype=torch.float64) * 0.7
    O = torch.randn(H, V, generator=g, dtype=torch.float64) * 1.5

    def fn(prefixes):
        h = torch.zeros(prefixes.shape[0], H, dtype=torch.float64)
        for j in range(prefixes.shape[1]):
            h = torch.tanh(h @ W + E[prefixes[:, j]])
        return torch.log_softmax(h @ O, -1)
    return fn


def ref_score(fn, bos, tokens, alpha):
    """Teacher-forced score of a hypothesis, computed independently of the search."""
    logp, prefix = 0.0, [bos]
    for tok in tokens:
        logp += fn(torch.tensor([prefix]))[0, tok].item()
        prefix.append(tok)
    return logp / (((5 + len(tokens)) / 6) ** alpha)


def ref_all(fn, bos, eos, max_len, alpha):
    """Every finished hypothesis with its score, by recursion (no beam, no pruning)."""
    out = []

    def rec(prefix):
        row = fn(torch.tensor([[bos] + prefix]))[0]
        for tok in range(row.shape[0]):
            if row[tok].item() == NEG_INF:
                continue
            seq = prefix + [tok]
            if tok == eos or len(seq) == max_len:
                out.append((seq, ref_score(fn, bos, seq, alpha)))
            else:
                rec(seq)
    rec([])
    return sorted(out, key=lambda h: -h[1])


def assert_same(got, want, tol=1e-9):
    assert [t for t, _ in got] == [t for t, _ in want]
    for (_, s1), (_, s2) in zip(got, want):
        assert s1 == pytest.approx(s2, abs=tol)


class Recorder:
    """Wraps a step function and records every call."""
    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def __call__(self, prefixes):
        self.calls.append(prefixes.clone())
        return self.fn(prefixes)


# ---------------------------------------------------------------- normalized_score
def test_normalized_score_values(impl):
    assert impl.normalized_score(-3.0, 7, 0.0) == pytest.approx(-3.0)
    assert impl.normalized_score(-3.0, 7, 1.0) == pytest.approx(-1.5)            # 12 / 6 = 2
    assert impl.normalized_score(-3.0, 1, 1.0) == pytest.approx(-3.0)            # lp(1) = 1
    assert impl.normalized_score(-3.0, 3, 0.6) == pytest.approx(-3.0 / (8 / 6) ** 0.6)
    assert isinstance(impl.normalized_score(-1.0, 2, 0.5), float)


# ---------------------------------------------------------------- hand-made toy with known answers
def test_toy_known_best_sequences(impl):
    fn = table_model(TOY)
    p_be, p_abe = .40 * .55, .45 * .85 * .55                                       # 0.22 and 0.21037
    res = impl.beam_search(fn, BOS, EOS, beam=2, max_len=3, length_penalty=0.0)
    assert [t for t, _ in res] == [[B, EOS], [A, B, EOS]]
    assert res[0][1] == pytest.approx(math.log(p_be)) and res[1][1] == pytest.approx(math.log(p_abe))
    # length normalization flips the winner: -1.559 / (8/6) beats -1.514 / (7/6)
    res = impl.beam_search(fn, BOS, EOS, beam=2, max_len=3, length_penalty=1.0)
    assert [t for t, _ in res] == [[A, B, EOS], [B, EOS]]
    assert res[0][1] == pytest.approx(math.log(p_abe) / (8 / 6))
    assert res[1][1] == pytest.approx(math.log(p_be) / (7 / 6))


def test_greedy_is_not_optimal_but_beam_two_is(impl):
    fn = table_model(TOY)
    greedy = impl.beam_search(fn, BOS, EOS, beam=1, max_len=3, length_penalty=0.0)
    assert [t for t, _ in greedy] == [[A, B, EOS]]                                 # a (.45) then b (.85) then EOS
    best = impl.beam_search(fn, BOS, EOS, beam=2, max_len=3, length_penalty=0.0)[0]
    assert best[0] == [B, EOS] and best[1] > greedy[0][1]


def test_low_ranked_eos_extensions_are_still_recorded(impl):
    # Every live hypothesis's EOS extension is a finished hypothesis, whatever its rank among the
    # candidates. Here (ids: 0 BOS, 1 a, 2 EOS) "a" always beats EOS as the next token, so with beam 1
    # the EOS extension is never in the top 1, yet the empty output [EOS] (log .4 = -0.92) beats every
    # other finished hypothesis: aEOS -1.43, aaEOS -1.94 and the truncated aaa -1.53 (and also at alpha 1).
    fn = table_model([[0, .6, .4], [0, .6, .4], [0, .6, .4]])
    for alpha in (0.0, 1.0):
        res = impl.beam_search(fn, 0, 2, beam=1, max_len=3, length_penalty=alpha)
        assert [t for t, _ in res] == [[2]]
        assert res[0][1] == pytest.approx(math.log(.4))
        assert_same(res, impl.exhaustive_search(fn, 0, 2, 3, alpha, top_n=1))


def test_first_step_eos_is_the_empty_hypothesis(impl):
    fn = table_model(TOY)
    res = impl.beam_search(fn, BOS, EOS, beam=8, max_len=3, length_penalty=0.0)
    assert any(t == [EOS] and s == pytest.approx(math.log(.15)) for t, s in res)


# ---------------------------------------------------------------- brute force agreement
@pytest.mark.parametrize("alpha", [0.0, 0.6, 1.0, 2.0])
def test_wide_beam_equals_exhaustive_search(impl, alpha):
    V, bos, eos, max_len = 4, 0, 3, 4
    for seed in range(6):
        fn = rnn_model(V, seed)
        want = ref_all(fn, bos, eos, max_len, alpha)[:5]
        assert_same(impl.exhaustive_search(fn, bos, eos, max_len, alpha, top_n=5), want)
        got = impl.beam_search(fn, bos, eos, beam=100, max_len=max_len, length_penalty=alpha)
        assert_same(got[:5], want)


def test_wide_beam_returns_every_finished_hypothesis_when_beam_is_larger(impl):
    fn = rnn_model(3, 7)                                  # V = 3, eos = 2: 2^(t) live hypotheses at most
    want = ref_all(fn, 0, 2, 3, 0.6)
    got = impl.beam_search(fn, 0, 2, beam=len(want) + 5, max_len=3, length_penalty=0.6)
    assert_same(got, want)


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_beam_one_scores_at_least_greedy(impl, alpha):
    for seed in range(10):
        fn = rnn_model(5, seed + 20)
        bos, eos, max_len = 0, 4, 5
        prefix, logp = [bos], 0.0                         # plain greedy decoding as the reference
        for step in range(max_len):
            row = fn(torch.tensor([prefix]))[0]
            tok = int(row.argmax())
            logp += row[tok].item()
            prefix.append(tok)
            if tok == eos:
                break
        greedy_score = logp / (((5 + len(prefix) - 1) / 6) ** alpha)
        best = impl.beam_search(fn, bos, eos, beam=1, max_len=max_len, length_penalty=alpha)[0]
        assert best[1] >= greedy_score - 1e-9


# ---------------------------------------------------------------- invariants of the returned hypotheses
def test_returned_hypotheses_are_well_formed_and_correctly_scored(impl):
    V, bos, eos, max_len, alpha = 5, 0, 4, 5, 0.7
    fn = rnn_model(V, 3)
    for beam in (1, 2, 3, 5):
        res = impl.beam_search(fn, bos, eos, beam, max_len, alpha)
        assert 1 <= len(res) <= beam
        assert [s for _, s in res] == sorted((s for _, s in res), reverse=True)
        assert len({tuple(t) for t, _ in res}) == len(res)                  # no duplicates
        for toks, score in res:
            assert all(isinstance(t, int) for t in toks) and isinstance(score, float)
            assert eos not in toks[:-1]                                     # EOS only as the last token
            assert len(toks) <= max_len and (toks[-1] == eos or len(toks) == max_len)
            assert score == pytest.approx(ref_score(fn, bos, toks, alpha), abs=1e-9)


def test_eos_hypotheses_are_never_extended_and_the_first_call_has_one_row(impl):
    rec = Recorder(rnn_model(4, 5))
    impl.beam_search(rec, 0, 3, beam=3, max_len=5, length_penalty=0.6)
    assert rec.calls[0].shape == (1, 1) and rec.calls[0][0, 0].item() == 0   # one live hypothesis, no copies
    for step, prefixes in enumerate(rec.calls, start=1):
        assert prefixes.shape[1] == step                                     # equal lengths, one token per step
        assert 1 <= prefixes.shape[0] <= 3                                   # never more rows than the beam
        assert (prefixes[:, 0] == 0).all()
        assert not (prefixes[:, 1:] == 3).any()                              # no finished hypothesis is fed back
    assert len(rec.calls) <= 5
    assert len({tuple(r.tolist()) for r in rec.calls[-1]}) == rec.calls[-1].shape[0]   # no duplicate live rows


def test_forbidden_tokens_never_enter_the_beam(impl):
    # BOS -> {a, b}; a -> {c}; b -> {c, EOS}; c -> {EOS}.  Ids: 0 BOS, 1 a, 2 b, 3 c, 4 EOS
    rows = [[0, .6, .4, 0, 0],
            [0, 0, 0, 1.0, 0],
            [0, 0, 0, .6, .4],
            [0, 0, 0, 0, 1.0],
            [0, 0, 0, 0, 1.0]]
    fn = table_model(rows)
    res = impl.beam_search(fn, 0, 4, beam=6, max_len=5, length_penalty=0.0)
    assert sorted(t for t, _ in res) == sorted([[1, 3, 4], [2, 3, 4], [2, 4]])
    assert all(math.isfinite(s) for _, s in res)
    for t, s in res:
        assert s == pytest.approx(ref_score(fn, 0, t, 0.0))
    assert_same(res, ref_all(fn, 0, 4, 5, 0.0))


def test_truncation_at_max_len_when_eos_is_impossible(impl):
    rows = [[0, .7, .3, 0], [0, .6, .4, 0], [0, .15, .85, 0], [0, 0, 0, 1.0]]
    fn = table_model(rows)                                # EOS has probability 0 everywhere
    res = impl.beam_search(fn, 0, 3, beam=2, max_len=3, length_penalty=1.0)
    assert len(res) == 2 and all(len(t) == 3 and 3 not in t for t, _ in res)
    for t, s in res:
        assert s == pytest.approx(ref_score(fn, 0, t, 1.0))
    wide = impl.beam_search(fn, 0, 3, beam=16, max_len=3, length_penalty=1.0)       # no pruning: exact
    assert_same(wide, ref_all(fn, 0, 3, 3, 1.0)[:16])


def test_everything_forbidden_returns_empty_and_wide_beam_does_not_crash(impl):
    fn = table_model([[0, 0, 0, 0]] * 4)                  # nothing has finite probability
    assert impl.beam_search(fn, 0, 3, beam=4, max_len=3) == []
    fn = rnn_model(4, 1)
    res = impl.beam_search(fn, 0, 3, beam=1000, max_len=2, length_penalty=0.5)   # beam >> candidates
    assert len(res) <= 1000 and len(res) > 0


def test_float32_and_batching_of_the_model_are_respected(impl):
    base = rnn_model(4, 11)
    fn32 = lambda p: base(p).float()
    want = ref_all(base, 0, 3, 3, 0.6)[:3]
    got = impl.beam_search(fn32, 0, 3, beam=50, max_len=3, length_penalty=0.6)[:3]
    assert_same(got, want, tol=1e-4)


def test_argument_validation(impl):
    fn = table_model(TOY)
    for kwargs in ({"beam": 0, "max_len": 3}, {"beam": 2, "max_len": 0}, {"beam": 2, "max_len": 3, "length_penalty": -0.5}):
        with pytest.raises(ValueError):
            impl.beam_search(fn, BOS, EOS, **kwargs)


# ---------------------------------------------------------------- early stopping
@pytest.mark.parametrize("alpha", [0.0, 0.6, 1.0])
def test_early_stop_never_changes_the_result(impl, alpha):
    for seed in range(12):
        fn = rnn_model(4, seed + 50)
        for beam in (1, 2, 3):
            for max_len in (3, 5, 6):
                full = impl.beam_search(fn, 0, 3, beam, max_len, alpha, early_stop=False)
                fast = impl.beam_search(fn, 0, 3, beam, max_len, alpha, early_stop=True)
                assert_same(fast, full)


def test_early_stop_saves_model_calls_when_eos_is_certain(impl):
    rows = [[0, .004, .006, .99], [0, .3, .3, .4], [0, .3, .3, .4], [0, .3, .3, .4]]
    for alpha in (0.0, 1.0):
        slow, fast = Recorder(table_model(rows)), Recorder(table_model(rows))
        r1 = impl.beam_search(slow, 0, 3, 1, 6, alpha, early_stop=False)
        r2 = impl.beam_search(fast, 0, 3, 1, 6, alpha, early_stop=True)
        assert_same(r2, r1)
        assert len(slow.calls) == 6 and len(fast.calls) == 1


def test_early_stop_is_not_fooled_by_length_normalization(impl):
    # With alpha > 0 a live hypothesis can still WIN later: its normalized score rises with length
    # when every extra token is nearly free. After step 1 the finished hypothesis [EOS] has score
    # log .6 = -0.51, and the live "a" has raw log-probability log .4 = -0.92, below it. But "a"
    # followed by seven more near-certain "a" tokens ends at -0.92 / lp(8) = -0.42 (alpha = 1), better.
    # Comparing raw scores, or normalizing by the CURRENT length, would stop too early.
    # Ids: 0 BOS, 1 a, 2 EOS
    rows = [[0, .4, .6], [0, .999, .001], [0, .5, .5]]
    fn = table_model(rows)
    for alpha in (1.0, 2.0):
        full = impl.beam_search(fn, 0, 2, 1, 8, alpha, early_stop=False)
        assert full[0][0] == [1] * 8                      # the long truncated hypothesis wins
        fast = impl.beam_search(fn, 0, 2, 1, 8, alpha, early_stop=True)
        assert_same(fast, full)


# ---------------------------------------------------------------- the oracle itself
def test_exhaustive_search_matches_reference_and_uses_batches_of_one(impl):
    fn = rnn_model(4, 9)
    rec = Recorder(fn)
    got = impl.exhaustive_search(rec, 0, 3, 3, 0.6, top_n=7)
    assert_same(got, ref_all(fn, 0, 3, 3, 0.6)[:7])
    assert all(c.shape[0] == 1 for c in rec.calls)
    toy = impl.exhaustive_search(table_model(TOY), BOS, EOS, 3, 0.0, top_n=1)
    assert toy[0][0] == [B, EOS]
