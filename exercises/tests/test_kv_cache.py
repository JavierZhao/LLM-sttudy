import pytest
import torch
import torch.nn.functional as F

MODULE = "kv_cache"

B, N_H, N_KV, D_H = 2, 4, 2, 8


def rand(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float64)


def reference_causal(q, k, v):
    """Full-sequence causal attention, GQA by explicit head repetition, PyTorch SDPA."""
    g = q.shape[1] // k.shape[1]
    return F.scaled_dot_product_attention(q, k.repeat_interleave(g, 1), v.repeat_interleave(g, 1), is_causal=True)


def new_cache(impl, max_len=16, n_layers=2, batch=B):
    return impl.KVCache(n_layers, batch, N_KV, max_len, D_H, dtype=torch.float64)


# ----------------------------------------------------------------------------- KVCache

def test_buffers_are_preallocated_with_the_documented_shape(impl):
    c = new_cache(impl, max_len=12, n_layers=3)
    assert len(c.k) == len(c.v) == 3
    for buf in c.k + c.v:
        assert buf.shape == (B, N_KV, 12, D_H) and buf.dtype == torch.float64
        assert torch.count_nonzero(buf) == 0
    assert [c.length(layer) for layer in range(3)] == [0, 0, 0]


def test_append_get_and_length(impl):
    c = new_cache(impl)
    k1, v1 = rand(B, N_KV, 3, D_H, seed=1), rand(B, N_KV, 3, D_H, seed=2)
    k2, v2 = rand(B, N_KV, 1, D_H, seed=3), rand(B, N_KV, 1, D_H, seed=4)
    c.append(0, k1, v1)
    assert c.length(0) == 3
    c.append(0, k2, v2)
    assert c.length(0) == 4
    K, V = c.get(0)
    assert K.shape == V.shape == (B, N_KV, 4, D_H)
    assert torch.equal(K, torch.cat([k1, k2], dim=2))
    assert torch.equal(V, torch.cat([v1, v2], dim=2))


def test_append_is_in_place_and_get_returns_views(impl):
    c = new_cache(impl)
    ptr = c.k[0].untyped_storage().data_ptr()
    buf = c.k[0]
    for i in range(4):
        c.append(0, rand(B, N_KV, 2, D_H, seed=i), rand(B, N_KV, 2, D_H, seed=10 + i))
    assert c.k[0] is buf and c.k[0].untyped_storage().data_ptr() == ptr        # never reallocated
    K, V = c.get(0)
    assert K.untyped_storage().data_ptr() == ptr                               # a view, not a copy
    assert V.untyped_storage().data_ptr() == c.v[0].untyped_storage().data_ptr()
    assert c.length(0) == 8


def test_layers_have_independent_lengths_and_contents(impl):
    c = new_cache(impl)
    k, v = rand(B, N_KV, 5, D_H, seed=5), rand(B, N_KV, 5, D_H, seed=6)
    c.append(1, k, v)
    assert c.length(0) == 0 and c.length(1) == 5
    assert c.get(0)[0].shape[2] == 0
    assert torch.equal(c.get(1)[0], k)


def test_overflow_raises_and_leaves_cache_unchanged(impl):
    c = new_cache(impl, max_len=6)
    c.append(0, rand(B, N_KV, 5, D_H, seed=7), rand(B, N_KV, 5, D_H, seed=8))
    before = c.get(0)[0].clone()
    with pytest.raises(ValueError):
        c.append(0, rand(B, N_KV, 2, D_H, seed=9), rand(B, N_KV, 2, D_H, seed=9))
    assert c.length(0) == 5
    assert torch.equal(c.get(0)[0], before)
    c.append(0, rand(B, N_KV, 1, D_H, seed=9), rand(B, N_KV, 1, D_H, seed=9))   # exactly full is fine
    assert c.length(0) == 6


# ----------------------------------------------------------------------------- attend_with_cache

def _inputs(S=9):
    q = rand(B, N_H, S, D_H, seed=20)
    k = rand(B, N_KV, S, D_H, seed=21)
    v = rand(B, N_KV, S, D_H, seed=22)
    return q, k, v


def _run_chunks(impl, q, k, v, chunks):
    c = new_cache(impl, max_len=q.shape[2])
    outs, start = [], 0
    for T in chunks:
        sl = slice(start, start + T)
        outs.append(impl.attend_with_cache(q[:, :, sl], k[:, :, sl], v[:, :, sl], c, layer=0))
        start += T
    return torch.cat(outs, dim=2), c


def test_prefill_then_decode_equals_full_causal_attention(impl):
    q, k, v = _inputs(9)
    out, c = _run_chunks(impl, q, k, v, [5, 1, 1, 1, 1])        # prefill 5 tokens, then 4 decode steps
    torch.testing.assert_close(out, reference_causal(q, k, v), atol=1e-9, rtol=1e-9)
    assert c.length(0) == 9


@pytest.mark.parametrize("chunks", [[9], [3, 3, 3], [1] * 9, [2, 4, 3]])
def test_any_chunking_gives_the_same_answer(impl, chunks):
    q, k, v = _inputs(9)
    out, _ = _run_chunks(impl, q, k, v, chunks)
    torch.testing.assert_close(out, reference_causal(q, k, v), atol=1e-9, rtol=1e-9)


def test_decode_step_sees_every_cached_key(impl):
    # one new query over a 7-token cache must attend to all 8 positions (no mask applies)
    q, k, v = _inputs(8)
    c = new_cache(impl, max_len=8)
    impl.attend_with_cache(q[:, :, :7], k[:, :, :7], v[:, :, :7], c, layer=0)
    out = impl.attend_with_cache(q[:, :, 7:], k[:, :, 7:], v[:, :, 7:], c, layer=0)
    g = N_H // N_KV
    scores = q[:, :, 7:] @ k.repeat_interleave(g, 1).transpose(-1, -2) / D_H**0.5
    expected = scores.softmax(-1) @ v.repeat_interleave(g, 1)
    torch.testing.assert_close(out, expected, atol=1e-9, rtol=1e-9)


def test_outputs_do_not_depend_on_later_tokens(impl):
    q, k, v = _inputs(8)
    out_a, _ = _run_chunks(impl, q, k, v, [4, 1, 1, 1, 1])
    q2, k2, v2 = q.clone(), k.clone(), v.clone()
    q2[:, :, -1], k2[:, :, -1], v2[:, :, -1] = rand(B, N_H, D_H, seed=30), rand(B, N_KV, D_H, seed=31), rand(B, N_KV, D_H, seed=32)
    out_b, _ = _run_chunks(impl, q2, k2, v2, [4, 1, 1, 1, 1])
    torch.testing.assert_close(out_a[:, :, :-1], out_b[:, :, :-1], atol=0, rtol=0)   # earlier positions unchanged
    assert not torch.allclose(out_a[:, :, -1], out_b[:, :, -1])


def test_attend_grows_only_the_requested_layer(impl):
    q, k, v = _inputs(4)
    c = new_cache(impl)
    impl.attend_with_cache(q, k, v, c, layer=1)
    assert c.length(1) == 4 and c.length(0) == 0
    impl.attend_with_cache(q[:, :, :1], k[:, :, :1], v[:, :, :1], c, layer=1)
    assert c.length(1) == 5


# ----------------------------------------------------------------------------- generate_greedy

VOCAB, D_MODEL = 11, 16


def make_model(seed=0, n_layers=2):
    g = torch.Generator().manual_seed(seed)

    def w(*shape, scale=0.5):
        return torch.randn(*shape, generator=g, dtype=torch.float64) * scale

    return {
        "emb": w(VOCAB, D_MODEL, scale=1.0),
        "pos": w(64, D_MODEL, scale=1.0),
        "head": w(D_MODEL, VOCAB, scale=1.0),
        "layers": [
            {"q": w(D_MODEL, N_H * D_H), "k": w(D_MODEL, N_KV * D_H), "v": w(D_MODEL, N_KV * D_H), "o": w(N_H * D_H, D_MODEL)}
            for _ in range(n_layers)
        ],
    }


def heads(x, n):
    return x.view(1, x.shape[1], n, D_H).transpose(1, 2)              # (1, T, n*d_h) -> (1, n, T, d_h)


def forward_uncached(m, ids):
    """Reference: recompute the whole sequence every time; logits of the last position."""
    T = ids.shape[1]
    x = m["emb"][ids] + m["pos"][:T]
    for lw in m["layers"]:
        q, k, v = heads(x @ lw["q"], N_H), heads(x @ lw["k"], N_KV), heads(x @ lw["v"], N_KV)
        o = reference_causal(q, k, v).transpose(1, 2).reshape(1, T, N_H * D_H)
        x = x + o @ lw["o"]
    return (x[:, -1] @ m["head"])[0]


def make_step_fn(impl, m):
    def step(ids, cache):
        start = cache.length(0)                                          # position of the first new token
        T = ids.shape[1]
        x = m["emb"][ids] + m["pos"][start:start + T]
        for layer, lw in enumerate(m["layers"]):
            q, k, v = heads(x @ lw["q"], N_H), heads(x @ lw["k"], N_KV), heads(x @ lw["v"], N_KV)
            o = impl.attend_with_cache(q, k, v, cache, layer).transpose(1, 2).reshape(1, T, N_H * D_H)
            x = x + o @ lw["o"]
        return (x[:, -1] @ m["head"])[0]

    return step


def test_cached_greedy_generation_matches_recomputing_everything(impl):
    m = make_model()
    prompt = [3, 1, 4, 1, 5]
    n_new = 12
    cache = impl.KVCache(len(m["layers"]), 1, N_KV, 32, D_H, dtype=torch.float64)
    got = impl.generate_greedy(make_step_fn(impl, m), prompt, n_new, eos_id=VOCAB, cache=cache)   # eos_id never produced
    ids = torch.tensor([prompt])
    want = []
    for _ in range(n_new):
        tok = int(forward_uncached(m, ids).argmax())
        want.append(tok)
        ids = torch.cat([ids, torch.tensor([[tok]])], dim=1)
    assert got == want
    assert all(isinstance(t, int) for t in got)
    # bookkeeping: prompt + every generated token except the last one (which is never fed back)
    assert [cache.length(layer) for layer in range(2)] == [len(prompt) + n_new - 1] * 2


class Script:
    """A step function that ignores the cache and emits a fixed token sequence, recording its inputs."""

    def __init__(self, tokens):
        self.tokens, self.calls = tokens, []

    def __call__(self, ids, cache):
        logits = torch.zeros(VOCAB)
        logits[self.tokens[len(self.calls)]] = 1.0
        self.calls.append(ids.clone())
        return logits


def test_stops_after_eos_and_includes_it(impl):
    step = Script([5, 3, 7, 2, 2])
    assert impl.generate_greedy(step, [1, 2], max_new_tokens=10, eos_id=7) == [5, 3, 7]
    assert len(step.calls) == 3


def test_respects_max_new_tokens_and_feeds_back_one_token_at_a_time(impl):
    step = Script([5, 3, 6, 2, 9, 9])
    out = impl.generate_greedy(step, [1, 2, 4], max_new_tokens=4, eos_id=7)
    assert out == [5, 3, 6, 2]
    assert len(step.calls) == 4                                       # n new tokens -> n calls, not n + 1
    assert step.calls[0].tolist() == [[1, 2, 4]]                      # prefill: the whole prompt at once
    assert [c.tolist() for c in step.calls[1:]] == [[[5]], [[3]], [[6]]]
    assert all(c.dtype == torch.long for c in step.calls)


def test_edge_cases(impl):
    step = Script([7])
    assert impl.generate_greedy(step, [1], max_new_tokens=5, eos_id=7) == [7]
    assert len(step.calls) == 1
    step = Script([1])
    assert impl.generate_greedy(step, [1], max_new_tokens=0, eos_id=7) == []
    assert step.calls == []
    step = Script([4, 5])
    assert impl.generate_greedy(step, [7, 7], max_new_tokens=2, eos_id=7) == [4, 5]   # EOS in the prompt is ignored
