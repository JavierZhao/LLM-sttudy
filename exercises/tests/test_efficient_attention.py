import math

import torch
import torch.nn.functional as F

MODULE = "efficient_attention"


def _phi(x):
    return F.elu(x) + 1.0


def _rand(B=2, H=3, T=9, dk=8, dv=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, T, dk, generator=g)
    k = torch.randn(B, H, T, dk, generator=g)
    v = torch.randn(B, H, T, dv, generator=g)
    return q, k, v


# ---------------------------------------------------------------- sliding window
def test_mask_exact_small_case(impl):
    m = impl.sliding_window_mask(5, 3)
    assert m.dtype == torch.bool and m.shape == (5, 5)
    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(m, expected)


def test_mask_limits(impl):
    T = 6
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    assert torch.equal(impl.sliding_window_mask(T, T), causal)
    assert torch.equal(impl.sliding_window_mask(T, 100), causal)
    assert torch.equal(impl.sliding_window_mask(T, 1), torch.eye(T, dtype=torch.bool))


def test_sliding_window_attention_matches_sdpa(impl):
    q, k, v = _rand(T=11, dk=8, dv=8)
    for w in (1, 3, 5, 11, 20):
        mask = impl.sliding_window_mask(11, w)  # checked above; reused to build the SDPA reference
        ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        torch.testing.assert_close(impl.sliding_window_attention(q, k, v, w), ref, atol=1e-5, rtol=1e-5)


def test_sliding_window_is_local(impl):
    # changing keys/values outside the window must not change the output at t
    q, k, v = _rand(T=12, dk=8, dv=8)
    w = 4
    out = impl.sliding_window_attention(q, k, v, w)
    k2, v2 = k.clone(), v.clone()
    k2[:, :, :5] += 10.0
    v2[:, :, :5] += 10.0
    out2 = impl.sliding_window_attention(q, k2, v2, w)
    # query t sees keys t-3..t; keys 0..4 are visible only to queries t <= 7
    torch.testing.assert_close(out[:, :, 8:], out2[:, :, 8:], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(out[:, :, :8], out2[:, :, :8])


def test_receptive_field_grows_with_depth(impl):
    # two stacked window-3 layers: position 0 reaches output t <= 2*(3-1) = 4, not t = 5
    T, w = 10, 3
    x = torch.randn(1, 1, T, 8)

    def two_layers(x):
        h = x + impl.sliding_window_attention(x, x, x, w)
        return h + impl.sliding_window_attention(h, h, h, w)

    base = two_layers(x)
    x2 = x.clone()
    x2[:, :, 0] += 5.0
    diff = (two_layers(x2) - base).abs().amax(dim=-1)[0, 0]
    assert diff[4] > 1e-6
    assert diff[5] < 1e-6


# ---------------------------------------------------------------- linear attention
def _linear_naive(q, k, v):
    B, H, T, _ = q.shape
    out = torch.zeros(B, H, T, v.shape[-1])
    for b in range(B):
        for h in range(H):
            for t in range(T):
                w = torch.stack([(_phi(q[b, h, t]) * _phi(k[b, h, s])).sum() for s in range(t + 1)])
                out[b, h, t] = (w[:, None] * v[b, h, : t + 1]).sum(0) / w.sum()
    return out


def test_linear_parallel_matches_naive(impl):
    q, k, v = _rand(B=1, H=2, T=7)
    torch.testing.assert_close(impl.linear_attention_parallel(q, k, v), _linear_naive(q, k, v), atol=1e-5, rtol=1e-5)


def test_linear_recurrent_equals_parallel(impl):
    q, k, v = _rand(T=16, dk=8, dv=5)
    torch.testing.assert_close(
        impl.linear_attention_recurrent(q, k, v), impl.linear_attention_parallel(q, k, v), atol=1e-5, rtol=1e-5
    )


def test_linear_is_causal(impl):
    q, k, v = _rand(T=10)
    for fn in (impl.linear_attention_parallel, impl.linear_attention_recurrent):
        out = fn(q, k, v)
        k2, v2 = k.clone(), v.clone()
        k2[:, :, 6:] = torch.randn_like(k2[:, :, 6:])
        v2[:, :, 6:] = torch.randn_like(v2[:, :, 6:])
        torch.testing.assert_close(fn(q, k2, v2)[:, :, :6], out[:, :, :6], atol=1e-6, rtol=1e-6)


def test_linear_normalizer_gives_convex_average(impl):
    # constant values: a normalized weighted average returns the same constant
    q, k, _ = _rand(T=8)
    v = torch.full((2, 3, 8, 4), 2.5)
    for fn in (impl.linear_attention_parallel, impl.linear_attention_recurrent):
        torch.testing.assert_close(fn(q, k, v), v, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------- delta rule
def _delta_naive(q, k, v, beta):
    B, H, T, dk = q.shape
    dv = v.shape[-1]
    out = torch.zeros(B, H, T, dv)
    for b in range(B):
        for h in range(H):
            S = torch.zeros(dk, dv)
            for t in range(T):
                kt = k[b, h, t][None, :]  # (1, dk) row
                vt = v[b, h, t][None, :]  # (1, dv) row
                S = S + beta[b, h, t] * kt.T @ (vt - kt @ S)
                out[b, h, t] = (q[b, h, t][None, :] @ S)[0]
    return out


def test_delta_rule_matches_naive(impl):
    q, k, v = _rand(B=2, H=2, T=10, dk=6, dv=5)
    k = F.normalize(k, dim=-1)
    beta = torch.rand(2, 2, 10)
    torch.testing.assert_close(impl.delta_rule_recurrent(q, k, v, beta), _delta_naive(q, k, v, beta), atol=1e-5, rtol=1e-5)


def test_delta_rule_beta_zero_writes_nothing(impl):
    q, k, v = _rand(T=6)
    out = impl.delta_rule_recurrent(q, k, v, torch.zeros(2, 3, 6))
    assert torch.count_nonzero(out) == 0


def test_delta_rule_overwrites_instead_of_accumulating(impl):
    # write v1 then v2 under the SAME unit key with beta = 1: reading that key returns v2.
    # (Linear attention without a normalizer would return v1 + v2.)
    dk, dv = 4, 3
    e0 = torch.zeros(dk)
    e0[0] = 1.0
    v1, v2 = torch.randn(dv), torch.randn(dv)
    k = torch.stack([e0, e0])[None, None]  # (1, 1, 2, dk)
    v = torch.stack([v1, v2])[None, None]
    beta = torch.ones(1, 1, 2)
    out = impl.delta_rule_recurrent(k, k, v, beta)  # q = k
    torch.testing.assert_close(out[0, 0, 0], v1, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(out[0, 0, 1], v2, atol=1e-6, rtol=1e-6)


def test_delta_rule_orthogonal_keys_do_not_interfere(impl):
    # distinct one-hot keys: each read returns exactly what was written under that key
    dk, dv, T = 5, 3, 5
    k = torch.eye(dk)[None, None]  # (1, 1, T, dk)
    v = torch.randn(1, 1, T, dv)
    beta = torch.ones(1, 1, T)
    out = impl.delta_rule_recurrent(k, k, v, beta)
    torch.testing.assert_close(out, v, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------- top-k sparse
def _topk_naive(q, k, v, scores, k_top):
    B, H, T, d = q.shape
    out = torch.zeros_like(q)
    for b in range(B):
        for t in range(T):
            cand = list(range(t + 1))
            cand.sort(key=lambda s: -scores[b, t, s].item())
            keep = cand[:k_top]
            for h in range(H):
                logits = torch.stack([(q[b, h, t] * k[b, h, s]).sum() / math.sqrt(d) for s in keep])
                out[b, h, t] = (logits.softmax(0)[:, None] * v[b, h, keep]).sum(0)
    return out


def test_topk_matches_naive(impl):
    q, k, v = _rand(B=2, H=2, T=10, dk=8, dv=8)
    scores = torch.randn(2, 10, 10)
    for kt in (1, 3, 6):
        torch.testing.assert_close(
            impl.topk_sparse_attention(q, k, v, scores, kt), _topk_naive(q, k, v, scores, kt), atol=1e-5, rtol=1e-5
        )


def test_topk_full_budget_equals_dense_causal(impl):
    q, k, v = _rand(T=8, dk=8, dv=8)
    scores = torch.randn(2, 8, 8)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    for kt in (8, 50):
        torch.testing.assert_close(impl.topk_sparse_attention(q, k, v, scores, kt), ref, atol=1e-5, rtol=1e-5)


def test_topk_one_key_returns_that_value(impl):
    q, k, v = _rand(B=1, H=2, T=8, dk=8, dv=8)
    scores = torch.randn(1, 8, 8)
    out = impl.topk_sparse_attention(q, k, v, scores, 1)
    for t in range(8):
        best = scores[0, t, : t + 1].argmax()
        torch.testing.assert_close(out[0, :, t], v[0, :, best], atol=1e-6, rtol=1e-6)


def test_topk_never_selects_future_keys(impl):
    q, k, v = _rand(B=1, H=1, T=6, dk=4, dv=4)
    scores = torch.randn(1, 6, 6)
    boosted = scores + torch.triu(torch.full((6, 6), 100.0), diagonal=1)  # future keys look best
    a = impl.topk_sparse_attention(q, k, v, boosted, 3)
    b = impl.topk_sparse_attention(q, k, v, scores, 3)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    # and changing future values cannot change earlier outputs
    v2 = v.clone()
    v2[:, :, 4:] += 3.0
    c = impl.topk_sparse_attention(q, k, v2, scores, 3)
    torch.testing.assert_close(c[:, :, :4], b[:, :, :4], atol=1e-6, rtol=1e-6)
