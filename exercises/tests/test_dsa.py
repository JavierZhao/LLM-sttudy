import math

import pytest
import torch

MODULE = "dsa"

NEG_INF = float("-inf")


# ----------------------------------------------------------------------------------------
# naive references (written inside the test on purpose: the tests never import solutions)
# ----------------------------------------------------------------------------------------
def _ref_scores(q, w, k):
    B, T, H, d = q.shape
    S = k.shape[1]
    out = torch.full((B, T, S), NEG_INF, dtype=q.dtype)
    for b in range(B):
        for t in range(T):
            for s in range(S - T + t + 1):
                acc = 0.0
                for j in range(H):
                    acc = acc + w[b, t, j] * max(float(q[b, t, j] @ k[b, s]), 0.0)
                out[b, t, s] = acc
    return out


def _dense_mqa(q_lat, q_rope, c, kr, scale, allowed):
    """Dense MQA-mode attention; `allowed` is a (B, T, S) bool mask of keys each query uses."""
    logits = (torch.einsum("bthc,bsc->bths", q_lat, c) + torch.einsum("bthr,bsr->bths", q_rope, kr)) * scale
    logits = logits.masked_fill(~allowed[:, :, None, :], NEG_INF)
    return torch.einsum("bths,bsc->bthc", logits.softmax(-1), c)


def _causal(B, T, S):
    qpos = torch.arange(T)[:, None] + (S - T)
    return (torch.arange(S)[None, :] <= qpos)[None].expand(B, T, S)


def _make(B=2, T=8, S=8, n_h=3, d_c=16, d_r=4, dtype=torch.float64):
    g = torch.Generator().manual_seed(1)
    r = lambda *shape: torch.randn(*shape, generator=g, dtype=dtype)
    return dict(q_lat=r(B, T, n_h, d_c), q_rope=r(B, T, n_h, d_r), c=r(B, S, d_c), kr=r(B, S, d_r),
                scale=1.0 / math.sqrt(d_c + d_r))


# ----------------------------------------------------------------------------------------
# lightning indexer
# ----------------------------------------------------------------------------------------
def test_index_scores_match_naive_prefill(impl):
    g = torch.Generator().manual_seed(0)
    q = torch.randn(2, 5, 4, 6, generator=g, dtype=torch.float64)
    w = torch.randn(2, 5, 4, generator=g, dtype=torch.float64)      # weights may be negative
    k = torch.randn(2, 5, 6, generator=g, dtype=torch.float64)
    torch.testing.assert_close(impl.lightning_index_scores(q, w, k), _ref_scores(q, w, k))


def test_index_scores_end_aligned_with_cache(impl):
    g = torch.Generator().manual_seed(1)
    q = torch.randn(1, 3, 2, 5, generator=g, dtype=torch.float64)   # last 3 of 7 positions
    w = torch.rand(1, 3, 2, generator=g, dtype=torch.float64)
    k = torch.randn(1, 7, 5, generator=g, dtype=torch.float64)
    out = impl.lightning_index_scores(q, w, k)
    assert out.shape == (1, 3, 7)
    torch.testing.assert_close(out, _ref_scores(q, w, k))
    assert torch.isinf(out[0, 0, 5:]).all() and (out[0, 0, 5:] < 0).all()   # query at position 4: keys 5, 6 hidden
    assert torch.isfinite(out[0, 2]).all()                                  # last query sees everything


def test_index_scores_are_relu_gated(impl):
    # one head, positive weight: a key anti-aligned with the query scores exactly 0, not negative
    q = torch.tensor([[[[1.0, 0.0]]]])
    w = torch.ones(1, 1, 1)
    k = torch.tensor([[[-1.0, 0.0]]])
    assert impl.lightning_index_scores(q, w, k).item() == 0.0
    # a negative weight flips the sign of the (non-negative) gated score
    out = impl.lightning_index_scores(q, -2.0 * w, torch.tensor([[[3.0, 0.0]]]))
    assert out.item() == pytest.approx(-6.0)


# ----------------------------------------------------------------------------------------
# sparse core attention in MQA mode
# ----------------------------------------------------------------------------------------
def test_dsa_equals_dense_when_k_covers_everything(impl):
    d = _make(T=8, S=8)
    scores = torch.randn(2, 8, 8, dtype=torch.float64).masked_fill(~_causal(2, 8, 8), NEG_INF)
    for k_top in (8, 20):
        out = impl.dsa_attention(d["q_lat"], d["q_rope"], d["c"], d["kr"], scores, k_top, d["scale"])
        ref = _dense_mqa(d["q_lat"], d["q_rope"], d["c"], d["kr"], d["scale"], _causal(2, 8, 8))
        torch.testing.assert_close(out, ref)


def test_dsa_matches_masked_dense_for_small_k(impl):
    d = _make(T=9, S=9)
    scores = torch.randn(2, 9, 9, dtype=torch.float64).masked_fill(~_causal(2, 9, 9), NEG_INF)
    k_top = 3
    idx = scores.topk(k_top, dim=-1).indices
    allowed = torch.zeros(2, 9, 9, dtype=torch.bool).scatter_(-1, idx, True) & _causal(2, 9, 9)
    out = impl.dsa_attention(d["q_lat"], d["q_rope"], d["c"], d["kr"], scores, k_top, d["scale"])
    torch.testing.assert_close(out, _dense_mqa(d["q_lat"], d["q_rope"], d["c"], d["kr"], d["scale"], allowed))


def test_dsa_respects_causality_when_fewer_than_k_keys_exist(impl):
    # early queries have < k_top usable keys: the filler picks must be masked, not attended to
    d = _make(T=6, S=6)
    scores = torch.randn(2, 6, 6, dtype=torch.float64).masked_fill(~_causal(2, 6, 6), NEG_INF)
    k_top = 4
    base = impl.dsa_attention(d["q_lat"], d["q_rope"], d["c"], d["kr"], scores, k_top, d["scale"])
    assert torch.isfinite(base).all()
    c2, kr2 = d["c"].clone(), d["kr"].clone()
    c2[:, 3:] += 100.0 * torch.randn_like(c2[:, 3:])                # corrupt keys/values at positions >= 3
    kr2[:, 3:] += 100.0 * torch.randn_like(kr2[:, 3:])
    pert = impl.dsa_attention(d["q_lat"], d["q_rope"], c2, kr2, scores, k_top, d["scale"])
    torch.testing.assert_close(pert[:, :3], base[:, :3])            # queries 0..2 cannot see positions >= 3
    assert not torch.allclose(pert[:, 3:], base[:, 3:])             # later queries do


def test_dsa_never_reads_unselected_entries(impl):
    d = _make(T=7, S=7)
    scores = torch.randn(2, 7, 7, dtype=torch.float64).masked_fill(~_causal(2, 7, 7), NEG_INF)
    k_top = 2
    base = impl.dsa_attention(d["q_lat"], d["q_rope"], d["c"], d["kr"], scores, k_top, d["scale"])
    # last query: find a usable key that it did NOT select and corrupt it
    sel = set(scores[0, 6].topk(k_top).indices.tolist())
    other = next(s for s in range(7) if s not in sel)
    c2 = d["c"].clone()
    c2[0, other] += 1e3
    pert = impl.dsa_attention(d["q_lat"], d["q_rope"], c2, d["kr"], scores, k_top, d["scale"])
    torch.testing.assert_close(pert[0, 6], base[0, 6])


def test_dsa_single_query_decode_with_cache(impl):
    d = _make(B=2, T=1, S=12)
    scores = torch.randn(2, 1, 12, dtype=torch.float64)             # last query sees all 12 keys
    out = impl.dsa_attention(d["q_lat"], d["q_rope"], d["c"], d["kr"], scores, 12, d["scale"])
    ref = _dense_mqa(d["q_lat"], d["q_rope"], d["c"], d["kr"], d["scale"], torch.ones(2, 1, 12, dtype=torch.bool))
    torch.testing.assert_close(out, ref)
    assert out.shape == (2, 1, 3, 16)


def test_dsa_selection_is_shared_across_heads(impl):
    # heads only differ in their queries: with k_top = 1 every head must return the same latent
    d = _make(T=5, S=5)
    scores = torch.randn(2, 5, 5, dtype=torch.float64).masked_fill(~_causal(2, 5, 5), NEG_INF)
    out = impl.dsa_attention(d["q_lat"], d["q_rope"], d["c"], d["kr"], scores, 1, d["scale"])
    best = scores.argmax(-1)                                        # (B, T)
    for b in range(2):
        for t in range(5):
            for h in range(3):
                torch.testing.assert_close(out[b, t, h], d["c"][b, best[b, t]])


def test_full_pipeline_shapes(impl):
    B, T, H, d_i = 2, 6, 4, 8
    q_idx, w, k_idx = torch.randn(B, T, H, d_i), torch.rand(B, T, H), torch.randn(B, T, d_i)
    scores = impl.lightning_index_scores(q_idx, w, k_idx)
    d = _make(B=B, T=T, S=T, dtype=torch.float32)
    out = impl.dsa_attention(d["q_lat"], d["q_rope"], d["c"], d["kr"], scores, 3, d["scale"])
    assert out.shape == (B, T, 3, 16) and torch.isfinite(out).all()


# ----------------------------------------------------------------------------------------
# indexer KL loss
# ----------------------------------------------------------------------------------------
def _random_attn(B, H, T, S, generator):
    logits = torch.randn(B, H, T, S, generator=generator, dtype=torch.float64)
    mask = _causal(B, T, S)[:, None]
    return logits.masked_fill(~mask, NEG_INF).softmax(-1)


def test_kl_is_zero_when_index_softmax_equals_target(impl):
    B, H, T = 2, 4, 6
    g = torch.Generator().manual_seed(3)
    logits = torch.randn(B, T, T, generator=g, dtype=torch.float64).masked_fill(~_causal(B, T, T), NEG_INF)
    p = logits.softmax(-1)
    attn = p[:, None].expand(B, H, T, T).contiguous()               # every head has the same rows
    assert impl.indexer_kl_loss(attn, logits).abs().item() < 1e-10
    # shift-invariance: softmax ignores an additive constant per row
    assert impl.indexer_kl_loss(attn, logits + 5.0).abs().item() < 1e-10


def test_kl_matches_reference_dense(impl):
    B, H, T, S = 2, 3, 5, 5
    g = torch.Generator().manual_seed(4)
    attn = _random_attn(B, H, T, S, g)
    scores = torch.randn(B, T, S, generator=g, dtype=torch.float64).masked_fill(~_causal(B, T, S), NEG_INF)
    total = 0.0
    for b in range(B):
        acc = 0.0
        for t in range(T):
            n = t + 1
            p = attn[b, :, t, :n].sum(0)
            p = p / p.sum()
            q = scores[b, t, :n].softmax(-1)
            acc += float((p * (p.log() - q.log())).sum())
        total += acc
    got = impl.indexer_kl_loss(attn, scores)
    assert got.item() == pytest.approx(total / B, rel=1e-9)
    assert got.item() > 0


def test_kl_target_is_head_sum_not_head_mean_of_logits(impl):
    # two heads with very different rows: target is the L1-normalized SUM of probabilities
    T = 4
    a1 = torch.zeros(1, 1, T, T, dtype=torch.float64)
    a2 = torch.zeros(1, 1, T, T, dtype=torch.float64)
    for t in range(T):
        a1[0, 0, t, 0] = 1.0                                        # head 1 looks at token 0
        a2[0, 0, t, t] = 1.0                                        # head 2 looks at itself
    attn = torch.cat([a1, a2], dim=1)
    target = (a1 + a2).sum(1)[0]                                    # rows: 0.5 / 0.5 (row 0: 1.0)
    scores = torch.log(target.clamp_min(1e-30)).masked_fill(target == 0, NEG_INF)[None]
    assert impl.indexer_kl_loss(attn, scores).abs().item() < 1e-9


def test_kl_sparse_variant_restricts_to_selected_keys(impl):
    B, H, T, S = 1, 2, 6, 6
    g = torch.Generator().manual_seed(5)
    attn = _random_attn(B, H, T, S, g)
    scores = torch.randn(B, T, S, generator=g, dtype=torch.float64).masked_fill(~_causal(B, T, S), NEG_INF)
    k_top = 3
    idx = scores.topk(k_top, dim=-1).indices
    selected = torch.zeros(B, T, S, dtype=torch.bool).scatter_(-1, idx, True) & _causal(B, T, S)
    acc = 0.0
    for t in range(T):
        keys = selected[0, t].nonzero().flatten()
        p = attn[0, :, t, keys].sum(0)
        p = p / p.sum()
        q = scores[0, t, keys].softmax(-1)
        acc += float((p * (p.log() - q.log())).sum())
    assert impl.indexer_kl_loss(attn, scores, selected).item() == pytest.approx(acc, rel=1e-9)
    # zero when the index softmax over the selected set equals the renormalized target
    tgt = attn.sum(1)
    tgt = (tgt * selected).clamp_min(1e-30)
    perfect = torch.log(tgt).masked_fill(~selected, NEG_INF)
    assert impl.indexer_kl_loss(attn, perfect, selected).abs().item() < 1e-9


def test_kl_gradients_flow_only_into_the_indexer(impl):
    B, H, T, S = 1, 2, 4, 4
    g = torch.Generator().manual_seed(6)
    attn = _random_attn(B, H, T, S, g).requires_grad_(True)
    scores = torch.randn(B, T, S, generator=g, dtype=torch.float64).masked_fill(~_causal(B, T, S), NEG_INF)
    scores = scores.detach().requires_grad_(True)
    impl.indexer_kl_loss(attn, scores).backward()
    assert attn.grad is None or float(attn.grad.abs().sum()) == 0.0   # target is a constant
    assert scores.grad is not None and torch.isfinite(scores.grad).all()
    assert float(scores.grad.abs().sum()) > 0
    assert float(scores.grad[0, 0, 1:].abs().sum()) == 0.0            # future keys of query 0 get no gradient


# ----------------------------------------------------------------------------------------
# cost model (page 28 worked numbers)
# ----------------------------------------------------------------------------------------
def test_flops_at_128k_and_1m(impl):
    r = impl.dsa_decode_flops(131072, 2048)
    assert r["dense"] == pytest.approx(278528 * 131072)
    assert r["sparse_core"] == pytest.approx(278528 * 2048)
    assert r["indexer"] == pytest.approx(16384 * 131072)
    assert r["ratio"] == pytest.approx(13.43, abs=0.01)
    r = impl.dsa_decode_flops(2**20, 2048)
    assert r["ratio"] == pytest.approx(16.45, abs=0.01)
    assert r["indexer"] / r["sparse_core"] == pytest.approx(30.1, abs=0.1)


def test_flops_short_context_and_asymptote(impl):
    short = impl.dsa_decode_flops(1024, 2048)                        # fewer keys than k: sparse core = dense
    assert short["sparse_core"] == short["dense"]
    assert short["ratio"] < 1.0                                      # the indexer is pure overhead here
    huge = impl.dsa_decode_flops(10**12, 2048)
    assert huge["ratio"] == pytest.approx(278528 / 16384, rel=1e-3)  # 17x ceiling


def test_break_even_length(impl):
    star = impl.dsa_break_even_length(2048)
    assert star == pytest.approx(2176.0)
    r_lo = impl.dsa_decode_flops(2100, 2048)["ratio"]
    r_hi = impl.dsa_decode_flops(2300, 2048)["ratio"]
    assert r_lo < 1.0 < r_hi
