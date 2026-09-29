import math

import pytest
import torch
import torch.nn.functional as F

MODULE = "sinks"

GIB = 2**30
MIB = 2**20


# ---------------------------------------------------------------- softmax_with_sink

def _ref_sink(scores, sink):
    # the sink is a fictitious extra key with a zero value: append its logit, softmax, drop the column
    sink = torch.as_tensor(sink, dtype=scores.dtype).expand(*scores.shape[:-1], 1)
    return torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]


def test_sink_minus_inf_recovers_softmax(impl):
    s = torch.randn(2, 3, 4, 7)
    torch.testing.assert_close(impl.softmax_with_sink(s, float("-inf")), s.softmax(-1))
    torch.testing.assert_close(impl.softmax_with_sink(s, torch.tensor(-float("inf"))), s.softmax(-1))


def test_rows_sum_below_one_with_finite_sink(impl):
    s = torch.randn(3, 5, 6, dtype=torch.float64)
    z = 0.7
    p = impl.softmax_with_sink(s, z)
    Z = s.exp().sum(-1)
    torch.testing.assert_close(p.sum(-1), Z / (Z + math.exp(z)))
    assert (p.sum(-1) < 1).all()
    # the weights themselves: exp(s_j) / (Z + e^z)
    torch.testing.assert_close(p, s.exp() / (Z[..., None] + math.exp(z)))


def test_sink_matches_zero_value_key_reference(impl):
    s = torch.randn(2, 4, 3, 9)
    sink = torch.randn(4, 1, 1)                      # one sink logit per head
    torch.testing.assert_close(impl.softmax_with_sink(s, sink), _ref_sink(s, sink), atol=1e-6, rtol=1e-5)


def test_large_sink_removes_all_attention(impl):
    s = torch.randn(2, 5, 5)
    p = impl.softmax_with_sink(s, 60.0)
    assert p.sum(-1).max() < 1e-20
    p = impl.softmax_with_sink(s, -60.0)
    torch.testing.assert_close(p.sum(-1), torch.ones(2, 5))


def test_stable_for_huge_logits_and_masked_rows(impl):
    s = torch.tensor([[1e4, 0.0, -1e4], [float("-inf")] * 3], dtype=torch.float32)
    p = impl.softmax_with_sink(s, 0.0)
    assert torch.isfinite(p).all()
    assert abs(p[0, 0].item() - 1.0) < 1e-6          # e^{1e4} dwarfs the sink
    assert torch.equal(p[1], torch.zeros(3))         # nothing to attend to: all mass on the sink


def test_dtype_preserved(impl):
    s = torch.randn(3, 4, dtype=torch.float64)
    assert impl.softmax_with_sink(s, 0.0).dtype == torch.float64


# ---------------------------------------------------------------- banded_causal_mask

def test_banded_mask_square(impl):
    m = impl.banded_causal_mask(6, 3)
    expected = torch.tensor([
        [1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1],
    ], dtype=torch.bool)
    assert m.dtype == torch.bool and torch.equal(m, expected)


def test_banded_mask_limits(impl):
    T = 7
    assert torch.equal(impl.banded_causal_mask(T, None), torch.tril(torch.ones(T, T, dtype=torch.bool)))
    assert torch.equal(impl.banded_causal_mask(T, T), torch.tril(torch.ones(T, T, dtype=torch.bool)))
    assert torch.equal(impl.banded_causal_mask(T, 1), torch.eye(T, dtype=torch.bool))
    assert impl.banded_causal_mask(T, 4).sum(-1).max().item() == 4     # at most `window` keys per query


def test_banded_mask_end_aligned(impl):
    m = impl.banded_causal_mask(2, 3, S=6)           # queries at absolute positions 4 and 5
    assert m.shape == (2, 6)
    assert m[0].nonzero().flatten().tolist() == [2, 3, 4]
    assert m[1].nonzero().flatten().tolist() == [3, 4, 5]
    m1 = impl.banded_causal_mask(1, None, S=9)       # single decode step sees the whole cache
    assert m1.all() and m1.shape == (1, 9)


# ---------------------------------------------------------------- layer_schedule

def test_layer_schedule_gemma3_gptoss_mistral(impl):
    s = impl.layer_schedule(62, "LLLLLG")
    assert len(s) == 62 and [i for i, c in enumerate(s) if c == "G"] == list(range(5, 60, 6))
    assert s[60:] == ["L", "L"]                      # 62 = 10 * 6 + 2: the model ends on local layers
    g = impl.layer_schedule(36, "LG")
    assert g[0] == "L" and g[1] == "G" and g.count("G") == 18
    assert impl.layer_schedule(32, "L") == ["L"] * 32
    assert impl.layer_schedule(46, "LG").count("G") == 23


def test_layer_schedule_validation(impl):
    for bad in [("", 4), ("LX", 4), ("lg", 4)]:
        with pytest.raises(ValueError):
            impl.layer_schedule(bad[1], bad[0])
    with pytest.raises(ValueError):
        impl.layer_schedule(0, "LG")


# ---------------------------------------------------------------- kv_bytes_hybrid

def test_kv_bytes_gemma3_27b_at_128k(impl):
    gqa = 2 * 16 * 128                                # K and V, 16 KV heads, head dim 128
    hybrid = impl.HybridCfg(62, "LLLLLG", 1024, gqa, gqa)
    allglobal = impl.HybridCfg(62, "G", 1024, gqa, gqa)
    T = 131072
    assert impl.kv_bytes_hybrid(allglobal, T) == 62 * GIB
    assert impl.kv_bytes_hybrid(hybrid, T) == 10 * GIB + 52 * 8192 * 1024
    assert impl.kv_bytes_hybrid(hybrid, T) / GIB == pytest.approx(10.40625)


def test_kv_bytes_short_sequences_do_not_benefit(impl):
    gqa = 2 * 16 * 128
    hybrid = impl.HybridCfg(62, "LLLLLG", 1024, gqa, gqa)
    allglobal = impl.HybridCfg(62, "G", 1024, gqa, gqa)
    for T in (1, 500, 1024):
        assert impl.kv_bytes_hybrid(hybrid, T) == impl.kv_bytes_hybrid(allglobal, T)


def test_kv_bytes_mistral_rolling_buffer(impl):
    e = 2 * 8 * 128
    swa = impl.HybridCfg(32, "L", 4096, e, e)
    full = impl.HybridCfg(32, "G", 4096, e, e)
    assert impl.kv_bytes_hybrid(swa, 32768) == 512 * MIB
    assert impl.kv_bytes_hybrid(full, 32768) == 8 * impl.kv_bytes_hybrid(swa, 32768)   # the paper's 8x


def test_kv_bytes_different_local_and_global_widths(impl):
    # Gemma 4 31B shape: local layers cache K and V (16 x 256 each), global layers cache one
    # tensor (K = V, 4 KV heads of dim 512); int8 cache; 5:1 pattern over 60 layers.
    cfg = impl.HybridCfg(60, "LLLLLG", 1024, 2 * 16 * 256, 4 * 512, bytes_per_elem=1)
    assert impl.kv_bytes_hybrid(cfg, 32768) == 50 * 8192 * 1024 + 10 * 2048 * 32768


def test_kv_bytes_gptoss_120b(impl):
    e = 2 * 8 * 64
    cfg = impl.HybridCfg(36, "LG", 128, e, e)
    T = 131072
    assert impl.kv_bytes_hybrid(cfg, T) == 2 * (18 * e * T + 18 * e * 128)
    assert impl.kv_bytes_hybrid(cfg, T) / GIB == pytest.approx(4.5 + 4.5 / 1024)


# ---------------------------------------------------------------- sink_attention

def _ref_sink_attention(q, k, v, sinks, window):
    B, n_h, T, d = q.shape
    n_kv, S = k.shape[1], k.shape[2]
    g = n_h // n_kv
    kk, vv = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
    s = q @ kk.transpose(-1, -2) / d**0.5
    qpos = torch.arange(T)[:, None] + (S - T)
    kpos = torch.arange(S)[None, :]
    ok = kpos <= qpos
    if window is not None:
        ok = ok & (qpos - kpos < window)
    s = s.masked_fill(~ok, float("-inf"))
    sink_col = sinks.view(1, n_h, 1, 1).expand(B, n_h, T, 1)
    p = torch.softmax(torch.cat([s, sink_col], dim=-1), dim=-1)[..., :-1]
    return p @ vv


@pytest.mark.parametrize("window", [None, 3])
def test_sink_attention_matches_reference(impl, window):
    B, n_h, n_kv, T, S, d = 2, 4, 2, 3, 8, 8
    q = torch.randn(B, n_h, T, d)
    k = torch.randn(B, n_kv, S, d)
    v = torch.randn(B, n_kv, S, d)
    sinks = torch.randn(n_h)
    out = impl.sink_attention(q, k, v, sinks, window=window)
    assert out.shape == (B, n_h, T, d)
    torch.testing.assert_close(out, _ref_sink_attention(q, k, v, sinks, window), atol=1e-5, rtol=1e-5)


def test_sink_attention_minus_inf_is_plain_attention(impl):
    q = torch.randn(1, 4, 5, 8)
    k = torch.randn(1, 2, 5, 8)
    v = torch.randn(1, 2, 5, 8)
    out = impl.sink_attention(q, k, v, torch.full((4,), -float("inf")))
    ref = F.scaled_dot_product_attention(q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), is_causal=True)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_sink_is_a_sigmoid_gate_on_the_head_output(impl):
    # o_sink = o_plain * sigmoid(logsumexp(scores) - z_sink): the sink rescales each head's output
    # by the fraction of the denominator that belongs to real keys.
    torch.manual_seed(1)
    B, n_h, T, d = 1, 2, 4, 8
    q, k, v = (torch.randn(B, n_h, T, d, dtype=torch.float64) for _ in range(3))
    sinks = torch.tensor([0.3, -1.2], dtype=torch.float64)
    plain = impl.sink_attention(q, k, v, torch.full((n_h,), -float("inf"), dtype=torch.float64))
    withsink = impl.sink_attention(q, k, v, sinks)
    s = (q @ k.transpose(-1, -2) / d**0.5).masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool)), float("-inf"))
    gate = torch.sigmoid(torch.logsumexp(s, -1) - sinks.view(1, n_h, 1))
    torch.testing.assert_close(withsink, plain * gate[..., None])


# ---------------------------------------------------------------- MXFP4

GRID = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _block(vals):
    vals = list(vals) + [0.0] * (32 - len(vals))
    return torch.tensor(vals, dtype=torch.float32)


def test_mxfp4_grid_values_are_fixed_points(impl):
    x = _block(GRID + [-g for g in GRID] + [0.0])        # max 6 => X = 2^(2 - 2) = 1
    torch.testing.assert_close(impl.mxfp4_quantize(x), x)
    y = 8 * x                                             # max 48 => X = 2^(5 - 2) = 8
    torch.testing.assert_close(impl.mxfp4_quantize(y), y)


def test_mxfp4_power_of_two_scale_and_clamping(impl):
    x = _block([12.0, 1.0, -3.1])                          # X = 2^(3-2) = 2: 12 -> 6*2, 1/2 = 0.5 -> 0.5*2, -3.1/2 = -1.55 -> -1.5*2
    out = impl.mxfp4_quantize(x)
    assert out[0].item() == 12.0 and out[1].item() == 1.0 and out[2].item() == -3.0
    z = _block([7.9, 1.0])                                 # floor(log2 7.9) = 2 => X = 1; 7.9 is clamped to 6, not rounded up
    assert impl.mxfp4_quantize(z)[0].item() == 6.0
    assert torch.equal(impl.mxfp4_quantize(torch.zeros(64)), torch.zeros(64))


def test_mxfp4_blocks_are_independent_and_error_is_bounded(impl):
    torch.manual_seed(0)
    x = torch.randn(4, 64) * torch.tensor([1.0, 10.0, 0.01, 100.0]).view(4, 1)
    out = impl.mxfp4_quantize(x)
    assert out.shape == x.shape and out.dtype == x.dtype
    xb, ob = x.reshape(-1, 32), out.reshape(-1, 32)
    X = 2.0 ** (torch.floor(torch.log2(xb.abs().amax(-1, keepdim=True))) - 2)
    assert ((xb - ob).abs() <= 2 * X + 1e-6).all()         # <= 1 grid step (2X) even when the block max is clamped
    assert (torch.sign(ob) * torch.sign(xb) >= 0).all()
    torch.testing.assert_close(impl.mxfp4_quantize(out), out)   # idempotent


# ---------------------------------------------------------------- weight_bytes_mxfp4

def test_weight_bytes_mxfp4_gptoss_120b(impl):
    assert impl.weight_bytes_mxfp4(8, 0) == pytest.approx(4.25)
    assert impl.weight_bytes_mxfp4(0, 5) == pytest.approx(10.0)
    # experts 114.71B params in MXFP4, attention + embeddings 0.96B + 1.16B in bf16 (model card Table 1)
    total = impl.weight_bytes_mxfp4(114.71e9, 0.96e9 + 1.16e9)
    assert total / GIB == pytest.approx(60.70, abs=0.01)
    assert total / GIB == pytest.approx(60.8, rel=0.003)   # the card lists 60.8 GiB
    assert 116.83e9 * 2 / total == pytest.approx(3.58, abs=0.01)
