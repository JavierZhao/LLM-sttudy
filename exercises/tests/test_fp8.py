import pytest
import torch

MODULE = "fp8"


def _expand(s, g):
    # self-contained reference expansion of group scales to element scales
    return s.repeat_interleave(g[0], 0).repeat_interleave(g[1], 1)


def _bulk_rel_rms(x, xh, mask):
    d = (xh - x)[mask]
    return (d.pow(2).mean().sqrt() / x[mask].pow(2).mean().sqrt()).item()


# ---------------------------------------------------------------- rounding primitives

def test_round_to_bits_known_values(impl):
    x = torch.tensor([1.0, 1.0 + 2.0**-10, 1.0 + 2.0**-4, 1.0 + 3 * 2.0**-5, -1.0 - 2.0**-10, 0.0, 6.0e-30])
    y = impl.round_to_bits(x, 4)
    # 4 significant bits: spacing 2**-3 in [1, 2)
    assert y[0] == 1.0 and y[1] == 1.0 and y[5] == 0.0
    assert y[2] == 1.0                      # 1 + 1/16 is a tie between 1.0 and 1.125 -> even mantissa (1.0)
    assert y[3] == 1.125                    # 1 + 3/32 is closer to 1.125 than to 1.0
    assert y[4] == -1.0                     # sign preserved
    assert torch.isfinite(y).all()


def test_round_to_bits_matches_bfloat16_and_identity(impl):
    x = torch.randn(4000) * torch.exp2(torch.randint(-20, 20, (4000,)).float())
    torch.testing.assert_close(impl.round_to_bits(x, 8), x.to(torch.bfloat16).float(), rtol=0, atol=0)
    torch.testing.assert_close(impl.round_to_bits(x, 24), x, rtol=0, atol=0)


def test_round_to_e4m3_known_values(impl):
    x = torch.tensor([0.0, 1.0, 448.0, 1000.0, -1000.0, 1.0625, 1.1875, 2.0**-9, 2.0**-10, 0.75 * 2.0**-9, 2.0**-6])
    y = impl.round_to_e4m3(x)
    assert y[1] == 1.0 and y[2] == 448.0
    assert y[3] == 448.0 and y[4] == -448.0     # saturation, not NaN
    assert y[5] == 1.0                           # tie between 1.0 and 1.125 goes to the even mantissa
    assert y[6] == 1.25                          # tie between 1.125 and 1.25 goes to the even mantissa
    assert y[7] == 2.0**-9                       # smallest subnormal is representable
    assert y[8] == 0.0                           # half of the smallest subnormal: tie to even (zero)
    assert y[9] == 2.0**-9                       # 0.75 * 2**-9 rounds up to 2**-9
    assert y[10] == 2.0**-6                      # smallest normal


def test_round_to_e4m3_matches_torch_float8(impl):
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("this torch build has no float8_e4m3fn")
    g = torch.Generator().manual_seed(1)
    x = torch.randn(20000, generator=g) * torch.exp2(torch.randint(-14, 11, (20000,), generator=g).float())
    ref = x.clamp(-448, 448).to(torch.float8_e4m3fn).float()
    assert torch.equal(impl.round_to_e4m3(x), ref)


def test_e4m3_has_127_nonnegative_values(impl):
    grid = torch.linspace(0, 460, 400001)
    vals = torch.unique(impl.round_to_e4m3(grid))
    assert len(vals) == 127 and vals.max() == 448.0     # 0, 7 subnormals, 119 normals (S.1111.111 is NaN)


# ---------------------------------------------------------------- scaling granularity

def test_per_tensor_scale_and_range(impl):
    x = torch.randn(64, 64) * 3
    q, s = impl.quantize_per_tensor(x)
    assert s.ndim == 0
    torch.testing.assert_close(s, x.abs().max() / 448.0)
    assert q.abs().max() == 448.0                       # the amax element maps to the top of the range
    assert torch.equal(q, impl.round_to_e4m3(q))        # entries are E4M3 values
    z, sz = impl.quantize_per_tensor(torch.zeros(4, 4))
    assert sz == 1.0 and (z == 0).all()


def test_tile_and_block_shapes_and_scales(impl):
    x = torch.randn(4, 256)
    q, s = impl.quantize_tiles(x, (1, 128))
    assert q.shape == (4, 256) and s.shape == (4, 2)
    torch.testing.assert_close(s[1, 1], x[1, 128:].abs().max() / 448.0)
    w = torch.randn(256, 384)
    qw, sw = impl.quantize_blocks(w)
    assert qw.shape == (256, 384) and sw.shape == (2, 3)
    torch.testing.assert_close(sw[1, 2], w[128:, 256:].abs().max() / 448.0)
    assert qw.abs().max() <= 448.0


def test_dequantize_roundtrip_error_is_small(impl):
    x = torch.randn(8, 256)
    q, s = impl.quantize_tiles(x)
    xh = impl.dequantize(q, s, (1, 128))
    rel = ((xh - x).norm() / x.norm()).item()
    assert 0.005 < rel < 0.05                           # 3 mantissa bits: a few percent, not exact, not garbage
    qp, sp = impl.quantize_per_tensor(x)
    torch.testing.assert_close(impl.dequantize(qp, sp), qp * sp)
    z = torch.zeros(2, 128)
    zq, zs = impl.quantize_tiles(z)
    assert (impl.dequantize(zq, zs, (1, 128)) == 0).all()


def test_fine_grained_scales_beat_per_tensor_with_token_outliers(impl):
    # 8 whole tokens are 1e5 times larger (token-correlated outliers, like activation gradients)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(256, 256, generator=g)
    rows = torch.randperm(256, generator=g)[:8]
    x[rows] *= 1e5
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[rows] = False
    qt, st = impl.quantize_per_tensor(x)
    qb, sb = impl.quantize_tiles(x, (128, 128))
    ql, sl = impl.quantize_tiles(x, (1, 128))
    e_tensor = _bulk_rel_rms(x, impl.dequantize(qt, st), mask)
    e_block = _bulk_rel_rms(x, impl.dequantize(qb, sb, (128, 128)), mask)
    e_tile = _bulk_rel_rms(x, impl.dequantize(ql, sl, (1, 128)), mask)
    assert e_tile < 0.04                                # tiles stay at the E4M3 noise floor (about 2.6%)
    assert e_tile < e_block < e_tensor                  # tiles isolate a bad token; blocks and tensors do not
    assert e_tensor > 5 * e_tile


def test_block_scales_confine_a_weight_outlier_to_its_block(impl):
    g = torch.Generator().manual_seed(2)
    w = torch.randn(256, 256, generator=g)
    w[0, 0] = 2e5                                        # one huge weight in block (0, 0)
    mask = torch.ones_like(w, dtype=torch.bool)
    mask[:128, :128] = False                             # judge the other three blocks
    qt, st = impl.quantize_per_tensor(w)
    qb, sb = impl.quantize_blocks(w)
    e_tensor = _bulk_rel_rms(w, impl.dequantize(qt, st), mask)
    e_block = _bulk_rel_rms(w, impl.dequantize(qb, sb, (128, 128)), mask)
    assert e_block < 0.04                                # untouched blocks keep the E4M3 noise floor
    assert e_tensor > 5 * e_block                        # one outlier degrades the whole tensor under one scale


# ---------------------------------------------------------------- GEMM with promotion

def _quantized_pair(M=6, K=384, N=256, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(M, K, generator=g), torch.randn(K, N, generator=g)


def test_tensor_core_mma_exact_and_narrow(impl):
    a, b = _quantized_pair(4, 128, 8)
    torch.testing.assert_close(impl.tensor_core_mma(a, b), a @ b)
    torch.testing.assert_close(impl.tensor_core_mma(a, b, acc_bits=24), a @ b, rtol=1e-4, atol=1e-4)
    err_narrow = (impl.tensor_core_mma(a, b, acc_bits=6) - a @ b).norm()
    err_wide = (impl.tensor_core_mma(a, b, acc_bits=14) - a @ b).norm()
    assert err_narrow > 10 * err_wide > 0               # a narrower accumulator is visibly worse


def test_fp8_matmul_equals_dequantized_matmul(impl):
    a, w = _quantized_pair()
    aq, asc = impl.quantize_tiles(a, (1, 128))
    wq, wsc = impl.quantize_blocks(w)
    ref = (aq * _expand(asc, (1, 128))) @ (wq * _expand(wsc, (128, 128)))
    out = impl.fp8_matmul(aq, asc, wq, wsc)
    assert out.shape == (6, 256) and out.dtype == torch.float32
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
    for pe in (64, 32):                                  # chunking only changes the order of the FP32 adds
        torch.testing.assert_close(impl.fp8_matmul(aq, asc, wq, wsc, promote_every=pe), ref, rtol=1e-4, atol=1e-4)


def test_fp8_matmul_close_to_fp32_reference(impl):
    a, w = _quantized_pair()
    aq, asc = impl.quantize_tiles(a, (1, 128))
    wq, wsc = impl.quantize_blocks(w)
    out = impl.fp8_matmul(aq, asc, wq, wsc)
    rel = ((out - a @ w).norm() / (a @ w).norm()).item()
    assert rel < 0.06                                    # two 3-bit-mantissa operands: a few percent in total
    assert rel > 0.005                                   # and it is not secretly exact


def test_fp8_matmul_rejects_bad_promotion_interval(impl):
    a, w = _quantized_pair(2, 128, 128)
    aq, asc = impl.quantize_tiles(a, (1, 128))
    wq, wsc = impl.quantize_blocks(w)
    with pytest.raises(ValueError):
        impl.fp8_matmul(aq, asc, wq, wsc, promote_every=96)


def test_promotion_repairs_a_narrow_accumulator(impl):
    # K = 2048, unit scales: isolate the accumulation effect from quantization error.
    g = torch.Generator().manual_seed(4)
    M, K, N = 8, 2048, 128
    a = impl.round_to_e4m3(torch.randn(M, K, generator=g))
    w = impl.round_to_e4m3(torch.randn(K, N, generator=g))
    exact = a.double() @ w.double()
    ones_a, ones_w = torch.ones(M, K // 128), torch.ones(K // 128, N // 128)
    naive = impl.tensor_core_mma(a, w, acc_bits=12).double()                       # one long narrow chain
    promoted = impl.fp8_matmul(a, ones_a, w, ones_w, promote_every=128, acc_bits=12).double()
    e_naive = ((naive - exact).norm() / exact.norm()).item()
    e_prom = ((promoted - exact).norm() / exact.norm()).item()
    assert e_prom < e_naive / 2                          # promotion every 128 elements cuts the error several-fold


def test_fp8_linear_matches_manual_pipeline(impl):
    a, w = _quantized_pair(4, 256, 128, seed=5)
    out = impl.fp8_linear(a, w)
    aq, asc = impl.quantize_tiles(a, (1, 128))
    wq, wsc = impl.quantize_blocks(w)
    torch.testing.assert_close(out, impl.fp8_matmul(aq, asc, wq, wsc))
    rel = ((out - a @ w).norm() / (a @ w).norm()).item()
    assert 0.005 < rel < 0.06
