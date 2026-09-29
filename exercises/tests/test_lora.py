import math

import pytest
import torch
import torch.nn as nn

MODULE = "lora"

# Expected NF4 grid (QLoRA paper, Appendix E). Written out again here so the tests never import solutions.
NF4 = torch.tensor(
    [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
     -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
     0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
     0.7229568362236023, 1.0]
)
MAX_HALF_GAP = float((NF4[1:] - NF4[:-1]).max()) / 2       # 0.1519: worst-case error / absmax


def _layer(d_in=24, d_out=16, bias=True):
    return nn.Linear(d_in, d_out, bias=bias)


# ---------------------------------------------------------------- LoRALinear


def test_zero_init_equals_base(impl):
    base = _layer()
    x = torch.randn(5, 24)
    expected = base(x).detach().clone()
    m = impl.LoRALinear(base, r=4, alpha=8)
    assert m.lora_A.shape == (4, 24) and m.lora_B.shape == (16, 4)
    assert torch.count_nonzero(m.lora_B) == 0
    assert torch.count_nonzero(m.lora_A) > 0
    assert m.lora_A.abs().max() <= 1 / math.sqrt(24) + 1e-7
    assert m.scaling == pytest.approx(2.0)
    assert torch.equal(m(x), expected)                                   # exactly the pretrained function


def test_forward_matches_explicit_formula(impl):
    base = _layer(bias=True)
    m = impl.LoRALinear(base, r=8, alpha=32)
    with torch.no_grad():
        m.lora_B.normal_()
    x = torch.randn(3, 7, 24)                                            # extra leading dims
    W = base.weight.detach() + (32 / 8) * (m.lora_B @ m.lora_A)
    expected = x @ W.T + base.bias.detach()
    torch.testing.assert_close(m(x), expected, atol=1e-5, rtol=1e-5)


def test_changing_r_changes_scaling(impl):
    base = _layer()
    assert impl.LoRALinear(base, r=4, alpha=16).scaling == pytest.approx(4.0)
    assert impl.LoRALinear(base, r=16, alpha=16).scaling == pytest.approx(1.0)


def test_only_lora_params_trainable(impl):
    base = _layer()
    m = impl.LoRALinear(base, r=4, alpha=8)
    assert {n for n, p in m.named_parameters() if p.requires_grad} == {"lora_A", "lora_B"}
    assert not base.weight.requires_grad and not base.bias.requires_grad
    m(torch.randn(6, 24)).square().sum().backward()
    assert base.weight.grad is None and base.bias.grad is None
    assert m.lora_A.grad is not None and m.lora_B.grad is not None


def test_first_step_reaches_only_B(impl):
    # With B = 0 the gradient of A is exactly zero; B gets a nonzero gradient. This is why A can be
    # random and B zero, and why initializing BOTH to zero would never learn.
    m = impl.LoRALinear(_layer(), r=4, alpha=8)
    x = torch.randn(10, 24)
    m(x).square().sum().backward()
    assert torch.count_nonzero(m.lora_A.grad) == 0
    assert m.lora_B.grad.abs().sum() > 0
    # after B moves, A receives gradient too
    with torch.no_grad():
        m.lora_B.add_(0.1 * torch.randn_like(m.lora_B))
    m.lora_A.grad = None
    m(x).square().sum().backward()
    assert m.lora_A.grad.abs().sum() > 0


def test_merge_equals_unmerged_and_unmerge_restores(impl):
    base = _layer()
    m = impl.LoRALinear(base, r=4, alpha=8)
    with torch.no_grad():
        m.lora_B.normal_()
    x = torch.randn(9, 24)
    w0 = base.weight.detach().clone()
    y_unmerged = m(x).detach().clone()
    m.merge()
    assert m.merged
    torch.testing.assert_close(base.weight.detach(), w0 + m.scaling * (m.lora_B @ m.lora_A).detach(), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(m(x), y_unmerged, atol=1e-5, rtol=1e-5)
    m.merge()                                                            # second merge must be a no-op
    torch.testing.assert_close(m(x), y_unmerged, atol=1e-5, rtol=1e-5)
    m.unmerge()
    assert not m.merged
    torch.testing.assert_close(base.weight.detach(), w0, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(m(x), y_unmerged, atol=1e-5, rtol=1e-5)


def test_no_bias_layer(impl):
    base = _layer(bias=False)
    m = impl.LoRALinear(base, r=2, alpha=2)
    with torch.no_grad():
        m.lora_B.normal_()
    x = torch.randn(4, 24)
    y = m(x).detach().clone()
    m.merge()
    torch.testing.assert_close(m(x), y, atol=1e-5, rtol=1e-5)


def test_lora_param_count_llama3_8b(impl):
    d, kv, ff, L = 4096, 1024, 14336, 32
    per_layer = [(d, d), (d, kv), (d, kv), (d, d), (d, ff), (d, ff), (ff, d)]   # q, k, v, o, gate, up, down
    assert impl.lora_param_count(per_layer * L, 16) == 41_943_040
    assert impl.lora_param_count([(d, d)], 16) == 131_072
    assert impl.lora_param_count([(d, d), (d, kv)], 1) == 2 * d + (d + kv)


# ---------------------------------------------------------------- NF4


def test_nf4_exact_on_the_grid(impl):
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, 16, (10, 64), generator=g)
    idx[:, 0] = 15                                                       # every block contains the level +1, so absmax == scale
    scale = torch.rand(10, generator=g) * 3 + 0.1
    w = (NF4[idx] * scale[:, None]).reshape(20, 32)
    codes, absmax = impl.nf4_quantize(w, block=64)
    assert codes.dtype == torch.uint8 and codes.shape == (w.numel(),)
    assert absmax.shape == (10,)
    assert int(codes.max()) <= 15
    torch.testing.assert_close(absmax, scale, atol=1e-6, rtol=1e-6)
    assert torch.equal(codes.long(), idx.reshape(-1))
    w2 = impl.nf4_dequantize(codes, absmax, w.shape, block=64)
    assert w2.shape == w.shape
    torch.testing.assert_close(w2, w, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("block", [64, 32])
def test_nf4_roundtrip_error_bound(impl, block):
    w = torch.randn(37, 53)                                              # 1961 values: needs padding
    codes, absmax = impl.nf4_quantize(w, block=block)
    assert absmax.shape == (math.ceil(w.numel() / block),)
    w2 = impl.nf4_dequantize(codes, absmax, w.shape, block=block)
    assert w2.shape == w.shape
    flat_err = (w - w2).abs().reshape(-1)
    scale_per_elem = absmax.repeat_interleave(block)[: w.numel()]
    # per element: |error| <= (largest half-gap between adjacent levels) * (its block's absmax)
    assert bool((flat_err <= (MAX_HALF_GAP + 1e-6) * scale_per_elem + 1e-7).all())
    assert float(flat_err.max()) > 0                                     # it is a lossy code


def test_nf4_all_zero_block_and_absmax(impl):
    w = torch.zeros(3, 64)
    w[1, 5] = -2.0
    codes, absmax = impl.nf4_quantize(w, block=64)
    assert absmax.tolist() == pytest.approx([0.0, 2.0, 0.0])
    w2 = impl.nf4_dequantize(codes, absmax, w.shape, block=64)
    assert torch.isfinite(w2).all()
    torch.testing.assert_close(w2, w, atol=1e-6, rtol=1e-6)              # -2 is the block absmax: exactly the -1 level


def test_nf4_beats_uniform_grid_on_gaussian(impl):
    torch.manual_seed(0)
    w = torch.randn(64, 256)
    codes, absmax = impl.nf4_quantize(w, block=64)
    err_nf4 = (w - impl.nf4_dequantize(codes, absmax, w.shape, block=64)).pow(2).mean()
    # naive reference: 16 evenly spaced levels on [-1, 1], same blockwise absmax scaling
    blocks = w.reshape(-1, 64)
    am = blocks.abs().amax(1, keepdim=True)
    uni = torch.linspace(-1, 1, 16)
    nearest = ((blocks / am)[..., None] - uni).abs().argmin(-1)
    err_uniform = (blocks - uni[nearest] * am).pow(2).mean()
    assert err_nf4 < err_uniform


def test_nf4_bits_per_param(impl):
    assert impl.nf4_bits_per_param() == pytest.approx(4 + 8 / 64 + 32 / (64 * 256))      # 4.126953125
    assert impl.nf4_bits_per_param() == pytest.approx(4.126953125)
    assert impl.nf4_bits_per_param(double_quant=False) == pytest.approx(4.5)
    assert impl.nf4_bits_per_param(block=128, double_quant=False) == pytest.approx(4.25)
    saved = impl.nf4_bits_per_param(double_quant=False) - impl.nf4_bits_per_param()
    assert saved == pytest.approx(0.373046875)


# ---------------------------------------------------------------- QLoRA forward


def test_qlora_linear_matches_dequant_plus_lora(impl):
    torch.manual_seed(1)
    d_out, d_in, r = 12, 40, 4
    W = torch.randn(d_out, d_in) * 0.1
    bias = torch.randn(d_out)
    codes, absmax = impl.nf4_quantize(W, block=64)
    A = (torch.rand(r, d_in) * 2 - 1) / math.sqrt(d_in)
    B = torch.randn(d_out, r) * 0.1
    x = torch.randn(3, 5, d_in)
    y = impl.qlora_linear(x, codes, absmax, (d_out, d_in), A, B, scaling=2.0, block=64, bias=bias)
    W_deq = impl.nf4_dequantize(codes, absmax, (d_out, d_in), block=64)
    expected = x @ (W_deq + 2.0 * B @ A).T + bias
    torch.testing.assert_close(y, expected, atol=1e-5, rtol=1e-5)
    # the frozen 4-bit base is only an approximation of W, so the result is close to (not equal to) the 16-bit layer
    y16 = x @ (W + 2.0 * B @ A).T + bias
    assert 0 < float((y - y16).abs().max()) < 0.5


def test_qlora_linear_gradients_reach_only_lora(impl):
    d_out, d_in, r = 8, 32, 4
    W = torch.randn(d_out, d_in)
    codes, absmax = impl.nf4_quantize(W, block=64)
    A = torch.randn(r, d_in, requires_grad=True)
    B = torch.randn(d_out, r, requires_grad=True)
    x = torch.randn(6, d_in, requires_grad=True)
    impl.qlora_linear(x, codes, absmax, (d_out, d_in), A, B, scaling=0.5).square().sum().backward()
    assert A.grad is not None and B.grad is not None and x.grad is not None
    assert A.grad.abs().sum() > 0 and B.grad.abs().sum() > 0
    assert not codes.requires_grad
