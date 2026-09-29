import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

MODULE = "attention"


# ------------------------------------------------------------------ softmax
def test_softmax_matches_torch(impl):
    x = torch.randn(3, 4, 5) * 3
    for dim in (-1, 0, 1, 2):
        torch.testing.assert_close(impl.softmax(x, dim=dim), torch.softmax(x, dim=dim), atol=1e-6, rtol=1e-5)


def test_softmax_large_logits_do_not_overflow(impl):
    x = torch.tensor([[1000.0, 1001.0, 1002.0], [-1000.0, -1001.0, -1002.0]])
    out = impl.softmax(x)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.softmax(x, -1), atol=1e-6, rtol=1e-5)


def test_softmax_shift_invariance(impl):
    x = torch.randn(4, 7)
    torch.testing.assert_close(impl.softmax(x + 123.0), impl.softmax(x), atol=1e-5, rtol=1e-4)


def test_softmax_neg_inf_entries_and_fully_masked_rows(impl):
    x = torch.tensor([[0.0, float("-inf"), 1.0], [float("-inf")] * 3])
    out = impl.softmax(x)
    assert out[0, 1].item() == 0.0
    torch.testing.assert_close(out[0, [0, 2]], torch.softmax(torch.tensor([0.0, 1.0]), -1))
    assert torch.equal(out[1], torch.zeros(3)), "an all -inf row must give zeros, not NaN"


def test_softmax_preserves_dtype(impl):
    x = torch.randn(2, 6).to(torch.bfloat16)
    out = impl.softmax(x)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), torch.softmax(x.float(), -1), atol=2e-2, rtol=2e-2)


def test_softmax_gradient(impl):
    x = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    w = torch.randn(3, 5, dtype=torch.float64)
    (impl.softmax(x) * w).sum().backward()
    g = x.grad.clone()
    x.grad = None
    (torch.softmax(x, -1) * w).sum().backward()
    torch.testing.assert_close(g, x.grad)


# ------------------------------------------------------------------ causal_mask
def test_causal_mask_square(impl):
    m = impl.causal_mask(4)
    assert m.dtype == torch.bool and m.shape == (4, 4)
    assert torch.equal(m, torch.tril(torch.ones(4, 4, dtype=torch.bool)))


def test_causal_mask_end_aligned(impl):
    m = impl.causal_mask(2, 5)   # two new queries after a 3-token cache
    expected = torch.tensor([[1, 1, 1, 1, 0],
                             [1, 1, 1, 1, 1]], dtype=torch.bool)
    assert torch.equal(m, expected)


def test_causal_mask_single_decode_step_sees_everything(impl):
    assert impl.causal_mask(1, 9).all()


def test_causal_mask_rejects_more_queries_than_keys(impl):
    with pytest.raises(ValueError):
        impl.causal_mask(5, 3)


# ------------------------------------------------------------------ scaled_dot_product_attention
def test_sdpa_matches_torch_no_mask(impl):
    q, k, v = torch.randn(2, 3, 5, 8), torch.randn(2, 3, 7, 8), torch.randn(2, 3, 7, 8)
    torch.testing.assert_close(impl.scaled_dot_product_attention(q, k, v),
                               F.scaled_dot_product_attention(q, k, v), atol=1e-5, rtol=1e-5)


def test_sdpa_value_dim_differs_from_key_dim(impl):
    q, k, v = torch.randn(2, 4, 8), torch.randn(2, 6, 8), torch.randn(2, 6, 5)
    out = impl.scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 4, 5)
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v), atol=1e-5, rtol=1e-5)


def test_sdpa_random_boolean_mask(impl):
    q, k, v = torch.randn(2, 2, 6, 8), torch.randn(2, 2, 9, 8), torch.randn(2, 2, 9, 8)
    mask = torch.rand(2, 1, 6, 9) > 0.5
    mask[..., 0] = True          # every query keeps at least one key
    torch.testing.assert_close(impl.scaled_dot_product_attention(q, k, v, mask),
                               F.scaled_dot_product_attention(q, k, v, attn_mask=mask), atol=1e-5, rtol=1e-5)


def test_sdpa_end_aligned_causal_mask(impl):
    T, S = 3, 8
    q, k, v = torch.randn(1, 2, T, 4), torch.randn(1, 2, S, 4), torch.randn(1, 2, S, 4)
    mask = torch.arange(S)[None, :] <= (torch.arange(T)[:, None] + S - T)
    torch.testing.assert_close(impl.scaled_dot_product_attention(q, k, v, mask),
                               F.scaled_dot_product_attention(q, k, v, attn_mask=mask), atol=1e-5, rtol=1e-5)


def test_sdpa_large_logits_stay_finite(impl):
    q, k, v = torch.randn(1, 4, 16) * 30, torch.randn(1, 6, 16) * 30, torch.randn(1, 6, 16)
    out = impl.scaled_dot_product_attention(q, k, v)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v), atol=1e-4, rtol=1e-4)


def test_sdpa_fully_masked_row_gives_zeros_and_finite_grads(impl):
    q = torch.randn(1, 3, 4, requires_grad=True)
    k = torch.randn(1, 5, 4, requires_grad=True)
    v = torch.randn(1, 5, 4, requires_grad=True)
    mask = torch.ones(3, 5, dtype=torch.bool)
    mask[1] = False              # query 1 may attend to nothing
    out = impl.scaled_dot_product_attention(q, k, v, mask)
    assert torch.isfinite(out).all()
    assert torch.equal(out[0, 1], torch.zeros(4))
    out.sum().backward()
    for t in (q, k, v):
        assert torch.isfinite(t.grad).all(), "NaN gradient: a fully masked row leaked NaN into the backward pass"


def test_sdpa_uses_scale_one_over_sqrt_d(impl):
    torch.manual_seed(1)
    q, k, v = torch.randn(1, 64), torch.randn(5, 64), torch.randn(5, 64)
    a = torch.softmax((q @ k.T) / 8.0, -1)          # sqrt(64) = 8
    torch.testing.assert_close(impl.scaled_dot_product_attention(q, k, v), a @ v, atol=1e-5, rtol=1e-5)


# ------------------------------------------------------------------ MultiHeadAttention
def _make(impl, d=16, h=4, dtype=torch.float64):
    torch.manual_seed(0)
    return impl.MultiHeadAttention(d, h).to(dtype)


def _ref_mha(m, x, n_heads, causal):
    """Explicit per-head loop written from the definition, using the module's own weights."""
    B, T, d = x.shape
    dh = d // n_heads
    Wq, Wk, Wv, Wo = (getattr(m, n).weight for n in ("q_proj", "k_proj", "v_proj", "o_proj"))  # (out, in)
    q, k, v = x @ Wq.T, x @ Wk.T, x @ Wv.T
    outs = []
    for i in range(n_heads):
        sl = slice(i * dh, (i + 1) * dh)
        z = q[..., sl] @ k[..., sl].transpose(-1, -2) / math.sqrt(dh)
        if causal:
            z = z + torch.triu(torch.full((T, T), float("-inf"), dtype=x.dtype), diagonal=1)
        outs.append(torch.softmax(z, -1) @ v[..., sl])
    return torch.cat(outs, -1) @ Wo.T


def test_mha_shape_and_parameter_count(impl):
    m = impl.MultiHeadAttention(32, 4)
    y = m(torch.randn(2, 5, 32))
    assert y.shape == (2, 5, 32)
    assert sum(p.numel() for p in m.parameters()) == 4 * 32 * 32     # W^Q, W^K, W^V, W^O and no biases


def test_mha_matches_per_head_reference(impl):
    m = _make(impl, 16, 4)
    x = torch.randn(3, 6, 16, dtype=torch.float64)
    for causal in (True, False):
        torch.testing.assert_close(m(x, causal=causal), _ref_mha(m, x, 4, causal))


def test_mha_matches_torch_nn_multihead_attention(impl):
    d, h, T = 16, 4, 7
    m = _make(impl, d, h)
    ref = nn.MultiheadAttention(d, h, bias=False, batch_first=True).double()
    with torch.no_grad():
        ref.in_proj_weight.copy_(torch.cat([m.q_proj.weight, m.k_proj.weight, m.v_proj.weight], 0))
        ref.out_proj.weight.copy_(m.o_proj.weight)
    x = torch.randn(2, T, d, dtype=torch.float64)
    blocked = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)   # nn.MultiheadAttention: True = NOT allowed
    expected, _ = ref(x, x, x, attn_mask=blocked, need_weights=False)
    torch.testing.assert_close(m(x, causal=True), expected)


def test_mha_is_causal(impl):
    m = _make(impl)
    x = torch.randn(2, 8, 16, dtype=torch.float64)
    y = m(x)
    x2 = x.clone()
    x2[:, 5:] = torch.randn_like(x2[:, 5:])          # change the future of positions 0..4
    y2 = m(x2)
    torch.testing.assert_close(y[:, :5], y2[:, :5])
    assert not torch.allclose(y[:, 5:], y2[:, 5:])


def test_mha_without_mask_is_permutation_equivariant(impl):
    m = _make(impl)
    x = torch.randn(2, 6, 16, dtype=torch.float64)
    perm = torch.randperm(6)
    torch.testing.assert_close(m(x[:, perm], causal=False), m(x, causal=False)[:, perm])
    # the causal mask breaks equivariance: order matters
    assert not torch.allclose(m(x[:, perm], causal=True), m(x, causal=True)[:, perm])


def test_mha_single_token(impl):
    m = _make(impl)
    x = torch.randn(2, 1, 16, dtype=torch.float64)
    torch.testing.assert_close(m(x), _ref_mha(m, x, 4, True))


def test_mha_rejects_indivisible_width(impl):
    with pytest.raises(ValueError):
        impl.MultiHeadAttention(10, 4)
