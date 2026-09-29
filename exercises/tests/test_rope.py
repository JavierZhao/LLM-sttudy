import math

import pytest
import torch

MODULE = "rope"


# ------------------------------------------------------------------ naive references (self-contained)
def _theta(d, base):
    return [base ** (-2 * i / d) for i in range(d // 2)]


def _ref_interleaved(x, pos, base):
    """Complex-number reference: view pairs (x[2i], x[2i+1]) as z = a + ib and multiply by e^{i pos theta_i}."""
    d = x.shape[-1]
    th = torch.tensor(_theta(d, base), dtype=torch.float64)
    ang = pos.to(torch.float64)[:, None] * th[None, :]                     # (T, d/2)
    if x.dim() == 4:
        ang = ang[:, None, :]
    z = torch.complex(x[..., 0::2].double(), x[..., 1::2].double()) * torch.polar(torch.ones_like(ang), ang)
    return torch.stack((z.real, z.imag), dim=-1).flatten(-2).to(x.dtype)


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def _ref_half(x, pos, base):
    """Hugging-Face-style reference: x * cos + rotate_half(x) * sin with cos = cat(freqs, freqs)."""
    d = x.shape[-1]
    th = torch.tensor(_theta(d, base), dtype=torch.float64)
    ang = pos.to(torch.float64)[:, None] * th[None, :]
    emb = torch.cat((ang, ang), dim=-1)                                    # (T, d)
    if x.dim() == 4:
        emb = emb[:, None, :]
    return (x * emb.cos().to(x.dtype) + _rotate_half(x) * emb.sin().to(x.dtype))


def _scores(qr, kr):
    """(B, T, d) rotated q and k -> (B, T, T) raw scores."""
    return qr @ kr.transpose(-1, -2)


# ------------------------------------------------------------------ frequencies
def test_frequencies(impl):
    th = impl.rope_frequencies(128, 10000.0)
    assert th.shape == (64,)
    assert th.dtype == torch.float64
    assert th[0].item() == pytest.approx(1.0)
    assert th[-1].item() == pytest.approx(10000.0 ** (-126 / 128))
    ratios = th[1:] / th[:-1]                                              # geometric progression
    torch.testing.assert_close(ratios, torch.full_like(ratios, 10000.0 ** (-2 / 128)))
    # slowest wavelength for Llama-3's base: 2*pi * b^(126/128) ~ 2.56M tokens
    slow = impl.rope_frequencies(128, 500000.0)[-1].item()
    assert 2 * math.pi / slow == pytest.approx(2.5592e6, rel=1e-3)


# ------------------------------------------------------------------ the two layouts against naive references
@pytest.mark.parametrize("shape", [(2, 7, 3, 16), (2, 7, 16)])
def test_interleaved_matches_complex_reference(impl, shape):
    x = torch.randn(*shape, dtype=torch.float64)
    pos = torch.arange(shape[1])
    torch.testing.assert_close(impl.apply_rope_interleaved(x, pos, 10000.0), _ref_interleaved(x, pos, 10000.0))


@pytest.mark.parametrize("shape", [(2, 7, 3, 16), (2, 7, 16)])
def test_half_matches_rotate_half_reference(impl, shape):
    x = torch.randn(*shape, dtype=torch.float64)
    pos = torch.arange(shape[1])
    torch.testing.assert_close(impl.apply_rope_half(x, pos, 500000.0), _ref_half(x, pos, 500000.0))


def test_position_zero_is_identity_and_dtype_preserved(impl):
    x = torch.randn(1, 3, 2, 8)
    pos = torch.zeros(3, dtype=torch.long)
    for fn in (impl.apply_rope_interleaved, impl.apply_rope_half):
        torch.testing.assert_close(fn(x, pos), x)
    xb = torch.randn(1, 4, 2, 8).to(torch.bfloat16)
    for fn in (impl.apply_rope_interleaved, impl.apply_rope_half):
        assert fn(xb, torch.arange(4)).dtype == torch.bfloat16
        assert fn(x.double(), torch.arange(3)).dtype == torch.float64


def test_angles_are_computed_before_casting_to_low_precision(impl):
    """Angles near 100_000 rad are meaningless in bfloat16: do the trig in high precision, cast cos/sin after."""
    x = torch.randn(1, 4, 2, 16).to(torch.bfloat16)
    pos = torch.tensor([0, 1, 100_000, 100_001])
    for fn, ref in ((impl.apply_rope_interleaved, _ref_interleaved), (impl.apply_rope_half, _ref_half)):
        got = fn(x, pos, 10000.0).float()
        want = ref(x.double(), pos, 10000.0).float()
        torch.testing.assert_close(got, want, atol=6e-2, rtol=0.0)


def test_rotation_preserves_norms(impl):
    x = torch.randn(3, 11, 4, 32, dtype=torch.float64)
    pos = torch.arange(11) * 137                                           # large, irregular positions
    for fn in (impl.apply_rope_interleaved, impl.apply_rope_half):
        y = fn(x, pos)
        torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1))         # whole vector
    # interleaved layout: every 2-D pair keeps its own norm
    y = impl.apply_rope_interleaved(x, pos)
    torch.testing.assert_close(y.view(*y.shape[:-1], 16, 2).norm(dim=-1), x.view(*x.shape[:-1], 16, 2).norm(dim=-1))


# ------------------------------------------------------------------ the relative-position property
@pytest.mark.parametrize("name", ["apply_rope_interleaved", "apply_rope_half"])
def test_scores_depend_only_on_relative_offset(impl, name):
    fn = getattr(impl, name)
    d, T = 32, 9
    q0, k0 = torch.randn(d, dtype=torch.float64), torch.randn(d, dtype=torch.float64)
    q = q0.expand(1, T, d).contiguous()                                    # same content at every position
    k = k0.expand(1, T, d).contiguous()
    pos = torch.arange(T)
    s = _scores(fn(q, pos), fn(k, pos))[0]                                 # (T, T)
    for t in range(T - 1):
        for j in range(T - 1):
            assert s[t, j].item() == pytest.approx(s[t + 1, j + 1].item(), abs=1e-9)   # Toeplitz
    # shifting BOTH positions by the same amount changes nothing (float64: exact to ~1e-9)
    s_shift = _scores(fn(q, pos + 4321), fn(k, pos + 4321))[0]
    torch.testing.assert_close(s, s_shift, atol=1e-8, rtol=1e-8)
    # shifting only one side does change the score
    s_one = _scores(fn(q, pos + 5), fn(k, pos))[0]
    assert not torch.allclose(s, s_one, atol=1e-3)


@pytest.mark.parametrize("name", ["apply_rope_interleaved", "apply_rope_half"])
def test_incremental_positions_match_full_sequence(impl, name):
    """KV-cache style: rotate a prefix, then later chunks with an offset, and get the same tensor."""
    fn = getattr(impl, name)
    x = torch.randn(2, 10, 3, 16, dtype=torch.float64)
    full = fn(x, torch.arange(10))
    part = torch.cat([fn(x[:, :6], torch.arange(6)), fn(x[:, 6:9], torch.arange(6, 9)), fn(x[:, 9:], torch.tensor([9]))], dim=1)
    torch.testing.assert_close(full, part)


def test_4d_equals_per_head_3d(impl):
    x = torch.randn(2, 5, 4, 8, dtype=torch.float64)
    pos = torch.tensor([3, 4, 5, 6, 7])
    for fn in (impl.apply_rope_interleaved, impl.apply_rope_half):
        y = fn(x, pos)
        for h in range(4):
            torch.testing.assert_close(y[:, :, h], fn(x[:, :, h], pos))


# ------------------------------------------------------------------ layouts are equivalent up to a permutation
def test_permutation_is_valid_and_has_expected_form(impl):
    perm = impl.interleaved_to_half_permutation(8)
    assert perm.dtype == torch.long
    assert perm.tolist() == [0, 2, 4, 6, 1, 3, 5, 7]
    assert sorted(impl.interleaved_to_half_permutation(64).tolist()) == list(range(64))


def test_layouts_agree_after_permutation_and_not_before(impl):
    d, T = 16, 12
    perm = impl.interleaved_to_half_permutation(d)
    pos = torch.arange(T)
    q = torch.randn(1, T, d, dtype=torch.float64)
    k = torch.randn(1, T, d, dtype=torch.float64)
    # output equivalence
    torch.testing.assert_close(impl.apply_rope_half(q[..., perm], pos), impl.apply_rope_interleaved(q, pos)[..., perm])
    # attention scores are identical when Q and K are permuted the same way (the permutation can be folded into W_Q, W_K)
    s_int = _scores(impl.apply_rope_interleaved(q, pos), impl.apply_rope_interleaved(k, pos))
    s_half = _scores(impl.apply_rope_half(q[..., perm], pos), impl.apply_rope_half(k[..., perm], pos))
    torch.testing.assert_close(s_int, s_half)
    # mixing the layouts (no permutation) silently gives different scores
    s_mixed = _scores(impl.apply_rope_half(q, pos), impl.apply_rope_half(k, pos))
    assert not torch.allclose(s_int, s_mixed, atol=1e-2)


# ------------------------------------------------------------------ partial RoPE
@pytest.mark.parametrize("interleaved", [False, True])
def test_partial_rope(impl, interleaved):
    x = torch.randn(2, 6, 3, 24, dtype=torch.float64)                      # e.g. d_head = 96, rotary_pct = 0.25 -> 24 dims
    pos = torch.arange(6) + 10
    y = impl.apply_partial_rope(x, pos, 8, 10000.0, interleaved)
    assert y.shape == x.shape
    torch.testing.assert_close(y[..., 8:], x[..., 8:])                     # untouched tail
    base_fn = impl.apply_rope_interleaved if interleaved else impl.apply_rope_half
    torch.testing.assert_close(y[..., :8], base_fn(x[..., :8], pos, 10000.0))   # head slice = 8-dim RoPE
    torch.testing.assert_close(impl.apply_partial_rope(x, pos, 24, 10000.0, interleaved), base_fn(x, pos, 10000.0))


# ------------------------------------------------------------------ sinusoidal
def test_sinusoidal_matches_definition(impl):
    T, d = 20, 16
    pe = impl.sinusoidal_embedding(T, d)
    assert pe.shape == (T, d) and pe.dtype == torch.float32
    for pos in (0, 1, 7, 19):
        for i in range(d // 2):
            w = pos / 10000 ** (2 * i / d)
            assert pe[pos, 2 * i].item() == pytest.approx(math.sin(w), abs=1e-5)
            assert pe[pos, 2 * i + 1].item() == pytest.approx(math.cos(w), abs=1e-5)
    assert pe[0, 0::2].abs().max() == 0 and (pe[0, 1::2] == 1).all()


def test_sinusoidal_dot_products_depend_only_on_offset(impl):
    """PE_p . PE_{p+k} = sum_i cos(k theta_i): the 'linear offset' property in disguise."""
    d = 32
    pe = impl.sinusoidal_embedding(60, d).double()
    th = torch.tensor(_theta(d, 10000.0), dtype=torch.float64)
    for k in (1, 5, 20):
        expected = torch.cos(k * th).sum()
        for p in (0, 3, 30):
            assert (pe[p] @ pe[p + k]).item() == pytest.approx(expected.item(), abs=1e-4)


# ------------------------------------------------------------------ ALiBi
def test_alibi_slopes_power_of_two(impl):
    s8 = impl.alibi_slopes(8)
    assert s8.dtype == torch.float32
    torch.testing.assert_close(s8, torch.tensor([2.0 ** -(i + 1) for i in range(8)]))          # 1/2, 1/4, ..., 1/256
    s16 = impl.alibi_slopes(16)
    torch.testing.assert_close(s16, torch.tensor([2.0 ** (-0.5 * (i + 1)) for i in range(16)]))  # starts at 2^-0.5
    torch.testing.assert_close(impl.alibi_slopes(1), torch.tensor([2.0 ** -8]))
    torch.testing.assert_close(impl.alibi_slopes(2), torch.tensor([2.0 ** -4, 2.0 ** -8]))


def test_alibi_slopes_non_power_of_two(impl):
    s12 = impl.alibi_slopes(12)
    expected = [2.0 ** -(i + 1) for i in range(8)] + [2.0 ** -0.5, 2.0 ** -1.5, 2.0 ** -2.5, 2.0 ** -3.5]
    torch.testing.assert_close(s12, torch.tensor(expected))


def test_alibi_bias_values_and_mask(impl):
    H, T = 8, 6
    b = impl.alibi_bias(H, T)
    assert b.shape == (H, T, T) and b.dtype == torch.float32
    m = impl.alibi_slopes(H)
    for h in range(H):
        for i in range(T):
            for j in range(T):
                if j > i:
                    assert b[h, i, j].item() == -math.inf
                else:
                    assert b[h, i, j].item() == pytest.approx(-m[h].item() * (i - j), rel=1e-6, abs=1e-7)
    torch.testing.assert_close(b[:, 1, 0], -m)                            # distance 1 -> -m_h
    assert (b.diagonal(dim1=1, dim2=2) == 0).all()


def test_alibi_softmax_is_causal_and_recency_biased(impl):
    H, T = 4, 10
    b = impl.alibi_bias(H, T)
    p = torch.softmax(torch.zeros(H, T, T) + b, dim=-1)                    # content-free scores
    torch.testing.assert_close(p.sum(-1), torch.ones(H, T))
    assert (p.triu(1) == 0).all()                                          # no attention to the future
    row = p[:, T - 1]                                                      # last query: weights fall with distance
    assert (row[:, 1:] >= row[:, :-1]).all()
    # steeper slope -> more mass on the newest token
    assert row[0, -1] > row[-1, -1]
    # translation invariance the reference code uses: adding +m*j (a per-row shift of -m*i) gives the same softmax
    m = impl.alibi_slopes(H)
    j = torch.arange(T)[None, None, :].float()
    mask = torch.triu(torch.full((T, T), -math.inf), diagonal=1)
    alt = torch.softmax(torch.zeros(H, T, T) + m[:, None, None] * j + mask, dim=-1)
    torch.testing.assert_close(p, alt, atol=1e-6, rtol=1e-5)
