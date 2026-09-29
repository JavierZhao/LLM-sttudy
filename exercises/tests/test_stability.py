import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

MODULE = "stability"


def _rms_ref(x, gain=None, eps=1e-6):
    x = x.double()
    y = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y if gain is None else y * gain.double()


# ------------------------------------------------------------------- rms_norm
def test_rms_norm_matches_formula(impl):
    x = torch.randn(3, 5, 16) * 7
    g = torch.rand(16) + 0.5
    torch.testing.assert_close(impl.rms_norm(x, g).double(), _rms_ref(x, g), atol=1e-5, rtol=1e-5)
    y = impl.rms_norm(x)
    torch.testing.assert_close(y.pow(2).mean(-1), torch.ones(3, 5), atol=1e-4, rtol=1e-4)


def test_rms_norm_eps_and_dtype(impl):
    z = torch.zeros(2, 8)
    out = impl.rms_norm(z, eps=1e-6)
    assert torch.isfinite(out).all() and out.abs().max() == 0
    x = (torch.randn(4, 32) * 3).to(torch.bfloat16)
    y = impl.rms_norm(x, torch.ones(32))
    assert y.dtype == torch.bfloat16 and y.shape == x.shape
    torch.testing.assert_close(y.float(), _rms_ref(x.float()).float(), atol=3e-2, rtol=3e-2)


def test_rms_norm_eps_inside_sqrt(impl):
    # mean-square ~ 1e-6, the same size as eps: eps added outside the square root gives a visibly different answer
    x = torch.randn(4, 16) * 1e-3
    torch.testing.assert_close(impl.rms_norm(x, eps=1e-6).double(), _rms_ref(x, eps=1e-6), atol=1e-5, rtol=1e-4)


def test_rms_norm_statistics_in_float32(impl):
    # 300^2 = 90,000 overflows float16 (max 65,504): the mean-square must be computed in float32
    x = torch.full((2, 16), 300.0, dtype=torch.float16)
    y = impl.rms_norm(x)
    assert y.dtype == torch.float16 and torch.isfinite(y).all()
    torch.testing.assert_close(y.float(), torch.ones(2, 16), atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------- qk_norm_attention
def _ref_attention(q, k, v, g_q, g_k, causal):
    qh = _rms_ref(q, g_q).to(q.dtype)
    kh = _rms_ref(k, g_k).to(k.dtype)
    return F.scaled_dot_product_attention(qh, kh, v, is_causal=causal)


@pytest.mark.parametrize("causal", [True, False])
def test_qk_norm_attention_matches_reference(impl, causal):
    B, H, T, d = 2, 3, 6, 16
    q, k, v = (torch.randn(B, H, T, d, dtype=torch.float64) for _ in range(3))
    g_q, g_k = torch.rand(d, dtype=torch.float64) + 0.5, torch.rand(d, dtype=torch.float64) + 0.5
    out = impl.qk_norm_attention(q, k, v, g_q, g_k, causal=causal)
    assert out.shape == (B, H, T, d)
    torch.testing.assert_close(out, _ref_attention(q, k, v, g_q, g_k, causal), atol=1e-5, rtol=1e-5)


def test_qk_norm_attention_weights_and_mask(impl):
    q, k, v = (torch.randn(1, 2, 5, 8) for _ in range(3))
    ones = torch.ones(8)
    out, w = impl.qk_norm_attention(q, k, v, ones, ones, causal=True, return_weights=True)
    assert w.shape == (1, 2, 5, 5)
    torch.testing.assert_close(w.sum(-1), torch.ones(1, 2, 5), atol=1e-6, rtol=1e-6)
    assert torch.triu(w, diagonal=1).abs().max() == 0
    torch.testing.assert_close(out, w @ v, atol=1e-6, rtol=1e-6)


def test_qk_norm_is_scale_invariant(impl):
    # this is the whole point: growing q and k cannot sharpen attention
    q, k, v = (torch.randn(1, 2, 7, 16, dtype=torch.float64) for _ in range(3))
    g = torch.ones(16, dtype=torch.float64)
    base = impl.qk_norm_attention(q, k, v, g, g)
    huge = impl.qk_norm_attention(1e4 * q, 3e3 * k, v, g, g)
    torch.testing.assert_close(base, huge, atol=1e-6, rtol=1e-6)
    # plain attention with the same inflated q, k collapses to (near) one-hot rows
    plain = ((1e4 * q) @ (3e3 * k).transpose(-1, -2) / 4.0).masked_fill(
        ~torch.ones(7, 7, dtype=torch.bool).tril(), float("-inf")).softmax(-1)
    assert plain.max(-1).values.min() > 0.999


def test_qk_norm_logit_bound(impl):
    # |logit| <= g^2 * sqrt(d_h), so within a row max weight / min weight <= exp(2 g^2 sqrt(d_h))
    d, g = 16, 1.0
    q, k, v = (torch.randn(2, 2, 12, d) * 100 for _ in range(3))
    gain = torch.full((d,), g)
    _, w = impl.qk_norm_attention(q, k, v, gain, gain, causal=False, return_weights=True)
    ratio = (w.max(-1).values / w.min(-1).values).max().item()
    assert ratio <= math.exp(2 * g * g * math.sqrt(d)) * (1 + 1e-4)


# --------------------------------------------------------------------- z_loss
def test_z_loss_value(impl):
    logits = torch.randn(3, 5, 11) * 4
    ref = 1e-4 * torch.logsumexp(logits.double(), -1).pow(2).mean()
    torch.testing.assert_close(impl.z_loss(logits).double(), ref, atol=1e-9, rtol=1e-5)
    torch.testing.assert_close(impl.z_loss(logits, coef=0.5).double(), 5000 * ref, atol=1e-6, rtol=1e-5)


def test_z_loss_zero_when_normalized_and_shift_sensitive(impl):
    logits = torch.randn(4, 9)
    normalized = logits - torch.logsumexp(logits, -1, keepdim=True)   # log Z = 0
    assert impl.z_loss(normalized).abs() < 1e-10
    # cross entropy cannot see a constant shift of all logits, z-loss can
    assert impl.z_loss(normalized + 10.0) > impl.z_loss(normalized) + 1e-3


def test_z_loss_gradient(impl):
    logits = torch.randn(6, 10, requires_grad=True)
    impl.z_loss(logits, coef=1e-2).backward()
    lz = torch.logsumexp(logits.detach(), -1, keepdim=True)
    expected = 2 * 1e-2 * lz * torch.softmax(logits.detach(), -1) / 6
    torch.testing.assert_close(logits.grad, expected, atol=1e-8, rtol=1e-5)


def test_z_loss_mask_and_bf16(impl):
    logits = torch.randn(2, 4, 7) * 3
    mask = torch.tensor([[True, True, False, False], [True, False, False, False]])
    ref = 1e-4 * torch.logsumexp(logits.double(), -1)[mask].pow(2).mean()
    torch.testing.assert_close(impl.z_loss(logits, mask=mask).double(), ref, atol=1e-9, rtol=1e-5)
    big = (torch.randn(4, 64) * 20 + 150).to(torch.bfloat16)          # coarse spacing near 150
    out = impl.z_loss(big)
    assert out.dtype == torch.float32
    ref = 1e-4 * torch.logsumexp(big.double(), -1).pow(2).mean()
    torch.testing.assert_close(out.double(), ref, atol=1e-6, rtol=1e-5)


# -------------------------------------------------------------------- softcap
def test_softcap_formula_and_bound(impl):
    x = torch.linspace(-300, 300, 601)
    y = impl.softcap(x, 50.0)
    torch.testing.assert_close(y, 50.0 * torch.tanh(x / 50.0), atol=1e-4, rtol=1e-5)
    assert y.abs().max() <= 50.0
    small = torch.linspace(-1, 1, 21)
    torch.testing.assert_close(impl.softcap(small, 30.0), small, atol=1e-3, rtol=1e-3)


def test_softcap_no_overflow_and_gradient(impl):
    z = torch.tensor([5e4, -5e4, 1e6], requires_grad=True)
    out = impl.softcap(z, 50.0)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.detach(), torch.tensor([50.0, -50.0, 50.0]))
    out.sum().backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().max() < 1e-6      # saturated: no signal back
    x = torch.tensor([0.0, 25.0, 100.0], requires_grad=True)
    impl.softcap(x, 50.0).sum().backward()
    expected = 1 - torch.tanh(x.detach() / 50.0) ** 2
    torch.testing.assert_close(x.grad, expected, atol=1e-6, rtol=1e-5)


def test_softcap_dtype(impl):
    x = torch.randn(5).to(torch.bfloat16) * 60
    assert impl.softcap(x, 30.0).dtype == torch.bfloat16


# ----------------------------------------------------------- attention_entropy
def test_entropy_uniform_and_onehot(impl):
    for m in (1, 2, 8, 100):
        p = torch.full((3, m), 1.0 / m)
        torch.testing.assert_close(impl.attention_entropy(p), torch.full((3,), math.log(m)), atol=1e-5, rtol=1e-5)
    onehot = torch.zeros(2, 4, 6)
    onehot[..., 2] = 1.0
    e = impl.attention_entropy(onehot)
    assert e.shape == (2, 4) and torch.isfinite(e).all() and e.abs().max() == 0


def test_entropy_causal_rows_and_gradient(impl):
    T = 6
    scores = torch.randn(1, 2, T, T, requires_grad=True)
    mask = torch.ones(T, T, dtype=torch.bool).tril()
    p = scores.masked_fill(~mask, float("-inf")).softmax(-1)
    e = impl.attention_entropy(p)
    assert e.shape == (1, 2, T)
    torch.testing.assert_close(e[0, 0, 0], torch.tensor(0.0), atol=1e-6, rtol=0)     # first row: one key
    ref = -(p * torch.log(p.clamp_min(1e-30))).sum(-1)
    torch.testing.assert_close(e, ref, atol=1e-5, rtol=1e-5)
    e.sum().backward()
    assert torch.isfinite(scores.grad).all()


def test_entropy_falls_as_logits_grow(impl):
    base = torch.randn(64, 128)
    ents = [impl.attention_entropy((s * base).softmax(-1)).mean().item() for s in (0.1, 1.0, 10.0, 100.0)]
    assert all(a > b for a, b in zip(ents, ents[1:]))
    assert ents[0] < math.log(128) + 1e-6 and ents[-1] < 0.05


# ------------------------------------------------------- MTPDepth and mtp_loss
def _rms(x, g, eps):
    return x / torch.sqrt(x.pow(2).mean() + eps) * g


def _ref_mtp(hidden, targets, mods, embed, head, lam):
    """Loop-based reference of DeepSeek-V3 eq. 21-25 (mean CE over the valid positions of each depth)."""
    B, T, d = hidden.shape
    D = len(mods)
    total = 0.0
    prev = [[hidden[b, i] for i in range(T)] for b in range(B)]
    for k, m in enumerate(mods, start=1):
        n = T - k
        cur = [[None] * n for _ in range(B)]
        losses = []
        for b in range(B):
            for i in range(n):
                e = embed(targets[b, i + k - 1])                      # token x_{i+k}
                cat = torch.cat([_rms(prev[b][i], m.norm_h, m.eps), _rms(e, m.norm_e, m.eps)])
                h = m.block(m.proj(cat))
                cur[b][i] = h
                logits = head(h)
                losses.append(F.cross_entropy(logits[None], targets[b, i + k][None]))
        total = total + torch.stack(losses).mean()
        prev = cur
    return lam / D * total


def _mods(impl, d, D):
    mods = []
    for _ in range(D):
        m = impl.MTPDepth(d, nn.Sequential(nn.Linear(d, d), nn.GELU()))
        with torch.no_grad():                       # non-trivial gains so wrong norm gains are caught
            m.norm_h.copy_(torch.rand(d) + 0.5)
            m.norm_e.copy_(torch.rand(d) + 0.5)
        mods.append(m)
    return mods


def test_mtp_depth_forward(impl):
    d = 8
    m = _mods(impl, d, 1)[0]
    h, e = torch.randn(2, 5, d), torch.randn(2, 5, d)
    out = m(h, e)
    assert out.shape == (2, 5, d)
    both = torch.cat([_rms_ref(h, m.norm_h).float(), _rms_ref(e, m.norm_e).float()], -1)
    torch.testing.assert_close(out, m.block(m.proj(both)), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("D,lam", [(1, 0.3), (2, 0.1)])
def test_mtp_loss_matches_reference(impl, D, lam):
    torch.manual_seed(1)
    B, T, d, V = 2, 7, 8, 13
    embed = nn.Embedding(V, d)
    head = nn.Linear(d, V, bias=False)
    mods = _mods(impl, d, D)
    hidden = torch.randn(B, T, d)
    targets = torch.randint(0, V, (B, T))
    got = impl.mtp_loss(hidden, targets, mods, embed, head, lam=lam)
    assert got.ndim == 0
    ref = _ref_mtp(hidden, targets, mods, embed, head, lam)
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


def test_mtp_loss_shared_parameters_get_gradients(impl):
    torch.manual_seed(2)
    B, T, d, V = 2, 6, 8, 11
    embed = nn.Embedding(V, d)
    head = nn.Linear(d, V, bias=False)
    mods = _mods(impl, d, 2)
    hidden = torch.randn(B, T, d, requires_grad=True)
    targets = torch.randint(0, V, (B, T))
    impl.mtp_loss(hidden, targets, mods, embed, head).backward()
    assert embed.weight.grad is not None and embed.weight.grad.abs().sum() > 0
    assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0
    assert hidden.grad is not None
    assert hidden.grad[:, -1].abs().sum() == 0 and hidden.grad[:, :-1].abs().sum() > 0   # last position has no label k steps ahead
    assert all(m.proj.weight.grad is not None for m in mods)
