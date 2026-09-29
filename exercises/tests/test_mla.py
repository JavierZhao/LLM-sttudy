import torch

MODULE = "mla"


def _reference(h, w, impl):
    """Slow, explicit per-head MLA straight from the DeepSeek-V2 equations (no batching tricks)."""
    B, T, d = h.shape
    pos = torch.arange(T)
    c = h @ w.W_dkv
    kr = impl.rope(h @ w.W_kr, pos)
    cq = h @ w.W_dq
    out = torch.zeros(B, T, w.n_h * w.d_v, dtype=h.dtype)
    for i in range(w.n_h):
        sl = slice(i * w.d_nope, (i + 1) * w.d_nope)
        q_c = cq @ w.W_uq[:, sl]
        q_r = impl.rope(cq @ w.W_qr[:, i * w.d_r:(i + 1) * w.d_r], pos)
        k_c = c @ w.W_uk[:, sl]
        v = c @ w.W_uv[:, i * w.d_v:(i + 1) * w.d_v]
        q = torch.cat([q_c, q_r], -1)
        k = torch.cat([k_c, kr], -1)
        s = q @ k.transpose(1, 2) / (w.d_nope + w.d_r) ** 0.5
        s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
        out[:, :, i * w.d_v:(i + 1) * w.d_v] = s.softmax(-1) @ v
    return out @ w.W_o


def test_forward_matches_reference(impl):
    w = impl.init_mla()
    h = torch.randn(2, 7, 64, dtype=torch.float64)
    out, c, kr = impl.mla_forward(h, w)
    assert out.shape == (2, 7, 64) and c.shape == (2, 7, 24) and kr.shape == (2, 7, 8)
    torch.testing.assert_close(out, _reference(h, w, impl))


def test_decode_matches_forward(impl):
    w = impl.init_mla()
    h = torch.randn(3, 9, 64, dtype=torch.float64)
    full, _, _ = impl.mla_forward(h, w)
    # prefill 6 tokens, then decode 3 tokens one at a time with the absorbed path
    _, c, kr = impl.mla_forward(h[:, :6], w)
    for t in range(6, 9):
        y, c, kr = impl.mla_decode_step(h[:, t:t + 1], w, c, kr, pos=t)
        assert y.shape == (3, 1, 64)
        torch.testing.assert_close(y[:, 0], full[:, t])
    assert c.shape == (3, 9, 24) and kr.shape == (3, 9, 8)


def test_cache_contents(impl):
    w = impl.init_mla()
    h = torch.randn(1, 5, 64, dtype=torch.float64)
    _, c, kr = impl.mla_forward(h, w)
    torch.testing.assert_close(c, h @ w.W_dkv)
    torch.testing.assert_close(kr, impl.rope(h @ w.W_kr, torch.arange(5)))


def test_kv_bytes(impl):
    f = impl.kv_cache_bytes_per_token
    assert f("gqa", 32, n_kv=8, d_h=128) == 131072                 # Llama-3-8B, bf16
    assert f("mha", 32, n_h=32, d_h=128) == 524288
    assert f("mqa", 32, d_h=128) == 16384
    assert f("mla", 61, d_c=512, d_r=64) == 70272                  # DeepSeek-V3
    assert f("mla", 61, d_c=512, d_r=64, bytes_per_el=1) == 35136  # fp8 cache
