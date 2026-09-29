import torch
import torch.nn.functional as F

MODULE = "gqa"


def _ref(q, k, v, causal):
    # reference: repeat heads explicitly and use PyTorch SDPA with an explicit mask
    B, n_h, T, d = q.shape
    n_kv, S = k.shape[1], k.shape[2]
    g = n_h // n_kv
    k = k.repeat_interleave(g, dim=1)
    v = v.repeat_interleave(g, dim=1)
    mask = None
    if causal:
        qpos = torch.arange(T)[:, None] + (S - T)
        mask = torch.arange(S)[None, :] <= qpos
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def test_repeat_kv_order(impl):
    x = torch.arange(3).float().view(1, 3, 1, 1).expand(2, 3, 5, 4)
    y = impl.repeat_kv(x, 4)
    assert y.shape == (2, 12, 5, 4)
    assert y[0, :, 0, 0].tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    assert torch.equal(impl.repeat_kv(x, 1), x)


def test_gqa_full_sequence(impl):
    q = torch.randn(2, 8, 7, 16)
    k = torch.randn(2, 2, 7, 16)
    v = torch.randn(2, 2, 7, 16)
    for causal in (True, False):
        torch.testing.assert_close(impl.gqa_attention(q, k, v, causal), _ref(q, k, v, causal), atol=1e-5, rtol=1e-5)


def test_gqa_decode_with_cache(impl):
    # one new query attending to a 9-token cache: must see ALL 9 keys (end-aligned mask)
    q = torch.randn(3, 4, 1, 8)
    k = torch.randn(3, 1, 9, 8)
    v = torch.randn(3, 1, 9, 8)
    out = impl.gqa_attention(q, k, v, causal=True)
    torch.testing.assert_close(out, _ref(q, k, v, causal=False), atol=1e-5, rtol=1e-5)


def test_gqa_chunked_prefill(impl):
    q = torch.randn(1, 6, 3, 8)
    k = torch.randn(1, 3, 10, 8)
    v = torch.randn(1, 3, 10, 8)
    torch.testing.assert_close(impl.gqa_attention(q, k, v, True), _ref(q, k, v, True), atol=1e-5, rtol=1e-5)


def test_gqa_equals_mha_when_heads_match(impl):
    q, k, v = (torch.randn(2, 4, 5, 8) for _ in range(3))
    torch.testing.assert_close(impl.gqa_attention(q, k, v, True),
                               F.scaled_dot_product_attention(q, k, v, is_causal=True), atol=1e-5, rtol=1e-5)


def test_mean_pool_conversion(impl):
    d, n_h, n_kv, d_h = 12, 8, 2, 4
    W = torch.randn(d, n_h * d_h)
    Wg = impl.mha_to_gqa_kv_weight(W, n_h, n_kv)
    assert Wg.shape == (d, n_kv * d_h)
    heads = W.view(d, n_h, d_h)
    torch.testing.assert_close(Wg[:, :d_h], heads[:, 0:4].mean(1))
    torch.testing.assert_close(Wg[:, d_h:], heads[:, 4:8].mean(1))
    # identical heads within a group -> conversion is exact
    same = heads[:, ::4].repeat_interleave(4, dim=1).reshape(d, n_h * d_h)
    torch.testing.assert_close(impl.mha_to_gqa_kv_weight(same, n_h, n_kv), heads[:, ::4].reshape(d, n_kv * d_h))
