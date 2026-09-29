import contextlib
from unittest import mock

import pytest
import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

MODULE = "flash"
NEG_INF = float("-inf")


# ---------------------------------------------------------------- helpers
def _dense(q, k, v, causal, scale=None):
    """Reference attention that materializes the whole score matrix."""
    d = q.shape[-1]
    scale = d ** -0.5 if scale is None else scale
    s = (q.double() @ k.double().transpose(-1, -2)) * scale
    Tq, Tk = s.shape[-2:]
    if causal:
        qpos = torch.arange(Tq)[:, None] + (Tk - Tq)
        s = s.masked_fill(torch.arange(Tk)[None, :] > qpos, NEG_INF)
    return torch.softmax(s, -1) @ v.double(), torch.logsumexp(s, -1)


def _qkv(B=2, H=3, Tq=37, Tk=37, d=16, dtype=torch.float32):
    return (torch.randn(B, H, Tq, d, dtype=dtype), torch.randn(B, H, Tk, d, dtype=dtype),
            torch.randn(B, H, Tk, d, dtype=dtype))


@contextlib.contextmanager
def _forbid_softmax():
    """Make the library softmax / logsumexp / SDPA raise while the implementation runs."""
    def boom(*a, **k):
        raise AssertionError("softmax/logsumexp/SDPA is not allowed here: build it from running statistics")
    targets = [(torch, "softmax"), (torch, "log_softmax"), (torch, "logsumexp"),
               (F, "softmax"), (F, "log_softmax"), (F, "scaled_dot_product_attention"),
               (torch.Tensor, "softmax"), (torch.Tensor, "log_softmax"), (torch.Tensor, "logsumexp")]
    with contextlib.ExitStack() as st:
        for obj, name in targets:
            st.enter_context(mock.patch.object(obj, name, boom))
        yield


class _SizeSpy(TorchFunctionMode):
    """Records the largest tensor produced by any torch operation."""

    def __init__(self):
        super().__init__()
        self.max_numel = 0
        self.total_numel = 0                         # rough measure of the work done

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for o in (out if isinstance(out, (tuple, list)) else (out,)):
            if isinstance(o, torch.Tensor):
                self.max_numel = max(self.max_numel, o.numel())
                self.total_numel += o.numel()
        return out


# ---------------------------------------------------------------- online softmax
def test_merge_stats_matches_direct_and_is_associative(impl):
    s = [torch.randn(4, n) * 3 for n in (5, 9, 2)]
    stats = [(x.amax(-1), torch.exp(x - x.amax(-1, keepdim=True)).sum(-1)) for x in s]
    allx = torch.cat(s, -1)
    m_ref = allx.amax(-1)
    l_ref = torch.exp(allx - m_ref[:, None]).sum(-1)
    (ma, la), (mb, lb), (mc, lc) = stats
    m1, l1 = impl.merge_softmax_stats(*impl.merge_softmax_stats(ma, la, mb, lb), mc, lc)
    m2, l2 = impl.merge_softmax_stats(ma, la, *impl.merge_softmax_stats(mb, lb, mc, lc))
    for m, l in ((m1, l1), (m2, l2)):
        torch.testing.assert_close(m, m_ref)
        torch.testing.assert_close(l, l_ref, rtol=1e-5, atol=1e-6)


def test_merge_stats_empty_set_is_identity(impl):
    m = torch.tensor([1.5, -2.0])
    l = torch.tensor([3.0, 0.5])
    empty_m = torch.full((2,), NEG_INF)
    empty_l = torch.zeros(2)
    for args in ((m, l, empty_m, empty_l), (empty_m, empty_l, m, l)):
        m_out, l_out = impl.merge_softmax_stats(*args)
        torch.testing.assert_close(m_out, m)
        torch.testing.assert_close(l_out, l)
    m_out, l_out = impl.merge_softmax_stats(empty_m, empty_l, empty_m, empty_l)
    assert torch.isfinite(l_out).all() and (l_out == 0).all()      # no NaN from -inf - -inf


def test_online_softmax_matches_softmax(impl):
    x = torch.randn(3, 5, 37)
    blocks = list(torch.split(x, [1, 10, 11, 15], dim=-1))
    with _forbid_softmax():
        y = impl.online_softmax(blocks)
    torch.testing.assert_close(y, torch.softmax(x, -1), rtol=1e-5, atol=1e-6)
    assert y.shape == x.shape


def test_online_softmax_is_stable_for_huge_logits_and_masked_entries(impl):
    x = torch.randn(4, 24) * 300                     # exp(x) alone would overflow float32
    x[:, 3:9] = NEG_INF                              # a run of masked entries
    x[0, :16] = NEG_INF                              # row 0: the whole first block is masked
    blocks = list(torch.split(x, [8, 8, 8], dim=-1))
    with _forbid_softmax():
        y = impl.online_softmax(blocks)
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y, torch.softmax(x, -1), rtol=1e-5, atol=1e-6)


def test_online_softmax_later_block_dominates(impl):
    # the running max changes a lot between blocks: the rescale factor exp(m_old - m_new) is tiny
    x = torch.cat([torch.zeros(2, 6), torch.full((2, 6), 60.0) + torch.randn(2, 6)], dim=-1)
    y = impl.online_softmax(list(torch.split(x, 6, dim=-1)))
    torch.testing.assert_close(y, torch.softmax(x, -1), rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------- forward
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("bq,bk", [(4, 4), (8, 16), (16, 8), (7, 5), (64, 64)])
def test_forward_matches_dense(impl, causal, bq, bk):
    q, k, v = _qkv()                                 # T = 37 is not a multiple of any block size
    with _forbid_softmax():
        out, lse = impl.flash_attention_forward(q, k, v, bq, bk, causal)
    ref_out, ref_lse = _dense(q, k, v, causal)
    assert out.shape == q.shape and lse.shape == q.shape[:-1]
    torch.testing.assert_close(out.double(), ref_out, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(lse.double(), ref_lse, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("Tq,Tk", [(5, 23), (1, 40), (12, 12)])
def test_forward_causal_is_end_aligned(impl, Tq, Tk):
    q, k, v = _qkv(B=1, H=2, Tq=Tq, Tk=Tk, d=8)
    out, lse = impl.flash_attention_forward(q, k, v, block_q=4, block_k=6, causal=True)
    ref_out, ref_lse = _dense(q, k, v, True)
    torch.testing.assert_close(out.double(), ref_out, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(lse.double(), ref_lse, rtol=1e-4, atol=1e-5)
    if Tq == 1:                                      # decoding: the new token sees every cached key
        torch.testing.assert_close(out.double(), _dense(q, k, v, False)[0], rtol=1e-4, atol=1e-5)


def test_forward_custom_scale_and_large_scores(impl):
    q, k, v = _qkv(B=1, H=2, Tq=20, Tk=20, d=8)
    q = q * 40                                       # logits of magnitude ~100s
    out, lse = impl.flash_attention_forward(q, k, v, 6, 6, True, scale=0.7)
    ref_out, ref_lse = _dense(q, k, v, True, scale=0.7)
    assert torch.isfinite(out).all() and torch.isfinite(lse).all()
    torch.testing.assert_close(out.double(), ref_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(lse.double(), ref_lse, rtol=1e-4, atol=1e-4)


def test_forward_low_precision_inputs_keep_dtype(impl):
    q, k, v = _qkv(B=1, H=2, Tq=24, Tk=24, d=16, dtype=torch.bfloat16)
    out, lse = impl.flash_attention_forward(q, k, v, 8, 8, True)
    assert out.dtype == torch.bfloat16
    assert lse.dtype in (torch.float32, torch.float64)
    ref_out, _ = _dense(q, k, v, True)
    torch.testing.assert_close(out.double(), ref_out, rtol=0, atol=3e-2)


def test_forward_never_materializes_the_full_score_matrix(impl):
    B, H, T, d, bq, bk = 1, 2, 64, 4, 8, 8
    q, k, v = _qkv(B, H, T, T, d)
    with _SizeSpy() as spy:
        impl.flash_attention_forward(q, k, v, bq, bk, True)
    limit = B * H * max(bq * bk, T * d)              # a score tile, or one (T, d) accumulator
    assert spy.max_numel <= limit, f"largest intermediate has {spy.max_numel} elements; tiles allow {limit}"
    assert B * H * T * T > limit                     # sanity: T x T would have violated this


def test_forward_causal_skips_fully_masked_tiles(impl):
    # With a causal mask about half of the (query tile, key tile) pairs lie above the diagonal. Visiting them and
    # masking afterwards is correct but does roughly twice the work; the spec says to skip them.
    q, k, v = _qkv(1, 1, 128, 128, 4)
    with _SizeSpy() as dense:
        impl.flash_attention_forward(q, k, v, 8, 8, False)
    with _SizeSpy() as causal:
        impl.flash_attention_forward(q, k, v, 8, 8, True)
    assert causal.total_numel < 0.85 * dense.total_numel, "causal=True must skip key tiles above the diagonal"


# ---------------------------------------------------------------- backward
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("bq,bk", [(5, 7), (16, 16)])
def test_backward_matches_autograd(impl, causal, bq, bk):
    q, k, v = (t.double().requires_grad_() for t in _qkv(B=2, H=2, Tq=21, Tk=21, d=8))
    dout = torch.randn(2, 2, 21, 8, dtype=torch.double)
    s = (q @ k.transpose(-1, -2)) * 8 ** -0.5
    if causal:
        s = s.masked_fill(torch.arange(21)[None, :] > torch.arange(21)[:, None], NEG_INF)
    ref_out = torch.softmax(s, -1) @ v
    ref_dq, ref_dk, ref_dv = torch.autograd.grad(ref_out, (q, k, v), dout)

    qd, kd, vd = q.detach(), k.detach(), v.detach()
    out, lse = impl.flash_attention_forward(qd, kd, vd, bq, bk, causal)
    dq, dk, dv = impl.flash_attention_backward(qd, kd, vd, out, lse, dout, bq, bk, causal)
    for got, ref in ((dq, ref_dq), (dk, ref_dk), (dv, ref_dv)):
        torch.testing.assert_close(got, ref, rtol=1e-8, atol=1e-9)


def test_backward_cross_length_causal(impl):
    q, k, v = (t.double().requires_grad_() for t in _qkv(B=1, H=2, Tq=6, Tk=19, d=8))
    dout = torch.randn(1, 2, 6, 8, dtype=torch.double)
    s = (q @ k.transpose(-1, -2)) * 8 ** -0.5
    s = s.masked_fill(torch.arange(19)[None, :] > torch.arange(6)[:, None] + 13, NEG_INF)
    ref = torch.autograd.grad(torch.softmax(s, -1) @ v, (q, k, v), dout)
    qd, kd, vd = q.detach(), k.detach(), v.detach()
    out, lse = impl.flash_attention_forward(qd, kd, vd, 4, 5, True)
    got = impl.flash_attention_backward(qd, kd, vd, out, lse, dout, 4, 5, True)
    for g, r in zip(got, ref):
        torch.testing.assert_close(g, r, rtol=1e-8, atol=1e-9)


def test_backward_never_materializes_the_full_score_matrix(impl):
    B, H, T, d, bq, bk = 1, 2, 64, 4, 8, 8
    q, k, v = _qkv(B, H, T, T, d)
    out, lse = impl.flash_attention_forward(q, k, v, bq, bk, True)
    dout = torch.randn_like(out)
    with _SizeSpy() as spy:
        impl.flash_attention_backward(q, k, v, out, lse, dout, bq, bk, True)
    limit = B * H * max(bq * bk, T * d)
    assert spy.max_numel <= limit, f"largest intermediate has {spy.max_numel} elements; tiles allow {limit}"


# ---------------------------------------------------------------- split-KV (FlashDecoding)
@pytest.mark.parametrize("num_splits", [1, 2, 3, 7, 40])
@pytest.mark.parametrize("Tq", [1, 4])
def test_split_kv_attention_matches_dense(impl, num_splits, Tq):
    q, k, v = _qkv(B=2, H=3, Tq=Tq, Tk=40, d=16)
    out = impl.split_kv_attention(q, k, v, num_splits)
    ref, _ = _dense(q, k, v, False)
    torch.testing.assert_close(out.double(), ref, rtol=1e-4, atol=1e-5)


def test_combine_partial_attention_is_exact(impl):
    q, k, v = _qkv(B=1, H=2, Tq=3, Tk=31, d=8)
    cuts = [0, 4, 5, 20, 31]                         # uneven chunks, one of length 1
    parts = [impl.flash_attention_forward(q, k[:, :, a:b], v[:, :, a:b]) for a, b in zip(cuts[:-1], cuts[1:])]
    out, lse = impl.combine_partial_attention([p[0] for p in parts], [p[1] for p in parts])
    ref_out, ref_lse = _dense(q, k, v, False)
    torch.testing.assert_close(out.double(), ref_out, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(lse.double(), ref_lse, rtol=1e-4, atol=1e-5)


def test_combine_handles_chunks_with_very_different_scores(impl):
    q, k, v = _qkv(B=1, H=1, Tq=2, Tk=16, d=8)
    k = k.clone()
    k[:, :, 8:] *= 25                                # the second chunk holds almost all the probability mass
    parts = [impl.flash_attention_forward(q, k[:, :, a:b], v[:, :, a:b]) for a, b in ((0, 8), (8, 16))]
    out, _ = impl.combine_partial_attention([p[0] for p in parts], [p[1] for p in parts])
    torch.testing.assert_close(out.double(), _dense(q, k, v, False)[0], rtol=1e-4, atol=1e-5)


def test_combine_is_stable_for_large_logsumexp(impl):
    # lse values in the hundreds (large logits): exp(lse) overflows float32, so log(sum(exp(lse))) would be inf
    outs = [torch.randn(2, 3, 5, 4) for _ in range(3)]
    lses = [200.0 + 10 * torch.randn(2, 3, 5) for _ in range(3)]
    out, lse = impl.combine_partial_attention(outs, lses)
    stacked = torch.stack(lses).double()
    ref_lse = torch.logsumexp(stacked, 0)
    ref_out = (torch.stack(outs).double() * torch.exp(stacked - ref_lse)[..., None]).sum(0)
    assert torch.isfinite(out).all() and torch.isfinite(lse).all()
    torch.testing.assert_close(lse.double(), ref_lse, rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(out.double(), ref_out, rtol=1e-4, atol=1e-5)
