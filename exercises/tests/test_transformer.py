import math

import pytest
import torch
import torch.nn.functional as F

MODULE = "transformer"

LLAMA3_8B = dict(vocab_size=128256, d_model=4096, n_layers=32, n_heads=32, n_kv_heads=8,
                 d_ff=14336, tie_embeddings=False, bias=False, max_seq_len=0)

SMALL = dict(vocab_size=50, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=48, max_seq_len=16)


def _n(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------- RMSNorm


def test_rmsnorm_matches_manual_formula(impl):
    x = 1e-2 * torch.randn(2, 5, 16)          # small scale, so eps placement matters
    m = impl.RMSNorm(16, eps=1e-5)
    assert torch.equal(m.weight, torch.ones(16))
    with torch.no_grad():
        m.weight.copy_(torch.randn(16))
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * m.weight
    torch.testing.assert_close(m(x), expected, atol=1e-5, rtol=1e-5)
    assert sum(p.numel() for p in m.parameters()) == 16      # gain only, no bias


def test_rmsnorm_scale_invariant_and_no_mean_subtraction(impl):
    x = torch.randn(3, 8) + 5.0                                # large mean: LayerNorm would remove it
    m = impl.RMSNorm(8, eps=0.0)
    torch.testing.assert_close(m(3.0 * x), m(x), atol=1e-5, rtol=1e-5)
    y = m(x)
    assert y.mean(-1).abs().min() > 0.5                        # mean survives
    torch.testing.assert_close(y.pow(2).mean(-1), torch.ones(3), atol=1e-5, rtol=1e-5)


def test_rmsnorm_dtype_and_zero_input(impl):
    m = impl.RMSNorm(16)
    xb = torch.randn(2, 4, 16).bfloat16()
    yb = m(xb)
    assert yb.dtype == torch.bfloat16
    ref = m(xb.float())
    torch.testing.assert_close(yb.float(), ref, atol=5e-2, rtol=5e-2)
    assert torch.equal(m(torch.zeros(2, 16)), torch.zeros(2, 16))   # eps keeps it finite


# ---------------------------------------------------------------- SwiGLU


def test_swiglu_matches_manual_formula(impl):
    m = impl.SwiGLU(16, 24)
    x = torch.randn(2, 5, 16)
    expected = (F.silu(x @ m.w1.weight.T) * (x @ m.w3.weight.T)) @ m.w2.weight.T
    torch.testing.assert_close(m(x), expected, atol=1e-5, rtol=1e-5)
    assert m.w1.bias is None and m.w3.bias is None and m.w2.bias is None
    assert _n(m) == 3 * 16 * 24
    mb = impl.SwiGLU(16, 24, bias=True)
    assert _n(mb) == 3 * 16 * 24 + 24 + 24 + 16


# ---------------------------------------------------------------- attention


def _naive_attention(m, x):
    """Explicit-loop causal attention using the module's own weights."""
    B, T, _ = x.shape
    nh, nkv, dh = m.n_heads, m.n_kv_heads, m.head_dim
    q = (x @ m.wq.weight.T).view(B, T, nh, dh)
    k = (x @ m.wk.weight.T).view(B, T, nkv, dh)
    v = (x @ m.wv.weight.T).view(B, T, nkv, dh)
    g = nh // nkv
    out = torch.zeros(B, T, nh, dh)
    for b in range(B):
        for h in range(nh):
            for t in range(T):
                scores = torch.stack([q[b, t, h] @ k[b, j, h // g] for j in range(t + 1)]) / math.sqrt(dh)
                w = scores.softmax(0)
                out[b, t, h] = sum(w[j] * v[b, j, h // g] for j in range(t + 1))
    return out.reshape(B, T, nh * dh) @ m.wo.weight.T


def test_attention_matches_naive_reference_mha_and_gqa(impl):
    for n_kv in (None, 2, 1):
        m = impl.CausalSelfAttention(32, 4, n_kv_heads=n_kv)
        with torch.no_grad():                                  # make all four projections non-trivial
            for lin in (m.wq, m.wk, m.wv, m.wo):
                lin.weight.normal_(0, 0.2)
        x = torch.randn(2, 6, 32)
        y = m(x)
        assert y.shape == (2, 6, 32)
        torch.testing.assert_close(y, _naive_attention(m, x), atol=1e-5, rtol=1e-5)
        nkv = 4 if n_kv is None else n_kv
        assert m.wk.weight.shape == (nkv * 8, 32) and m.wv.weight.shape == (nkv * 8, 32)


def test_attention_is_causal_and_head_dim_override(impl):
    m = impl.CausalSelfAttention(32, 4, n_kv_heads=2, head_dim=16)     # n_heads * head_dim = 64 != d_model
    assert m.wq.weight.shape == (64, 32) and m.wo.weight.shape == (32, 64)
    x = torch.randn(1, 9, 32)
    x2 = x.clone()
    x2[:, 5:] = torch.randn(1, 4, 32)
    y, y2 = m(x), m(x2)
    torch.testing.assert_close(y[:, :5], y2[:, :5], atol=1e-6, rtol=1e-6)
    assert (y[:, 5:] - y2[:, 5:]).abs().max() > 1e-4


# ---------------------------------------------------------------- block


def test_block_is_prenorm_residual(impl):
    blk = impl.Block(SMALL)
    x = torch.randn(2, 7, 32)
    h = x + blk.attn(blk.attn_norm(x))
    expected = h + blk.ffn(blk.ffn_norm(h))
    torch.testing.assert_close(blk(x), expected, atol=1e-6, rtol=1e-6)
    with torch.no_grad():                                      # kill both sublayer outputs
        blk.attn.wo.weight.zero_()
        blk.ffn.w2.weight.zero_()
    torch.testing.assert_close(blk(x), x, atol=0, rtol=0)      # the stream passes through untouched


# ---------------------------------------------------------------- GPT


def test_gpt_shapes_loss_and_init(impl):
    torch.manual_seed(0)
    model = impl.GPT(SMALL)
    idx = torch.randint(0, 50, (3, 10))
    targets = torch.randint(0, 50, (3, 10))
    logits, loss = model(idx, targets)
    assert logits.shape == (3, 10, 50)
    torch.testing.assert_close(loss, F.cross_entropy(logits.reshape(-1, 50), targets.reshape(-1)))
    assert abs(loss.item() - math.log(50)) < 0.1               # near-uniform predictions at init
    logits2, loss2 = model(idx)
    assert loss2 is None
    torch.testing.assert_close(logits2, logits)


def test_gpt_init_scales(impl):
    cfg = dict(vocab_size=200, d_model=64, n_layers=4, n_heads=4, d_ff=96, max_seq_len=8, bias=True)
    model = impl.GPT(cfg)
    assert abs(model.tok_emb.weight.std().item() - 0.02) < 0.002
    assert abs(model.blocks[0].attn.wq.weight.std().item() - 0.02) < 0.002
    expected = 0.02 / math.sqrt(2 * 4)
    for w in (model.blocks[1].attn.wo.weight, model.blocks[3].ffn.w2.weight):
        assert abs(w.std().item() - expected) < 0.15 * expected
    assert model.blocks[0].attn.wq.bias.abs().max() == 0
    assert torch.equal(model.final_norm.weight, torch.ones(64))


def test_gpt_is_causal(impl):
    model = impl.GPT(SMALL).eval()
    idx = torch.randint(0, 50, (2, 12))
    idx2 = idx.clone()
    idx2[:, 6] = (idx[:, 6] + 1) % 50                          # change ONE token, at position 6
    with torch.no_grad():
        l1, _ = model(idx)
        l2, _ = model(idx2)
    torch.testing.assert_close(l1[:, :6], l2[:, :6], atol=1e-6, rtol=1e-6)   # earlier positions unaffected
    assert (l1[:, 6] - l2[:, 6]).abs().max() > 1e-5
    assert (l1[:, 7:] - l2[:, 7:]).abs().max() > 1e-6                        # later positions do see it


def test_gpt_tied_embeddings(impl):
    tied = impl.GPT({**SMALL, "tie_embeddings": True})
    untied = impl.GPT({**SMALL, "tie_embeddings": False})
    assert tied.lm_head.weight is tied.tok_emb.weight
    assert untied.lm_head.weight is not untied.tok_emb.weight
    assert _n(untied) - _n(tied) == 50 * 32
    logits, _ = tied(torch.randint(0, 50, (2, 5)))
    assert logits.shape == (2, 5, 50)


def test_gpt_without_position_table(impl):
    model = impl.GPT({**SMALL, "max_seq_len": 0})
    assert model.pos_emb is None
    logits, _ = model(torch.randint(0, 50, (2, 20)))           # any length works
    assert logits.shape == (2, 20, 50)


# ---------------------------------------------------------------- counting


@pytest.mark.parametrize("tie", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("n_kv,head_dim,max_len", [(None, None, 0), (2, None, 16), (1, 6, 0), (4, 12, 24)])
def test_count_params_matches_module(impl, tie, bias, n_kv, head_dim, max_len):
    cfg = dict(vocab_size=37, d_model=24, n_layers=3, n_heads=4, d_ff=40, tie_embeddings=tie, bias=bias,
               max_seq_len=max_len)
    if n_kv is not None:
        cfg["n_kv_heads"] = n_kv
    if head_dim is not None:
        cfg["head_dim"] = head_dim
    assert impl.count_params(cfg) == _n(impl.GPT(cfg))


def test_count_params_llama3_8b(impl):
    assert impl.count_params(LLAMA3_8B) == 8_030_261_248
    assert impl.count_params({**LLAMA3_8B, "tie_embeddings": True}) == 8_030_261_248 - 128256 * 4096
    with torch.device("meta"):                                  # build the real module without allocating 32 GB
        model = impl.GPT(LLAMA3_8B)
    assert _n(model) == 8_030_261_248


def test_count_params_hand_formula_with_biases_positions_and_tying(impl):
    # GPT-2-small-sized shapes (12 layers, d=768, V=50257, 1024 learned positions, tied
    # embeddings, biases) but with our RMSNorm + SwiGLU block, counted by hand.
    cfg = dict(vocab_size=50257, d_model=768, n_layers=12, n_heads=12, d_ff=3072, tie_embeddings=True,
               bias=True, max_seq_len=1024)
    n = impl.count_params(cfg)
    emb = 50257 * 768 + 1024 * 768
    assert n - emb == 12 * (4 * 768 * 768 + 4 * 768 + 3 * 768 * 3072 + 2 * 3072 + 768 + 2 * 768) + 768


# ---------------------------------------------------------------- FLOPs


def test_train_flops(impl):
    assert impl.train_flops(8.03e9, 15e12) == pytest.approx(7.227e23, rel=1e-9)
    assert impl.train_flops(7e9, 2e12) == pytest.approx(8.4e22, rel=1e-9)
    assert isinstance(impl.train_flops(1, 1), float)


def test_forward_flops_per_token_llama3_8b(impl):
    lin = impl.forward_flops_per_token(LLAMA3_8B, 0)
    assert lin == 2 * (32 * 218_103_808 + 128256 * 4096)               # 2 * matmul weights, no norms
    assert lin == pytest.approx(2 * (8_030_261_248 - 128256 * 4096), rel=1e-4)
    # tying the embedding does not change the FLOPs: the LM head still multiplies
    assert impl.forward_flops_per_token({**LLAMA3_8B, "tie_embeddings": True}, 0) == lin
    T = 8192
    full = impl.forward_flops_per_token(LLAMA3_8B, T, causal=False)
    causal = impl.forward_flops_per_token(LLAMA3_8B, T, causal=True)
    assert full - lin == pytest.approx(4 * 32 * T * 4096, rel=1e-12)
    assert causal - lin == pytest.approx(2 * 32 * T * 4096, rel=1e-12)


def test_attention_flop_fraction_llama3_8b(impl):
    f = impl.attention_flop_fraction
    assert f(LLAMA3_8B, 8192, causal=True) == pytest.approx(0.1252, abs=5e-4)
    assert f(LLAMA3_8B, 8192, causal=False) == pytest.approx(0.2225, abs=5e-4)
    assert f(LLAMA3_8B, 131072, causal=True) == pytest.approx(0.6960, abs=5e-4)
    assert f(LLAMA3_8B, 131072, causal=False) == pytest.approx(0.8207, abs=5e-4)
    assert f(LLAMA3_8B, 0) == 0.0
    # attention reaches half of the forward FLOPs at T = N_matmul / (L d) for the causal count
    t_star = round((32 * 218_103_808 + 128256 * 4096) / (32 * 4096))
    assert f(LLAMA3_8B, t_star, causal=True) == pytest.approx(0.5, abs=1e-3)
