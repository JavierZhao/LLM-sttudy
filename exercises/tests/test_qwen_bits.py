import pytest
import torch

MODULE = "qwen_bits"


# ------------------------------------------------------------------ gated attention
def test_gate_plus_infinity_recovers_ungated(impl):
    # x = 1 and a large positive W_gate push every pre-activation to +1600: sigmoid == 1 exactly
    attn = torch.randn(2, 5, 4, 8)
    x = torch.ones(2, 5, 16)
    W = torch.full((16, 4 * 8), 100.0)
    out = impl.gated_attention_output(attn, x, W)
    assert out.shape == attn.shape
    torch.testing.assert_close(out, attn)


def test_gate_minus_infinity_and_zero(impl):
    attn = torch.randn(1, 3, 2, 4)
    x = torch.ones(1, 3, 6)
    torch.testing.assert_close(impl.gated_attention_output(attn, x, torch.full((6, 8), -100.0)),
                               torch.zeros_like(attn), atol=1e-6, rtol=0)
    # zero pre-activation: sigmoid(0) = 1/2
    torch.testing.assert_close(impl.gated_attention_output(attn, x, torch.zeros(6, 8)), 0.5 * attn)


def test_gate_matches_per_token_reference(impl):
    B, T, H, D, d = 2, 4, 3, 5, 7
    attn, x, W = torch.randn(B, T, H, D), torch.randn(B, T, d), torch.randn(d, H * D)
    out = impl.gated_attention_output(attn, x, W)
    for b in range(B):
        for t in range(T):
            g = torch.sigmoid(x[b, t] @ W).view(H, D)      # head h, channel c uses column h * D + c
            torch.testing.assert_close(out[b, t], attn[b, t] * g)


def test_gate_is_token_local(impl):
    # changing another token's hidden state must not change this token's output
    attn, x, W = torch.randn(1, 6, 2, 4), torch.randn(1, 6, 9), torch.randn(9, 8)
    base = impl.gated_attention_output(attn, x, W)
    x2 = x.clone()
    x2[0, 3] += 5.0
    out2 = impl.gated_attention_output(attn, x2, W)
    torch.testing.assert_close(out2[0, :3], base[0, :3])
    assert not torch.allclose(out2[0, 3], base[0, 3])


def test_headwise_gate(impl):
    B, T, H, D, d = 2, 3, 4, 6, 5
    attn, x, W = torch.randn(B, T, H, D), torch.randn(B, T, d), torch.randn(d, H)
    out = impl.gated_attention_output(attn, x, W)
    assert out.shape == (B, T, H, D)
    torch.testing.assert_close(out, attn * torch.sigmoid(x @ W)[..., None])


def test_gate_gradients(impl):
    attn = torch.randn(1, 3, 2, 4, requires_grad=True)
    x = torch.randn(1, 3, 6, requires_grad=True)
    W = torch.randn(6, 8, requires_grad=True)
    impl.gated_attention_output(attn, x, W).sum().backward()
    for p in (attn, x, W):
        assert p.grad is not None and p.grad.abs().sum() > 0


# ------------------------------------------------------------------ balance losses
def _random_group(n, E, k, gen):
    logits = torch.randn(n, E, generator=gen)
    probs = logits.softmax(-1)
    idx = probs.topk(k, dim=-1).indices
    return probs, idx


def _ref_losses(probs_list, idx_list, E):
    """Loop reference. f counts (token, slot) assignments divided by tokens * k."""
    Np = len(probs_list)
    tot_counts = [0.0] * E
    tot_assign = 0
    for idx in idx_list:
        for e in idx.reshape(-1).tolist():
            tot_counts[e] += 1
        tot_assign += idx.numel()
    micro, glob = 0.0, 0.0
    for p, idx in zip(probs_list, idx_list):
        counts = [0.0] * E
        for e in idx.reshape(-1).tolist():
            counts[e] += 1
        m_terms, g_terms = 0.0, 0.0
        for i in range(E):
            Pji = p[:, i].mean()
            m_terms = m_terms + (counts[i] / idx.numel()) * Pji
            g_terms = g_terms + (tot_counts[i] / tot_assign) * Pji
        micro = micro + E * m_terms / Np
        glob = glob + E * g_terms / Np
    return micro, glob


def test_uniform_router_gives_one(impl):
    E, k, n = 4, 2, 8
    probs = torch.full((n, E), 1.0 / E)
    idx = torch.tensor([[0, 1], [2, 3]] * 4)          # every expert gets exactly n * k / E = 4 slots
    for fn in (impl.micro_batch_balance_loss, impl.global_batch_balance_loss):
        torch.testing.assert_close(fn([probs, probs], [idx, idx], E), torch.tensor(1.0))


def test_matches_reference_with_unequal_groups(impl):
    gen = torch.Generator().manual_seed(0)
    E, k = 6, 2
    groups = [_random_group(n, E, k, gen) for n in (5, 9, 3)]
    probs_list, idx_list = [g[0] for g in groups], [g[1] for g in groups]
    micro_ref, glob_ref = _ref_losses(probs_list, idx_list, E)
    torch.testing.assert_close(impl.micro_batch_balance_loss(probs_list, idx_list, E), micro_ref.to(torch.float32))
    torch.testing.assert_close(impl.global_batch_balance_loss(probs_list, idx_list, E), glob_ref.to(torch.float32))


def test_single_group_global_equals_micro(impl):
    gen = torch.Generator().manual_seed(1)
    p, idx = _random_group(11, 5, 2, gen)
    torch.testing.assert_close(impl.global_batch_balance_loss([p], [idx], 5), impl.micro_batch_balance_loss([p], [idx], 5))


def test_equal_when_every_microbatch_has_the_same_statistics(impl):
    gen = torch.Generator().manual_seed(2)
    p, idx = _random_group(10, 8, 2, gen)
    same_p, same_idx = [p, p.clone(), p.clone()], [idx, idx.clone(), idx.clone()]
    torch.testing.assert_close(impl.global_batch_balance_loss(same_p, same_idx, 8),
                               impl.micro_batch_balance_loss(same_p, same_idx, 8))


def test_domain_specific_microbatches_are_penalized_only_by_micro_loss(impl):
    # Two experts. Micro-batch 0 is all "code" and goes to expert 0, micro-batch 1 is all "prose"
    # and goes to expert 1. Balanced over the global batch, collapsed inside every micro-batch.
    p0 = torch.tensor([[1.0, 0.0]] * 4)
    p1 = torch.tensor([[0.0, 1.0]] * 4)
    i0 = torch.zeros(4, 1, dtype=torch.long)
    i1 = torch.ones(4, 1, dtype=torch.long)
    micro = impl.micro_batch_balance_loss([p0, p1], [i0, i1], 2)
    glob = impl.global_batch_balance_loss([p0, p1], [i0, i1], 2)
    torch.testing.assert_close(micro, torch.tensor(2.0))      # E * (1 * 1) in each micro-batch
    torch.testing.assert_close(glob, torch.tensor(1.0))       # fbar = (1/2, 1/2), P^j one-hot: 2 * 1/2
    assert glob < micro


def test_global_frequency_is_token_weighted(impl):
    # group 0: one token routed to expert 0. group 1: three tokens routed to expert 1.
    # Token-weighted fbar = (1/4, 3/4). A group-averaged frequency would be (1/2, 1/2) instead.
    i0 = torch.zeros(1, 1, dtype=torch.long)
    i1 = torch.ones(3, 1, dtype=torch.long)
    # both groups have P = (1, 0): each group term is E * fbar_0 * 1 = 2 * 1/4 = 0.5 (it would be 1.0 with (1/2, 1/2))
    p0 = torch.tensor([[1.0, 0.0]])
    p1 = torch.tensor([[1.0, 0.0]] * 3)
    torch.testing.assert_close(impl.global_batch_balance_loss([p0, p1], [i0, i1], 2), torch.tensor(0.5))
    # the micro-batch version uses each group's own frequency: group 0 -> 2 * 1 * 1, group 1 -> 2 * 0 * 1, mean 1.0
    torch.testing.assert_close(impl.micro_batch_balance_loss([p0, p1], [i0, i1], 2), torch.tensor(1.0))


def test_gradient_flows_only_through_local_probabilities(impl):
    gen = torch.Generator().manual_seed(3)
    E, k = 4, 2
    groups = [_random_group(n, E, k, gen) for n in (3, 6)]
    probs = [g[0].clone().requires_grad_(True) for g in groups]
    idx = [g[1] for g in groups]
    loss = impl.global_batch_balance_loss(probs, idx, E)
    grads = torch.autograd.grad(loss, probs)
    counts = sum(torch.bincount(i.reshape(-1), minlength=E).float() for i in idx)
    fbar = counts / sum(i.numel() for i in idx)
    for g, p in zip(grads, probs):
        # dL/dp[t, i] = (1 / N_P) * E * fbar_i / n_j, with fbar treated as a constant
        expected = (E * fbar / (len(probs) * p.shape[0])).expand_as(p)
        torch.testing.assert_close(g, expected)


# ------------------------------------------------------------------ parameter and cache calculators
def test_qwen3_235b_param_counts(impl):
    r = impl.qwen3_moe_param_counts(n_layers=94, d=4096, n_heads=64, n_kv_heads=4, head_dim=128,
                                    d_expert=1536, n_experts=128, top_k=8, vocab=151936)
    assert r["total"] == 235_093_634_560            # equals the Hugging Face safetensors total
    assert r["active"] == 22_190_763_520            # about 22B when embedding and LM head are counted
    assert r["attention_per_layer"] == 71_303_168
    assert r["expert_per_layer"] == 2_415_919_104


def test_qwen3_30b_param_counts(impl):
    r = impl.qwen3_moe_param_counts(n_layers=48, d=2048, n_heads=32, n_kv_heads=4, head_dim=128,
                                    d_expert=768, n_experts=128, top_k=8, vocab=151936)
    assert r["total"] == 30_532_122_624
    assert r["active"] == 3_353_032_704


def test_tied_embeddings_save_one_matrix(impl):
    kw = dict(n_layers=2, d=16, n_heads=4, n_kv_heads=2, head_dim=4, d_expert=8, n_experts=4, top_k=2, vocab=50)
    assert impl.qwen3_moe_param_counts(**kw)["total"] - impl.qwen3_moe_param_counts(tie_embeddings=True, **kw)["total"] == 50 * 16


def test_qwen3_next_cache(impl):
    r = impl.hybrid_cache_bytes(seq_len=131072, n_layers=48, full_attn_interval=4, n_kv_heads=2,
                                head_dim=256, n_state_heads=32, d_k=128, d_v=128)
    assert r["kv_per_token"] == 24_576               # 12 full layers * 2 * 2 * 256 * 2 bytes
    assert r["kv_total"] == 3 * 2**30                # 3 GiB at 128K tokens
    assert r["state_total"] == 36 * 32 * 128 * 128 * 4   # 72 MiB, fp32, independent of length
    assert r["total"] == r["kv_total"] + r["state_total"]
    r2 = impl.hybrid_cache_bytes(seq_len=2 * 131072, n_layers=48, full_attn_interval=4, n_kv_heads=2,
                                 head_dim=256, n_state_heads=32, d_k=128, d_v=128)
    assert r2["state_total"] == r["state_total"] and r2["kv_total"] == 2 * r["kv_total"]


def test_interval_one_is_a_plain_transformer(impl):
    # Qwen3-32B: 64 layers, 8 KV heads of 128: 256 KiB per token, 32 GiB at 128K, no recurrent state
    r = impl.hybrid_cache_bytes(seq_len=131072, n_layers=64, full_attn_interval=1, n_kv_heads=8,
                                head_dim=128, n_state_heads=1, d_k=1, d_v=1)
    assert r["kv_per_token"] == 262_144 and r["kv_total"] == 32 * 2**30 and r["state_total"] == 0
