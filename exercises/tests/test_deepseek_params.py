import math

import pytest
import torch
import torch.nn as nn

MODULE = "deepseek_params"

# Relevant keys of the released config.json files (Hugging Face: deepseek-ai/DeepSeek-V2, -V2-Lite, -V3).
V3 = dict(num_hidden_layers=61, hidden_size=7168, num_attention_heads=128, q_lora_rank=1536,
          kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
          first_k_dense_replace=3, intermediate_size=18432, moe_intermediate_size=2048,
          n_routed_experts=256, n_shared_experts=1, num_experts_per_tok=8, vocab_size=129280)
V2 = dict(num_hidden_layers=60, hidden_size=5120, num_attention_heads=128, q_lora_rank=1536,
          kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
          first_k_dense_replace=1, intermediate_size=12288, moe_intermediate_size=1536,
          n_routed_experts=160, n_shared_experts=2, num_experts_per_tok=6, vocab_size=102400)
V2_LITE = dict(num_hidden_layers=27, hidden_size=2048, num_attention_heads=16, q_lora_rank=None,
               kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
               first_k_dense_replace=1, intermediate_size=10944, moe_intermediate_size=1408,
               n_routed_experts=64, n_shared_experts=2, num_experts_per_tok=6, vocab_size=102400)
TINY = dict(num_hidden_layers=4, hidden_size=32, num_attention_heads=4, q_lora_rank=16,
            kv_lora_rank=8, qk_nope_head_dim=8, qk_rope_head_dim=4, v_head_dim=6,
            first_k_dense_replace=1, intermediate_size=48, moe_intermediate_size=12,
            n_routed_experts=10, n_shared_experts=2, num_experts_per_tok=3, vocab_size=50)


def _reference_model(c):
    """An independent count: build the modules with nn.Linear / nn.Parameter and sum numel."""
    d, nh = c["hidden_size"], c["num_attention_heads"]
    dn, dr, dv = c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]
    kvr, qr = c["kv_lora_rank"], c["q_lora_rank"]

    def lin(i, o):
        return nn.Linear(i, o, bias=False)

    def ffn(width):
        return nn.ModuleList([lin(d, width), lin(d, width), lin(width, d)])   # gate, up, down

    layers = nn.ModuleList()
    for i in range(c["num_hidden_layers"]):
        m = nn.ModuleDict()
        m["input_norm"] = nn.ParameterList([nn.Parameter(torch.ones(d))])
        m["post_norm"] = nn.ParameterList([nn.Parameter(torch.ones(d))])
        if qr:
            m["q_a"] = lin(d, qr)
            m["q_a_norm"] = nn.ParameterList([nn.Parameter(torch.ones(qr))])
            m["q_b"] = lin(qr, nh * (dn + dr))
        else:
            m["q"] = lin(d, nh * (dn + dr))
        m["kv_a"] = lin(d, kvr + dr)
        m["kv_a_norm"] = nn.ParameterList([nn.Parameter(torch.ones(kvr))])
        m["kv_b"] = lin(kvr, nh * (dn + dv))
        m["o"] = lin(nh * dv, d)
        if i < c["first_k_dense_replace"]:
            m["ffn"] = ffn(c["intermediate_size"])
        else:
            m["router"] = lin(d, c["n_routed_experts"])
            m["routed"] = nn.ModuleList([ffn(c["moe_intermediate_size"]) for _ in range(c["n_routed_experts"])])
            m["shared"] = nn.ModuleList([ffn(c["moe_intermediate_size"]) for _ in range(c["n_shared_experts"])])
        layers.append(m)
    model = nn.ModuleDict(dict(layers=layers, embed=nn.Embedding(c["vocab_size"], d),
                               head=lin(d, c["vocab_size"]), final_norm=nn.ParameterList([nn.Parameter(torch.ones(d))])))
    return sum(p.numel() for p in model.parameters()), model


def test_mla_params_check_values(impl):
    assert impl.mla_params(V3) == 187_107_328          # page 07 table (includes the 2,048 latent-norm weights)
    assert impl.mla_params(V2) == 149_227_520
    assert impl.mla_params(V2_LITE) == 13_763_072      # no query compression: one d x n_h(d_nope+d_rope) matrix


def test_moe_layer_params(impl):
    total, active = impl.moe_layer_params(V3)
    assert total == 257 * 44_040_192 + 7168 * 256 == 11_320_164_352
    assert active == 9 * 44_040_192 + 7168 * 256 == 398_196_736
    total, active = impl.moe_layer_params(V2)
    assert total == 162 * 23_592_960 + 5120 * 160
    assert active == 8 * 23_592_960 + 5120 * 160 == 189_562_880


def test_total_matches_independent_module_count(impl):
    ref_total, _ = _reference_model(TINY)
    total, active = impl.count_params(TINY, "none")
    assert total == ref_total
    # activated = total minus the routed experts a token does not use, minus both vocabulary matrices
    n_moe = TINY["num_hidden_layers"] - TINY["first_k_dense_replace"]
    idle = (TINY["n_routed_experts"] - TINY["num_experts_per_tok"]) * 3 * TINY["hidden_size"] * TINY["moe_intermediate_size"] * n_moe
    assert active == ref_total - idle - 2 * TINY["vocab_size"] * TINY["hidden_size"]


def test_tiny_without_query_compression(impl):
    cfg = dict(TINY, q_lora_rank=None)
    ref_total, _ = _reference_model(cfg)
    assert impl.count_params(cfg)[0] == ref_total


def test_v3_matches_reported_size(impl):
    total, active = impl.count_params(V3)
    assert total == 671_026_404_352
    assert total == pytest.approx(671.03e9, rel=0.005)
    assert active == 37_552_282_624
    assert active == pytest.approx(37.55e9, rel=0.005)


def test_v2_matches_reported_size(impl):
    total, active = impl.count_params(V2)
    assert total == 235_741_434_880                     # equals the Hugging Face safetensors parameter count
    assert total == pytest.approx(236e9, rel=0.01)
    assert active == 21_375_800_320                     # both vocabulary matrices counted
    assert active == pytest.approx(21e9, rel=0.02)


def test_v2_activated_conventions(impl):
    _, both = impl.count_params(V2, "both")
    _, head = impl.count_params(V2, "head")
    _, none = impl.count_params(V2, "none")
    assert none == 20_327_224_320
    assert head == none + 102400 * 5120
    assert both == head + 102400 * 5120
    assert head == pytest.approx(21e9, rel=0.01)        # the head-only convention lands within 1% of the reported 21B
    with pytest.raises(ValueError):
        impl.count_params(V2, "input")


def test_v2_lite_matches_hf_metadata(impl):
    total, active = impl.count_params(V2_LITE, "head")
    assert total == 15_706_484_224
    assert active == pytest.approx(2.4e9, rel=0.03)


def test_sparsity_structure(impl):
    # total - activated (non-embedding) = idle routed experts, for any config
    for cfg in (V3, V2, V2_LITE, TINY):
        total, none = impl.count_params(cfg, "none")
        n_moe = cfg["num_hidden_layers"] - cfg["first_k_dense_replace"]
        idle = (cfg["n_routed_experts"] - cfg["num_experts_per_tok"]) * 3 * cfg["hidden_size"] * cfg["moe_intermediate_size"] * n_moe
        assert total - none == idle + 2 * cfg["vocab_size"] * cfg["hidden_size"]


def test_model_scale_flops(impl):
    # DeepSeek LLM Table 3 (vocab 102,400, l_seq 4,096)
    assert impl.model_scale_flops(8, 512, 4096) == pytest.approx(352e6, rel=0.005)
    assert impl.model_scale_flops(32, 4096, 4096) == pytest.approx(45.1e9, rel=0.005)
    assert impl.model_scale_flops(80, 8192, 4096) == pytest.approx(419e9, rel=0.005)
    # first term is exactly 6 * (12 L d^2); the attention term is linear in l_seq
    L, d = 30, 4096
    assert impl.model_scale_flops(L, d, 0) == 6 * 12 * L * d * d
    assert impl.model_scale_flops(L, d, 8192) - impl.model_scale_flops(L, d, 4096) == 12 * L * d * 4096


def test_hp_scaling_values(impl):
    B, eta = impl.hp_scaling(1e22)
    assert B == pytest.approx(4.5876e6, rel=1e-3) and eta == pytest.approx(5.5447e-4, rel=1e-3)
    B, eta = impl.hp_scaling(1e24)
    assert B == pytest.approx(2.0691e7, rel=1e-3) and eta == pytest.approx(3.118e-4, rel=1e-3)


def test_hp_scaling_power_law_shape(impl):
    B1, e1 = impl.hp_scaling(1e21)
    B2, e2 = impl.hp_scaling(1e22)
    assert B2 / B1 == pytest.approx(10**0.3271, rel=1e-9)
    assert e2 / e1 == pytest.approx(10**-0.125, rel=1e-9)


def test_hp_scaling_reproduces_deepseek_models(impl):
    # compute C = M * D with D = 2T tokens; compare with Table 2 (batch in sequences of 4,096 tokens)
    for (L, d, B_seq, lr) in [(30, 4096, 2304, 4.2e-4), (95, 8192, 4608, 3.2e-4)]:
        C = impl.model_scale_flops(L, d, 4096) * 2e12
        B, eta = impl.hp_scaling(C)
        assert B == pytest.approx(B_seq * 4096, rel=0.10)
        assert eta == pytest.approx(lr, rel=0.03)


def test_optimal_allocation(impl):
    for C in (1e18, 1e22, 1e24):
        M, D = impl.optimal_allocation(C)
        assert M * D == pytest.approx(C, rel=1e-3)      # exponents 0.5243 + 0.4757 = 1
    M, D = impl.optimal_allocation(1e24)
    assert M == pytest.approx(6.568e11, rel=1e-3)
    assert D == pytest.approx(1.523e12, rel=1e-3)
