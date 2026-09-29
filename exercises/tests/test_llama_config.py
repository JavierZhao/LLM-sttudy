MODULE = "llama_config"

# Ground truth: parameter totals are the sums of the tensor sizes in the released checkpoints
# (Hugging Face safetensors metadata of Meta's checkpoints or exact conversions of them; rotary
# inv_freq buffers are not parameters and are excluded). Config values come from the released
# config.json / params.json files.


def _cfg(impl, d, L, nh, nkv, V, mult_of, mult=None, tie=False, head_dim=None):
    return impl.LlamaConfig(d, L, nh, nkv, V, impl.ffn_hidden_dim(d, mult_of, mult),
                            head_dim=head_dim, tie_embeddings=tie)


def test_ffn_hidden_dim_known_models(impl):
    f = impl.ffn_hidden_dim
    assert f(4096, 256) == 11008                 # Llama 1/2 7B
    assert f(5120, 256) == 13824                 # 13B
    assert f(6656, 256) == 17920                 # Llama 1 33B
    assert f(8192, 256) == 22016                 # Llama 1 65B
    assert f(8192, 4096, 1.3) == 28672           # Llama 2 70B and Llama 3 70B
    assert f(4096, 1024, 1.3) == 14336           # Llama 3 8B (the number this page is about)
    assert f(16384, 4096, 1.2) == 53248          # Llama 3 405B
    assert f(2048, 256, 1.5) == 8192             # Llama 3.2 1B
    assert f(3072, 256, 1.0) == 8192             # Llama 3.2 3B


def test_ffn_hidden_dim_truncation_order(impl):
    # multiple_of=1 exposes the two truncations and their order.
    # d=4096: int(2*16384/3) = 10922; int(1.3 * 10922) = int(14198.6) = 14198 (rounding would give 14199).
    assert impl.ffn_hidden_dim(4096, 1) == 10922
    assert impl.ffn_hidden_dim(4096, 1, 1.3) == 14198
    # d=100: int(2*400/3) = 266, then int(1.5 * 266) = 399. Applying the multiplier first would give 400.
    assert impl.ffn_hidden_dim(100, 1, 1.5) == 399
    # the result is always a multiple of multiple_of and less than one multiple above the unrounded value
    for d, m in [(1000, 64), (4096, 128), (777, 32)]:
        h = impl.ffn_hidden_dim(d, m)
        raw = int(2 * (4 * d) / 3)
        assert h % m == 0 and 0 <= h - raw < m


def test_llama_params_exact_dense_models(impl):
    cases = {
        "llama1_7b": (_cfg(impl, 4096, 32, 32, 32, 32000, 256), 6_738_415_616),
        "llama1_13b": (_cfg(impl, 5120, 40, 40, 40, 32000, 256), 13_015_864_320),
        "llama1_33b": (_cfg(impl, 6656, 60, 52, 52, 32000, 256), 32_528_943_616),
        "llama1_65b": (_cfg(impl, 8192, 80, 64, 64, 32000, 256), 65_285_660_672),
        "llama2_70b": (_cfg(impl, 8192, 80, 64, 8, 32000, 4096, 1.3), 68_976_648_192),
        "llama3_8b": (_cfg(impl, 4096, 32, 32, 8, 128256, 1024, 1.3), 8_030_261_248),
        "llama3_70b": (_cfg(impl, 8192, 80, 64, 8, 128256, 4096, 1.3), 70_553_706_496),
        "llama3_405b": (_cfg(impl, 16384, 126, 128, 8, 128256, 4096, 1.2), 405_853_388_800),
    }
    for name, (cfg, expected) in cases.items():
        assert impl.llama_params(cfg) == expected, name


def test_tied_embeddings_and_explicit_head_dim(impl):
    # Llama 3.2 1B and 3B tie the output head to the embedding
    one_b = _cfg(impl, 2048, 16, 32, 8, 128256, 256, 1.5, tie=True)      # head_dim 64
    three_b = _cfg(impl, 3072, 28, 24, 8, 128256, 256, 1.0, tie=True)    # head_dim 128
    assert impl.llama_params(one_b) == 1_235_814_400
    assert impl.llama_params(three_b) == 3_212_749_824
    untied = impl.LlamaConfig(2048, 16, 32, 8, 128256, one_b.ffn_hidden, tie_embeddings=False)
    assert impl.llama_params(untied) - impl.llama_params(one_b) == 128256 * 2048
    # an explicit head_dim that differs from dim // n_heads changes only the attention projections
    a = impl.LlamaConfig(1024, 2, 8, 2, 100, 512)                 # d_h = 128
    b = impl.LlamaConfig(1024, 2, 8, 2, 100, 512, head_dim=64)
    assert impl.param_breakdown(a)["attention"] == 2 * (2 * 1024 * 8 * 128 + 2 * 1024 * 2 * 128)
    assert impl.param_breakdown(b)["attention"] == 2 * (2 * 1024 * 8 * 64 + 2 * 1024 * 2 * 64)
    assert impl.param_breakdown(a)["ffn"] == impl.param_breakdown(b)["ffn"]


def test_breakdown_llama3_8b(impl):
    cfg = _cfg(impl, 4096, 32, 32, 8, 128256, 1024, 1.3)
    b = impl.param_breakdown(cfg)
    assert b["embedding"] == b["lm_head"] == 128256 * 4096 == 525_336_576
    assert b["attention"] == 32 * (2 * 4096 * 4096 + 2 * 4096 * 1024)
    assert b["ffn"] == 32 * 3 * 4096 * 14336
    assert b["norm"] == 65 * 4096
    assert b["total"] == sum(v for k, v in b.items() if k != "total") == 8_030_261_248
    # GQA saves exactly 2 * d * (n_h - n_kv) * d_h per layer versus plain MHA
    mha = impl.LlamaConfig(4096, 32, 32, 32, 128256, 14336)
    assert impl.llama_params(mha) - impl.llama_params(cfg) == 32 * 2 * 4096 * (32 - 8) * 128


def _l4(impl, moe_layers, n_experts, V=202048):
    return impl.Llama4Config(dim=5120, n_layers=48, n_heads=40, n_kv_heads=8, head_dim=128, vocab_size=V,
                             dense_ffn=16384, expert_ffn=8192, n_experts=n_experts,
                             moe_layers=tuple(moe_layers), top_k=1)


def test_llama4_scout_and_maverick(impl):
    scout = _l4(impl, range(48), 16)                 # every layer is MoE
    maverick = _l4(impl, range(1, 48, 2), 128)       # every other layer is MoE
    s_total, s_active = impl.llama4_params(scout)
    m_total, m_active = impl.llama4_params(maverick)
    assert (s_total, s_active) == (107_769_861_120, 17_172_894_720)
    assert (m_total, m_active) == (400_711_848_960, 17_184_691_200)
    # The Hugging Face checkpoints hold 108,641,793,536 and 401,583,781,376 tensors in total. Both differ from
    # the text backbone by the same 871,932,416 parameters (the vision encoder and projector), which is a
    # strong check that the accounting above is exact.
    assert 108_641_793_536 - s_total == 401_583_781_376 - m_total == 871_932_416


def test_llama4_active_is_smaller_and_top_k_scales(impl):
    cfg = _l4(impl, range(1, 48, 2), 128)
    t1, a1 = impl.llama4_params(cfg)
    t2, a2 = impl.llama4_params(impl.Llama4Config(**{**cfg.__dict__, "top_k": 2}))
    assert t1 == t2                                   # top_k never changes the total
    assert a2 - a1 == 24 * 3 * 5120 * 8192            # one more routed expert in each of the 24 MoE layers
    dense = impl.Llama4Config(**{**cfg.__dict__, "moe_layers": ()})
    td, ad = impl.llama4_params(dense)
    assert td == ad                                   # no MoE layers: everything is active


def test_kv_cache_dense_models(impl):
    kv = impl.kv_cache_bytes
    assert kv(32, 32, 128, 1) == 524_288              # Llama 2 7B, plain MHA: 512 KiB per token
    assert kv(32, 8, 128, 1) == 131_072               # Llama 3 8B: 128 KiB per token
    assert kv(80, 8, 128, 1) == 327_680               # Llama 2/3 70B: 320 KiB per token
    assert kv(126, 8, 128, 1) == 516_096              # Llama 3 405B: 504 KiB per token
    assert kv(32, 8, 128, 131072) == 16 * 2**30       # 16 GiB at 128K
    assert kv(32, 8, 128, 10, bytes_per_el=1) == 32 * 2 * 8 * 128 * 10


def test_kv_cache_hybrid_local_global(impl):
    kv = impl.kv_cache_bytes
    per = 2 * 8 * 128 * 2                              # bytes per token per layer
    # Llama 4 Scout at 10M tokens: 12 global layers keep everything, 36 local layers keep 8,192 tokens
    T = 10_485_760
    got = kv(48, 8, 128, T, n_global_layers=12, local_window=8192)
    assert got == per * (12 * T + 36 * 8192) == 516_604_035_072
    assert abs(got / 2**30 - 481.125) < 1e-9
    assert kv(48, 8, 128, T) == 4 * per * 12 * T       # all-global would be 4x the global part
    # shorter than the window: local layers hold everything, so nothing is saved
    assert kv(48, 8, 128, 1000, n_global_layers=12, local_window=8192) == kv(48, 8, 128, 1000)


def test_breakdown_llama3_405b_matches_worked_example(impl):
    # The worked example on the page: one layer is 3,187,703,808 parameters, the whole model 405,853,388,800.
    cfg = _cfg(impl, 16384, 126, 128, 8, 128256, 4096, 1.2)
    assert cfg.ffn_hidden == 53248
    b = impl.param_breakdown(cfg)
    per_layer = (2 * 16384 * 16384 + 2 * 16384 * 1024) + 3 * 16384 * 53248 + 2 * 16384
    assert per_layer == 3_187_703_808
    assert b["attention"] == 126 * (2 * 16384 * 16384 + 2 * 16384 * 1024)
    assert b["ffn"] == 126 * 3 * 16384 * 53248
    assert b["norm"] == (2 * 126 + 1) * 16384
    assert b["embedding"] == b["lm_head"] == 128256 * 16384
    assert b["total"] == impl.llama_params(cfg) == 126 * per_layer + 2 * 128256 * 16384 + 16384 == 405_853_388_800


def test_llama4_tied_embeddings(impl):
    cfg = _l4(impl, range(1, 48, 2), 128)
    tied = impl.Llama4Config(**{**cfg.__dict__, "tie_embeddings": True})
    (t0, a0), (t1, a1) = impl.llama4_params(cfg), impl.llama4_params(tied)
    assert t0 - t1 == a0 - a1 == 202048 * 5120        # the output head is shared with the embedding


def test_kv_cache_edge_cases(impl):
    kv = impl.kv_cache_bytes
    per = 2 * 8 * 128 * 2
    # no global layers: every layer is local, so nothing grows past the window
    assert kv(48, 8, 128, 100_000, n_global_layers=0, local_window=8192) == 48 * per * 8192
    # every layer global: the window is irrelevant
    assert kv(48, 8, 128, 100_000, n_global_layers=48, local_window=8192) == kv(48, 8, 128, 100_000)
    # a window exactly as long as the sequence saves nothing; an empty sequence costs nothing
    assert kv(48, 8, 128, 8192, n_global_layers=12, local_window=8192) == kv(48, 8, 128, 8192)
    assert kv(48, 8, 128, 0, n_global_layers=12, local_window=8192) == 0
    # element size scales every term, local and global
    assert kv(48, 8, 128, 20_000, n_global_layers=12, local_window=8192, bytes_per_el=1) * 2 == \
        kv(48, 8, 128, 20_000, n_global_layers=12, local_window=8192, bytes_per_el=2)
    # per-token cost is exactly 2 * n_kv * d_h * bytes per layer, independent of the model width
    assert kv(1, 3, 5, 7, bytes_per_el=4) == 2 * 3 * 5 * 4 * 7
