import pytest
import torch
import torch.nn as nn

MODULE = "config_zoo"

# Ground truth: parameter totals are the sums of the tensor sizes in the released checkpoints (Hugging Face
# safetensors metadata; rotary buffers excluded), except where a comment says otherwise. Config values are the
# relevant keys of each released config.json (Meta's and Google's repositories are gated: public mirrors).
LLAMA3_8B = dict(model_type="llama", hidden_size=4096, num_hidden_layers=32, num_attention_heads=32,
                 num_key_value_heads=8, intermediate_size=14336, vocab_size=128256, tie_word_embeddings=False)
LLAMA31_405B = dict(model_type="llama", hidden_size=16384, num_hidden_layers=126, num_attention_heads=128,
                    num_key_value_heads=8, intermediate_size=53248, vocab_size=128256, tie_word_embeddings=False)
MISTRAL_7B = dict(model_type="mistral", hidden_size=4096, num_hidden_layers=32, num_attention_heads=32,
                  num_key_value_heads=8, intermediate_size=14336, vocab_size=32000, sliding_window=4096,
                  tie_word_embeddings=False)
MIXTRAL_8X7B = dict(model_type="mixtral", hidden_size=4096, num_hidden_layers=32, num_attention_heads=32,
                    num_key_value_heads=8, intermediate_size=14336, num_local_experts=8, num_experts_per_tok=2,
                    vocab_size=32000, tie_word_embeddings=False)
QWEN25_72B = dict(model_type="qwen2", hidden_size=8192, num_hidden_layers=80, num_attention_heads=64,
                  num_key_value_heads=8, intermediate_size=29568, vocab_size=152064, tie_word_embeddings=False,
                  sliding_window=131072, use_sliding_window=False)          # the window is switched off
QWEN3_32B = dict(model_type="qwen3", hidden_size=5120, num_hidden_layers=64, num_attention_heads=64,
                 num_key_value_heads=8, head_dim=128, intermediate_size=25600, vocab_size=151936,
                 tie_word_embeddings=False)
QWEN3_235B = dict(model_type="qwen3_moe", hidden_size=4096, num_hidden_layers=94, num_attention_heads=64,
                  num_key_value_heads=4, head_dim=128, intermediate_size=12288, moe_intermediate_size=1536,
                  num_experts=128, num_experts_per_tok=8, decoder_sparse_step=1, mlp_only_layers=[],
                  vocab_size=151936, tie_word_embeddings=False)
OLMO2_32B = dict(model_type="olmo2", hidden_size=5120, num_hidden_layers=64, num_attention_heads=40,
                 num_key_value_heads=8, intermediate_size=27648, vocab_size=100352, tie_word_embeddings=False)
DSV3 = dict(model_type="deepseek_v3", hidden_size=7168, num_hidden_layers=61, num_attention_heads=128,
            num_key_value_heads=128, intermediate_size=18432, moe_intermediate_size=2048, n_routed_experts=256,
            n_shared_experts=1, num_experts_per_tok=8, first_k_dense_replace=3, moe_layer_freq=1, q_lora_rank=1536,
            kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128, vocab_size=129280,
            tie_word_embeddings=False)
DSV2 = dict(model_type="deepseek_v2", hidden_size=5120, num_hidden_layers=60, num_attention_heads=128,
            num_key_value_heads=128, intermediate_size=12288, moe_intermediate_size=1536, n_routed_experts=160,
            n_shared_experts=2, num_experts_per_tok=6, first_k_dense_replace=1, moe_layer_freq=1, q_lora_rank=1536,
            kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128, vocab_size=102400,
            tie_word_embeddings=False)
KIMI_K2 = dict(DSV3, model_type="kimi_k2", num_attention_heads=64, num_key_value_heads=64, n_routed_experts=384,
               first_k_dense_replace=1, vocab_size=163840)
GLM45 = dict(model_type="glm4_moe", hidden_size=5120, num_hidden_layers=92, num_attention_heads=96,
             num_key_value_heads=8, head_dim=128, intermediate_size=12288, moe_intermediate_size=1536,
             n_routed_experts=160, n_shared_experts=1, num_experts_per_tok=8, first_k_dense_replace=3,
             attention_bias=True, use_qk_norm=True, vocab_size=151552, tie_word_embeddings=False)


def _total(impl, cfg):
    return impl.params(impl.load_config(cfg))[0]


# ---------------------------------------------------------------- parameters, dense

def test_params_llama3_exact(impl):
    assert impl.params(impl.load_config(LLAMA3_8B)) == (8_030_261_248, 8_030_261_248)
    assert _total(impl, LLAMA31_405B) == 405_853_388_800


def test_params_dense_families(impl):
    # bias (Qwen2), QK-norm (Qwen3), full-width QK-norm (OLMo 2), no KV heads key (defaults to MHA)
    assert _total(impl, QWEN25_72B) == 72_706_203_648
    assert _total(impl, QWEN3_32B) == 32_762_123_264
    assert _total(impl, OLMO2_32B) == 32_234_279_936
    assert _total(impl, MISTRAL_7B) == 7_241_732_096
    mha = {k: v for k, v in LLAMA3_8B.items() if k != "num_key_value_heads"}
    assert _total(impl, mha) - _total(impl, LLAMA3_8B) == 32 * 2 * 4096 * (32 - 8) * 128    # GQA saves K and V columns


def _reference_count(d, L, n_h, n_kv, hd, d_ff, V, tie, qkv_bias=False, qk_norm=False, experts=0, d_e=0, shared=0):
    """Independent count: build modules with nn.Linear / nn.Parameter and sum numel."""
    def lin(i, o, bias=False):
        return nn.Linear(i, o, bias=bias)
    layers = nn.ModuleList()
    for _ in range(L):
        m = nn.ModuleDict(dict(
            q=lin(d, n_h * hd, qkv_bias), k=lin(d, n_kv * hd, qkv_bias), v=lin(d, n_kv * hd, qkv_bias),
            o=lin(n_h * hd, d)))
        m["norms"] = nn.ParameterList([nn.Parameter(torch.ones(d)) for _ in range(2)])
        if qk_norm:
            m["qk"] = nn.ParameterList([nn.Parameter(torch.ones(hd)) for _ in range(2)])
        if experts:
            m["router"] = lin(d, experts)
            m["experts"] = nn.ModuleList([nn.ModuleList([lin(d, d_e), lin(d, d_e), lin(d_e, d)]) for _ in range(experts)])
            if shared:
                m["shared"] = nn.ModuleList([lin(d, shared), lin(d, shared), lin(shared, d)])
        else:
            m["ffn"] = nn.ModuleList([lin(d, d_ff), lin(d, d_ff), lin(d_ff, d)])
        layers.append(m)
    top = nn.ModuleDict(dict(layers=layers, embed=nn.Embedding(V, d)))
    if not tie:
        top["head"] = lin(d, V)
    top["final_norm"] = nn.ParameterList([nn.Parameter(torch.ones(d))])
    return sum(p.numel() for p in top.parameters())


def test_params_against_torch_modules(impl):
    base = dict(hidden_size=24, num_hidden_layers=3, num_attention_heads=6, num_key_value_heads=2, head_dim=8,
                intermediate_size=40, vocab_size=50)
    cases = [
        (dict(base, model_type="llama", tie_word_embeddings=False), dict(tie=False)),
        (dict(base, model_type="llama", tie_word_embeddings=True), dict(tie=True)),
        (dict(base, model_type="llama"), dict(tie=True)),                            # key missing: tied by default
        (dict(base, model_type="llama", tie_word_embeddings=False, attention_bias=False), dict(tie=False)),
        (dict(base, model_type="qwen2", tie_word_embeddings=False), dict(tie=False, qkv_bias=True)),
        (dict(base, model_type="qwen3", tie_word_embeddings=False), dict(tie=False, qk_norm=True)),
    ]
    for cfg, kw in cases:
        got = impl.params(impl.load_config(cfg))[0]
        assert got == _reference_count(24, 3, 6, 2, 8, 40, 50, **kw), (cfg["model_type"], kw)


def test_params_multimodal_nesting(impl):
    wrapped = {"model_type": "llava", "tie_word_embeddings": False, "text_config": {k: v for k, v in LLAMA3_8B.items() if k != "tie_word_embeddings"}}
    assert _total(impl, wrapped) == 8_030_261_248                     # untied flag sits outside text_config


# ---------------------------------------------------------------- parameters, MoE and MLA

def test_params_moe_against_torch_modules(impl):
    cfg = dict(model_type="mixtral", hidden_size=16, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
               intermediate_size=24, num_local_experts=5, num_experts_per_tok=2, vocab_size=30, tie_word_embeddings=False)
    total, active = impl.params(impl.load_config(cfg))
    assert total == _reference_count(16, 2, 4, 4, 4, 0, 30, False, experts=5, d_e=24)
    assert total - active == 2 * 3 * (3 * 16 * 24)                    # 3 idle experts in each of 2 layers
    cfg2 = dict(model_type="deepseek_v3", hidden_size=16, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
                intermediate_size=48, moe_intermediate_size=12, n_routed_experts=6, n_shared_experts=2,
                num_experts_per_tok=2, first_k_dense_replace=1, kv_lora_rank=8, q_lora_rank=0, qk_nope_head_dim=4,
                qk_rope_head_dim=2, v_head_dim=4, vocab_size=30, tie_word_embeddings=False)
    s = impl.load_config(cfg2)
    assert s.moe_layers == (1, 2, 3) and s.d_shared == 24 and s.d_ff == 48 and s.q_lora_rank == 0
    total2, active2 = impl.params(s)
    dense_layer = 3 * 16 * 48
    moe_layer = 16 * 6 + 6 * 3 * 16 * 12 + 3 * 16 * 24
    mla = 16 * 4 * 6 + (16 * 10 + 8) + 8 * 4 * 8 + 4 * 4 * 16         # q, W_DKV|W_KR + norm, W_UK|W_UV, W_O
    assert total2 == 4 * (mla + 2 * 16) + dense_layer + 3 * moe_layer + 2 * 30 * 16 + 16
    assert total2 - active2 == 3 * 4 * (3 * 16 * 12)


def test_params_deepseek_v3(impl):
    total, both = impl.params(impl.load_config(DSV3), "both")
    assert total == 671_026_404_352
    assert both == 37_552_282_624
    assert impl.params(impl.load_config(DSV3), "head")[1] == 37_552_282_624 - 129280 * 7168 == 36_625_603_584
    assert impl.params(impl.load_config(DSV3), "none")[1] == 35_698_924_544      # page 08's non-embedding count
    with pytest.raises(ValueError):
        impl.params(impl.load_config(DSV3), "all")


def test_params_more_moe_models(impl):
    assert impl.params(impl.load_config(DSV2))[0] == 235_741_434_880
    assert impl.params(impl.load_config(MIXTRAL_8X7B)) == (46_702_792_704, 12_879_925_248)
    total, active = impl.params(impl.load_config(QWEN3_235B))
    assert (total, active) == (235_093_634_560, 22_190_763_520)
    # Kimi K2's config recount: the checkpoint holds 62.5M more elements (FP8 scale factors), and the report's 1.04T
    # is what you get if the dense first layer is wrongly counted as MoE
    assert impl.params(impl.load_config(KIMI_K2))[0] == 1_026_408_209_408
    # GLM-4.5: 351.2B without embeddings; the report's 355B adds a 3.9B MTP layer
    total, _ = impl.params(impl.load_config(GLM45))
    assert total == 352_797_814_784 and total - 2 * 151552 * 5120 == 351_245_922_304


# ---------------------------------------------------------------- KV cache

def test_kv_bytes_per_token(impl):
    assert impl.kv_bytes_per_token(impl.load_config(LLAMA3_8B)) == 131_072                # 128 KiB
    assert impl.kv_bytes_per_token(impl.load_config(LLAMA31_405B)) == 126 * 2 * 8 * 128 * 2
    assert impl.kv_bytes_per_token(impl.load_config(DSV3)) == 61 * (512 + 64) * 2 == 70_272
    assert impl.kv_bytes_per_token(impl.load_config(KIMI_K2)) == 70_272                   # 64 heads, same latent
    assert impl.kv_bytes_per_token(impl.load_config(QWEN3_235B)) == 94 * 2 * 4 * 128 * 2 == 192_512
    assert impl.kv_bytes_per_token(impl.load_config(LLAMA3_8B), bytes_per_el=1) == 65_536   # fp8 cache
    mha = {k: v for k, v in LLAMA3_8B.items() if k != "num_key_value_heads"}
    assert impl.kv_bytes_per_token(impl.load_config(mha)) == 4 * 131_072                  # MHA: 32 KV heads
    # GPT-3 175B (MHA, 96 layers, 96 heads of 128): 4.5 MiB per token
    gpt3 = impl.Spec(n_layers=96, d=12288, vocab=50257, tie_embeddings=True, n_heads=96, n_kv_heads=96, head_dim=128, d_ff=49152)
    assert impl.kv_bytes_per_token(gpt3) == 4_718_592


def test_kv_cache_windows_and_layouts(impl):
    mistral = impl.load_config(MISTRAL_7B)
    assert mistral.n_local == 32 and mistral.n_global == 0 and mistral.window == 4096
    assert impl.kv_bytes_per_token(mistral) == 0                                            # bounded, not growing
    assert impl.kv_cache_bytes(mistral, 100) == 100 * 32 * 2 * 8 * 128 * 2                  # window not yet full
    assert impl.kv_cache_bytes(mistral, 131_072) == 32 * 2 * 8 * 128 * 2 * 4096 == 2**29     # 0.5 GiB, flat
    llama = impl.load_config(LLAMA3_8B)
    assert impl.kv_cache_bytes(llama, 131_072) == 16 * 2**30                                # page 07: 16 GiB at 128K
    # Gemma-3-27B-like: 62 layers, 10 global, window 1,024, 16 KV heads of 128
    g3 = impl.Spec(n_layers=62, d=5376, vocab=262208, tie_embeddings=True, n_heads=32, n_kv_heads=16, head_dim=128,
                   d_ff=21504, n_global=10, n_local=52, window=1024)
    per_layer_token = 2 * 16 * 128 * 2
    assert impl.kv_bytes_per_token(g3) == 10 * per_layer_token
    assert impl.kv_cache_bytes(g3, 131_072) == per_layer_token * (10 * 131_072 + 52 * 1024)   # 10.41 GiB
    # a hybrid keeps a cache only in its softmax layers: Qwen3-Next has 12 of 48 with 2 KV heads of 256
    hybrid = impl.Spec(n_layers=48, d=2048, vocab=151936, tie_embeddings=False, n_heads=16, n_kv_heads=2, head_dim=256,
                       n_global=12)
    assert impl.kv_bytes_per_token(hybrid) == 12 * 2 * 2 * 256 * 2 == 24_576
    with pytest.raises(NotImplementedError):
        impl.params(hybrid)


def test_kv_cache_gemma4_shared_kv(impl):
    # Gemma 4 31B: 60 layers, 50 local (16 KV heads x 256, window 1,024), 10 global (4 KV heads x 512, keys reused as
    # values). The report's Table 3 lists +1.10 GB for an int8 cache at 32K tokens.
    g4 = impl.Spec(n_layers=60, d=5376, vocab=262144, tie_embeddings=True, n_heads=32, n_kv_heads=16, head_dim=256,
                   d_ff=21504, norms_per_layer=4, qk_norm="head", n_global=10, n_local=50, window=1024,
                   global_n_kv_heads=4, global_head_dim=512, global_k_eq_v=True)
    assert impl.kv_bytes_per_token(g4) == 10 * 4 * 512 * 2 == 40_960
    got = impl.kv_cache_bytes(g4, 32_768, bytes_per_el=1)
    assert got == 50 * 8192 * 1024 + 10 * 2048 * 32768
    assert abs(got / 1e9 - 1.10) < 0.01
    # storing V separately would double the global part: 1.76 GB, which the report's table rules out
    assert impl.kv_cache_bytes(impl.Spec(**{**g4.__dict__, "global_k_eq_v": False}), 32_768, bytes_per_el=1) / 1e9 > 1.7


# ---------------------------------------------------------------- FLOPs

def test_flops_llama3_8b(impl):
    s = impl.load_config(LLAMA3_8B)
    T = 8192
    matmul = 8_030_261_248 - 128256 * 4096          # the input embedding is a gather, not a matmul
    attn = 32 * 2 * 32 * (128 + 128) * (T / 2)      # scores + values, average T/2 keys
    assert impl.attention_flops_per_token(s, T) == attn == 2_147_483_648
    assert impl.flops_per_token(s, T) == 2 * matmul + attn == 17_157_332_992
    # attention grows linearly with T, the parameter term does not
    assert impl.attention_flops_per_token(s, 2 * T) == 2 * attn
    assert impl.flops_per_token(s, 2 * T) - impl.flops_per_token(s, T) == attn


def test_flops_mla_windows_and_moe(impl):
    v3 = impl.load_config(DSV3)
    T = 8192
    attn = 61 * 2 * 128 * (192 + 128) * (T / 2)      # decompressed MLA shapes: 192-dim keys, 128-dim values
    assert impl.attention_flops_per_token(v3, T) == attn
    total, act_head = impl.params(v3, "head")
    assert impl.flops_per_token(v3, T) == 2 * act_head + attn
    assert 2 * act_head < 2 * total / 15                                            # sparsity: far fewer than 2N
    # a window shorter than T cuts the local layers' term: window - window^2/(2T) keys instead of T/2
    s = impl.Spec(n_layers=4, d=64, vocab=100, tie_embeddings=True, n_heads=4, n_kv_heads=2, head_dim=16, d_ff=128,
                  n_global=1, n_local=3, window=1024)
    per_key = 2 * 4 * (16 + 16)
    keys_local = 1024 - 1024**2 / (2 * 8192)
    assert impl.attention_flops_per_token(s, 8192) == pytest.approx(per_key * (4096 + 3 * keys_local))
    assert impl.attention_flops_per_token(s, 512) == pytest.approx(per_key * 4 * 256)      # window >= T: all layers see T/2
    # GQA does not change attention FLOPs: query heads still each score every key
    mha = impl.Spec(n_layers=4, d=64, vocab=100, tie_embeddings=True, n_heads=4, n_kv_heads=4, head_dim=16, d_ff=128)
    gqa = impl.Spec(n_layers=4, d=64, vocab=100, tie_embeddings=True, n_heads=4, n_kv_heads=1, head_dim=16, d_ff=128)
    assert impl.attention_flops_per_token(mha, 1000) == impl.attention_flops_per_token(gqa, 1000)


# ---------------------------------------------------------------- load_config details

def test_load_config_fields(impl):
    s = impl.load_config(QWEN3_235B)
    assert (s.n_layers, s.d, s.n_heads, s.n_kv_heads, s.head_dim) == (94, 4096, 64, 4, 128)
    assert s.qk_norm == "head" and not s.qkv_bias and s.d_ff == 0 and s.d_shared == 0
    assert s.moe_layers == tuple(range(94)) and (s.n_experts, s.top_k, s.d_expert) == (128, 8, 1536)
    s = impl.load_config(DSV3)
    assert s.moe_layers == tuple(range(3, 61)) and s.d_ff == 18432 and s.d_shared == 2048
    assert (s.q_lora_rank, s.kv_lora_rank, s.qk_nope_dim, s.qk_rope_dim, s.v_dim) == (1536, 512, 128, 64, 128)
    s = impl.load_config(QWEN25_72B)
    assert s.qkv_bias and not s.o_bias and s.n_global is None       # the switched-off window is ignored
    s = impl.load_config(OLMO2_32B)
    assert s.qk_norm == "full"
    s = impl.load_config(GLM45)
    assert s.qkv_bias and not s.o_bias and s.moe_layers == tuple(range(3, 92)) and s.d_shared == 1536
    # Qwen3-MoE layer placement: mlp_only_layers and decoder_sparse_step
    q = dict(QWEN3_235B, num_hidden_layers=6, mlp_only_layers=[0], decoder_sparse_step=2)
    assert impl.load_config(q).moe_layers == (1, 3, 5) and impl.load_config(q).d_ff == 12288
    q = dict(QWEN3_235B, num_hidden_layers=6, mlp_only_layers=[1, 5], decoder_sparse_step=1)
    assert impl.load_config(q).moe_layers == (0, 2, 3, 4)
    # DeepSeek layer frequency
    v = dict(DSV3, num_hidden_layers=8, first_k_dense_replace=2, moe_layer_freq=2)
    assert impl.load_config(v).moe_layers == (2, 4, 6)
    # None-valued keys count as missing
    assert impl.load_config(dict(LLAMA3_8B, num_key_value_heads=None)).n_kv_heads == 32
