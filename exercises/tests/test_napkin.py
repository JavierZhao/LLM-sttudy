import math

MODULE = "napkin"

GB = 1e9
GiB = 2**30
H100_BF16 = 989.4e12
H100_FP8 = 1978.9e12
H100_BW = 3.35e12
NVLINK = 450e9
IB = 50e9

LLAMA3_8B = dict(L=32, d=4096, n_h=32, n_kv=8, d_h=128, d_ff=14336, V=128256)
LLAMA3_70B = dict(L=80, d=8192, n_h=64, n_kv=8, d_h=128, d_ff=28672, V=128256)
LLAMA3_405B = dict(L=126, d=16384, n_h=128, n_kv=8, d_h=128, d_ff=53248, V=128256)
DSV3_MLA = dict(d=7168, n_h=128, d_nope=128, d_rope=64, d_v=128, d_c=512, d_cq=1536)
DSV3_MOE = dict(L=61, L_moe=58, d=7168, attn=187_107_328, n_routed=256, n_shared=1, k=8,
                d_e=2048, d_dense=18432, V=129280)


def close(a, b, rel=1e-3):
    assert math.isclose(a, b, rel_tol=rel), f"{a} vs {b}"


# ---- parameters and FLOPs (page drills 1 to 5) ---------------------------------------------------
def test_llama3_8b_params_exact(impl):
    assert impl.params(**LLAMA3_8B) == 8_030_261_248


def test_llama3_70b_and_405b_params(impl):
    assert impl.params(**LLAMA3_70B) == 70_553_706_496
    assert impl.params(**LLAMA3_405B) == 405_853_388_800


def test_tied_embeddings_save_one_vocab_matrix(impl):
    n = impl.params(**LLAMA3_8B)
    assert n - impl.params(**LLAMA3_8B, tied=True) == 128256 * 4096


def test_per_layer_coefficient_is_not_13(impl):
    # (attention + FFN) / d^2 is 13 for 8B, 12.75 for 70B, 11.875 for 405B: rederive it per model
    for cfg, coef in ((LLAMA3_8B, 13.0), (LLAMA3_70B, 12.75), (LLAMA3_405B, 11.875)):
        n = impl.params(**cfg)
        rest = n - 2 * cfg["V"] * cfg["d"] - cfg["d"]
        per_layer = rest / cfg["L"] - 2 * cfg["d"]
        close(per_layer / cfg["d"] ** 2, coef, 1e-9)


def test_mla_layer_params(impl):
    assert impl.mla_attn_params(**DSV3_MLA) == 187_107_328
    # tiny config with d_v != d_nope: 96 + 72 + 112 + 70 + 96 + 11 (matrices, then the two latent norms)
    assert impl.mla_attn_params(d=16, n_h=2, d_nope=4, d_rope=2, d_v=3, d_c=5, d_cq=6) == 457


def test_deepseek_v3_total_and_activated(impl):
    total, active = impl.moe_params(**DSV3_MOE)
    assert total == 671_026_404_352
    assert active == 37_552_282_624
    # non-embedding convention: drop both vocabulary matrices
    assert active - 2 * 129280 * 7168 == 35_698_924_544


def test_flops_per_token(impl):
    n_mm = 7_504_658_432   # Llama-3-8B blocks plus LM head, no embedding table, no norms
    close(impl.flops(8_030_261_248), 16.06e9, 1e-3)
    fwd = impl.flops(n_mm, L=32, d=4096, T=8192, causal=False)   # 4LTd counting
    close(fwd, 2 * n_mm + 4.294967296e9, 1e-9)
    close(4.294967296e9 / fwd, 0.2225, 2e-3)
    close(impl.flops(n_mm, L=32, d=4096, T=8192, causal=True, training=True), 51.4704e9, 1e-4)
    assert impl.flops(n_mm, training=True) == 3 * impl.flops(n_mm)


# ---- memory (drills 6 to 9) ----------------------------------------------------------------------
def test_training_states_16_bytes_per_param(impl):
    n70 = 70_553_706_496
    close(impl.train_memory(n70) / GB, 1128.86, 1e-4)
    close(impl.train_memory(n70) / 80e9, 14.11, 1e-3)


def test_zero_stages_llama_8b_on_64_gpus(impl):
    n = 8_030_261_248
    got = [impl.train_memory(n, s, 64) / GB for s in range(4)]
    for g, want in zip(got, (128.48, 33.63, 17.82, 2.01)):
        close(g, want, 2e-3)
    assert got[0] > got[1] > got[2] > got[3]


def test_zero_matches_paper_table(impl):
    # ZeRO paper Table 1: 7.5B parameters, 64 GPUs
    got = [impl.train_memory(7.5e9, s, 64) / GB for s in range(4)]
    for g, want in zip(got, (120.0, 31.41, 16.64, 1.875)):
        close(g, want, 1e-3)


def test_activation_memory_llama_8b(impl):
    args = dict(s=8192, b=1, h=4096, a=32, n_layers=32)
    close(impl.activation_memory(**args) / GB, 380.10, 1e-4)
    close(impl.activation_memory(**args, mode="selective") / GB, 36.51, 1e-3)
    close(impl.activation_memory(**args, mode="recompute") / GB, 2.147, 1e-3)


def test_405b_states_and_min_gpus(impl):
    states = impl.train_memory(405e9)
    close(states / 1e12, 6.48, 1e-9)
    close(states / 80e9, 81.0, 1e-9)
    close(impl.train_memory(405e9) / 128 / GB, 50.6, 2e-3)   # TP8 x PP16 splits states 128 ways


# ---- training time and cost (drills 10 to 14) ----------------------------------------------------
def test_llama_8b_gpu_hours_and_cost(impl):
    C = impl.train_flops(8_030_261_248, 15e12)
    close(C, 7.2272e23, 1e-4)
    gh = impl.gpu_hours(C, H100_BF16, 0.40)
    close(gh, 507_268, 1e-4)
    close(impl.dollar_cost(gh), 1_014_537, 1e-4)
    close(impl.train_time(C, 1024, H100_BF16, 0.40) / 86400, 20.64, 1e-3)


def test_405b_training_time_on_16k_gpus(impl):
    C = impl.train_flops(405e9, 15.6e12)
    close(C, 3.7908e25, 1e-4)
    for mfu, days in ((0.40, 67.67), (0.38, 71.23), (0.43, 62.94)):
        close(impl.train_time(C, 16384, H100_BF16, mfu) / 86400, days, 1e-3)
    # the model card's 30.84M GPU-hours imply an MFU of about 34.5%
    implied = C / (30.84e6 * 3600 * H100_BF16)
    close(implied, 0.345, 5e-3)


def test_deepseek_v3_implied_mfu_and_cost(impl):
    C = impl.train_flops(37e9, 14.8e12)
    close(C, 3.2856e24, 1e-4)
    close(impl.gpu_hours(C, H100_BF16, 0.3463), 2.664e6, 1e-3)
    tok_per_s = 1e12 / (180_000 * 3600)
    close(tok_per_s, 1543.2, 1e-4)
    close(tok_per_s * 6 * 37e9 / H100_BF16, 0.3463, 1e-3)
    close(tok_per_s * 6 * 37e9 / H100_FP8, 0.1731, 1e-3)
    close(impl.dollar_cost(2.788e6), 5.576e6, 1e-9)


def test_rl_rollout_cost(impl):
    tokens = 1e12
    gh = tokens / 2500 / 3600
    close(gh, 111_111, 1e-5)
    close(impl.dollar_cost(gh), 222_222, 1e-5)


def test_gpus_for_a_30_day_70b_run(impl):
    C = impl.train_flops(70_553_706_496, 15e12)
    n_gpu = C / (30 * 86400 * H100_BF16 * 0.40)
    close(n_gpu, 6190, 1e-3)
    close(impl.train_time(C, 1024, H100_BF16, 0.40) / 86400, 181.35, 1e-3)
    close(impl.dollar_cost(impl.gpu_hours(C, H100_BF16, 0.40)), 8.91e6, 2e-3)


# ---- inference (drills 15 to 21) -----------------------------------------------------------------
W8 = 2 * 8_030_261_248
KAPPA8 = 131_072


def test_decode_batch_1_and_64(impl):
    t1 = impl.decode_time(W8, 0, H100_BW)
    close(t1 * 1e3, 4.794, 1e-3)
    close(1 / t1, 208.6, 1e-3)
    t1k = impl.decode_time(W8, 4096 * KAPPA8, H100_BW)
    close(1 / t1k, 201.8, 1e-3)
    t64 = impl.decode_time(W8, impl.kv_bytes(32, 8, 128, 4096, 64), H100_BW)
    close(t64 * 1e3, 15.05, 1e-3)
    close(64 / t64, 4252, 1e-3)
    close(1 / t64, 66.4, 1e-3)


def test_decode_is_memory_bound_at_batch_64(impl):
    n_mm = 7_504_658_432
    flops = 64 * (2 * n_mm + 4 * 32 * 4096 * 4096)
    t = impl.decode_time(W8, impl.kv_bytes(32, 8, 128, 4096, 64), H100_BW, flops, H100_BF16)
    close(t, 15.05e-3, 1e-3)                       # the memory term wins
    t_c = impl.decode_time(0, 0, H100_BW, flops, H100_BF16)
    close(t_c, flops / H100_BF16, 1e-12)           # compute alone is about 1.1 ms
    assert t_c < 2e-3
    # a compute-bound step: 1e15 FLOPs at 1e15 FLOP/s is 1 s, far above 1 GB / 1 TB/s = 1 ms
    close(impl.decode_time(1e9, 0, 1e12, 1e15, 1e15), 1.0, 1e-12)
    close(impl.decode_time(1e9, 0, 1e12), 1e-3, 1e-12)   # default peak_flops: no compute term


def test_prefill_2048_tokens(impl):
    close(impl.prefill_time(8_030_261_248, 2048, H100_BF16, 1.0) * 1e3, 33.24, 1e-3)
    close(impl.prefill_time(8_030_261_248, 2048, H100_BF16, 0.5) * 1e3, 66.49, 1e-3)


def test_tpot_slo_batch_limit(impl):
    slo = 0.020
    per_seq = impl.kv_bytes(32, 8, 128, 4096)
    b = 1
    while impl.decode_time(W8, impl.kv_bytes(32, 8, 128, 4096, b + 1), H100_BW) <= slo:
        b += 1
    assert b == 94
    t = impl.decode_time(W8, b * per_seq, H100_BW)
    close(b / t, 4733, 1e-3)


def test_cost_per_million_tokens(impl):
    t = impl.decode_time(W8, impl.kv_bytes(32, 8, 128, 4096, 104), H100_BW)
    close(impl.cost_per_million_tokens(2.0, 1, 104 / t), 0.1146, 2e-3)
    t1 = impl.decode_time(W8, impl.kv_bytes(32, 8, 128, 4096, 1), H100_BW)
    close(impl.cost_per_million_tokens(2.0, 1, 1 / t1), 2.752, 2e-3)
    # tokens_per_s is the TOTAL rate: 8 GPUs at 8x the rate cost the same per token
    close(impl.cost_per_million_tokens(2.0, 8, 8 / t1), impl.cost_per_million_tokens(2.0, 1, 1 / t1), 1e-12)
    # DeepSeek's decode node: 8 H800 at $2 = $16 per node-hour, 14.8k output tokens/s (page 17)
    close(impl.cost_per_million_tokens(2.0, 8, 14_800), 0.3003, 1e-3)


# ---- KV cache (drills 22 to 25) ------------------------------------------------------------------
def test_kv_per_token_and_128k(impl):
    assert impl.kv_bytes(32, 8, 128, 1) == 131_072
    assert impl.kv_bytes(32, 8, 128, 131_072) / GiB == 16.0
    assert impl.kv_bytes(32, 32, 128, 131_072) / GiB == 64.0       # MHA
    assert impl.kv_bytes(32, 8, 128, 1, bytes_per_el=1) == 65_536   # fp8


def test_kv_70b_32_users_32k(impl):
    kv = impl.kv_bytes(80, 8, 128, 32_768, batch=32)
    assert kv / GiB == 320.0
    close(kv / GB, 343.6, 1e-4)
    weights = 2 * 70_553_706_496
    close((kv + weights) / GB, 484.7, 1e-4)
    close((kv + weights) / 72e9, 6.73, 2e-3)               # at 90% of 80 GB
    close(impl.kv_bytes(80, 8, 128, 32_768, batch=32, bytes_per_el=1) / GB, 171.8, 1e-3)


def test_mla_versus_gqa_kv(impl):
    assert impl.kv_bytes(61, 0, 0, 1, latent_dim=576) == 70_272
    mla = impl.kv_bytes(61, 0, 0, 131_072, latent_dim=576)
    gqa = impl.kv_bytes(61, 8, 128, 131_072)
    close(mla / GiB, 8.578, 1e-3)
    close(gqa / GiB, 30.5, 1e-3)
    close(gqa / mla, 3.556, 1e-3)


def test_kv_transfer_time(impl):
    kv = impl.kv_bytes(80, 8, 128, 2048)
    close(kv / 1e6, 671.1, 1e-3)
    close(kv / IB * 1e3, 13.4, 5e-3)
    mla = impl.kv_bytes(61, 0, 0, 2048, latent_dim=576)
    close(mla / 1e6, 143.9, 1e-3)


# ---- communication (drills 26 to 29) -------------------------------------------------------------
def test_allreduce_8b_over_8_gpus_nvlink(impl):
    S = 2 * 8_030_261_248
    close(impl.allreduce_time(S, 8, NVLINK) * 1e3, 62.46, 1e-3)
    close(impl.allreduce_time(S, 8, 2 * NVLINK) * 1e3, 31.23, 1e-3)    # the bidirectional-number mistake
    close(impl.allreduce_time(2 * S, 8, NVLINK) * 1e3, 124.9, 1e-3)    # fp32 gradients


def test_allreduce_bytes_do_not_grow_with_n(impl):
    a = impl.allreduce_time(1e9, 8, 1e9)
    b = impl.allreduce_time(1e9, 1024, 1e9)
    assert a < b < 2.0 * 1.0001                                   # tends to 2 S / BW
    close(impl.allreduce_time(1e9, 2, 1e9), 1.0, 1e-12)


def test_ddp_allreduce_over_64_gpus(impl):
    S = 2 * 8_030_261_248
    t = impl.allreduce_time(S, 64, IB)
    close(t, 0.632, 2e-3)
    step = 6 * 8_030_261_248 * 65_536 / (0.40 * H100_BF16)
    close(step, 7.98, 2e-3)
    close(t / step, 0.0793, 5e-3)


def test_pipeline_bubble(impl):
    rel, idle = impl.pipeline_bubble(16, 64)
    close(rel, 15 / 64, 1e-12)
    close(idle, 15 / 79, 1e-12)
    rel4, idle4 = impl.pipeline_bubble(16, 64, v=4)
    close(rel4, 15 / 256, 1e-12)
    close(idle4, 15 / 271, 1e-12)
    assert impl.pipeline_bubble(1, 8) == (0.0, 0.0)


def test_moe_alltoall_bytes(impl):
    assert impl.moe_alltoall_bytes(7168) == 21_504
    assert impl.moe_alltoall_bytes(7168, n_dest=4) == 86_016
    per_token_fwd = 58 * impl.moe_alltoall_bytes(7168, n_dest=4)
    assert per_token_fwd == 4_988_928
    tok_per_s = 1e12 / (180_000 * 3600)
    close(2 * tok_per_s * per_token_fwd / 1e9, 15.40, 1e-3)       # forward + backward, GB/s per GPU


# ---- scaling laws (drills 30 to 32) --------------------------------------------------------------
def test_chinchilla_rule_of_thumb(impl):
    N, D = impl.chinchilla_opt(1e23, tokens_per_param=20)
    close(N / 1e9, 28.87, 1e-3)
    close(D / 1e9, 577.35, 1e-3)
    close(6 * N * D, 1e23, 1e-12)


def test_chinchilla_closed_form_epoch(impl):
    for C, n, d in ((1e22, 9.045, 184.3), (1e23, 29.45, 566.0), (1e24, 95.86, 1738.6), (3.8e25, 618.7, 10237.1)):
        N, D = impl.chinchilla_opt(C)
        close(N / 1e9, n, 1e-3)
        close(D / 1e9, d, 1e-3)
        close(6 * N * D, C, 1e-9)


def test_chinchilla_printed_constants_give_the_artifact(impl):
    # Hoffmann's rounded Approach-3 constants imply ~78 tokens/param at 1e23 (see page 13)
    N, D = impl.chinchilla_opt(1e23, A=406.4, B=410.7, alpha=0.34, beta=0.28)
    close(N / 1e9, 14.6, 5e-3)
    close(D / N, 78.2, 5e-3)


def test_overtraining_ratio_llama_8b(impl):
    C = impl.train_flops(8_030_261_248, 15e12)
    N, D = impl.chinchilla_opt(C, tokens_per_param=20)
    close(N / 1e9, 77.6, 1e-3)
    close(15e12 / 8_030_261_248, 1867.9, 1e-4)
    close(15e12 / 8_030_261_248 / 20, 93.4, 1e-3)


# ---- MoE serving (drills 33 and 34) --------------------------------------------------------------
def test_v3_weight_stream_on_8_h200(impl):
    t = impl.decode_time(671_026_404_352, 0, 8 * 4.8e12)
    close(t * 1e3, 17.47, 1e-3)
    close(1 / t, 57.2, 2e-3)
    # batch 128 touches 98% of the routed experts, not all: page 17's 17.2 ms
    routed = 58 * 256 * 3 * 7168 * 2048
    frac = impl.experts_touched(256, 8, 128) / 256
    t128 = impl.decode_time(671_026_404_352 - (1 - frac) * routed, 0, 8 * 4.8e12)
    close(t128 * 1e3, 17.18, 1e-3)
    close(671.03e9 / 80e9, 8.39, 1e-3)


def test_experts_touched(impl):
    close(impl.experts_touched(256, 8, 1), 8.0, 1e-12)
    close(impl.experts_touched(256, 8, 64), 222.44, 1e-4)
    close(impl.experts_touched(256, 8, 64) / 256, 0.869, 1e-3)
    assert impl.experts_touched(256, 8, 100_000) > 255.99
    # tokens per expert reach the 295 ridge only at b = 295 * 256 / 8 tokens in flight
    assert 295 * 256 // 8 == 9440
