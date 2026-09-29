"""Reference solutions: DeepSeek parameter accounting and hyperparameter laws (page 26)."""
from typing import Mapping, Tuple


def mla_params(cfg: Mapping) -> int:
    """Parameters of one MLA layer, latent RMSNorms included."""
    d, n_h = cfg["hidden_size"], cfg["num_attention_heads"]
    d_nope, d_rope, d_v = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"], cfg["v_head_dim"]
    d_c, d_cq = cfg["kv_lora_rank"], cfg.get("q_lora_rank")
    if d_cq:
        q = d * d_cq + d_cq + d_cq * n_h * (d_nope + d_rope)   # W_DQ, its RMSNorm, fused W_UQ | W_QR
    else:
        q = d * n_h * (d_nope + d_rope)                         # no query compression (V2-Lite)
    kv = d * (d_c + d_rope) + d_c                               # fused W_DKV | W_KR, KV latent RMSNorm
    kv += d_c * n_h * (d_nope + d_v)                            # fused W_UK | W_UV
    o = n_h * d_v * d                                           # output projection
    return q + kv + o


def moe_layer_params(cfg: Mapping) -> Tuple[int, int]:
    """(total, activated) of one MoE FFN layer."""
    d, de = cfg["hidden_size"], cfg["moe_intermediate_size"]
    n_r, n_s, k = cfg["n_routed_experts"], cfg["n_shared_experts"], cfg["num_experts_per_tok"]
    expert = 3 * d * de                                         # SwiGLU: gate, up, down
    router = d * n_r                                            # every token scores all routed experts
    return (n_r + n_s) * expert + router, (k + n_s) * expert + router


def count_params(cfg: Mapping, active_embeddings: str = "both") -> Tuple[int, int]:
    if active_embeddings not in ("both", "head", "none"):
        raise ValueError(f"active_embeddings must be 'both', 'head' or 'none', got {active_embeddings!r}")
    L, d, V = cfg["num_hidden_layers"], cfg["hidden_size"], cfg["vocab_size"]
    n_dense = cfg["first_k_dense_replace"]
    n_moe = L - n_dense
    dense_ffn = 3 * d * cfg["intermediate_size"]
    moe_total, moe_active = moe_layer_params(cfg)
    shared = L * (mla_params(cfg) + 2 * d) + d + n_dense * dense_ffn   # attention, norms, dense FFN: always active
    total = shared + n_moe * moe_total + 2 * V * d
    n_emb = {"both": 2, "head": 1, "none": 0}[active_embeddings]
    active = shared + n_moe * moe_active + n_emb * V * d
    return total, active


def model_scale_flops(n_layer: int, d_model: int, l_seq: int) -> float:
    return 72 * n_layer * d_model**2 + 12 * n_layer * d_model * l_seq


def hp_scaling(C: float) -> Tuple[float, float]:
    return 0.2920 * C**0.3271, 0.3118 * C**-0.1250


def optimal_allocation(C: float) -> Tuple[float, float]:
    return 0.1715 * C**0.5243, 5.8316 * C**0.4757
