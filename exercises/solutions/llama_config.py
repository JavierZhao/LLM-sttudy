"""Reference solutions: Llama configs, FFN rounding rule, parameter counts, KV cache (page 29)."""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LlamaConfig:
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    ffn_hidden: int
    head_dim: Optional[int] = None
    tie_embeddings: bool = False


@dataclass(frozen=True)
class Llama4Config:
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    dense_ffn: int
    expert_ffn: int
    n_experts: int
    moe_layers: tuple
    top_k: int = 1
    tie_embeddings: bool = False


def ffn_hidden_dim(d: int, multiple_of: int, ffn_dim_multiplier: Optional[float] = None) -> int:
    h = int(2 * (4 * d) / 3)                      # 2/3 of 4d, truncated
    if ffn_dim_multiplier is not None:
        h = int(ffn_dim_multiplier * h)           # truncate again after the multiplier
    return multiple_of * ((h + multiple_of - 1) // multiple_of)   # ceil to a multiple


def _attn_params(dim, n_heads, n_kv_heads, head_dim):
    # q and o are d x (n_h d_h); k and v are d x (n_kv d_h): GQA only shrinks k and v
    return dim * n_heads * head_dim * 2 + dim * n_kv_heads * head_dim * 2


def param_breakdown(cfg: LlamaConfig) -> dict:
    d, L = cfg.dim, cfg.n_layers
    d_h = cfg.head_dim if cfg.head_dim is not None else d // cfg.n_heads
    out = {
        "embedding": cfg.vocab_size * d,
        "attention": L * _attn_params(d, cfg.n_heads, cfg.n_kv_heads, d_h),
        "ffn": L * 3 * d * cfg.ffn_hidden,          # gate, up, down
        "norm": (2 * L + 1) * d,                    # attn-norm and ffn-norm per layer, plus the final norm
        "lm_head": 0 if cfg.tie_embeddings else cfg.vocab_size * d,
    }
    out["total"] = sum(out.values())
    return out


def llama_params(cfg: LlamaConfig) -> int:
    return param_breakdown(cfg)["total"]


def llama4_params(cfg: Llama4Config) -> tuple:
    d, V = cfg.dim, cfg.vocab_size
    attn = _attn_params(d, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim)
    expert = 3 * d * cfg.expert_ffn
    dense = 3 * d * cfg.dense_ffn
    shared = V * d + (0 if cfg.tie_embeddings else V * d) + d   # embedding + head + final norm
    total = active = shared
    moe = set(cfg.moe_layers)
    for i in range(cfg.n_layers):
        total += attn + 2 * d
        active += attn + 2 * d
        if i in moe:
            router = d * cfg.n_experts
            total += cfg.n_experts * expert + expert + router      # all routed + shared + router
            active += cfg.top_k * expert + expert + router         # top_k routed + shared + router
        else:
            total += dense
            active += dense
    return total, active


def kv_cache_bytes(n_layers: int, n_kv_heads: int, head_dim: int, n_tokens: int,
                   n_global_layers: Optional[int] = None, local_window: Optional[int] = None,
                   bytes_per_el: int = 2) -> int:
    per_token_per_layer = 2 * n_kv_heads * head_dim * bytes_per_el   # K and V
    if n_global_layers is None:
        return n_layers * per_token_per_layer * n_tokens
    assert local_window is not None, "local layers need a window"
    n_local = n_layers - n_global_layers
    return per_token_per_layer * (n_global_layers * n_tokens + n_local * min(n_tokens, local_window))
