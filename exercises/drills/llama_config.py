"""Drill (page 29): Llama configs, Meta's FFN rounding rule, exact parameter counts, KV cache.

Implement the functions below. Run:
    pytest exercises/tests/test_llama_config.py
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LlamaConfig:
    """A dense Llama-style decoder: pre-RMSNorm, GQA, SwiGLU FFN, RoPE, no biases anywhere."""
    dim: int                        # model width d
    n_layers: int                   # L
    n_heads: int                    # query heads n_h
    n_kv_heads: int                 # key/value heads n_kv (n_kv == n_h means plain multi-head attention)
    vocab_size: int                 # V
    ffn_hidden: int                 # SwiGLU hidden width, already rounded (see ffn_hidden_dim)
    head_dim: Optional[int] = None  # d_h; None means dim // n_heads
    tie_embeddings: bool = False    # True: the output projection reuses the input embedding matrix


@dataclass(frozen=True)
class Llama4Config:
    """Text backbone of a Llama-4-style MoE (vision encoder excluded)."""
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    dense_ffn: int                  # SwiGLU width of the dense (non-MoE) layers
    expert_ffn: int                 # SwiGLU width of every routed expert AND of the shared expert
    n_experts: int                  # routed experts per MoE layer
    moe_layers: tuple               # 0-based indices of the layers whose FFN is an MoE block
    top_k: int = 1                  # routed experts used per token
    tie_embeddings: bool = False


def ffn_hidden_dim(d: int, multiple_of: int, ffn_dim_multiplier: Optional[float] = None) -> int:
    """Hidden width of the SwiGLU FFN, following the rounding rule in Meta's Llama reference code.

    Start from h = 4 * d. Then, in this order:
      1. h = int(2 * h / 3)                       (the 2/3 rule that keeps parameters close to a 4d GELU FFN)
      2. if ffn_dim_multiplier is not None: h = int(ffn_dim_multiplier * h)   (truncate toward zero)
      3. round h UP to the next multiple of multiple_of.
    Returns the resulting int. Example: d=4096, multiple_of=256, no multiplier -> 11008 (Llama 1 7B).
    """
    raise NotImplementedError


def param_breakdown(cfg: LlamaConfig) -> dict:
    """Exact parameter counts of a dense Llama, split into groups.

    Keys (all ints, all summed over layers where relevant):
      "embedding": token embedding matrix, V * d
      "attention": q, k, v and output projections. q is d x (n_h*d_h), k and v are each d x (n_kv*d_h),
                   the output projection is (n_h*d_h) x d
      "ffn":       three SwiGLU matrices (gate, up, down), each d x ffn_hidden
      "norm":      two RMSNorm weight vectors (size d) per layer plus one final RMSNorm (size d)
      "lm_head":   output projection V * d, or 0 if cfg.tie_embeddings
      "total":     the sum of the five groups above
    There are no biases and RoPE has no parameters.
    """
    raise NotImplementedError


def llama_params(cfg: LlamaConfig) -> int:
    """Total number of parameters of the model (an exact int; equals param_breakdown(cfg)["total"])."""
    raise NotImplementedError


def llama4_params(cfg: Llama4Config) -> tuple:
    """(total, active) parameter counts of the text backbone of a Llama-4-style MoE.

    Every layer has attention (q, k, v, o as in LlamaConfig, no biases) and two RMSNorm vectors.
    Dense layers have a SwiGLU FFN of width dense_ffn. Layers listed in cfg.moe_layers instead have:
      n_experts routed SwiGLU experts of width expert_ffn, ONE shared expert of the same width,
      and a router matrix d x n_experts (no bias).
    Embedding, untied output head (unless tie_embeddings) and a final RMSNorm are counted as usual.
    total  = every parameter.
    active = parameters touched by one token: all attention, norm, embedding and head parameters,
             dense FFNs, and in each MoE layer the shared expert, top_k routed experts and the router.
    """
    raise NotImplementedError


def kv_cache_bytes(n_layers: int, n_kv_heads: int, head_dim: int, n_tokens: int,
                   n_global_layers: Optional[int] = None, local_window: Optional[int] = None,
                   bytes_per_el: int = 2) -> int:
    """Bytes of KV cache for ONE sequence of n_tokens tokens.

    Each cached token of one layer stores a key and a value: 2 * n_kv_heads * head_dim elements.
    If n_global_layers is None, all layers are global and keep every token.
    Otherwise n_global_layers layers keep every token and the remaining (n_layers - n_global_layers)
    local layers keep only the most recent min(n_tokens, local_window) tokens (local_window is then required).
    Returns an exact int.
    """
    raise NotImplementedError
