"""Drill (page 33): the config zoo. Read a config.json, count parameters, KV cache and FLOPs.

Implement the six functions below. `Spec` is given. Run:
    pytest exercises/tests/test_config_zoo.py

Conventions: a multiply-add is 2 FLOPs; the default cache element is bf16 (2 bytes); norm gains are
parameters (RMSNorm: one vector of size d); rotary tables and other buffers are not; FFNs are gated
(SwiGLU or GeGLU: three matrices of shape d x width); no biases unless a field says so.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple


@dataclass(frozen=True)
class Spec:
    # backbone
    n_layers: int
    d: int
    vocab: int
    tie_embeddings: bool
    # attention (GQA family; MHA is n_kv_heads == n_heads)
    n_heads: int
    n_kv_heads: int
    head_dim: int
    qkv_bias: bool = False
    o_bias: bool = False
    qk_norm: str = "none"            # "none" | "head" (q and k gains of size head_dim) | "full" (n_heads*head_dim and n_kv_heads*head_dim)
    norms_per_layer: int = 2         # RMSNorm gain vectors of size d per layer (Gemma: 4)
    # MLA (kv_lora_rank == 0 means the model is not MLA; head_dim is then unused)
    q_lora_rank: int = 0             # 0: queries are one d x n_heads*(qk_nope_dim+qk_rope_dim) matrix
    kv_lora_rank: int = 0
    qk_nope_dim: int = 0
    qk_rope_dim: int = 0
    v_dim: int = 0
    # FFN: dense layers have width d_ff; MoE layers are listed by index
    d_ff: int = 0
    moe_layers: Tuple[int, ...] = ()
    n_experts: int = 0               # routed experts per MoE layer (router: d x n_experts)
    top_k: int = 0
    d_expert: int = 0                # width of one routed expert
    d_shared: int = 0                # total width of the always-on branch of each MoE layer (0: none)
    # attention layout: n_global full-context layers, n_local windowed layers; any layer that is
    # neither is a linear-attention or SSM layer (no KV cache, no T-dependent FLOPs)
    n_global: Optional[int] = None   # None means every layer is global
    n_local: int = 0
    window: int = 0
    global_n_kv_heads: int = 0       # global layers with a different shape (0: same as the local shape)
    global_head_dim: int = 0
    global_k_eq_v: bool = False      # global layers have no V projection and cache K once


def params(s: Spec, active_embeddings: str = "both") -> Tuple[int, int]:
    """(total, activated) parameter counts of the decoder (no vision tower, no MTP module).

    Per layer: attention, `norms_per_layer` norm gains of size d, and either a dense FFN (3*d*d_ff) or,
    for layers listed in `moe_layers`, a router (d*n_experts), `n_experts` routed experts of 3*d*d_expert
    and an always-on branch of 3*d*d_shared. Attention parameters:
      * GQA: W_q (d x n_heads*head_dim), W_k and W_v (d x n_kv_heads*head_dim), W_o (n_heads*head_dim x d);
        biases if qkv_bias (q, k, v) or o_bias (d numbers); QK-norm gains as described on Spec.
      * MLA (kv_lora_rank > 0): W_DQ (d x q_lora_rank) with an RMSNorm of q_lora_rank gains and W_UQ|W_QR
        (q_lora_rank x n_heads*(qk_nope_dim+qk_rope_dim)); W_DKV|W_KR (d x (kv_lora_rank+qk_rope_dim)) with an
        RMSNorm of kv_lora_rank gains; W_UK|W_UV (kv_lora_rank x n_heads*(qk_nope_dim+v_dim));
        W_O (n_heads*v_dim x d). With q_lora_rank == 0 the query is a single d x n_heads*(qk_nope_dim+qk_rope_dim)
        matrix and has no norm.
      * The n_global global layers use global_head_dim and global_n_kv_heads when global_head_dim > 0
        (else the shared head_dim and n_kv_heads); with global_k_eq_v they have no W_v, so K is counted once.
    Add the input embedding (vocab*d), the output head (another vocab*d unless tie_embeddings) and one final
    norm of size d. Raise NotImplementedError if n_local + n_global != n_layers (linear-attention layers; n_global
    None counts as n_layers).

    "Activated" is total minus the routed experts a token does not select ((n_experts - top_k)*3*d*d_expert per
    MoE layer), then adjusted for embeddings by `active_embeddings`:
      "both": keep the embedding table and the head (a tied matrix counts once);
      "head": drop the input table, keep the head matmul (a tied matrix still counts once);
      "none": drop both.
    Raise ValueError for any other value. Must give 8,030,261,248 for Llama-3-8B and
    (671,026,404,352, 37,552,282,624) for DeepSeek-V3.
    """
    raise NotImplementedError


def kv_bytes_per_token(s: Spec, bytes_per_el: int = 2) -> int:
    """Marginal KV-cache growth per token once every window is full: only the n_global layers count.

    A softmax layer caches, per token: 2 * n_kv_heads * head_dim numbers (GQA; the global shape when
    global_head_dim > 0; only n_kv_heads * head_dim when global_k_eq_v), or kv_lora_rank + qk_rope_dim
    numbers (MLA). n_global None means n_layers. Windowed layers add nothing per token here.
    """
    raise NotImplementedError


def kv_cache_bytes(s: Spec, T: int, bytes_per_el: int = 2) -> int:
    """Cache of one sequence of T tokens: each global layer holds T tokens at the global shape (as in
    kv_bytes_per_token), each local layer min(T, window) tokens at the local shape (2 * n_kv_heads * head_dim
    numbers per token, or the MLA size)."""
    raise NotImplementedError


def attention_flops_per_token(s: Spec, T: int) -> float:
    """Softmax-attention FLOPs per token, averaged over a causal sequence of length T.

    One query head against one key costs 2*d_qk (scores) plus 2*d_v (value mixing) FLOPs. For GQA d_qk = d_v =
    the layer's head dim (global_head_dim for global layers when set); for MLA use the decompressed shapes
    d_qk = qk_nope_dim + qk_rope_dim and d_v = v_dim. All n_heads query heads count, whatever n_kv_heads is.
    A token in a length-T causal sequence sees on average T/2 keys in a global layer, and
    window - window**2 / (2*T) keys in a local layer when window < T (T/2 when window >= T).
    Sum over the n_global global layers and the n_local local layers; linear-attention layers add nothing.
    """
    raise NotImplementedError


def flops_per_token(s: Spec, T: int) -> float:
    """Forward FLOPs per token at sequence length T: 2 * (activated parameters, "head" convention) plus
    attention_flops_per_token(s, T). The input embedding is a lookup, the output head is a matmul."""
    raise NotImplementedError


def load_config(cfg: Mapping) -> Spec:
    """Normalize a Hugging Face config.json dict into a Spec.

    Let c = cfg["text_config"] if present (multimodal releases nest the language model), else cfg, and let
    mt = c["model_type"]. Rules, in this order:
      * n_layers, d, vocab, n_heads from num_hidden_layers, hidden_size, vocab_size, num_attention_heads;
        n_kv_heads from num_key_value_heads (missing or None: n_heads); tie_embeddings from
        tie_word_embeddings looked up in c, then in cfg, else True (the Hugging Face default).
      * MLA: if kv_lora_rank is present and nonzero, fill q_lora_rank (None becomes 0), kv_lora_rank,
        qk_nope_dim = qk_nope_head_dim, qk_rope_dim = qk_rope_head_dim, v_dim = v_head_dim, and head_dim = v_head_dim.
        Otherwise head_dim = c["head_dim"] if present, else d // n_heads.
      * Biases: mt == "qwen2" always has q, k, v biases (its config has no key for it). Otherwise a true
        attention_bias sets qkv_bias and o_bias, except for mt == "glm4_moe", where it sets only qkv_bias.
      * QK-norm: "head" for mt in ("qwen3", "qwen3_moe"), and for mt == "glm4_moe" when use_qk_norm is true;
        "full" for mt == "olmo2".
      * Experts: E is the first nonzero of n_routed_experts, num_local_experts, num_experts (else the model is
        dense and d_ff = intermediate_size). With experts: top_k = num_experts_per_tok; d_expert =
        moe_intermediate_size if present, else intermediate_size; d_shared = n_shared_experts * d_expert if
        n_shared_experts is present, else 0. MoE layers: if first_k_dense_replace is present, layers with
        index >= first_k_dense_replace and index % moe_layer_freq == 0 (moe_layer_freq missing means 1); elif
        decoder_sparse_step is present (Qwen3-MoE), layers with (index + 1) % step == 0 that are not in
        mlp_only_layers; else every layer (Mixtral).
        d_ff = intermediate_size if some layer is dense, else 0.
      * Layout: mt == "mistral" with a sliding_window: every layer is local (n_global = 0, n_local = n_layers,
        window = sliding_window). Otherwise all layers are global (n_global None).
    The reference solution also understands Gemma 2/3/4, Llama 4, gpt-oss and MiniMax keys (used for the atlas
    tables); the tests do not require them.
    """
    raise NotImplementedError
