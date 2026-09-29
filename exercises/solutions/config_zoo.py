"""Reference solutions: the config zoo (page 33).

Read a Hugging Face config.json, count parameters, KV-cache bytes and FLOPs per token.
Conventions (say them out loud in an interview): a multiply-add is 2 FLOPs; bf16 is 2 bytes per
element; norm gains are counted as parameters; rotary tables and other buffers are not.
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
    qk_norm: str = "none"            # "none" | "head" (q and k gains of size head_dim) | "full" (n_h*d_h and n_kv*d_h)
    norms_per_layer: int = 2         # RMSNorm gain vectors of size d per layer (Gemma: 4)
    # MLA (kv_lora_rank == 0 means the layer is not MLA)
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_nope_dim: int = 0
    qk_rope_dim: int = 0
    v_dim: int = 0
    # FFN: dense SwiGLU/GeGLU (3 matrices) of width d_ff; MoE layers listed by index
    d_ff: int = 0
    moe_layers: Tuple[int, ...] = ()
    n_experts: int = 0
    top_k: int = 0
    d_expert: int = 0
    d_shared: int = 0                # total width of the always-on branch of each MoE layer (0 = none)
    # attention layout: n_global full-context layers, n_local windowed layers; any layer that is
    # neither is a linear-attention or SSM layer (no KV cache, no T-dependent FLOPs)
    n_global: Optional[int] = None   # None means every layer is global
    n_local: int = 0
    window: int = 0
    global_n_kv_heads: int = 0       # 0 means "same as the local shape"
    global_head_dim: int = 0
    global_k_eq_v: bool = False      # global layers have no V projection and cache K once


def _global_layers(s: Spec) -> int:
    return s.n_layers if s.n_global is None else s.n_global


def _is_mla(s: Spec) -> bool:
    return s.kv_lora_rank > 0


def _gqa_shape(s: Spec, is_global: bool) -> Tuple[int, int]:
    """(kv heads, head dim) of a GQA layer of the given kind."""
    if is_global and s.global_head_dim:
        return s.global_n_kv_heads or s.n_kv_heads, s.global_head_dim
    return s.n_kv_heads, s.head_dim


def _attn_params(s: Spec, is_global: bool) -> int:
    d, n_h = s.d, s.n_heads
    if _is_mla(s):
        dq = s.qk_nope_dim + s.qk_rope_dim
        if s.q_lora_rank:
            q = d * s.q_lora_rank + s.q_lora_rank + s.q_lora_rank * n_h * dq   # W_DQ, its RMSNorm, W_UQ | W_QR
        else:
            q = d * n_h * dq
        kv = d * (s.kv_lora_rank + s.qk_rope_dim) + s.kv_lora_rank             # W_DKV | W_KR, latent RMSNorm
        kv += s.kv_lora_rank * n_h * (s.qk_nope_dim + s.v_dim)                 # W_UK | W_UV
        return q + kv + n_h * s.v_dim * d
    n_kv, hd = _gqa_shape(s, is_global)
    k_eq_v = is_global and s.global_k_eq_v
    n = d * n_h * hd + d * n_kv * hd * (1 if k_eq_v else 2) + n_h * hd * d
    if s.qkv_bias:
        n += n_h * hd + n_kv * hd * (1 if k_eq_v else 2)
    if s.o_bias:
        n += d
    if s.qk_norm == "head":
        n += 2 * hd
    elif s.qk_norm == "full":
        n += n_h * hd + n_kv * hd
    return n


def params(s: Spec, active_embeddings: str = "both") -> Tuple[int, int]:
    """(total, activated) parameter counts of the decoder (no vision tower, no MTP module)."""
    if active_embeddings not in ("both", "head", "none"):
        raise ValueError(f"active_embeddings must be 'both', 'head' or 'none', got {active_embeddings!r}")
    if s.n_local + _global_layers(s) != s.n_layers:
        raise NotImplementedError("linear-attention layers are not modeled")
    n_glob = _global_layers(s)
    total = n_glob * _attn_params(s, True) + s.n_local * _attn_params(s, False) + s.n_layers * s.norms_per_layer * s.d
    moe = set(s.moe_layers)
    idle = 0                                                     # expert weights a token does not touch
    for i in range(s.n_layers):
        if i in moe:
            expert = 3 * s.d * s.d_expert                        # SwiGLU: gate, up, down
            total += s.d * s.n_experts + s.n_experts * expert + 3 * s.d * s.d_shared   # router, routed, always-on branch
            idle += (s.n_experts - s.top_k) * expert
        else:
            total += 3 * s.d * s.d_ff
    n_emb = 1 if s.tie_embeddings else 2
    total += n_emb * s.vocab * s.d + s.d                         # embedding (+ head) and the final norm
    active = total - idle
    if active_embeddings == "head":
        active -= (n_emb - 1) * s.vocab * s.d                    # the input table is a lookup; a tied matrix still serves as the head
    elif active_embeddings == "none":
        active -= n_emb * s.vocab * s.d
    return total, active


def _cached_elems(s: Spec, is_global: bool) -> int:
    """K and V numbers one token adds to one softmax layer's cache."""
    if _is_mla(s):
        return s.kv_lora_rank + s.qk_rope_dim
    n_kv, hd = _gqa_shape(s, is_global)
    return (1 if is_global and s.global_k_eq_v else 2) * n_kv * hd


def kv_bytes_per_token(s: Spec, bytes_per_el: int = 2) -> int:
    """Marginal cache growth per token once every window is full: only the global layers count."""
    return _global_layers(s) * _cached_elems(s, True) * bytes_per_el


def kv_cache_bytes(s: Spec, T: int, bytes_per_el: int = 2) -> int:
    """Cache of one sequence of T tokens: global layers hold T tokens, local layers min(T, window)."""
    glob = _global_layers(s) * _cached_elems(s, True) * T
    local = s.n_local * _cached_elems(s, False) * min(T, s.window)
    return (glob + local) * bytes_per_el


def _avg_keys(T: int, window: Optional[int]) -> float:
    """Mean number of keys a query sees in a causal sequence of length T (continuous form)."""
    if window is None or window >= T:
        return T / 2
    return window - window**2 / (2 * T)


def attention_flops_per_token(s: Spec, T: int) -> float:
    """Softmax-attention FLOPs per token at sequence length T (scores and value mixing, all softmax layers)."""
    def per_layer(is_global: bool) -> float:
        if _is_mla(s):
            d_qk, d_v = s.qk_nope_dim + s.qk_rope_dim, s.v_dim     # decompressed ("MHA mode") shapes
        else:
            d_qk = d_v = _gqa_shape(s, is_global)[1]
        # 2 FLOPs per multiply-add; QK^T uses d_qk numbers per key, the weighted sum uses d_v
        return 2 * s.n_heads * (d_qk + d_v) * _avg_keys(T, None if is_global else s.window)
    return _global_layers(s) * per_layer(True) + s.n_local * per_layer(False)


def flops_per_token(s: Spec, T: int) -> float:
    """Forward FLOPs per token at sequence length T: 2 * (matmul parameters) + attention."""
    matmul = params(s, "head")[1]                                # activated, input embedding is a lookup, head is a matmul
    return 2 * matmul + attention_flops_per_token(s, T)


def load_config(cfg: Mapping) -> Spec:
    """Normalize a Hugging Face config.json dict (Llama, Mistral, Qwen2/3, Mixtral, Qwen3-MoE, DeepSeek, Gemma styles)."""
    c = cfg.get("text_config", cfg)                              # multimodal releases nest the LM config
    mt = c.get("model_type", cfg.get("model_type", ""))
    L, d, V = c["num_hidden_layers"], c["hidden_size"], c["vocab_size"]
    n_h = c["num_attention_heads"]
    kw = dict(n_layers=L, d=d, vocab=V, tie_embeddings=c.get("tie_word_embeddings", cfg.get("tie_word_embeddings", True)),   # HF default is tied
              n_heads=n_h, n_kv_heads=c.get("num_key_value_heads") or n_h)
    # attention
    if c.get("kv_lora_rank"):
        kw.update(head_dim=c["v_head_dim"], q_lora_rank=c.get("q_lora_rank") or 0, kv_lora_rank=c["kv_lora_rank"],
                  qk_nope_dim=c["qk_nope_head_dim"], qk_rope_dim=c["qk_rope_head_dim"], v_dim=c["v_head_dim"])
    else:
        kw["head_dim"] = c.get("head_dim") or d // n_h
    if mt == "qwen2":
        kw["qkv_bias"] = True                                    # always, the config has no key for it
    elif mt == "glm4_moe":
        kw["qkv_bias"] = bool(c.get("attention_bias"))           # q, k, v only
    elif c.get("attention_bias"):
        kw["qkv_bias"] = kw["o_bias"] = True
    if mt in ("qwen3", "qwen3_moe", "gemma3_text", "gemma4_text") or (mt == "glm4_moe" and c.get("use_qk_norm")):
        kw["qk_norm"] = "head"
    elif mt in ("olmo2", "minimax_m2"):
        kw["qk_norm"] = "full"                                   # norm over the whole projected width, before the head split
    if mt.startswith("gemma"):
        kw["norms_per_layer"] = 4
    # FFN and experts
    E = next((c[k] for k in ("n_routed_experts", "num_local_experts", "num_experts") if c.get(k)), 0)
    if E:
        top_k = next(c[k] for k in ("num_experts_per_tok", "top_k_experts", "experts_per_token") if c.get(k))
        d_e = c.get("moe_intermediate_size") or c["intermediate_size"]
        if mt == "llama4_text":
            moe = tuple(c["moe_layers"])
            shared, kw["d_ff"] = d_e, c["intermediate_size_mlp"]
        elif mt.startswith("gemma"):                             # a dense MLP runs in parallel with the experts
            moe, shared, kw["d_ff"] = tuple(range(L)), c["intermediate_size"], 0
        else:
            if "first_k_dense_replace" in c:                     # DeepSeek, Kimi, GLM
                k0, freq = c["first_k_dense_replace"], c.get("moe_layer_freq", 1)
                moe = tuple(i for i in range(L) if i >= k0 and i % freq == 0)
            elif "decoder_sparse_step" in c:                     # Qwen3-MoE
                skip = set(c.get("mlp_only_layers", []))
                moe = tuple(i for i in range(L) if i not in skip and (i + 1) % c["decoder_sparse_step"] == 0)
            else:                                                # Mixtral, gpt-oss, MiniMax: every layer
                moe = tuple(range(L))
            shared = c.get("n_shared_experts", 0) * d_e or c.get("shared_expert_intermediate_size", 0)
            kw["d_ff"] = c.get("intermediate_size", 0) if len(moe) < L else 0
        kw.update(moe_layers=moe, n_experts=E, top_k=top_k, d_expert=d_e, d_shared=shared)
    else:
        kw["d_ff"] = c["intermediate_size"]
    # layout: windowed layers, global layers, linear layers
    types = c.get("layer_types")
    if types:
        n_loc = sum(t == "sliding_attention" for t in types)
        n_glob = sum(t == "full_attention" for t in types)
        kw.update(n_global=n_glob, n_local=n_loc, window=c.get("sliding_window") or 0)
    elif c.get("sliding_window") and c.get("sliding_window_pattern"):    # Gemma 3: every 6th layer is global
        n_glob = L // c["sliding_window_pattern"]
        kw.update(n_global=n_glob, n_local=L - n_glob, window=c["sliding_window"])
    elif c.get("sliding_window") and mt == "mistral":            # every layer windowed
        kw.update(n_global=0, n_local=L, window=c["sliding_window"])
    elif c.get("sliding_window") and mt == "gemma2":             # local and global layers alternate 1:1
        kw.update(n_global=L // 2, n_local=L - L // 2, window=c["sliding_window"])
    elif mt == "llama4_text":                                    # NoPE layers are global, the rest see 8,192-token chunks
        n_glob = sum(1 for r in c["no_rope_layers"] if not r)
        kw.update(n_global=n_glob, n_local=L - n_glob, window=c["attention_chunk_size"])
    elif c.get("full_attention_interval"):                       # Qwen3-Next: 1 softmax layer per interval
        kw.update(n_global=L // c["full_attention_interval"])
    if mt.startswith("gemma4") and c.get("global_head_dim"):
        kw.update(global_head_dim=c["global_head_dim"], global_n_kv_heads=c["num_global_key_value_heads"],
                  global_k_eq_v=bool(c.get("attention_k_eq_v")))
    return Spec(**kw)
