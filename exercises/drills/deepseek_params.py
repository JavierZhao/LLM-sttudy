"""Drill (page 26): DeepSeek parameter accounting and the DeepSeek LLM hyperparameter laws.

Implement the functions below. Run:
    pytest exercises/tests/test_deepseek_params.py

Configs are plain dicts with the key names of the released Hugging Face `config.json`
files of DeepSeek-V2, DeepSeek-V2-Lite and DeepSeek-V3:

    num_hidden_layers        L, number of transformer layers
    hidden_size              d
    num_attention_heads      n_h (MLA has no KV-head grouping)
    q_lora_rank              d_c' (query latent size) or None (no query compression, as in V2-Lite)
    kv_lora_rank             d_c (KV latent size)
    qk_nope_head_dim         per-head content query/key size
    qk_rope_head_dim         per-head RoPE size d_h^R (the RoPE key is shared by all heads)
    v_head_dim               per-head value size
    first_k_dense_replace    the first k layers use a dense FFN, the rest use MoE
    intermediate_size        hidden width of the dense FFN
    moe_intermediate_size    hidden width d_e of ONE expert
    n_routed_experts, n_shared_experts, num_experts_per_tok (routed experts active per token)
    vocab_size               V (input embedding and LM head are separate matrices, untied)

What is counted (all matrices have no bias; every FFN and expert is SwiGLU with 3 matrices d x width):
  * MLA per layer: W_DQ (d x d_c'), q latent RMSNorm (d_c'), W_UQ and W_QR fused
    (d_c' x n_h (d_nope + d_rope)), W_DKV and W_KR fused (d x (d_c + d_rope)), KV latent
    RMSNorm (d_c), W_UK and W_UV fused (d_c x n_h (d_nope + d_v)), W_O (n_h d_v x d).
    If q_lora_rank is None there is no W_DQ and no q latent norm, and the query projection
    is a single d x n_h (d_nope + d_rope) matrix.
  * Two RMSNorm weight vectors of size d per layer, plus one final RMSNorm of size d.
  * Dense layer: 3 d * intermediate_size. MoE layer: (n_routed + n_shared) experts of
    3 d * moe_intermediate_size, plus a router matrix d x n_routed.
  * Input embedding V x d and LM head V x d.
  * NOT counted: the multi-token-prediction module of V3, V3's aux-loss-free routing bias, buffers.
"""
from typing import Mapping, Optional, Tuple


def mla_params(cfg: Mapping) -> int:
    """Parameters of ONE multi-head latent attention layer, including its latent RMSNorms.

    Returns an int. Check values: DeepSeek-V3 187,107,328; DeepSeek-V2 149,227,520.
    """
    raise NotImplementedError


def moe_layer_params(cfg: Mapping) -> Tuple[int, int]:
    """(total, activated) parameters of ONE MoE FFN layer: experts plus router.

    total counts all shared and routed experts and the router. activated counts the shared
    experts, the `num_experts_per_tok` routed experts that a token uses, and the whole router.
    """
    raise NotImplementedError


def count_params(cfg: Mapping, active_embeddings: str = "both") -> Tuple[int, int]:
    """Return (total, activated) parameter counts for a DeepSeek-style MLA + MoE model.

    total: everything listed in the module docstring (both vocabulary matrices included).
    activated: the parameters one token uses: all attention, norms and dense-FFN weights, the
      shared and top-k routed experts of every MoE layer, every router, and the embeddings
      selected by `active_embeddings`:
        "both": the input embedding and the LM head are counted (default),
        "head": only the LM head is counted,
        "none": no vocabulary matrix is counted (non-embedding parameters).
    Raise ValueError for any other value of `active_embeddings`.
    Both numbers are ints.
    """
    raise NotImplementedError


def model_scale_flops(n_layer: int, d_model: int, l_seq: int) -> float:
    """DeepSeek LLM's model scale M: non-embedding FLOPs per token (training, forward + backward).

    M = 72 * n_layer * d_model**2 + 12 * n_layer * d_model * l_seq
    (the compute budget is then C = M * D for D training tokens).
    """
    raise NotImplementedError


def hp_scaling(C: float) -> Tuple[float, float]:
    """DeepSeek LLM's fitted hyperparameter laws for compute budget C in FLOPs.

    Returns (B_opt, eta_opt): the optimal batch size in TOKENS and the optimal peak learning
    rate, from  eta_opt = 0.3118 * C**-0.1250  and  B_opt = 0.2920 * C**0.3271.
    """
    raise NotImplementedError


def optimal_allocation(C: float) -> Tuple[float, float]:
    """DeepSeek LLM's fitted compute-optimal allocation for the 'current data' fit.

    Returns (M_opt, D_opt): non-embedding FLOPs per token and training tokens, from
    M_opt = 0.1715 * C**0.5243 and D_opt = 5.8316 * C**0.4757.
    """
    raise NotImplementedError
