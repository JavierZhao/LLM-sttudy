"""Drill (page 28): DeepSeek Sparse Attention (DSA), as in DeepSeek-V3.2.

DSA sits on top of MLA's MQA mode (page 07). A cheap "lightning indexer" scores every
earlier token, a top-k selector keeps k of them, and core attention reads only the
selected latents. The indexer is trained as a student of the model's own dense attention
with a KL loss. Implement the pieces below. Run:
    pytest exercises/tests/test_dsa.py

Conventions: batch B, T new query positions that are the LAST T of S total positions
(T = S for prefill, T = 1 for decoding with a cache). Query t sits at absolute position
S - T + t and may use only keys s <= S - T + t. Scores use -inf (never 0) for keys a
query may not use.
"""
from typing import Optional

import torch
from torch import Tensor


def lightning_index_scores(q_idx: Tensor, w_idx: Tensor, k_idx: Tensor) -> Tensor:
    """Lightning-indexer scores I[b, t, s] = sum_j w[b, t, j] * ReLU(q_idx[b, t, j] . k_idx[b, s]).

    Args:
        q_idx: (B, T, H, d) indexer queries, H indexer heads (64 in V3.2, d = 128).
        w_idx: (B, T, H) per-head weights produced from the query token. Plain real
               numbers: they may be negative, so do not clamp them.
        k_idx: (B, S, d) indexer keys for all S positions (cache + new), S >= T. One key
               per token, shared by all indexer heads.
    Returns:
        (B, T, S) scores. Entries for keys the query may not use (s > S - T + t) must be
        -inf so that a later top-k or softmax can never pick them.
    """
    raise NotImplementedError


def dsa_attention(
    q_lat: Tensor,
    q_rope: Tensor,
    c_cache: Tensor,
    kr_cache: Tensor,
    scores: Tensor,
    k_top: int,
    scale: float,
) -> Tensor:
    """MQA-mode MLA attention restricted to the top-k latents chosen by the indexer.

    Args:
        q_lat: (B, T, n_h, d_c) absorbed content queries (q^C W^UK per head).
        q_rope: (B, T, n_h, d_r) RoPE queries, one per head.
        c_cache: (B, S, d_c) cached latents. One per token, shared by all heads, and they
                 are both the content key and the value.
        kr_cache: (B, S, d_r) cached RoPE keys, one per token, shared by all heads.
        scores: (B, T, S) indexer scores (-inf on keys a query may not use).
        k_top: number of keys each query attends to. A query with fewer than k_top usable
               keys attends to all of them (never to a masked key). The selection is made
               once per query and shared by all n_h heads.
        scale: softmax scale; logits = (q_lat . c + q_rope . kr) * scale.
    Returns:
        (B, T, n_h, d_c) latent-space outputs sum_s p[b, t, h, s] * c[b, s], with the
        softmax over the selected keys only. (The per-head W^UV up-projection is applied
        by the caller and is not part of this function.)
    Gather the selected latents and run attention over (B, T, k_top) entries; do not
    build a dense (T, S) mask and attend over everything.
    """
    raise NotImplementedError


def indexer_kl_loss(
    attn_probs: Tensor,
    index_scores: Tensor,
    selected: Optional[Tensor] = None,
) -> Tensor:
    """Indexer training loss L^I = sum_t KL( p_t || softmax(I_t) ), averaged over the batch.

    Args:
        attn_probs: (B, H, T, S) post-softmax attention probabilities of the main (dense)
                    model, per head. Rows sum to 1 over the usable keys and are 0 elsewhere.
        index_scores: (B, T, S) indexer scores, -inf on keys a query may not use.
        selected: optional (B, T, S) bool mask of the keys chosen by top-k.
                  None: dense variant (warm-up stage), the sets are all usable keys.
                  Given: sparse variant, both distributions live on the selected keys only.
    Returns:
        A scalar: the sum over t of KL(p_t || q_t), then the mean over the batch.
        - Target p_t: attn_probs summed over heads, restricted to the set (all usable keys,
          or the selected ones), then L1-normalized over that set. It is a constant:
          gradients must flow only into index_scores, never into attn_probs.
        - q_t = softmax of index_scores over the same set.
        - KL(p || q) = sum_s p_s (log p_s - log q_s), with 0 log 0 = 0.
    """
    raise NotImplementedError


def dsa_decode_flops(
    L: int,
    k_top: int,
    n_h: int = 128,
    d_c: int = 512,
    d_r: int = 64,
    H_I: int = 64,
    d_I: int = 128,
) -> dict:
    """Per-layer FLOPs to decode one token at context length L (2 FLOPs per multiply-add).

    Core attention in MLA's absorbed MQA mode costs, per attended key and per head, a dot
    product over d_c + d_r dimensions (scores) plus a weighted sum over d_c dimensions.
    The lightning indexer costs, per scored key, one d_I-dimensional dot product for each of
    its H_I heads (ignore ReLU, the weights w and the softmax).

    Returns a dict with float values:
        "dense":       core attention over all L keys,
        "sparse_core": core attention over min(k_top, L) keys,
        "indexer":     indexer over all L keys,
        "total_sparse": sparse_core + indexer,
        "ratio":       dense / total_sparse.
    """
    raise NotImplementedError


def dsa_break_even_length(
    k_top: int,
    n_h: int = 128,
    d_c: int = 512,
    d_r: int = 64,
    H_I: int = 64,
    d_I: int = 128,
) -> float:
    """Context length L* (float) above which DSA (sparse core + indexer) costs fewer FLOPs
    than dense core attention, for L >= k_top, using the cost model of dsa_decode_flops.
    Solve dense(L) = k_top * core_per_key + L * indexer_per_key for L.
    """
    raise NotImplementedError
