"""Reference solution for the DSA drill (page 28). See drills/dsa.py for the specification."""
from typing import Optional

import torch
from torch import Tensor


def lightning_index_scores(q_idx: Tensor, w_idx: Tensor, k_idx: Tensor) -> Tensor:
    B, T, H, d = q_idx.shape
    S = k_idx.shape[1]
    # per-head dot products, ReLU, then a learned per-head weighting and a sum over heads
    dots = torch.einsum("bthd,bsd->bths", q_idx, k_idx)             # (B, T, H, S)
    scores = (w_idx[..., None] * torch.relu(dots)).sum(dim=2)       # (B, T, S)
    qpos = torch.arange(T, device=q_idx.device)[:, None] + (S - T)  # absolute query positions
    kpos = torch.arange(S, device=q_idx.device)[None, :]
    return scores.masked_fill(kpos > qpos, float("-inf"))           # end-aligned causal mask


def dsa_attention(
    q_lat: Tensor,
    q_rope: Tensor,
    c_cache: Tensor,
    kr_cache: Tensor,
    scores: Tensor,
    k_top: int,
    scale: float,
) -> Tensor:
    B, T, n_h, d_c = q_lat.shape
    S = c_cache.shape[1]
    k = min(k_top, S)
    top_scores, idx = scores.topk(k, dim=-1)                        # (B, T, k)
    # rows with fewer than k usable keys get -inf filler picks: remember them and mask later
    usable = torch.isfinite(top_scores)                             # (B, T, k)
    b = torch.arange(B, device=scores.device)[:, None, None]
    c_sel = c_cache[b, idx]                                         # (B, T, k, d_c): gather, no dense mask
    kr_sel = kr_cache[b, idx]                                       # (B, T, k, d_r)
    logits = (torch.einsum("bthc,btkc->bthk", q_lat, c_sel)
              + torch.einsum("bthr,btkr->bthk", q_rope, kr_sel)) * scale
    logits = logits.masked_fill(~usable[:, :, None, :], float("-inf"))
    p = logits.softmax(dim=-1)                                      # softmax over the selected set only
    return torch.einsum("bthk,btkc->bthc", p, c_sel)                # values are the latents themselves


def indexer_kl_loss(
    attn_probs: Tensor,
    index_scores: Tensor,
    selected: Optional[Tensor] = None,
) -> Tensor:
    usable = torch.isfinite(index_scores)
    mask = usable if selected is None else (selected & usable)      # the set both distributions live on
    p = attn_probs.detach().sum(dim=1)                              # (B, T, S): sum over heads, constant target
    p = p.masked_fill(~mask, 0.0)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-30)            # L1-normalize over the set
    log_q = index_scores.masked_fill(~mask, float("-inf")).log_softmax(dim=-1)
    log_q = log_q.masked_fill(~mask, 0.0)                           # avoid 0 * (-inf) = nan
    log_p = torch.log(p.clamp_min(1e-30))
    kl = torch.where(p > 0, p * (log_p - log_q), torch.zeros_like(p))
    return kl.sum(dim=(1, 2)).mean()                                # sum over t (and s), mean over batch


def _core_per_key(n_h: int, d_c: int, d_r: int) -> float:
    # scores over d_c + d_r dims, weighted sum over d_c dims, for each of n_h heads, 2 FLOPs per MAC
    return 2.0 * n_h * ((d_c + d_r) + d_c)


def _indexer_per_key(H_I: int, d_I: int) -> float:
    return 2.0 * H_I * d_I


def dsa_decode_flops(
    L: int,
    k_top: int,
    n_h: int = 128,
    d_c: int = 512,
    d_r: int = 64,
    H_I: int = 64,
    d_I: int = 128,
) -> dict:
    core = _core_per_key(n_h, d_c, d_r)
    idx = _indexer_per_key(H_I, d_I)
    dense = core * L
    sparse_core = core * min(k_top, L)
    indexer = idx * L
    total = sparse_core + indexer
    return {"dense": dense, "sparse_core": sparse_core, "indexer": indexer,
            "total_sparse": total, "ratio": dense / total}


def dsa_break_even_length(
    k_top: int,
    n_h: int = 128,
    d_c: int = 512,
    d_r: int = 64,
    H_I: int = 64,
    d_I: int = 128,
) -> float:
    core = _core_per_key(n_h, d_c, d_r)
    idx = _indexer_per_key(H_I, d_I)
    # core * L = core * k + idx * L  =>  L = core * k / (core - idx)
    return core * k_top / (core - idx)
