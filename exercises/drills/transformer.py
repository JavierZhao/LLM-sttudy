"""Drill (page 04): a decoder-only transformer, exact parameter counts, and FLOPs.

Build the pieces of a Llama-style model from scratch, then write the two calculations
every interview asks for. Run:
    pytest exercises/tests/test_transformer.py

Model config: a plain dict with these keys.
    vocab_size, d_model, n_layers, n_heads, d_ff      required ints
    n_kv_heads    int, number of KV heads (GQA). Default: n_heads (plain MHA).
    head_dim      int, per-head width. Default: d_model // n_heads.
    max_seq_len   int, rows of a learned absolute position table added to the token
                  embeddings. 0 (default) means no position parameters at all.
    tie_embeddings  bool, share the token embedding matrix with the LM head. Default False.
    bias          bool, add biases to every attention and FFN Linear layer. Default False.
                  The LM head never has a bias, and RMSNorm never has a bias.
    norm_eps      float, RMSNorm epsilon. Default 1e-5.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RMSNorm(nn.Module):
    """Root-mean-square normalization over the last dimension.

    y = x / sqrt(mean(x^2, dim=-1) + eps) * weight

    Attributes: `weight`, an nn.Parameter of shape (dim,) initialized to ones. There is
    no bias and no mean subtraction. eps goes inside the square root.
    The statistics must be computed in float32 even when x is bfloat16 or float16.
    The output has the same dtype as the input x.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """x: (..., dim) -> (..., dim)."""
        raise NotImplementedError


class SwiGLU(nn.Module):
    """Gated feed-forward layer: FFN(x) = (SiLU(x W1) * (x W3)) W2, with * elementwise.

    Attributes (three nn.Linear layers, in this naming):
        w1: d_model -> d_ff   (the gate, passed through SiLU)
        w3: d_model -> d_ff   (the up projection)
        w2: d_ff -> d_model   (the down projection)
    All three take bias=bias.
    """

    def __init__(self, d_model: int, d_ff: int, bias: bool = False):
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, d_model) -> (B, T, d_model)."""
        raise NotImplementedError


class CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention with optional grouped-query attention (GQA).

    Attributes (four nn.Linear layers, all with bias=bias, in this naming):
        wq: d_model -> n_heads * head_dim
        wk: d_model -> n_kv_heads * head_dim
        wv: d_model -> n_kv_heads * head_dim
        wo: n_heads * head_dim -> d_model
    Defaults: n_kv_heads = n_heads, head_dim = d_model // n_heads. n_heads must be
    divisible by n_kv_heads. Query head i uses KV head i // (n_heads // n_kv_heads).
    Also expose the resolved integers as attributes n_heads, n_kv_heads and head_dim.
    Position i may attend to positions j <= i only; the scale is 1/sqrt(head_dim).
    The module adds no positional information itself.
    Use F.scaled_dot_product_attention for the attention itself.
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: Optional[int] = None,
                 head_dim: Optional[int] = None, bias: bool = False):
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, d_model) -> (B, T, d_model)."""
        raise NotImplementedError


class Block(nn.Module):
    """Pre-norm decoder block. Built from a config dict (see the module docstring).

    Attributes, in this naming: attn_norm (RMSNorm), attn (CausalSelfAttention),
    ffn_norm (RMSNorm), ffn (SwiGLU).
        h = x + attn(attn_norm(x))
        y = h + ffn(ffn_norm(h))
    """

    def __init__(self, cfg: dict):
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, d_model) -> (B, T, d_model)."""
        raise NotImplementedError


class GPT(nn.Module):
    """Decoder-only language model built from a config dict.

    Attributes, in this naming:
        tok_emb      nn.Embedding(vocab_size, d_model)
        pos_emb      nn.Embedding(max_seq_len, d_model) if max_seq_len > 0, else None
        blocks       nn.ModuleList of n_layers Blocks
        final_norm   RMSNorm(d_model)
        lm_head      nn.Linear(d_model, vocab_size, bias=False). If tie_embeddings, its
                     weight must BE tok_emb.weight (the same Parameter object).

    Initialization: every Linear and Embedding weight ~ N(0, 0.02^2), all biases zero,
    except the output projection of each sublayer (blocks[i].attn.wo and blocks[i].ffn.w2),
    whose weights ~ N(0, (0.02 / sqrt(2 * n_layers))^2). RMSNorm weights stay at one.
    """

    def __init__(self, cfg: dict):
        raise NotImplementedError

    def forward(self, idx: Tensor, targets: Optional[Tensor] = None):
        """idx: (B, T) int64 token ids, T <= max_seq_len when a position table exists.

        Returns (logits, loss). logits: (B, T, vocab_size). loss is None when targets is
        None; otherwise it is the mean cross-entropy between logits and targets
        ((B, T) int64), averaged over all B*T positions.
        """
        raise NotImplementedError


def count_params(cfg: dict) -> int:
    """Closed-form number of parameters of GPT(cfg), without building the model.

    Must equal sum(p.numel() for p in GPT(cfg).parameters()) for every valid config
    (parameters shared by weight tying are counted once). Must return 8,030,261,248 for
    the Llama-3-8B config: vocab_size=128256, d_model=4096, n_layers=32, n_heads=32,
    n_kv_heads=8, d_ff=14336, untied, no biases, no position table.
    """
    raise NotImplementedError


def train_flops(n_params: float, n_tokens: float) -> float:
    """Training compute in FLOPs by the standard approximation C = 6 * N * D.

    n_params: N, the parameter count. n_tokens: D, the number of training tokens.
    """
    raise NotImplementedError


def forward_flops_per_token(cfg: dict, seq_len: int, causal: bool = True) -> float:
    """Forward-pass matmul FLOPs for ONE token at context length seq_len.

    Count a multiply-add as 2 FLOPs and include only matrix multiplications:
      - every Linear weight in the blocks (Wq, Wk, Wv, Wo, W1, W3, W2), plus the LM head
        (vocab_size * d_model weights, whether or not it is tied to the embedding);
      - the attention matmuls QK^T and (softmax) V: 4 * seq_len * n_heads * head_dim
        FLOPs per layer if every token attends to all seq_len positions. With
        causal=True the average token sees half of them, so halve the attention term.
    Embedding lookups, norms, softmax, SiLU and biases cost 0 in this count.
    """
    raise NotImplementedError


def attention_flop_fraction(cfg: dict, seq_len: int, causal: bool = True) -> float:
    """Fraction of forward_flops_per_token(cfg, seq_len, causal) spent in the attention
    matmuls QK^T and AV (the term that grows with seq_len). A float in [0, 1)."""
    raise NotImplementedError
