"""Reference solutions: KV cache, incremental attention, greedy loop (page 06)."""
from typing import Callable, List, Optional, Tuple

import torch
from torch import Tensor


class KVCache:
    """Preallocated per-layer K/V buffers of shape (batch, n_kv, max_len, d_h), filled in place."""

    def __init__(
        self,
        n_layers: int,
        batch: int,
        n_kv: int,
        max_len: int,
        d_h: int,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        shape = (batch, n_kv, max_len, d_h)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.max_len = max_len
        self._len = [0] * n_layers                      # one length per layer

    def length(self, layer: int = 0) -> int:
        return self._len[layer]

    def append(self, layer: int, k_new: Tensor, v_new: Tensor) -> None:
        start, T = self._len[layer], k_new.shape[2]
        if start + T > self.max_len:
            raise ValueError(f"cache overflow: {start} + {T} > {self.max_len}")   # checked before any write
        self.k[layer][:, :, start:start + T] = k_new    # in-place copy into the preallocated buffer
        self.v[layer][:, :, start:start + T] = v_new
        self._len[layer] = start + T

    def get(self, layer: int) -> Tuple[Tensor, Tensor]:
        n = self._len[layer]
        return self.k[layer][:, :, :n], self.v[layer][:, :, :n]     # slices are views: no copy


def attend_with_cache(q_t: Tensor, k_t: Tensor, v_t: Tensor, cache: KVCache, layer: int) -> Tensor:
    """Append the new K/V, then attend over the whole cache with an end-aligned causal mask."""
    B, n_h, T, d_h = q_t.shape
    cache.append(layer, k_t, v_t)
    K, V = cache.get(layer)                                         # (B, n_kv, S, d_h)
    n_kv, S = K.shape[1], K.shape[2]
    g = n_h // n_kv                                                 # query heads per KV head
    K = K.repeat_interleave(g, dim=1)                               # (B, n_h, S, d_h); kernels index instead of copying
    V = V.repeat_interleave(g, dim=1)
    scores = q_t @ K.transpose(-1, -2) / d_h**0.5                   # (B, n_h, T, S)
    qpos = torch.arange(T, device=q_t.device)[:, None] + (S - T)    # absolute position of each new query
    kpos = torch.arange(S, device=q_t.device)[None, :]
    scores = scores.masked_fill(kpos > qpos, float("-inf"))
    return scores.softmax(dim=-1) @ V                               # (B, n_h, T, d_h)


def generate_greedy(
    step_fn: Callable[[Tensor, Optional[KVCache]], Tensor],
    prompt_ids: List[int],
    max_new_tokens: int,
    eos_id: int,
    cache: Optional[KVCache] = None,
) -> List[int]:
    """Prefill once, then feed back one token at a time; stop after EOS or max_new_tokens."""
    out: List[int] = []
    if max_new_tokens <= 0:
        return out
    ids = torch.tensor([prompt_ids], dtype=torch.long)              # (1, P): the prefill call
    while True:
        logits = step_fn(ids, cache).reshape(-1)                    # logits of the last position, (V,)
        tok = int(logits.argmax())
        out.append(tok)
        if tok == eos_id or len(out) >= max_new_tokens:
            return out                                              # the last token is never fed back
        ids = torch.tensor([[tok]], dtype=torch.long)               # (1, 1): decode step
