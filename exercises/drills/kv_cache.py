"""Drill (page 06): a preallocated KV cache, incremental attention, and a greedy decoding loop.

Implement the `KVCache` class, `attend_with_cache` and `generate_greedy`. Run:
    pytest exercises/tests/test_kv_cache.py

Keep this file self-contained: do not import other drills.
"""
from typing import Callable, List, Optional, Tuple

import torch
from torch import Tensor


class KVCache:
    """Per-layer key/value buffers, allocated once and filled in place.

    Public attributes (the tests read them):
        k, v: lists with n_layers tensors each, every one of shape (batch, n_kv, max_len, d_h),
              created zero-filled with the given dtype and device.
        max_len: capacity in positions.
    Each layer has its own length: the number of positions written to that layer so far.
    """

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
        """Allocate all buffers up front. No tensor may be reallocated later."""
        raise NotImplementedError

    def length(self, layer: int = 0) -> int:
        """Number of cached positions in `layer`."""
        raise NotImplementedError

    def append(self, layer: int, k_new: Tensor, v_new: Tensor) -> None:
        """Write new keys and values into `layer` and advance its length.

        Args:
            layer: layer index.
            k_new, v_new: (batch, n_kv, T, d_h) tensors, T >= 1.
        Behavior: copy them into positions [length, length + T) of the layer's preallocated
        buffers, in place (no torch.cat, no new buffer). If length + T would exceed max_len,
        raise ValueError and leave the cache unchanged.
        """
        raise NotImplementedError

    def get(self, layer: int) -> Tuple[Tensor, Tensor]:
        """Return (K, V), each of shape (batch, n_kv, length, d_h).

        They must be views of the filled part of the buffers (no copy), so that reading the
        cache costs no extra memory traffic.
        """
        raise NotImplementedError


def attend_with_cache(q_t: Tensor, k_t: Tensor, v_t: Tensor, cache: KVCache, layer: int) -> Tensor:
    """Causal attention for one layer over the cached tokens plus T new ones.

    Args:
        q_t: (B, n_h, T, d_h) queries of the T new tokens. T = 1 is a decode step, T = prompt
             length is prefill. n_h must be a multiple of the cache's n_kv (grouped-query
             attention: query head i reads KV head i // (n_h // n_kv)).
        k_t, v_t: (B, n_kv, T, d_h) keys and values of the same new tokens. They are already
             projected (and rotated, if the model uses RoPE); the cache stores them as given.
        cache: the KVCache; layer: which layer's buffers to use.
    Steps:
        1. append k_t, v_t to the cache for `layer`;
        2. let S = the layer's new length and attend from the T queries over all S cached
           keys with scale 1 / sqrt(d_h);
        3. causal mask, end-aligned: query i sits at absolute position S - T + i and may only
           see keys at positions <= S - T + i.
    Returns:
        (B, n_h, T, d_h). Side effect: the layer's cache length grows by T.
    """
    raise NotImplementedError


def generate_greedy(
    step_fn: Callable[[Tensor, Optional[KVCache]], Tensor],
    prompt_ids: List[int],
    max_new_tokens: int,
    eos_id: int,
    cache: Optional[KVCache] = None,
) -> List[int]:
    """Greedy decoding with incremental (cached) forward passes.

    Args:
        step_fn: step_fn(ids, cache) runs the model on `ids`, a LongTensor of shape (1, T), and
            returns the logits of the LAST position, shape (V,) or (1, V). It reads and writes
            `cache` itself (normally through attend_with_cache); generate_greedy only passes the
            object through and never touches its contents.
        prompt_ids: non-empty list of prompt token ids.
        max_new_tokens: maximum number of tokens to generate (0 means none).
        eos_id: stop token. A prompt token equal to eos_id has no effect.
        cache: optional KVCache handed to every step_fn call (may be None).
    Protocol:
        * Call step_fn once with the whole prompt, shape (1, len(prompt_ids)) (prefill), and take
          the argmax as the first new token.
        * Then repeatedly call step_fn with ONLY the newest token, shape (1, 1), and take the argmax.
        * Stop right after a token equal to eos_id is produced (it is included in the output), or
          once max_new_tokens tokens have been produced.
        * Never call step_fn for a token whose logits you will not use: producing n new tokens
          takes exactly n calls (1 prefill and n - 1 single-token steps).
    Returns:
        The generated token ids as a list of Python ints (the prompt is not included).
    """
    raise NotImplementedError
