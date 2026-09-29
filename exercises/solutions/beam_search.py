"""Drill (page 35): beam search with length normalization and EOS handling (reference solution)."""
from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

NEG_INF = float("-inf")


def normalized_score(sum_logprob: float, length: int, alpha: float) -> float:
    return sum_logprob / (((5.0 + length) / 6.0) ** alpha)      # GNMT: lp = ((5 + |Y|) / 6) ** alpha


def beam_search(
    step_logprobs_fn: Callable[[Tensor], Tensor],
    bos: int,
    eos: int,
    beam: int,
    max_len: int,
    length_penalty: float = 0.0,
    early_stop: bool = False,
) -> list[tuple[list[int], float]]:
    if beam < 1 or max_len < 1:
        raise ValueError("beam and max_len must be at least 1")
    if length_penalty < 0:
        raise ValueError("length_penalty must be >= 0")
    alpha = length_penalty
    prefixes = torch.tensor([[bos]], dtype=torch.long)           # (n, t) live hypotheses, all the same length
    scores = torch.zeros(1, dtype=torch.float64)                 # (n,) cumulative log-probability
    finished: list[tuple[float, list[int]]] = []                 # (normalized score, tokens without bos)

    def trim() -> None:
        finished.sort(key=lambda h: -h[0])                       # stable: earlier finishers win ties
        del finished[beam:]

    for step in range(1, max_len + 1):                           # step = number of generated tokens so far
        logp = step_logprobs_fn(prefixes).to(torch.float64)      # (n, V), one model call per step
        n, V = logp.shape
        total = scores[:, None] + logp                           # cumulative score of every extension
        # 1. every EOS extension is a finished hypothesis and is never expanded again
        for i in range(n):
            s = total[i, eos].item()
            if s > NEG_INF:
                finished.append((normalized_score(s, step, alpha), prefixes[i, 1:].tolist() + [eos]))
        # 2. the live beam is the top-`beam` non-EOS extensions, ranked by RAW cumulative log-prob
        #    (all candidates have the same length here, so normalizing would not change the order)
        total[:, eos] = NEG_INF
        vals, idx = total.reshape(-1).topk(min(beam, n * V))     # k may exceed n * V for a wide beam
        keep = vals > NEG_INF                                    # forbidden tokens never enter the beam
        vals, idx = vals[keep], idx[keep]
        if vals.numel() == 0:
            prefixes = prefixes[:0]
            break
        parent, tok = idx // V, idx % V                          # flat index -> (which hypothesis, which token)
        prefixes = torch.cat([prefixes[parent], tok[:, None]], dim=1)   # this gather is the "cache reorder"
        scores = vals
        trim()
        # 3. optional exact stopping rule: no descendant can beat the worst kept finished hypothesis
        if early_stop and step < max_len and len(finished) == beam:
            bound = scores.max().item() / (((5.0 + max_len) / 6.0) ** alpha)
            if bound < finished[-1][0]:
                prefixes = prefixes[:0]
                break
    else:
        # max_len reached: the survivors are truncated hypotheses (no EOS), scored at length max_len
        for i in range(prefixes.shape[0]):
            finished.append((normalized_score(scores[i].item(), max_len, alpha), prefixes[i, 1:].tolist()))
    trim()
    return [(toks, s) for s, toks in finished]


def exhaustive_search(
    step_logprobs_fn: Callable[[Tensor], Tensor],
    bos: int,
    eos: int,
    max_len: int,
    length_penalty: float = 0.0,
    top_n: int = 1,
) -> list[tuple[list[int], float]]:
    results: list[tuple[float, list[int]]] = []

    def dfs(prefix: list[int], logp: float) -> None:
        row = step_logprobs_fn(torch.tensor([[bos] + prefix]))[0].to(torch.float64)
        for tok in range(row.shape[0]):
            lp = row[tok].item()
            if lp == NEG_INF:
                continue
            seq = prefix + [tok]
            if tok == eos or len(seq) == max_len:                # finished by EOS or truncated at max_len
                results.append((normalized_score(logp + lp, len(seq), length_penalty), seq))
            else:
                dfs(seq, logp + lp)

    dfs([], 0.0)
    results.sort(key=lambda h: -h[0])
    return [(toks, s) for s, toks in results[:top_n]]
