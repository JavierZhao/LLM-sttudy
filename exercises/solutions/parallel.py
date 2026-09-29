"""Reference solutions: distributed training by simulation (page 15)."""
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def ring_allreduce(chunks_per_rank: List[List[Tensor]]) -> Tuple[List[List[Tensor]], List[int]]:
    n = len(chunks_per_rank)
    buf = [[c.clone() for c in rank] for rank in chunks_per_rank]      # never touch the input
    sent = [0] * n
    if n == 1:
        return buf, sent
    # Reduce-scatter: after n-1 steps rank r owns the full sum of chunk (r + 1) % n.
    for s in range(n - 1):
        # Snapshot the messages first: every rank sends what it held at the start of the step.
        msgs = [((r - s) % n, buf[r][(r - s) % n].clone()) for r in range(n)]
        for r in range(n):
            c, payload = msgs[r]
            dst = (r + 1) % n
            buf[dst][c] = buf[dst][c] + payload
            sent[r] += payload.numel() * payload.element_size()
    # All-gather: circulate the finished chunks; the receiver overwrites.
    for s in range(n - 1):
        msgs = [((r + 1 - s) % n, buf[r][(r + 1 - s) % n].clone()) for r in range(n)]
        for r in range(n):
            c, payload = msgs[r]
            dst = (r + 1) % n
            buf[dst][c] = payload
            sent[r] += payload.numel() * payload.element_size()
    return buf, sent


def column_row_parallel_mlp(
    x: Tensor,
    W1: Tensor,
    W2: Tensor,
    tp: int,
    act: Callable[[Tensor], Tensor] = F.gelu,
) -> Tuple[Tensor, List[Tensor]]:
    d_ff = W1.shape[1]
    assert d_ff % tp == 0
    s = d_ff // tp
    partials = []
    for r in range(tp):
        A_r = W1[:, r * s:(r + 1) * s]        # column shard: (d, d_ff/tp)
        B_r = W2[r * s:(r + 1) * s, :]        # matching row shard: (d_ff/tp, d)
        # act is elementwise, so it commutes with the column split: no communication needed here.
        partials.append(act(x @ A_r) @ B_r)   # (..., d): a partial sum of the true output
    y = torch.stack(partials).sum(dim=0)      # the all-reduce
    return y, partials


def head_parallel_attention(
    x: Tensor,
    Wq: Tensor,
    Wk: Tensor,
    Wv: Tensor,
    Wo: Tensor,
    n_heads: int,
    tp: int,
) -> Tuple[Tensor, List[Tensor]]:
    B, T, d = x.shape
    assert n_heads % tp == 0 and d % n_heads == 0
    d_h = d // n_heads
    hp = n_heads // tp                         # heads per rank
    partials = []
    for r in range(tp):
        cols = slice(r * hp * d_h, (r + 1) * hp * d_h)
        q = (x @ Wq[:, cols]).view(B, T, hp, d_h).transpose(1, 2)      # (B, hp, T, d_h)
        k = (x @ Wk[:, cols]).view(B, T, hp, d_h).transpose(1, 2)
        v = (x @ Wv[:, cols]).view(B, T, hp, d_h).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)    # each head is independent
        o = o.transpose(1, 2).reshape(B, T, hp * d_h)
        partials.append(o @ Wo[cols, :])       # row-parallel output projection: a partial sum
    y = torch.stack(partials).sum(dim=0)
    return y, partials


def training_memory_bytes(
    n_params: float,
    zero_stage: int,
    dp: int,
    param_bytes: float = 2,
    grad_bytes: float = 2,
    optim_bytes: float = 12,
    mp: int = 1,
) -> Dict[str, float]:
    if zero_stage not in (0, 1, 2, 3):
        raise ValueError(f"zero_stage must be 0, 1, 2 or 3, got {zero_stage}")
    n = n_params / mp                                        # what one model-parallel rank holds
    params = n * param_bytes / (dp if zero_stage >= 3 else 1)
    grads = n * grad_bytes / (dp if zero_stage >= 2 else 1)
    optim = n * optim_bytes / (dp if zero_stage >= 1 else 1)
    return {"params": params, "grads": grads, "optim": optim, "total": params + grads + optim}


def zero_comm_bytes_per_step(n_params: float, zero_stage: int, dp: int, bytes_per_elem: float = 2) -> float:
    if zero_stage not in (0, 1, 2, 3):
        raise ValueError(f"zero_stage must be 0, 1, 2 or 3, got {zero_stage}")
    volume = 3.0 if zero_stage == 3 else 2.0                 # in units of n_params elements
    return volume * n_params * bytes_per_elem * (dp - 1) / dp


def pipeline_bubble(p: int, m: int, v: int = 1, of: str = "total") -> float:
    bubble = (p - 1) / v                # in units of one microbatch's (t_f + t_b)
    ideal = m
    if of == "total":
        return bubble / (ideal + bubble)
    if of == "ideal":
        return bubble / ideal
    raise ValueError("of must be 'total' or 'ideal'")


def simulate_1f1b(
    p: int,
    m: int,
    t_fwd: float = 1.0,
    t_bwd: float = 2.0,
    schedule: str = "1f1b",
) -> Tuple[float, List[int]]:
    if schedule not in ("1f1b", "gpipe"):
        raise ValueError("schedule must be '1f1b' or 'gpipe'")
    ops: List[List[Tuple[str, int]]] = []
    for i in range(p):
        if schedule == "gpipe":
            seq = [("F", j) for j in range(m)] + [("B", j) for j in range(m)]
        else:
            warm = min(p - 1 - i, m)
            seq = [("F", j) for j in range(warm)]
            f, b = warm, 0
            while f < m:                # steady state: one forward, one backward
                seq.append(("F", f)); f += 1
                seq.append(("B", b)); b += 1
            while b < m:                # cool-down: drain the remaining backwards
                seq.append(("B", b)); b += 1
        ops.append(seq)

    end: Dict[Tuple[str, int, int], float] = {}
    start: Dict[Tuple[str, int, int], float] = {}
    free = [0.0] * p
    ptr = [0] * p
    total = sum(len(s) for s in ops)
    done = 0
    while done < total:
        progressed = False
        for i in range(p):
            if ptr[i] == len(ops[i]):
                continue
            kind, j = ops[i][ptr[i]]
            if kind == "F":
                dep_key = ("F", i - 1, j) if i > 0 else None
                dur = t_fwd
            else:
                dep_key = ("B", i + 1, j) if i < p - 1 else ("F", i, j)
                dur = t_bwd
            if dep_key is not None and dep_key not in end:
                continue                # dependency not finished yet: try another stage
            t0 = max(free[i], end[dep_key] if dep_key is not None else 0.0)
            start[(kind, i, j)] = t0
            end[(kind, i, j)] = free[i] = t0 + dur
            ptr[i] += 1
            done += 1
            progressed = True
        assert progressed, "deadlock: the schedule is not executable"

    peak = []
    for i in range(p):
        events = []                     # (time, +1 when a forward starts, -1 when a backward ends)
        for j in range(m):
            events.append((start[("F", i, j)], 1))
            events.append((end[("B", i, j)], -1))
        events.sort(key=lambda e: (e[0], e[1]))   # at a tie, free the finished backward first
        cur = best = 0
        for _, delta in events:
            cur += delta
            best = max(best, cur)
        peak.append(best)
    return max(free), peak


def model_flops_per_token(
    n_params: float,
    n_layers: int = 0,
    d_model: int = 0,
    seq_len: int = 0,
    causal: bool = True,
) -> float:
    attn = 12.0 * n_layers * d_model * seq_len
    if causal:
        attn /= 2
    return 6.0 * n_params + attn


def mfu(tokens_per_sec_per_gpu: float, flops_per_token: float, peak_flops: float) -> float:
    return tokens_per_sec_per_gpu * flops_per_token / peak_flops
