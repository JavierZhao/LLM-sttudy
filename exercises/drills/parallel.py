"""Drill (page 15): distributed training by simulation on one CPU.

No real process group is used. "Ranks" are entries of Python lists, and a collective is
a function that reads all ranks' tensors and returns all ranks' results. Implement the
nine functions below. Run:
    pytest exercises/tests/test_parallel.py

Keep this file self-contained: do not import other drills.
"""
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def ring_allreduce(chunks_per_rank: List[List[Tensor]]) -> Tuple[List[List[Tensor]], List[int]]:
    """Simulate a ring all-reduce (sum) over n ranks, one message at a time.

    Args:
        chunks_per_rank: chunks_per_rank[r][c] is rank r's local tensor for chunk c. There
            are n ranks and every rank splits its data into the same n chunks: chunk c has
            the same shape on every rank, but different chunks may have different sizes.
    Returns:
        (result, bytes_sent)
        result[r][c]: what rank r holds for chunk c at the end. It equals the sum over all
            ranks r' of chunks_per_rank[r'][c]. Return new tensors, never views of the input,
            and do not modify the input.
        bytes_sent[r]: the total number of bytes rank r sent to its ring neighbor (r + 1) % n,
            counting chunk.numel() * chunk.element_size() for every message.
    Use exactly this schedule. Rank r only ever sends to rank (r + 1) % n. All ranks act
    at the same time within a step, so a rank sends the value it held at the START of the step.
        Reduce-scatter, steps s = 0 .. n-2: rank r sends its current copy of chunk (r - s) % n.
            The receiver adds it to its own copy of that chunk.
        All-gather, steps s = 0 .. n-2: rank r sends its current copy of chunk (r + 1 - s) % n.
            The receiver overwrites its own copy of that chunk.
    With n = 1 nothing is sent: return copies of the input and [0].
    """
    raise NotImplementedError


def column_row_parallel_mlp(
    x: Tensor,
    W1: Tensor,
    W2: Tensor,
    tp: int,
    act: Callable[[Tensor], Tensor] = F.gelu,
) -> Tuple[Tensor, List[Tensor]]:
    """Megatron-style tensor-parallel MLP: y = act(x @ W1) @ W2, split over `tp` ranks.

    Args:
        x: (..., d) activations, replicated on every rank.
        W1: (d, d_ff), split by COLUMNS: rank r owns columns [r * d_ff/tp, (r + 1) * d_ff/tp).
        W2: (d_ff, d), split by ROWS with the matching index range.
        tp: tensor-parallel degree; d_ff is divisible by tp.
        act: elementwise activation (default: PyTorch's exact GELU).
    Returns:
        (y, partials): partials[r] is rank r's output (..., d), computed ONLY from x and rank r's
        two weight shards (no communication before the activation). y is their sum, the
        simulated all-reduce, and must equal act(x @ W1) @ W2.
    """
    raise NotImplementedError


def head_parallel_attention(
    x: Tensor,
    Wq: Tensor,
    Wk: Tensor,
    Wv: Tensor,
    Wo: Tensor,
    n_heads: int,
    tp: int,
) -> Tuple[Tensor, List[Tensor]]:
    """Tensor-parallel causal multi-head attention: heads are split over `tp` ranks.

    Args:
        x: (B, T, d) activations, replicated on every rank.
        Wq, Wk, Wv, Wo: (d, d) matrices in the row-vector convention (q = x @ Wq). Head h uses
            columns [h * d_h, (h + 1) * d_h) of Wq, Wk, Wv with d_h = d / n_heads, and rows
            [h * d_h, (h + 1) * d_h) of Wo.
        n_heads: number of heads, divisible by tp. Rank r owns the contiguous heads
            [r * n_heads/tp, (r + 1) * n_heads/tp).
        tp: tensor-parallel degree.
    Returns:
        (y, partials): partials[r] is rank r's (B, T, d) contribution: its heads' causal
        attention outputs multiplied by its rows of Wo. y is their sum and must equal ordinary
        causal multi-head attention with softmax scale 1/sqrt(d_h).
    """
    raise NotImplementedError


def training_memory_bytes(
    n_params: float,
    zero_stage: int,
    dp: int,
    param_bytes: float = 2,
    grad_bytes: float = 2,
    optim_bytes: float = 12,
    mp: int = 1,
) -> Dict[str, float]:
    """Per-GPU bytes for weights, gradients and optimizer state (no activations, no buffers).

    Args:
        n_params: total parameter count.
        zero_stage: 0 = plain data parallelism (everything replicated), 1 = shard optimizer
            state over dp, 2 = also shard gradients, 3 = also shard the parameters.
        dp: data-parallel degree (the group ZeRO shards over).
        param_bytes, grad_bytes: bytes per parameter for the working weights and the gradients.
        optim_bytes: bytes per parameter of optimizer state (fp32 master copy + two Adam
            moments = 4 + 4 + 4 = 12).
        mp: model-parallel degree (tensor x pipeline). The parameters are first split evenly
            over mp, then ZeRO shards what each model-parallel rank holds over dp.
    Returns:
        dict with float values for the keys "params", "grads", "optim" and "total"
        (the sum of the other three).
    Raises:
        ValueError if zero_stage is not 0, 1, 2 or 3.
    """
    raise NotImplementedError


def zero_comm_bytes_per_step(n_params: float, zero_stage: int, dp: int, bytes_per_elem: float = 2) -> float:
    """Bytes each rank sends per optimizer step for ZeRO stage 0..3 with ring collectives.

    Stages 0, 1, 2 move 2 * n_params elements per step in the large-dp limit (an all-reduce, or
    a reduce-scatter plus an all-gather); stage 3 moves 3 * n_params (two parameter
    all-gathers and one gradient reduce-scatter). A ring collective over dp ranks sends the
    fraction (dp - 1) / dp of the tensor from each rank. Assume one microbatch per step.
    Returns the per-rank bytes as a float. Raise ValueError for an invalid stage.
    """
    raise NotImplementedError


def pipeline_bubble(p: int, m: int, v: int = 1, of: str = "total") -> float:
    """Pipeline bubble for a synchronous 1F1B/GPipe schedule (uniform stage times).

    Args:
        p: number of pipeline stages. m: microbatches per step. v: model chunks per device
            (v > 1 is the interleaved schedule, which divides the bubble by v).
        of: "total" returns the idle fraction of the step, bubble / (ideal + bubble).
            "ideal" returns bubble time relative to the ideal compute time, bubble / ideal.
    Returns a float. Raise ValueError if `of` is neither "total" nor "ideal".
    """
    raise NotImplementedError


def simulate_1f1b(
    p: int,
    m: int,
    t_fwd: float = 1.0,
    t_bwd: float = 2.0,
    schedule: str = "1f1b",
) -> Tuple[float, List[int]]:
    """Event-simulate a synchronous pipeline of p stages and m microbatches.

    Stage i runs its operations in a fixed order and one at a time. F(i, j), the forward pass of
    microbatch j on stage i, needs F(i-1, j) to have finished (no dependency for i = 0). B(i, j),
    the backward pass, needs B(i+1, j) (for the last stage: its own F(i, j)). Ignore
    communication time. Operation order on stage i (0-indexed):
        "gpipe": F(i, 0..m-1), then B(i, 0..m-1).
        "1f1b":  w = min(p - 1 - i, m) warm-up forwards, then alternate one forward and one
                 backward until every forward has run, then the remaining backwards.
    Every op starts as soon as both its stage and its dependency are free.
    Returns:
        (makespan, peak_inflight): the time when the last op finishes, and for every stage the
        maximum number of microbatches whose forward has started but whose backward has not
        yet finished (the activations that stage must hold). When one op ends exactly when
        another starts, count the end first.
    """
    raise NotImplementedError


def model_flops_per_token(
    n_params: float,
    n_layers: int = 0,
    d_model: int = 0,
    seq_len: int = 0,
    causal: bool = True,
) -> float:
    """Model FLOPs to train on one token: 6 * n_params plus the attention-score term.

    The attention term is 12 * n_layers * d_model * seq_len for full attention (PaLM,
    appendix B) and half of that for a causal mask. It is zero when n_layers is 0.
    """
    raise NotImplementedError


def mfu(tokens_per_sec_per_gpu: float, flops_per_token: float, peak_flops: float) -> float:
    """Model FLOPs utilization: achieved model FLOP/s per GPU divided by the GPU's peak FLOP/s."""
    raise NotImplementedError
