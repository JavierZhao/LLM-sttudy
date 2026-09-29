"""Reference solutions: back-of-envelope formulas (page 34).

Pure Python, no dependencies. Conventions used throughout (state them out loud in an interview):
  - a multiply-add is 2 FLOPs; "peak" means the DENSE peak (not the 2:4-sparsity headline number)
  - GB = 1e9 bytes (GiB = 2**30 is used only where a docstring says so)
  - bandwidths are per direction (NVLink 4 is 900 GB/s bidirectional = 450 GB/s each way)
  - time is in seconds, sizes in bytes, rates in FLOP/s or bytes/s
"""
from __future__ import annotations

import math

GB = 1e9
GiB = 2**30

# Hardware constants used by the page (verified against NVIDIA datasheets, see page 34).
H100_BF16 = 989.4e12       # dense bf16 FLOP/s (NVIDIA's 1,979 TFLOP/s headline includes 2:4 sparsity)
H100_FP8 = 1978.9e12       # dense fp8 FLOP/s
H100_BW = 3.35e12          # HBM3 bytes/s
H100_HBM = 80e9            # bytes
NVLINK4 = 450e9            # bytes/s per direction per GPU (900 GB/s bidirectional)
IB_400G = 50e9             # bytes/s per GPU (400 Gb/s)
EPOCH = dict(A=482.01, B=2085.43, alpha=0.3478, beta=0.3658)   # Epoch AI refit of Chinchilla's law (page 13)


def params(L: int, d: int, n_h: int, n_kv: int, d_h: int, d_ff: int, V: int, tied: bool = False) -> int:
    """Exact parameter count of a Llama-style dense decoder.

    Assumptions: GQA attention with W_q (d, n_h*d_h), W_k and W_v (d, n_kv*d_h), W_o (n_h*d_h, d);
    SwiGLU FFN with three (d, d_ff) matrices; two RMSNorm gains (d each) per layer plus one final
    RMSNorm; no biases; RoPE (so no position table). The input embedding is (V, d) and the LM head
    is another (V, d) unless tied=True (then it is shared and counted once).
    Returns an int. Llama-3-8B must give 8,030,261,248.
    """
    attn = 2 * d * n_h * d_h + 2 * d * n_kv * d_h      # q and o, k and v
    ffn = 3 * d * d_ff                                  # gate, up, down
    per_layer = attn + ffn + 2 * d                      # + two norm gains
    vocab = V * d * (1 if tied else 2)
    return vocab + L * per_layer + d                    # + final norm


def mla_attn_params(d: int, n_h: int, d_nope: int, d_rope: int, d_v: int, d_c: int, d_cq: int) -> int:
    """Parameters of one multi-head latent attention layer (DeepSeek-V2/V3, page 07).

    Matrices: W_DQ (d, d_cq); W_UQ and W_QR together (d_cq, n_h*(d_nope+d_rope));
    W_DKV and W_KR together (d, d_c+d_rope); W_UK and W_UV together (d_c, n_h*(d_nope+d_v));
    W_O (n_h*d_v, d); plus the two latent RMSNorm gains (d_cq + d_c). DeepSeek-V3 must give 187,107,328.
    """
    return (d * d_cq
            + d_cq * n_h * (d_nope + d_rope)
            + d * (d_c + d_rope)
            + d_c * n_h * (d_nope + d_v)
            + n_h * d_v * d
            + d_cq + d_c)


def moe_params(L: int, L_moe: int, d: int, attn: int, n_routed: int, n_shared: int, k: int,
               d_e: int, d_dense: int, V: int) -> tuple[int, int]:
    """(total, activated) parameters of a DeepSeek-style MoE, untied embeddings, page 08 recipe.

    L layers, of which the LAST L_moe are MoE and the first L - L_moe are dense SwiGLU FFNs of width
    d_dense. attn is the attention parameters per layer (already includes any latent norms).
    Each MoE layer has n_routed routed and n_shared shared SwiGLU experts of width d_e (three
    (d, d_e) matrices each) and a router (d, n_routed). Norms: (2L + 1) gains of size d.
    'Activated' counts, for each token: all attention and dense layers, k routed + n_shared shared
    experts and the router per MoE layer, the norms, and BOTH vocabulary matrices (the 37.55B
    convention of page 08; subtract V*d for the 36.63B count without the input table, or 2*V*d for
    the 35.70B non-embedding count).
    DeepSeek-V3 must give (671,026,404,352, 37,552,282,624).
    """
    expert = 3 * d * d_e
    common = L * attn + (L - L_moe) * 3 * d * d_dense + 2 * V * d + (2 * L + 1) * d
    router = d * n_routed
    total = common + L_moe * ((n_routed + n_shared) * expert + router)
    active = common + L_moe * ((k + n_shared) * expert + router)
    return total, active


def flops(n_params: float, L: int = 0, d: int = 0, T: int = 0, causal: bool = True,
          training: bool = False) -> float:
    """FLOPs per TOKEN: matmul weights plus the attention-score term.

    n_params is the number of parameters that take part in matmuls (page 04: pass the total minus the
    input embedding table if you want to be exact; 6ND uses the total). Forward is 2*n_params + attn
    with attn = 4*L*T*d over the full T x T scores, or 2*L*T*d if causal=True (the average token
    sees T/2 keys). Set L, d, T (all > 0) to include attention; leave them 0 for the plain 2N.
    training=True multiplies the forward count by 3 (forward + input-gradient + weight-gradient),
    giving 6N + 3*attn. No recomputation is counted.
    """
    attn = (2 if causal else 4) * L * T * d
    fwd = 2 * n_params + attn
    return 3 * fwd if training else fwd


def train_flops(n_params: float, tokens: float) -> float:
    """C = 6 N D: total training FLOPs (forward + backward, no recomputation, no attention term)."""
    return 6.0 * n_params * tokens


def train_time(C: float, n_gpu: int, peak_flops: float, mfu: float) -> float:
    """Wall-clock seconds to execute C FLOPs on n_gpu GPUs: C / (n_gpu * peak * MFU).

    peak_flops is the DENSE per-GPU peak in the number format you train in. mfu is model-FLOPs
    utilization in (0, 1]. Assumes perfect scaling and no downtime.
    """
    return C / (n_gpu * peak_flops * mfu)


def gpu_hours(C: float, peak_flops: float, mfu: float) -> float:
    """GPU-hours to execute C FLOPs at the given dense peak and MFU (independent of GPU count)."""
    return C / (peak_flops * mfu) / 3600.0


def dollar_cost(gpu_hours_: float, price_per_gpu_hour: float = 2.0) -> float:
    """Rental cost in dollars: GPU-hours times price per GPU-hour ($2 is the page's assumption)."""
    return gpu_hours_ * price_per_gpu_hour


def cost_per_million_tokens(price_per_gpu_hour: float, n_gpu: int, tokens_per_s: float) -> float:
    """Dollars per million tokens: price * n_gpu / (3600 * tokens_per_s) * 1e6.

    tokens_per_s is the TOTAL rate of the n_gpu GPUs (counting only tokens that met the SLO).
    """
    return price_per_gpu_hour * n_gpu / (3600.0 * tokens_per_s) * 1e6


def train_memory(n_params: float, zero_stage: int = 0, n_dp: int = 1, w: int = 2, g: int = 2, k: int = 12) -> float:
    """Model-state bytes PER GPU for mixed-precision Adam with the ZeRO accounting (page 15).

    w bytes of working weights, g bytes of gradients and k bytes of optimizer state (fp32 master +
    Adam m and v = 12) per parameter, so w + g + k = 16 by default. zero_stage 0 is plain DDP (nothing
    sharded); 1 shards the k term over n_dp ranks; 2 also shards gradients; 3 also shards weights.
    Activations, buffers and fragmentation are NOT included.
    """
    if zero_stage not in (0, 1, 2, 3):
        raise ValueError("zero_stage must be 0, 1, 2 or 3")
    weights = w * n_params / (n_dp if zero_stage >= 3 else 1)
    grads = g * n_params / (n_dp if zero_stage >= 2 else 1)
    opt = k * n_params / (n_dp if zero_stage >= 1 else 1)
    return weights + grads + opt


def activation_memory(s: int, b: int, h: int, a: int, n_layers: int, mode: str = "full") -> float:
    """Activation bytes for a GPT-style stack, Korthikanti et al. (page 15), no model parallelism.

    s = sequence length, b = micro-batch, h = hidden size, a = attention heads. Per layer:
    mode 'full' stores everything: s*b*h*(34 + 5*a*s/h); 'selective' (or a fused attention kernel that
    never stores the score matrix): 34*s*b*h; 'recompute' (full recomputation, keep each layer's
    input only): 2*s*b*h. Returns total bytes over n_layers.
    """
    sbh = s * b * h
    if mode == "full":
        per_layer = sbh * (34 + 5 * a * s / h)
    elif mode == "selective":
        per_layer = 34 * sbh
    elif mode == "recompute":
        per_layer = 2 * sbh
    else:
        raise ValueError("mode must be 'full', 'selective' or 'recompute'")
    return n_layers * per_layer


def kv_bytes(n_layers: int, n_kv: int, d_h: int, seq_len: int, batch: int = 1, bytes_per_el: float = 2,
             latent_dim: int | None = None) -> float:
    """Total KV-cache bytes for `batch` sequences of `seq_len` tokens.

    Standard attention (MHA n_kv = n_h, GQA, MQA n_kv = 1): n_layers * 2 * n_kv * d_h * bytes_per_el
    per token. If latent_dim is given, use MLA instead: n_layers * latent_dim * bytes_per_el per token
    with latent_dim = d_c + d_rope (576 for DeepSeek-V3); n_kv and d_h are then ignored.
    bytes_per_el is 2 for bf16, 1 for fp8. Llama-3-8B gives 131,072 B per token.
    """
    per_layer = latent_dim if latent_dim is not None else 2 * n_kv * d_h
    return n_layers * per_layer * bytes_per_el * seq_len * batch


def decode_time(weight_bytes: float, kv_bytes_total: float, bandwidth: float,
                n_flops: float = 0.0, peak_flops: float = math.inf) -> float:
    """Roofline lower bound on one decode step, in seconds: max(memory time, compute time).

    Memory time = (weight_bytes + kv_bytes_total) / bandwidth: every step streams all weights once
    (shared by the whole batch) plus every sequence's KV cache. Compute time = n_flops / peak_flops
    (both optional: with the default peak_flops = inf the compute term is zero). Real engines reach
    a fraction of this bound (80% in one measured H100 NVL point, page 06).
    """
    return max((weight_bytes + kv_bytes_total) / bandwidth, n_flops / peak_flops)


def prefill_time(n_params: float, n_tokens: int, peak_flops: float, mfu: float = 0.5) -> float:
    """Seconds to prefill n_tokens: 2 * n_params * n_tokens / (peak_flops * mfu).

    Compute-bound, so time is FLOPs over achieved FLOP/s. Ignores the attention-score term (add it
    yourself at long context) and the (small) weight-read time.
    """
    return 2.0 * n_params * n_tokens / (peak_flops * mfu)


def allreduce_time(size_bytes: float, n_ranks: int, bandwidth: float) -> float:
    """Bandwidth-optimal ring all-reduce: 2 * (n - 1) / n * size / bandwidth, in seconds.

    bandwidth is the per-GPU link bandwidth in ONE direction. Ignores latency, which matters for
    small messages. For reduce-scatter, all-gather or all-to-all halve the result.
    """
    return 2.0 * (n_ranks - 1) / n_ranks * size_bytes / bandwidth


def pipeline_bubble(p: int, m: int, v: int = 1) -> tuple[float, float]:
    """Pipeline bubble for p stages, m micro-batches, v interleaved chunks per stage.

    Returns (relative, idle): relative = (p - 1) / (v * m), the bubble as a fraction of the ideal
    compute time (Narayanan et al.); idle = (p - 1) / (v * m + p - 1), the fraction of wall-clock
    time a GPU sits idle. GPipe and 1F1B have v = 1.
    """
    return (p - 1) / (v * m), (p - 1) / (v * m + p - 1)


def moe_alltoall_bytes(d: int, dispatch_bytes: float = 1, combine_bytes: float = 2, n_dest: int = 1) -> float:
    """Bytes one token moves through one MoE layer's dispatch and combine.

    Per destination (node or GPU) the token's hidden state (d elements) goes out once at dispatch_bytes
    per element and its expert output comes back once at combine_bytes per element. DeepSeek-V3:
    FP8 dispatch (1), BF16 combine (2), so 3 * d = 21,504 B per destination at d = 7168.
    """
    return n_dest * d * (dispatch_bytes + combine_bytes)


def experts_touched(n_experts: int, k: int, batch: int) -> float:
    """Expected number of distinct routed experts read in one layer by `batch` tokens.

    Assumes each token picks k of n_experts uniformly and independently:
    n_experts * (1 - (1 - k / n_experts) ** batch). Real routing is correlated, which lowers it.
    """
    return n_experts * (1.0 - (1.0 - k / n_experts) ** batch)


def chinchilla_opt(C: float, tokens_per_param: float | None = None, A: float = EPOCH["A"],
                   B: float = EPOCH["B"], alpha: float = EPOCH["alpha"],
                   beta: float = EPOCH["beta"]) -> tuple[float, float]:
    """Compute-optimal (N, D) for a budget of C FLOPs, with C = 6 N D.

    If tokens_per_param is given (say 20), use the rule of thumb: N = sqrt(C / (6 r)), D = r N.
    Otherwise minimize E + A/N^alpha + B/D^beta in closed form (page 13):
    N = G (C/6)^a with G = (alpha A / (beta B))^(1/(alpha+beta)) and a = beta / (alpha + beta),
    D = C / (6 N). The default constants are the Epoch AI refit. Returns (N, D) as floats.
    """
    if tokens_per_param is not None:
        N = math.sqrt(C / (6.0 * tokens_per_param))
        return N, tokens_per_param * N
    G = (alpha * A / (beta * B)) ** (1.0 / (alpha + beta))
    a = beta / (alpha + beta)
    N = G * (C / 6.0) ** a
    return N, C / (6.0 * N)
