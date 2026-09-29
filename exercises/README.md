# Exercises

Every page of the course has one or more drills here. Each drill has three files:

| File | What it is |
|---|---|
| `drills/<name>.py` | Function signatures and docstrings. **You** fill in the bodies. |
| `solutions/<name>.py` | Reference implementation. Don't open it until you have a passing attempt or are truly stuck. |
| `tests/test_<name>.py` | Tests. By default they run against your code in `drills/`. |

```bash
pip install -r exercises/requirements.txt       # CPU torch is enough

pytest exercises/tests/test_attention.py         # test YOUR implementation
pytest exercises --solutions                     # sanity-check every reference solution
LLM_SOLUTIONS=1 pytest exercises                 # same, via env var
```

The tests are self-contained: they compare against PyTorch built-ins or check mathematical properties, so they never import the solutions.

Interview practice tip: time-box each drill to what an interview gives you (usually 20 to 40 minutes), write it without looking anything up, and only then run the tests.

## Index

| Page | Drill | File |
|---|---|---|
| 01 | Language-modeling loss and metrics (hand-written log-softmax, perplexity, bits per byte) | `drills/lm_loss.py` |
| 02 | Byte-level BPE from scratch (train, encode, decode) | `drills/bpe.py` |
| 03 | Attention from scratch: stable softmax, end-aligned causal mask, SDPA, multi-head attention | `drills/attention.py` |
| 04 | A GPT from scratch: RMSNorm, SwiGLU, GQA block, exact parameter count and FLOPs | `drills/transformer.py` |
| 05 | RoPE in both layouts (interleaved and rotate-half), sinusoidal, ALiBi | `drills/rope.py` |
| 06 | Sampling from logits: temperature, top-k, top-p, min-p | `drills/sampling.py` |
| 06 | KV cache, incremental attention, greedy decoding | `drills/kv_cache.py` |
| 07 | GQA with a KV cache; MHA→GQA mean-pool conversion | `drills/gqa.py` |
| 07 | MLA: decompressed forward and absorbed decode; KV bytes per token | `drills/mla.py` |
| 08 | Mixture of Experts: router, MoE layer, balance losses, aux-loss-free bias, capacity mask | `drills/moe.py` |
| 09 | Beyond full attention: sliding window, linear/delta-rule recurrence, top-k sparse attention | `drills/efficient_attention.py` |
| 10 | RoPE scaling for context extension: PI, NTK-aware, dynamic NTK, YaRN, Llama 3.1 rule | `drills/rope_scaling.py` |
| 11 | Training stability primitives (RMSNorm, QK-norm attention, z-loss, soft-cap, attention entropy) and multi-token prediction | `drills/stability.py` |
| 12 | Web-data pipeline: Gopher quality filter, MinHash, LSH deduplication | `drills/data_pipeline.py` |
| 13 | Scaling laws: fit, optimum, inference-aware sizing | `drills/scaling.py` |
| 14 | Optimizers: AdamW step, LR schedules (cosine/WSD/step), Newton-Schulz, Muon step | `drills/optim.py` |
| 15 | Distributed training by simulation: ring all-reduce, tensor-parallel MLP and attention, ZeRO memory, pipeline bubble, MFU | `drills/parallel.py` |
