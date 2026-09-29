// Single source of truth for the curriculum. Loaded as a script (not fetched as JSON)
// so navigation works when pages are opened directly from disk (file://).
// status: "planned" pages appear greyed out in the sidebar until they are written.
window.LLM_MANIFEST = {
  title: "LLM Interview Prep",
  parts: [
    {
      id: "I", title: "Foundations",
      pages: [
        { n: "01", slug: "01-big-picture", title: "The big picture: language modeling", tier: "P0", minutes: 45, status: "written",
          desc: "Autoregressive factorization, cross-entropy and perplexity, the pretrain→SFT→RL pipeline, why decoder-only won." },
        { n: "02", slug: "02-tokenization", title: "Tokenization", tier: "P0", minutes: 50, status: "planned",
          desc: "BPE from scratch, byte-level BPE, Unigram/SentencePiece, vocab-size trade-offs, chat templates, tokenizer failure modes." },
        { n: "03", slug: "03-attention", title: "Attention from scratch", tier: "P0", minutes: 50, status: "written",
          desc: "QKV, the 1/√d scale, causal masking, multi-head attention, O(n²) cost, a numerically stable implementation." },
        { n: "04", slug: "04-transformer-block", title: "The Transformer block, params & FLOPs", tier: "P0", minutes: 70, status: "written",
          desc: "Residual stream, pre-norm, RMSNorm, SwiGLU, weight tying, exact parameter and FLOP counting, a ~150-line GPT." },
        { n: "05", slug: "05-positional-encoding", title: "Positional encoding & RoPE", tier: "P0", minutes: 55, status: "written",
          desc: "Sinusoidal, learned, relative bias, ALiBi, full RoPE derivation, partial RoPE, NoPE, length generalization." },
        { n: "06", slug: "06-decoding-kv-cache", title: "Decoding & the KV cache", tier: "P0", minutes: 55, status: "written",
          desc: "Sampling (temperature, top-k, top-p, min-p), beam search, KV cache memory, prefill vs decode, arithmetic intensity." }
      ]
    },
    {
      id: "II", title: "Modern architecture",
      pages: [
        { n: "07", slug: "07-kv-efficient-attention", title: "KV-efficient attention: MQA, GQA, MLA", tier: "P0", minutes: 70, status: "written",
          desc: "MQA, GQA and DeepSeek’s Multi-head Latent Attention: low-rank KV compression, weight absorption, decoupled RoPE." },
        { n: "08", slug: "08-mixture-of-experts", title: "Mixture of Experts", tier: "P0", minutes: 75, status: "planned",
          desc: "Routing, load balancing (aux loss and aux-loss-free), fine-grained and shared experts, capacity, expert parallelism." },
        { n: "09", slug: "09-beyond-full-attention", title: "Beyond full attention: sparse, linear, SSM, hybrid", tier: "P1", minutes: 75, status: "planned",
          desc: "Sliding window, sparse attention (NSA, DSA), linear attention, Mamba, Gated DeltaNet, and hybrid stacks." },
        { n: "10", slug: "10-long-context", title: "Long context: extension & evaluation", tier: "P0", minutes: 60, status: "planned",
          desc: "RoPE scaling (PI, NTK, YaRN), staged context extension, long-context data, attention sinks, NIAH and RULER." },
        { n: "11", slug: "11-stability-tweaks", title: "Training stability & modern tweaks", tier: "P1", minutes: 50, status: "planned",
          desc: "Loss spikes, QK-norm, z-loss, logit softcapping, norm placement, gated attention, multi-token prediction." }
      ]
    },
    {
      id: "III", title: "Pretraining & systems",
      pages: [
        { n: "12", slug: "12-pretraining-data", title: "Pretraining data", tier: "P0", minutes: 60, status: "planned",
          desc: "Common Crawl pipelines, heuristic and model-based filtering, dedup, mixtures, annealing, synthetic data, contamination." },
        { n: "13", slug: "13-scaling-laws", title: "Scaling laws", tier: "P0", minutes: 60, status: "planned",
          desc: "Kaplan vs Chinchilla, IsoFLOP analysis, C≈6ND, overtraining for inference, data-constrained and MoE scaling." },
        { n: "14", slug: "14-optimization", title: "Optimizers, schedules & hyperparameters", tier: "P0", minutes: 65, status: "planned",
          desc: "AdamW details, warmup, cosine vs WSD, batch size, clipping, μP, Muon/MuonClip, a loss-spike playbook." },
        { n: "15", slug: "15-distributed-training", title: "Memory, precision & parallelism", tier: "P0", minutes: 80, status: "planned",
          desc: "Bytes per parameter, activation memory, bf16/fp8, ZeRO/FSDP, tensor/pipeline/expert/context parallelism, MFU." },
        { n: "16", slug: "16-gpus-kernels", title: "GPUs, kernels & FlashAttention", tier: "P1", minutes: 60, status: "planned",
          desc: "GPU memory hierarchy, roofline, kernel fusion, FlashAttention 1→3 and online softmax, Triton basics." },
        { n: "17", slug: "17-inference-serving", title: "Inference optimization & serving", tier: "P1", minutes: 60, status: "planned",
          desc: "Quantization, speculative decoding, continuous batching, PagedAttention, prefix caching, disaggregated serving." },
        { n: "18", slug: "18-evaluation", title: "Evaluation", tier: "P1", minutes: 45, status: "planned",
          desc: "Benchmark taxonomy, pass@k, likelihood vs generative evals, contamination, eval noise, LLM-as-judge." }
      ]
    },
    {
      id: "IV", title: "Post-training: SFT & preferences",
      pages: [
        { n: "19", slug: "19-sft-distillation", title: "SFT, instruction tuning & distillation", tier: "P0", minutes: 55, status: "planned",
          desc: "Chat formatting, loss masking, packing, data quality, rejection sampling, logit/sequence/on-policy distillation." },
        { n: "20", slug: "20-peft-lora", title: "Parameter-efficient finetuning (LoRA, QLoRA)", tier: "P1", minutes: 40, status: "planned",
          desc: "LoRA math and initialization, α/r, QLoRA and NF4, DoRA, what low-rank updates cannot do." },
        { n: "21", slug: "21-preference-optimization", title: "Reward models & preference optimization (DPO family)", tier: "P0", minutes: 65, status: "planned",
          desc: "Bradley-Terry reward models, over-optimization, the full DPO derivation, IPO/KTO/ORPO/SimPO, online DPO." }
      ]
    },
    {
      id: "V", title: "Reinforcement learning",
      pages: [
        { n: "22", slug: "22-rl-foundations", title: "RL foundations for LLMs: policy gradient to PPO", tier: "P0", minutes: 70, status: "planned",
          desc: "Token-level MDP, REINFORCE, baselines, GAE, importance sampling, the PPO clip, KL estimators." },
        { n: "23", slug: "23-rlhf", title: "RLHF in practice", tier: "P0", minutes: 50, status: "planned",
          desc: "The InstructGPT pipeline, 4-model memory footprint, reward hacking, length bias, RLAIF, PRMs vs ORMs." },
        { n: "24", slug: "24-rlvr-reasoning", title: "RLVR & reasoning models: GRPO and successors", tier: "P0", minutes: 80, status: "planned",
          desc: "GRPO, R1-Zero, verifiers, DAPO, Dr. GRPO, GSPO, CISPO, entropy collapse, test-time scaling." },
        { n: "25", slug: "25-rl-systems-agentic", title: "RL systems & agentic RL", tier: "P1", minutes: 50, status: "planned",
          desc: "Rollout/trainer split, async staleness, train-inference mismatch, multi-turn tool-use RL." }
      ]
    },
    {
      id: "VI", title: "Model case studies",
      pages: [
        { n: "26", slug: "26-deepseek-1-foundations", title: "DeepSeek I: LLM, MoE, Math, V2", tier: "P0", minutes: 60, status: "planned",
          desc: "DeepSeek LLM hyperparameter scaling, DeepSeekMoE, DeepSeekMath and GRPO, DeepSeek-V2." },
        { n: "27", slug: "27-deepseek-2-v3", title: "DeepSeek II: V3 end to end", tier: "P0", minutes: 75, status: "planned",
          desc: "V3 architecture, aux-loss-free MoE, MTP, FP8 training, DualPipe, the cost accounting." },
        { n: "28", slug: "28-deepseek-3-reasoning", title: "DeepSeek III: R1 to V3.2 and beyond", tier: "P0", minutes: 65, status: "planned",
          desc: "R1-Zero and R1, distillation, V3.1 hybrid thinking, V3.2 sparse attention and agentic RL, later releases." },
        { n: "29", slug: "29-llama", title: "Llama 1 to 4", tier: "P1", minutes: 45, status: "planned",
          desc: "The Llama recipe, GQA, 15T-token overtraining, Llama 3 long context, Llama 4 MoE and iRoPE." },
        { n: "30", slug: "30-qwen", title: "Qwen 1 to 3 and Qwen3-Next", tier: "P1", minutes: 45, status: "planned",
          desc: "Qwen2/2.5/3, QK-norm, hybrid thinking, global-batch balancing, Qwen3-Next Gated DeltaNet hybrid." },
        { n: "31", slug: "31-mistral-gemma-gptoss", title: "Mistral, Gemma, gpt-oss", tier: "P1", minutes: 45, status: "planned",
          desc: "Sliding window, Mixtral MoE, Gemma local/global and distillation, gpt-oss attention sinks and MXFP4." },
        { n: "32", slug: "32-frontier-open-models", title: "Kimi, GLM, MiniMax, OLMo, Nemotron", tier: "P1", minutes: 50, status: "planned",
          desc: "Kimi K2 and MuonClip, GLM-4.5, MiniMax lightning attention and CISPO, OLMo open recipes, Nemotron-H." },
        { n: "33", slug: "33-architecture-atlas", title: "Architecture atlas", tier: "P0", minutes: 35, status: "planned",
          desc: "Side-by-side configs for ~20 models, a timeline, and what converged and why." }
      ]
    },
    {
      id: "VII", title: "Interview toolkit",
      pages: [
        { n: "34", slug: "34-back-of-envelope", title: "Back-of-envelope calculations", tier: "P0", minutes: 45, status: "planned",
          desc: "Parameters, FLOPs, training time, memory, KV cache, MFU, with worked drills." },
        { n: "35", slug: "35-coding-drills", title: "Coding drills", tier: "P0", minutes: 90, status: "planned",
          desc: "From-scratch implementations you should be able to write in an interview, with tests." },
        { n: "36", slug: "36-question-bank", title: "Question bank & mock scenarios", tier: "P0", minutes: 120, status: "planned",
          desc: "All page questions in one place, plus open-ended design and debugging scenarios." },
        { n: "37", slug: "37-reading-list-glossary", title: "Reading list & glossary", tier: "P2", minutes: 20, status: "planned",
          desc: "Prioritized papers and an English/Chinese glossary." }
      ]
    }
  ]
};
