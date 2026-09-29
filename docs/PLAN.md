# Plan: LLM interview-prep course

## Goal

A ground-up series of static HTML pages that gets a reader with solid ML fundamentals
(and experience training encoder transformers for jet tagging) ready for model-building
interviews at frontier labs. The focus is architecture, pretraining, finetuning and RL,
with the famous open models (DeepSeek first) as case studies.

## Decisions (agreed with the user)

| Topic | Decision |
|---|---|
| Role emphasis | Balance research scientist and research engineer loops |
| Starting point | Has trained transformers (jet tagging), never language models: Part I is taught fully |
| Writers | Sonnet subagents, one page each |
| Review | A Sonnet fact-check pass on **every** page, against primary sources |
| Mechanical work | Haiku (question-bank aggregation, glossary, link fixes) |
| Exemplar + audit | Opus writes the exemplar page (07) and audits derivation-heavy P0 pages |
| Exercises | Yes: `exercises/` with drills, reference solutions, and pytest tests |
| Language | English, with Chinese terms for important concepts only |
| Out of scope | Multimodal/VLMs, safety/interpretability, RAG and agent applications, diffusion LMs |

## Curriculum

The canonical list (titles, tiers, minutes, one-line descriptions) is
`assets/manifest.js`. 37 pages in 7 parts plus `index.html`:

| Part | Pages |
|---|---|
| I · Foundations | 01 big picture · 02 tokenization · 03 attention · 04 transformer block · 05 positional encoding · 06 decoding & KV cache |
| II · Modern architecture | 07 MQA/GQA/MLA · 08 MoE · 09 beyond full attention · 10 long context · 11 stability tweaks |
| III · Pretraining & systems | 12 data · 13 scaling laws · 14 optimization · 15 memory/precision/parallelism · 16 GPUs & kernels · 17 inference & serving · 18 evaluation |
| IV · SFT & preferences | 19 SFT & distillation · 20 PEFT · 21 reward models & DPO family |
| V · RL | 22 RL foundations · 23 RLHF · 24 RLVR & reasoning · 25 RL systems & agentic RL |
| VI · Case studies | 26–28 DeepSeek I–III · 29 Llama · 30 Qwen · 31 Mistral/Gemma/gpt-oss · 32 Kimi/GLM/MiniMax/OLMo/Nemotron · 33 architecture atlas |
| VII · Interview toolkit | 34 back-of-envelope · 35 coding drills · 36 question bank · 37 reading list & glossary |

P0 pages (know cold): 01–08, 10, 12–15, 19, 21–24, 26–28, 33–36.

## Pipeline

1. **Scaffold (Opus):** shared CSS/JS, vendored KaTeX and highlight.js (offline-capable),
   manifest-driven navigation, page template, `tools/qa.py`, exercises harness,
   `docs/STYLE_GUIDE.md`.
2. **Exemplar (Opus):** page 07 (MQA/GQA/MLA) sets the bar for depth, verification, diagrams, and drills.
3. **Briefs (Opus):** `docs/briefs/NN.md` per page, written just before each batch:
   must-cover list, required worked examples, required questions, primary sources.
4. **Write (Sonnet, parallel batches):** one agent per page, following the style guide,
   exemplar, and brief; verifies facts via alphaXiv/web; writes the page and its drills; runs QA and tests.
5. **Fact-check (Sonnet, one per page):** adversarial review against primary sources;
   fixes errors in place and reports a changelog.
6. **Audit (Opus):** reads the review reports, spot-checks the derivation-heavy P0 pages
   (RoPE, MLA, DPO, PPO/GRPO, scaling laws, memory math), flips the page to `written`
   in the manifest, commits, and pushes. The push triggers CI (QA + drill tests) and, on
   success, deploys the finished pages to GitHub Pages (`.github/workflows/site.yml`).
7. **Aggregate (Haiku):** question bank (36), glossary (37), and README index at the end.

Batches: I (01–06) → user checkpoint → II (08–11) → III (12–18) → IV+V (19–25) →
VI (26–33) → VII (34–37).

## Status

| Batch | Pages | Status |
|---|---|---|
| Scaffold + exemplar | 07 | done (updated with DeepSeek-V4, Apr 2026) |
| I | 01–06 | done: written, fact-checked, audited (RoPE, memory math) |
| 2 | 08–15 | done: written, fact-checked; 09, 13, 15 audited by coordinator |
| 3 | 16–21 | done: written, fact-checked; 21 (DPO) audited by coordinator |
| 4 | 22–25 | done: written, fact-checked; 22 (PPO) and 24 (GRPO) audited by coordinator |
| 5 | 26–28 | done: written, fact-checked; 28 extended to V4.1-Flash (Sep 2026) |
| 6 | 29, 30, 34 | done: written, fact-checked |
| 7 | 31–33, 35–37 | 31, 32, 35 being written; then 33, 36, 37 |

Fact-check yield so far (errors fixed before commit): page 01 (vacuous test, citation scope),
02 (4 errors: digit grouping, Unigram unk, vocab-extension count, byte-level claim),
03 (3 errors: misattributed table, MLA projection claim, reversed TP rule),
04 (initial-loss formula, tying rule of thumb), 05 (figure caption direction, "values can't be
relative" claim), 06 (CTRL penalty direction, irreproducible simulation claim), 13 (about 15 imprecisions:
Llama 3 6ND footnote, Kaplan exponent sources, Hoffmann appendix sizes, Sardana & Frankle scope,
figure captions; drill validation tightened).
