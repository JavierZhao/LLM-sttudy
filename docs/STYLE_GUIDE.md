# Style guide for course pages

Every page writer (human or agent) follows this document. The exemplar page is
`pages/07-kv-efficient-attention.html`: when in doubt, match what it does.

## 1. Reader and goal

- **Reader:** strong ML fundamentals (optimization, backprop, regularization,
  probability). Has trained **encoder-style transformers for jet tagging** in particle
  physics: set-structured inputs, permutation-invariant, no positional encoding,
  classification head, no autoregressive generation. Has **never** worked with language
  models: tokenizers, causal LM, KV caches, pretraining at scale, SFT, RLHF are all new.
- **Goal:** pass interviews for core model-building roles at frontier labs (research
  scientist *and* research engineer loops, weighted equally). That means three things:
  1. **Exactness**: formulas, shapes, parameter/FLOP/memory counts that are right.
  2. **Why**: the problem each design solves, its alternatives, the evidence, the cost.
  3. **Fluency**: being able to implement core pieces from scratch and reason about
     trade-offs out loud.
- Do not re-teach generic ML (what a gradient is, what overfitting is). Do teach every
  LLM-specific concept rigorously, even if it looks simple.
- Use the jet-tagging background where it gives a real contrast (for example: "your
  tagger was permutation-invariant by design; a language model must not be"). Use a
  `callout bridge` for it, at most 2 or 3 per page, and never force it.

## 2. Voice

- Precise, dense, calm. No hype ("revolutionary", "game-changing"), no filler, no praise.
- Every design choice gets a **why**. "X uses Y" alone is never enough.
- Prefer concrete numbers to adjectives: "cuts the KV cache 8×", not "much smaller".
- **House style: never use a pair of dashes (— … — or -- … --) to wrap a clause.** Use a
  colon, commas, or parentheses. A single em dash is allowed but rare.
- American English. Present tense. Use "we" for derivations and "you" sparingly.
- When the field is uncertain or results conflict, say so and give both sides.

## 3. Length and depth

| Tier | Body text target | Interview questions | Diagrams |
|---|---|---|---|
| P0 | 4,500 to 8,000 words | 12 to 16 | 2 to 4 |
| P1 | 3,000 to 6,000 words | 10 to 14 | 1 to 3 |

Depth beats breadth: if something is in the brief's must-cover list, explain it well
enough that the reader could re-derive it on a whiteboard. Cut what is not in the brief
before you cut depth.

## 4. Page structure (required, in this order)

Start from `pages/_template.html`. Keep its `<head>`, scripts, and layout wrapper
exactly. Required elements (the QA script checks the ids):

1. `header.page-header`: crumbs (`Part X · Part title · Page NN`), `h1`, `.meta` with tier
   badge, minutes, prerequisite links, and a `p.lede`.
2. `aside.callout.lens#interview-lens`: 3 to 6 bullets on what interviewers probe here.
3. `section#objectives`: "After this page you can", 4 to 7 concrete, testable bullets.
4. **Body**: 4 to 8 `h2` sections, each with an `id`. Within a topic, go
   **intuition → math → code → diagram → worked numbers**. Put a `.decision` card on
   every major design choice.
5. `section#pitfalls`: "Common misconceptions", 5 to 8 bullets, each misconception in
   bold and then corrected.
6. `section#questions`: "Interview questions", tiered `details.qa` with
   `data-tier="warmup" | "core" | "deep"`. Roughly 25% warm-up, 50% core, 25% deep.
   Answers are complete: what a strong candidate would say in 1 to 3 minutes, with math
   where needed. Include at least one "calculate X" question and at least one "design /
   trade-off" question.
7. `section#exercises`: "Hands-on", linking the drill files you created (see §9).
8. `section#papers`: "Primary sources", an `ol.papers` with 4 to 10 entries, each with a
   one-line "why read it".
9. `section#cheatsheet`: "Cheat sheet", one screen of the facts and formulas to memorize.
10. `nav.pager` (empty; site.js fills it).

The file is `pages/NN-slug.html`, using the slug from `assets/manifest.js`, and
`<body data-page="NN" data-root="..">`.

## 5. Components (copy these exactly)

```html
<aside class="callout note"><div class="callout-title">Note</div><p>…</p></aside>
<aside class="callout key"><div class="callout-title">Key idea</div><p>…</p></aside>
<aside class="callout warn"><div class="callout-title">Pitfall</div><p>…</p></aside>
<aside class="callout bridge"><div class="callout-title">From your jet tagger</div><p>…</p></aside>

<div class="decision">
  <div class="decision-title">Grouped-query attention</div>
  <dl>
    <dt>Problem</dt><dd>…</dd>
    <dt>Idea</dt><dd>…</dd>
    <dt>Why it works</dt><dd>…</dd>
    <dt>Evidence</dt><dd>… <a href="https://arxiv.org/abs/2305.13245">[Ainslie et al. 2023, Tab. 1]</a></dd>
    <dt>Cost</dt><dd>…</dd>
    <dt>Used by</dt><dd>…</dd>
  </dl>
</div>

<div class="table-wrap"><table class="data">…</table></div>   <!-- use td.num for numbers -->

<div class="eq-box">\[ … \]<div class="eq-label">Result</div></div>   <!-- for key results -->

<details class="qa" data-tier="core">
  <summary><span class="tier">Core</span> Question?</summary>
  <div class="answer"><p>…</p></div>
</details>

<span class="zh">多头注意力</span>                      <!-- Chinese term, see §8 -->
<span class="unverified" title="why unverified">claim</span>   <!-- see §7 -->
```

## 6. Math, notation, code

**Math** is rendered by KaTeX. Inline: `\( … \)`. Display: `\[ … \]`. **Never use single
`$`** (dollar amounts appear in the text). Inside HTML, write `<`, `>`, `&` as `&lt;`,
`&gt;`, `&amp;` even inside math, or use `\lt` and `\gt`. Available macros: `\R \E
\softmax \KL \dmodel \dff \concat \argmax \argmin \RMSNorm \LayerNorm \RoPE \clip \sg`.
Math inside `<pre>`/`<code>` is not rendered.

**Shared notation** (use it consistently across all pages):

| Symbol | Meaning |
|---|---|
| \(B\) | batch size (sequences) |
| \(T\) | sequence length in tokens (\(t\) indexes positions) |
| \(d\) | model width (\(\dmodel\)) |
| \(n_h\) | number of query heads; \(d_h\) head dimension (usually \(d/n_h\)) |
| \(n_{kv}\) | number of KV heads (GQA); \(g = n_h/n_{kv}\) group size |
| \(\dff\) | FFN hidden width |
| \(L\) | number of layers |
| \(V\) | vocabulary size |
| \(N\) | parameter count; \(N_{\text{act}}\) activated parameters (MoE) |
| \(D\) | training tokens |
| \(C\) | training compute in FLOPs |
| \(x_t \in \R^{d}\) | hidden state of token \(t\), a **row vector**; linear maps are \(x W\) |

Softmax is over the last axis. Log is natural log unless stated. FLOPs count a
multiply-add as 2 FLOPs.

**Code**: PyTorch 2.x, minimal and runnable. Put shapes in comments: `# (B, T, n_h, d_h)`.
Keep inline snippets under about 60 lines; full implementations go into the exercises.
Use `<pre><code class="language-python">`. Escape `<`, `>`, `&` inside `<pre>`.

## 7. Facts, citations, verification (most important section)

This is interview prep: a confidently wrong fact is worse than no fact.

- **Every quantitative claim taken from a paper** (parameter counts, token counts,
  hyperparameters, benchmark numbers, ablation deltas, dates) **must be checked against
  the primary source** before it goes on the page. Tools:
  `mcp__alphaXiv__get_paper_content` (full text), `mcp__alphaXiv__answer_pdf_queries`
  (ask questions about a PDF), `mcp__alphaXiv__discover_papers`,
  `WebFetch` on `arxiv.org/abs/…` or official model cards and blogs, `WebSearch`. If a tool
  isn't loaded, load it with `ToolSearch` (for example `select:mcp__alphaXiv__get_paper_content`).
- Cite inline with a link to the primary source and a location when possible:
  `<a href="https://arxiv.org/abs/2412.19437">[DeepSeek-V3, §2.1.1]</a>`.
- **Never invent an arXiv ID, a table number, or a number.** Check every arXiv ID you link.
- If you cannot verify a claim you believe is correct, keep it only if it matters, and
  wrap it in `<span class="unverified" title="reason">…</span>`.
- Derived numbers (for example a KV-cache size you compute from a config) must show the
  arithmetic so a reader can check it.
- Prefer primary sources: papers, official tech reports, official model cards or
  config.json, official blogs for models without papers. Secondary blogs are only for
  "further reading".
- Releases after many models' training data that the coordinator has confirmed exist (use
  them where relevant, and still verify any detail you cite): Gemma 4 technical report
  (arXiv 2607.02770, 2026), DiffusionGemma (2608.00146, Jul 2026; diffusion LM fine-tuned from
  Gemma 4 26B A4B), LLaDA 2.0 (2512.15745), FlashAttention-4 (Zadouri et al., MLSys 2026),
  EAGLE-3, Gemma 4 multi-token-prediction drafters (Google blog, 2026), **DeepSeek-V4**
  (arXiv 2606.19348, April 2026 preview: V4-Pro 1.6T/49B and V4-Flash 284B/13B, 1M context,
  CSA/HCA compressed attention replacing MLA, mHC residuals, Muon, FP4 QAT experts, on-policy
  distillation instead of mixed RL), mHC (Xie et al. 2026). Other models referenced in 2026
  reports: GPT-5.4, Gemini 3.1 Pro, Claude Opus 4.6, Kimi K2.6, GLM-5.1. Search for others.
- Today is late September 2026. For fast-moving topics (latest model releases, newest RL
  variants), search for developments after your training data and say "as of <month
  year>". Do not present a model as the latest without checking.

## 8. Chinese terms

The reader is a native Chinese speaker. Add the standard mainland Chinese technical term
**only for important concepts**, about 5 to 15 per page, on the concept's first
definition: `multi-head attention <span class="zh">多头注意力</span>`. Use the
established translation the Chinese ML community actually uses (for example
自回归, 困惑度, 分词器, 键值缓存, 混合专家, 旋转位置编码, 监督微调, 强化学习,
奖励模型, 直接偏好优化). Do not translate model names or sentences.

## 9. Exercises

Each page ships 1 to 3 drills in `exercises/`, one Python module per drill named after
the concept (`attention`, `bpe`, `rope`, …, never with a page-number prefix):

- `exercises/drills/<name>.py`: imports, function or class signatures with type hints,
  a precise docstring (shapes, dtypes, edge cases, what to return), and a body that is
  just `raise NotImplementedError`. No hints in the stub beyond the docstring.
- `exercises/solutions/<name>.py`: same signatures, clean reference implementation with
  short comments on the non-obvious lines.
- `exercises/tests/test_<name>.py`: starts with `MODULE = "<name>"`, and uses the `impl`
  fixture from `exercises/conftest.py` (`def test_x(impl): impl.my_fn(...)`). Tests must be
  **self-contained** (compare against PyTorch built-ins, a naive reference written inside
  the test, or mathematical properties), must never import from `solutions`, must run on
  CPU in under 30 seconds total, and must use small shapes.
- Verify: `python3 -m pytest exercises/tests/test_<name>.py --solutions -q` passes, and
  without `--solutions` it fails with `NotImplementedError`.
- Do **not** edit `exercises/README.md` (several writers run in parallel). Instead, list one
  row per drill in your final report as `| NN | Drill title | drills/<name>.py |`, and the
  coordinator adds them.
- In the page's `#exercises` section, describe each drill in 1 to 2 sentences, name its
  path, and say what an interviewer would expect you to finish in 20 to 40 minutes.

Interview-realistic drills beat toy ones: "implement causal multi-head attention with
a KV cache", not "compute a dot product".

## 10. Diagrams

Inline SVG inside `<figure class="fig">` with a `<figcaption>` that starts with
`<b>Figure N.</b>` and says what to notice. Rules:

- `viewBox` set, no fixed `width`/`height` attributes, roughly 720 units wide.
- Include `<title id="fig-NN-k-title">` and `aria-labelledby`. Ids must be unique on the
  page, and so must marker ids (`arrow-07-1`, …).
- **Colors only through the theme classes** so diagrams work in dark mode: `s-box`,
  `s-plain`, `s-a` … `s-e` (tinted fill plus stroke), `s-fill-a` … (solid fill),
  `s-line`, `s-line-a` … (strokes), `s-dash`, `s-arrowhead`; text classes `s-muted`,
  `s-small`, `s-big`, `s-bold`, `s-mono`, `s-ta` … `s-td`. Do not hard-code hex colors.
- Text at least 11px. Diagrams explain a **mechanism** (data flow, shapes, what gets
  cached, where the savings come from). No decorative art.
- Check them in the browser via the QA script and look at the numbers: a wrong shape
  label in a diagram is as bad as a wrong formula.

## 11. Links

- Link to other course pages by filename (`05-positional-encoding.html#rope`), including
  pages not written yet (the slug list is in `assets/manifest.js`). Link generously to
  prerequisites and to where a concept is treated in depth.
- External links: primary sources (arXiv abs pages, official reports). Open in the same
  tab (no `target`).

## 12. Reference models for worked numbers

Use these two configs for numeric examples so the numbers are comparable across pages.
They were checked against the official configs and reports. Other models may appear
where they are the better example.

**Llama-3-8B** (dense, GQA): \(L=32\), \(d=4096\), \(n_h=32\), \(n_{kv}=8\), \(d_h=128\),
\(\dff=14336\) (SwiGLU), \(V=128256\), RoPE \(\theta=500{,}000\), untied embeddings,
RMSNorm pre-norm, ~8.03B params, pretrained on ~15T tokens, 8K context
(128K in Llama 3.1 via RoPE scaling).

**DeepSeek-V3** (MoE, MLA): \(L=61\), \(d=7168\), \(n_h=128\);
MLA with KV compression dim \(d_c=512\), query compression dim \(d_c'=1536\),
per-head no-RoPE q/k dim 128, decoupled RoPE dim \(d_h^R=64\) (shared across heads for K),
v head dim 128; first 3 layers dense FFN (\(\dff=18432\)), other 58 layers MoE with
1 shared + 256 routed experts, expert hidden 2048, 8 routed experts active per token,
routing limited to at most 4 nodes (config: 8 expert groups, top-4 groups);
sigmoid gating with aux-loss-free bias balancing, normalized top-k weights, routed
scaling factor 2.5; \(V=129280\); untied embeddings; 671B total and 37B activated
params; 1 MTP module; pretrained on 14.8T tokens; RoPE \(\theta=10{,}000\), context
4K → 32K → 128K via YaRN (config: factor 40 over 4,096 original positions).

Sources: Llama 3 paper ([2407.21783](https://arxiv.org/abs/2407.21783), Table 3) and
the `config.json` of `meta-llama/Llama-3.1-8B`; DeepSeek-V3 report
([2412.19437](https://arxiv.org/abs/2412.19437), §2 and §4.2) and the `config.json` of
`deepseek-ai/DeepSeek-V3`.

Handy derived numbers (bf16, 2 bytes/element, per token):

| Model | Cached elements per layer | Per token, all layers |
|---|---|---|
| Llama-3-8B (GQA) | \(2 \cdot n_{kv} \cdot d_h = 2048\) | \(2048 \cdot 32 \cdot 2\) B = 131,072 B = 128 KiB |
| DeepSeek-V3 (MLA) | \(d_c + d_h^R = 576\) | \(576 \cdot 61 \cdot 2\) B = 70,272 B ≈ 68.6 KiB |

## 13. Workflow for an agent writing a page

1. Read this guide, `pages/_template.html`, the exemplar
   `pages/07-kv-efficient-attention.html`, your brief in `docs/briefs/NN.md`, and
   `assets/manifest.js` (for slugs of pages you link to).
2. Pull the primary sources listed in the brief and verify facts before writing.
3. Write the page and the drills.
4. Run `python3 tools/qa.py pages/NN-*.html` and fix every ERROR and every warning that
   is not about links to not-yet-written pages. Run the drill tests.
5. **Do not edit** shared files: `assets/*`, `index.html`, `docs/*`, `exercises/README.md`,
   `exercises/conftest.py`, or other pages. Put requests for shared changes in your final report.
   Do not commit; the coordinator commits.
6. Final report: files created, facts you could not verify, anything in the brief you
   chose to skip and why, requested shared-file changes.
