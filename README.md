# LLM Interview Prep

A ground-up course on how frontier language models are built (architecture, pretraining,
finetuning, RL), written as static HTML pages for frontier-lab interview preparation.

- **Read online:** https://javierzhao.github.io/LLM-sttudy/ (finished pages only; updates automatically).
- **Read offline:** open `index.html` in a browser (works straight from disk).
- **Practice:** `exercises/` has drills with tests (see `exercises/README.md`).
- **How it is built:** `docs/PLAN.md` (curriculum and pipeline) and `docs/STYLE_GUIDE.md`.
- **QA:** `python3 tools/qa.py` checks HTML, links, math rendering, and required sections.

## Publishing (CI/CD)

`.github/workflows/site.yml` runs on every push:

1. **check**: `tools/qa.py --written-only` and the drill tests of finished pages.
2. **build**: `tools/build_site.py` assembles `_site/` with only the pages whose `status` is
   `"written"` in `assets/manifest.js`, so drafts committed mid-work are never published.
3. **deploy**: publishes `_site/` to GitHub Pages from the default branch. Requires
   *Settings → Pages → Build and deployment → Source: GitHub Actions*.

A page goes live when the coordinator flips its manifest status to `"written"` and pushes.
