#!/usr/bin/env python3
"""Assemble the publishable site: only pages marked "written" in assets/manifest.js.

Usage:
    python3 tools/build_site.py _site          # build into _site/
    python3 tools/build_site.py --list-tests   # print drill test files of written pages

Drafts that are committed while a page is still being written or fact-checked are never
copied, so the deployed site only ever shows finished pages. Links from finished pages to
planned ones land on 404.html, which explains that the page is not published yet.
"""
from __future__ import annotations

import datetime as dt
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import manifest  # noqa: E402

ROOT = manifest.ROOT


def commit_info() -> str:
    sha = os.environ.get("GITHUB_SHA")
    if not sha:
        try:
            sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            sha = "unknown"
    return sha[:7]


def build(out: str) -> None:
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "pages"))
    shutil.copytree(os.path.join(ROOT, "assets"), os.path.join(out, "assets"))
    # regenerate the question bank from the finished pages so page 36 is never stale
    import build_bank  # noqa: E402
    open(os.path.join(out, "assets", "question_bank.js"), "w", encoding="utf-8").write(build_bank.render(build_bank.collect()))
    written = manifest.pages("written")
    total = len(manifest.pages())
    for p in written:
        src = os.path.join(ROOT, "pages", p["slug"] + ".html")
        if not os.path.exists(src):
            raise SystemExit(f"manifest marks {p['n']} as written but {src} is missing")
        shutil.copy2(src, os.path.join(out, "pages"))
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    info = f"Site built {stamp} from commit {commit_info()} · {len(written)} of {total} pages published."
    index = open(os.path.join(ROOT, "index.html"), encoding="utf-8").read().replace("<!--BUILD_INFO-->", info)
    open(os.path.join(out, "index.html"), "w", encoding="utf-8").write(index)
    shutil.copy2(os.path.join(ROOT, "404.html"), out)
    open(os.path.join(out, ".nojekyll"), "w").close()
    print(f"built {out}: {len(written)} pages ({', '.join(p['n'] for p in written)})")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--list-tests":
        print("\n".join(os.path.relpath(t, ROOT) for t in manifest.written_test_paths()))
        return
    build(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "_site"))


if __name__ == "__main__":
    main()
