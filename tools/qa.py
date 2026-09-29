#!/usr/bin/env python3
"""Static + rendered QA for the course pages.

Usage:
    python3 tools/qa.py                      # all pages/NN-*.html and index.html
    python3 tools/qa.py pages/07-*.html      # specific pages
    python3 tools/qa.py --no-render ...      # skip the headless-browser pass

Checks (ERROR fails the run, WARN is advisory):
  static   - HTML tag balance, required section ids, internal links resolve,
             data-page matches the filename, Q&A tiers present, arXiv links well-formed,
             em-dash parenthetical pairs (house style), stray single-$ math.
  rendered - page loads without JS errors, KaTeX produced no .katex-error,
             sidebar/toc were built, every <svg> has a <title>.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from html.parser import HTMLParser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUIRED_IDS = ["interview-lens", "objectives", "pitfalls", "questions", "exercises", "papers", "cheatsheet"]
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
# SVG elements that are commonly self-closed; the parser reports these via handle_startendtag.
OPTIONAL_CLOSE = {"p", "li", "dt", "dd", "tr", "td", "th", "thead", "tbody", "option"}


class Balance(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, int]] = []
        self.errors: list[str] = []
        self.ids: dict[str, int] = {}
        self.links: list[tuple[str, int]] = []
        self.in_svg = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        line = self.getpos()[0]
        if "id" in a and a["id"]:
            if a["id"] in self.ids:
                self.errors.append(f"line {line}: duplicate id '{a['id']}' (first at line {self.ids[a['id']]})")
            self.ids[a["id"]] = line
        if tag == "a" and a.get("href"):
            self.links.append((a["href"], line))
        if tag in VOID:
            return
        if tag == "svg":
            self.in_svg += 1
        self.stack.append((tag, line))

    def handle_startendtag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a and a["id"]:
            line = self.getpos()[0]
            if a["id"] in self.ids:
                self.errors.append(f"line {line}: duplicate id '{a['id']}'")
            self.ids[a["id"]] = line

    def handle_endtag(self, tag):
        line = self.getpos()[0]
        if tag in VOID:
            return
        if tag == "svg":
            self.in_svg = max(0, self.in_svg - 1)
        # pop implicitly-closed optional tags
        while self.stack and self.stack[-1][0] != tag and self.stack[-1][0] in OPTIONAL_CLOSE:
            self.stack.pop()
        if self.stack and self.stack[-1][0] == tag:
            self.stack.pop()
        else:
            opened = self.stack[-1] if self.stack else None
            self.errors.append(f"line {line}: </{tag}> does not match open <{opened[0]}> from line {opened[1]}" if opened
                               else f"line {line}: stray </{tag}>")

    def close(self):
        super().close()
        for tag, line in self.stack:
            if tag not in OPTIONAL_CLOSE:
                self.errors.append(f"line {line}: <{tag}> never closed")


def static_checks(path: str) -> tuple[list[str], list[str]]:
    errs, warns = [], []
    src = open(path, encoding="utf-8").read()
    name = os.path.basename(path)
    is_page = re.match(r"^\d\d-", name) is not None

    p = Balance()
    try:
        p.feed(src)
        p.close()
    except Exception as e:  # pragma: no cover
        errs.append(f"parser crashed: {e}")
    errs += p.errors[:15]
    if len(p.errors) > 15:
        errs.append(f"... {len(p.errors) - 15} more tag errors")

    if is_page:
        m = re.search(r'<body[^>]*data-page="(\d\d)"', src)
        if not m or m.group(1) != name[:2]:
            errs.append("body data-page does not match filename number")
        for rid in REQUIRED_IDS:
            if rid not in p.ids:
                errs.append(f"missing required section id '{rid}'")
        tiers = set(re.findall(r'<details class="qa" data-tier="(\w+)"', src))
        nq = len(re.findall(r'<details class="qa"', src))
        if nq < 8:
            warns.append(f"only {nq} interview questions (target 10-15)")
        for t in ("warmup", "core", "deep"):
            if t not in tiers:
                warns.append(f"no '{t}' tier questions")
        bad_tiers = tiers - {"warmup", "core", "deep"}
        if bad_tiers:
            errs.append(f"unknown Q&A tiers: {bad_tiers}")
        text = re.sub(r"<(script|style|pre|code)[\s\S]*?</\1>", " ", src)
        text = re.sub(r"<[^>]+>", " ", text)
        words = len(text.split())
        if words < 2500:
            warns.append(f"body is short: ~{words} words")
        if src.count("<figure") == 0:
            warns.append("no diagrams (<figure>)")
        if re.search(r"(?<![\\$])\$(?!\$)[^$\n]{1,80}\\[a-zA-Z]+[^$\n]{0,80}\$(?!\$)", text):
            warns.append("possible single-$ inline math (use \\( ... \\) instead)")
        # em-dash parentheticals: two em dashes in one sentence
        for sent in re.split(r"(?<=[.!?])\s+", text):
            if sent.count("—") >= 2:
                warns.append(f"em-dash pair (house style: use ':' or ','): '{sent.strip()[:90]}...'")
                break
        if re.search(r"TODO|TBD|lorem ipsum|XXXX\.XXXXX", src):
            errs.append("placeholder text left in page (TODO/TBD/XXXX.XXXXX)")

    base = os.path.dirname(path)
    for href, line in p.links:
        if href.startswith(("http://", "https://", "mailto:", "#", "javascript:")):
            if "arxiv.org" in href and not re.search(r"arxiv\.org/(abs|pdf|html)/\d{4}\.\d{4,5}(v\d+)?", href):
                warns.append(f"line {line}: suspicious arXiv link {href}")
            if href.startswith("#") and href[1:] and href[1:] not in p.ids:
                warns.append(f"line {line}: in-page anchor {href} not found (may be generated by site.js)")
            continue
        target = href.split("#")[0]
        if not target:
            continue
        tp = os.path.normpath(os.path.join(base, target))
        if not os.path.exists(tp):
            # links to planned pages are allowed but reported
            if re.search(r"pages/\d\d-[\w-]+\.html$", tp.replace(os.sep, "/")) or re.match(r"^\d\d-[\w-]+\.html$", target):
                warns.append(f"line {line}: link to not-yet-written page {target}")
            else:
                errs.append(f"line {line}: broken link {href}")
    return errs, warns


def find_chrome() -> str | None:
    cands = sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux*/chrome"))
    return cands[-1] if cands else None


def render_checks(paths: list[str]) -> dict[str, tuple[list[str], list[str]]]:
    out: dict[str, tuple[list[str], list[str]]] = {}
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed; skipping render checks (pip install playwright)")
        return out
    with sync_playwright() as pw:
        kw = {}
        exe = find_chrome()
        try:
            browser = pw.chromium.launch()
        except Exception:
            if not exe:
                print("no chromium found; skipping render checks")
                return out
            browser = pw.chromium.launch(executable_path=exe)
        for path in paths:
            errs, warns = [], []
            page = browser.new_page(viewport={"width": 1400, "height": 900})
            logs: list[str] = []
            page.on("pageerror", lambda e, logs=logs: logs.append(f"JS error: {e}"))
            page.on("console", lambda m, logs=logs: logs.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
            page.goto("file://" + os.path.abspath(path))
            page.wait_for_timeout(300)
            errs += logs
            kerr = page.eval_on_selector_all(".katex-error", "els => els.map(e => e.getAttribute('title') || e.textContent)")
            for k in kerr[:10]:
                errs.append(f"KaTeX: {k[:160]}")
            if len(kerr) > 10:
                errs.append(f"... {len(kerr) - 10} more KaTeX errors")
            nmath = page.eval_on_selector_all(".katex", "els => els.length")
            raw = page.evaluate(r"""() => {
                const c = document.getElementById('content'); if (!c) return [];
                const w = document.createTreeWalker(c, NodeFilter.SHOW_TEXT);
                const bad = []; let n;
                while ((n = w.nextNode())) {
                  if (n.parentElement.closest('pre, code, .katex, script, style')) continue;
                  if (/\\\(|\\\[|\\frac|\\sqrt|\\mathbf|\\text\{/.test(n.textContent)) bad.push(n.textContent.trim().slice(0, 100));
                }
                return bad;
            }""")
            for r in raw[:5]:
                errs.append(f"unrendered math: {r}")
            side = page.eval_on_selector_all("#sidebar a, #sidebar span.planned", "els => els.length")
            if side < 10:
                errs.append("sidebar did not build")
            no_title = page.eval_on_selector_all("figure svg", "els => els.filter(s => !s.querySelector('title')).length")
            if no_title:
                warns.append(f"{no_title} <svg> without <title>")
            overflow = page.evaluate("() => document.documentElement.scrollWidth > window.innerWidth + 2")
            if overflow:
                warns.append("horizontal page overflow at 1400px")
            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_timeout(100)
            if page.evaluate("() => document.documentElement.scrollWidth > window.innerWidth + 2"):
                warns.append("horizontal page overflow at phone width (390px)")
            warns.append(f"info: {nmath} math expressions rendered")
            out[path] = (errs, warns)
            page.close()
        browser.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("-q", "--quiet", action="store_true", help="hide warnings")
    args = ap.parse_args()
    paths = args.paths or sorted(glob.glob(os.path.join(ROOT, "pages", "[0-9][0-9]-*.html"))) + [os.path.join(ROOT, "index.html")]
    paths = [p for p in paths if os.path.basename(p) != "_template.html"]
    results = {p: static_checks(p) for p in paths}
    if not args.no_render:
        for p, (e, w) in render_checks(paths).items():
            results[p][0].extend(e)
            results[p][1].extend(w)
    n_err = 0
    for p in paths:
        errs, warns = results[p]
        n_err += len(errs)
        status = "FAIL" if errs else "ok"
        print(f"[{status}] {os.path.relpath(p, ROOT)}")
        for e in errs:
            print(f"   ERROR {e}")
        if not args.quiet:
            for w in warns:
                print(f"   warn  {w}")
    print(f"\n{len(paths)} file(s), {n_err} error(s)")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
