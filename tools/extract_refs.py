#!/usr/bin/env python3
"""Extract the reading list and glossary raw material from the finished pages (for page 37).

Usage:
    python3 tools/extract_refs.py OUT.json

Writes {"papers": [...], "terms": [...]}:
  papers: one entry per unique link in every page's ol.papers list (merged by URL), with the
          citation text, every page that lists it and each page's "why" note, plus how many pages
          cite the URL anywhere (a rough importance signal for curating the must-read list).
  terms:  one entry per span.zh term, with the English term in the <strong> or text just before it,
          and every page where it appears (first occurrence per page).
"""
from __future__ import annotations

import html
import json
import os
import re
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import manifest  # noqa: E402

ROOT = manifest.ROOT


def strip(s: str) -> str:
    return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s))).strip()


def english_before(src: str, start: int) -> str:
    before = src[max(0, start - 400):start]
    m = re.search(r"<strong>([^<]{1,80})</strong>\s*$", before)
    if m:
        return strip(m.group(1))
    words = strip(before).split()
    return " ".join(words[-6:])


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "refs.json"
    papers: "OrderedDict[str, dict]" = OrderedDict()
    terms: "OrderedDict[str, dict]" = OrderedDict()
    cite_count: dict[str, set] = {}
    for part in manifest.load()["parts"]:
        for p in part["pages"]:
            if p["status"] != "written" or p["n"] in ("36", "37"):
                continue
            src = open(os.path.join(ROOT, "pages", p["slug"] + ".html"), encoding="utf-8").read()
            for url in set(re.findall(r'href="(https?://[^"]+)"', src)):
                cite_count.setdefault(url.split("#")[0].rstrip("/"), set()).add(p["n"])
            sec = re.search(r'<ol class="papers">([\s\S]*?)</ol>', src)
            if sec:
                for li in re.findall(r"<li>([\s\S]*?)</li>", sec.group(1)):
                    a = re.search(r'<a href="([^"]+)">([\s\S]*?)</a>', li)
                    if not a:
                        continue
                    url = a.group(1).split("#")[0].rstrip("/")
                    why = re.search(r'<span class="why">([\s\S]*?)</span>', li)
                    e = papers.setdefault(url, {"url": a.group(1), "citation": strip(a.group(2)), "pages": [], "why": {}})
                    if p["n"] not in e["pages"]:
                        e["pages"].append(p["n"])
                    if why:
                        e["why"][p["n"]] = strip(why.group(1))
            seen_here = set()
            for m in re.finditer(r'<span class="zh">([^<]+)</span>', src):
                zh = m.group(1).strip()
                if zh in seen_here:
                    continue
                seen_here.add(zh)
                t = terms.setdefault(zh, {"zh": zh, "english": english_before(src, m.start()), "pages": []})
                t["pages"].append(p["n"])
    for e in papers.values():
        e["cited_on_pages"] = len(cite_count.get(e["url"].split("#")[0].rstrip("/"), set()))
    data = {"papers": list(papers.values()), "terms": list(terms.values())}
    json.dump(data, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(data['papers'])} unique primary sources, {len(data['terms'])} unique Chinese terms -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
