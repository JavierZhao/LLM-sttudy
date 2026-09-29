#!/usr/bin/env python3
"""Generate the extended reading list (every unique primary source in refs.json), grouped by course part.

Reads refs.json (from tools/extract_refs.py) and assets/manifest.js (via tools/manifest.py).
Writes extended.html (an HTML fragment to paste into pages/37-reading-list-glossary.html) and
extended_stats.json.  A source is placed in the part of the LOWEST page number that lists it.
"""
import html, json, os, re, sys

ROOT = "/home/user/LLM-sttudy"
sys.path.insert(0, os.path.join(ROOT, "tools"))
import manifest  # noqa: E402

MUST = {  # arXiv abs URL -> must-read rank (filled from must_read.py so the star matches the top list)
}
try:
    from must_read import MUST_URLS
    MUST = {u: i + 1 for i, u in enumerate(MUST_URLS)}
except Exception:
    pass

esc = lambda s: html.escape(s, quote=True)

def norm(u):
    return u.split("#")[0].rstrip("/")

# pages that link each URL anywhere (same rule as tools/extract_refs.py, but we keep the page list)
cite_pages = {}
for part in manifest.load()["parts"]:
    for p in part["pages"]:
        if p["status"] != "written" or p["n"] in ("36", "37"):
            continue
        src = open(os.path.join(ROOT, "pages", p["slug"] + ".html"), encoding="utf-8").read()
        for u in set(re.findall(r'href="(https?://[^"]+)"', src)):
            cite_pages.setdefault(norm(u), set()).add(p["n"])

data = json.load(open("/tmp/w37/refs.json", encoding="utf-8"))
m = manifest.load()
page_info = {}
part_of = {}
for part in m["parts"]:
    for p in part["pages"]:
        page_info[p["n"]] = p
        part_of[p["n"]] = part

# group
groups = {part["id"]: [] for part in m["parts"]}
for idx, paper in enumerate(data["papers"]):
    low = min(paper["pages"], key=int)
    groups[part_of[low]["id"]].append((int(low), idx, paper))

out = []
stats = {}
total = 0
for part in m["parts"]:
    items = sorted(groups[part["id"]], key=lambda t: (t[0], t[1]))
    if not items:
        continue
    total += len(items)
    first, last = part["pages"][0]["n"], part["pages"][-1]["n"]
    stats[part["id"]] = len(items)
    hid = "ext-part-" + part["id"].lower()
    out.append(f'<h3 id="{hid}">Part {part["id"]} · {esc(part["title"])} <span class="muted small">(pages {first} to {last}, {len(items)} sources)</span></h3>')
    out.append('<div class="table-wrap"><table class="data">')
    out.append('<thead><tr><th>Source</th><th>Pages</th></tr></thead><tbody>')
    for low, idx, paper in items:
        url = paper["url"]
        star = ""
        base = url.split("#")[0].rstrip("/")
        if base in MUST:
            star = f' <a href="#must-{MUST[base]:02d}" class="small" title="On the must-read 25 list (item {MUST[base]})">★ must-read</a>'
        pages = sorted(paper["pages"], key=int)
        links = " · ".join(
            f'<a href="{page_info[n]["slug"]}.html#papers" title="{esc(page_info[n]["title"])}">{n}</a>' for n in pages
        )
        also = sorted(cite_pages.get(base, set()) - set(pages), key=int)
        assert len(cite_pages.get(base, set())) == paper["cited_on_pages"], (url, cite_pages.get(base), paper["cited_on_pages"])
        if also:
            links += ' <span class="muted small">; also ' + ", ".join(
                f'<a href="{page_info[n]["slug"]}.html" title="{esc(page_info[n]["title"])}">{n}</a>' for n in also) + "</span>"
        out.append(
            f'<tr><td><a href="{esc(url)}">{esc(paper["citation"])}</a>{star}</td><td>{links}</td></tr>'
        )
    out.append("</tbody></table></div>")

open("/tmp/w37/extended.html", "w", encoding="utf-8").write("\n".join(out) + "\n")
json.dump({"total": total, "by_part": stats}, open("/tmp/w37/extended_stats.json", "w"))
print("sources:", total, stats)
