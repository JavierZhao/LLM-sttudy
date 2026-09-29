#!/usr/bin/env python3
"""Assemble pages/37-reading-list-glossary.html from page_template.html and the generated fragments."""
import json, re, sys, html
sys.path.insert(0, "/tmp/w37")
from cheat import CHEAT

T = open("/tmp/w37/page_template.html", encoding="utf-8").read()
rd = lambda f: open(f"/tmp/w37/{f}", encoding="utf-8").read().rstrip("\n")
must_stats = json.load(open("/tmp/w37/must_stats.json"))
ext_stats = json.load(open("/tmp/w37/extended_stats.json"))
gl_stats = json.load(open("/tmp/w37/glossary_stats.json"))
gl_html = rd("glossary.html")

cheat = []
ids = set(re.findall(r'<tr id="(gl-[^"]+)"', gl_html))
for label, zh, gid, text in CHEAT:
    assert gid in ids, gid
    cheat.append(f'      <li><strong><a href="#{gid}">{label}</a></strong> <span class="zh">{zh}</span>: {text}</li>')

t1 = must_stats["tier_min"]["1"]
tot = sum(must_stats["tier_min"].values())
rep = {
    "@@N_SOURCES@@": str(ext_stats["total"]),
    "@@N_TERMS@@": str(gl_stats["entries"]),
    "@@T1_H@@": f"{t1/60:.0f}",
    "@@ALL_H@@": f"{tot/60:.0f}",
    "@@PATH_MIN@@": str(must_stats["path_min"]),
    "@@BUDGETS@@": rd("budgets.html"),
    "@@MUST@@": rd("must.html"),
    "@@EXTENDED@@": rd("extended.html"),
    "@@GLOSSARY@@": gl_html,
    "@@CONVENTIONS@@": rd("conventions.html"),
    "@@CHEAT@@": "\n".join(cheat),
}
for k, v in rep.items():
    assert k in T, k
    T = T.replace(k, v)
assert "@@" not in T
open("/home/user/LLM-sttudy/pages/37-reading-list-glossary.html", "w", encoding="utf-8").write(T)
print("written", len(T), "bytes")
