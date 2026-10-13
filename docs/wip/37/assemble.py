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

refs = json.load(open("/tmp/w37/refs.json", encoding="utf-8"))
cited = {q["url"]: q["cited_on_pages"] for q in refs["papers"]}
c_v4, c_v3, c_l3 = (cited["https://arxiv.org/abs/2606.19348"], cited["https://arxiv.org/abs/2412.19437"],
                    cited["https://arxiv.org/abs/2407.21783"])
top3 = sorted(cited.values(), reverse=True)[:3]
assert top3 == [c_v4, c_v3, c_l3], ("most-cited ranking changed", top3, c_v4, c_v3, c_l3)   # the template says V4, then V3, then Llama 3
t1 = must_stats["tier_min"]["1"]
tot = sum(must_stats["tier_min"].values())
rep = {
    "@@N_SOURCES@@": str(ext_stats["total"]),
    "@@N_TERMS@@": str(gl_stats["entries"]),
    "@@N_TEXT_ONLY@@": str(ext_stats["text_only"]),
    "@@CITE_V4@@": str(c_v4), "@@CITE_V3@@": str(c_v3), "@@CITE_L3@@": str(c_l3),
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
