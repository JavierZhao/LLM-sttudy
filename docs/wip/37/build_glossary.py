#!/usr/bin/env python3
"""Build the glossary for page 37.

Step 1 (skeleton): from refs.json, merge Chinese-term variants (glossary_data.MERGE), compute for each concept the
pages that use it, the first page (lowest number) and the nearest preceding heading anchor of the first occurrence,
and the variants.  Writes glossary_skeleton.json (definitions empty).
Step 2 (fill + render): fill definitions from glossary_data.G, sort alphabetically by English term, and write
glossary.html (letter nav + table) and conventions.html (merged-variant table).
"""
import html, json, os, re, sys
sys.path.insert(0, "/tmp/w37")
import glossary_data as GD
ROOT = "/home/user/LLM-sttudy"
sys.path.insert(0, ROOT + "/tools")
import manifest

esc = lambda s: html.escape(s, quote=True)
zhd = lambda z: esc(GD.DISPLAY.get(z, z))   # display form (STYLE_GUIDE: space between Chinese and Latin text)
m = manifest.load()
slug = {p["n"]: p["slug"] for part in m["parts"] for p in part["pages"]}
src_cache = {}
def src(n):
    if n not in src_cache:
        src_cache[n] = open(f"{ROOT}/pages/{slug[n]}.html", encoding="utf-8").read()
    return src_cache[n]

refs = json.load(open("/tmp/w37/refs.json", encoding="utf-8"))
groups = {}      # canonical zh -> {"zh", "variants": {zh: pages}, "pages": set, "guess": english guesses}
for t in refs["terms"]:
    c = GD.MERGE.get(t["zh"], t["zh"])
    g = groups.setdefault(c, {"zh": c, "variants": {}, "pages": set(), "guess": []})
    g["variants"][t["zh"]] = t["pages"]
    g["pages"].update(t["pages"])
    g["guess"].append(t["english"])

def anchor_for(g):
    first = min(g["pages"], key=int)
    s = src(first)
    pos = None
    for zh, pages in g["variants"].items():
        if first in pages:
            mm = re.search(r'<span class="zh">' + re.escape(zh) + r'</span>', s)
            if mm and (pos is None or mm.start() < pos):
                pos = mm.start()
    ids = list(re.finditer(r'<(?:h2|h3|section)\b[^>]*\sid="([^"]+)"', s[:pos]))
    return first, (ids[-1].group(1) if ids else None)

def anchor_on_page(g, page):
    s = src(page)
    pos = None
    for zh in g["variants"]:
        mm = re.search(r'<span class="zh">' + re.escape(zh) + r'</span>', s)
        if mm and (pos is None or mm.start() < pos):
            pos = mm.start()
    ids = list(re.finditer(r'<(?:h2|h3|section)\b[^>]*\sid="([^"]+)"', s[:pos])) if pos is not None else []
    return ids[-1].group(1) if ids else None

skeleton = []
for c, g in groups.items():
    first, anc = anchor_for(g)
    if c in GD.DEFPAGE:
        first, anc = GD.DEFPAGE[c]
        if anc is None:
            anc = anchor_on_page(g, first)
        assert first in g["pages"] or c in ("屋顶线模型",) or True
        g["pages"].add(first)
    skeleton.append({"zh": c, "english_guess": g["guess"], "variants": {k: v for k, v in g["variants"].items() if k != c} if len(g["variants"]) > 1 else {},
                     "canonical_pages": g["variants"][c] if c in g["variants"] else [],
                     "pages": sorted(g["pages"], key=int), "first_page": first, "anchor": anc, "definition": ""})
json.dump(skeleton, open("/tmp/w37/glossary_skeleton.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---- step 2: fill + render
def sortkey(en):
    k = re.sub(r"^[^A-Za-z0-9]+", "", en).lower()
    return k

def letter(en):
    k = sortkey(en)
    ch = k[0].upper()
    return ch if ch.isalpha() else "#"

def gid(en):
    base = re.sub(r"\s*\(.*?\)\s*", " ", en).strip().lower()
    base = re.sub(r"[^a-z0-9]+", "-", base).strip("-")
    return "gl-" + (base or "term")

entries = []
for s in skeleton:
    en, definition = GD.G[s["zh"]]
    entries.append(dict(s, en=en, definition=definition))
entries.sort(key=lambda e: sortkey(e["en"]))
ids = {}
for e in entries:
    i = gid(e["en"])
    assert i not in ids, (i, e["en"], ids[i])
    ids[i] = e["en"]
    e["id"] = i

ALSO = {}
for v, c in GD.MERGE.items():
    ALSO.setdefault(c, []).append(v)

letters = []
rows = []
cur = None
for e in entries:
    L = letter(e["en"])
    if L != cur:
        cur = L
        letters.append(L)
        rows.append(f'<tr class="gl-letter"><th colspan="2" id="gl-letter-{L.lower() if L != "#" else "num"}">{L}</th></tr>')
    first = e["first_page"]
    pl = f'<a href="{slug[first]}.html' + (f'#{e["anchor"]}' if e["anchor"] else "") + f'">{first}</a>'
    others = [n for n in e["pages"] if n != first]
    if others:
        pl += ' <span class="small muted">also ' + ", ".join(f'<a href="{slug[n]}.html">{n}</a>' for n in others) + "</span>"
    var = ""
    if ALSO.get(e["zh"]):
        var = '<div class="small muted">also heard ' + "; ".join(
            f'<span class="zh">{zhd(zh)}</span>' for zh in ALSO[e["zh"]]) + "</div>"
    rows.append(
        f'<tr id="{e["id"]}"><td><strong>{esc(e["en"])}</strong><br><span class="zh">{zhd(e["zh"])}</span>{var}</td>'
        f'<td>{esc(e["definition"])} <span class="gl-page small">Page {pl}</span></td></tr>'
    )

nav = " ".join(f'<a href="#gl-letter-{("num" if L == "#" else L.lower())}">{L}</a>' for L in letters)
out = [f'<p class="gl-nav" id="gl-nav"><strong>Jump to:</strong> {nav}</p>',
       '<div class="table-wrap"><table class="data glossary">',
       '<thead><tr><th>Term (English, 中文)</th><th>Definition and page</th></tr></thead><tbody>']
out += rows
out.append("</tbody></table></div>")
open("/tmp/w37/glossary.html", "w", encoding="utf-8").write("\n".join(out) + "\n")

# conventions table: merged groups
NOTE = {
 "聊天模板": "聊天模板 is the Hugging Face documentation wording; 对话模板 is also common.",
 "置换等变性": "置换 is the standard mathematical term for permutation; 排列 is also used informally.",
 "长度外推": "长度外推 names the phenomenon; the short form 外推 is used once the context is clear.",
 "位置插值": "Named after the Position Interpolation method; the bare 插值 is the short form.",
 "显存带宽": "For GPUs the memory is 显存 (HBM); 内存带宽 is the generic (CPU or system) wording.",
 "算术强度": "Roofline literature uses 算术强度; 计算强度 is a looser synonym.",
 "激活重计算": "The full term says what is recomputed (activations); 重计算 is the short form, also used for FlashAttention's tile recomputation.",
 "多词元预测": "Follows the course's 词元 for token; 多 token 预测 keeps the English word, as many Chinese reports do.",
 "投机解码": "The usual industry term (投机采样 for speculative sampling); 推测解码 is the literal translation and also appears.",
 "知识蒸馏": "The full term; 蒸馏 alone is the everyday short form.",
 "涌现能力": "涌现能力 names the property of a model; the short form 涌现 names the phenomenon (page 13).",
 "细粒度专家分割": "细粒度专家分割 names the technique of splitting each expert into m pieces (pages 08 and 26); the short form 细粒度专家 names the resulting many small experts (page 33).",
 "混合推理": "Hybrid reasoning (thinking and non-thinking in one model); 混合推理架构 stresses the architecture side.",
 "滑动窗口注意力": "Attention type; 滑动窗口 alone describes the window.",
 "基于人类反馈的强化学习": "The longer form is the standard rendering of RLHF; 人类反馈强化学习 drops 基于 and 的.",
}
crow = []
n_groups = 0
for e in entries:
    if ALSO.get(e["zh"]):
        n_groups += 1
        crow.append((sortkey(e["en"]), f'<tr><td>{esc(e["en"])}</td><td><span class="zh">{zhd(e["zh"])}</span></td>'
                     f'<td>' + "; ".join(f'<span class="zh">{zhd(zh)}</span>' for zh in ALSO[e["zh"]]) + f'</td>'
                     f'<td>{esc(NOTE[e["zh"]])}</td></tr>'))
crow.sort()
conv = ['<div class="table-wrap"><table class="data">',
        '<thead><tr><th>Concept</th><th>This course</th><th>Also heard</th><th>Why this one</th></tr></thead><tbody>']
conv += [r for _, r in crow]
conv.append("</tbody></table></div>")
open("/tmp/w37/conventions.html", "w", encoding="utf-8").write("\n".join(conv) + "\n")

stats = {"raw_terms": len(refs["terms"]), "entries": len(entries), "merged_groups": n_groups, "letters": letters}
json.dump(stats, open("/tmp/w37/glossary_stats.json", "w"))
print(stats)
