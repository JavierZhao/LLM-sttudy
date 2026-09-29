#!/usr/bin/env python3
"""Render the Must-read 25 as grouped ol.papers lists (fragment must.html) from must_read.py."""
import html, json, os, sys
sys.path.insert(0, "/tmp/w37")
from must_read import GROUPS, P, minutes
sys.path.insert(0, "/home/user/LLM-sttudy/tools")
import manifest

esc = lambda s: html.escape(s, quote=False)
m = manifest.load()
slug = {p["n"]: p["slug"] for part in m["parts"] for p in part["pages"]}
SHORT = {"01": "Big picture", "02": "Tokenization", "03": "Attention", "04": "Transformer block", "05": "RoPE",
         "06": "Decoding & KV cache", "07": "KV-efficient attention", "08": "MoE", "09": "Beyond full attention",
         "10": "Long context", "11": "Stability", "12": "Pretraining data", "13": "Scaling laws", "14": "Optimizers",
         "15": "Parallelism", "16": "GPUs & kernels", "17": "Inference", "18": "Evaluation", "19": "SFT & distillation",
         "20": "LoRA", "21": "Preference optimization", "22": "RL foundations", "23": "RLHF", "24": "RLVR",
         "25": "RL systems", "26": "DeepSeek I", "27": "DeepSeek II", "28": "DeepSeek III", "29": "Llama",
         "30": "Qwen", "31": "Mistral, Gemma, gpt-oss", "32": "Frontier open models", "33": "Architecture atlas", "34": "Back-of-envelope"}
d = json.load(open("/tmp/w37/refs.json", encoding="utf-8"))
cites = {p["url"]: p["cited_on_pages"] for p in d["papers"]}

def fmt_pages(x):
    return f"{x:g}"

def lc(s):
    w = s.split()[0]
    if any(c.isupper() for c in w[1:]) or any(c.isdigit() for c in w):
        return s
    return s[0].lower() + s[1:]

out = []
n = 0
tier_min = {1: 0, 2: 0, 3: 0}
for gid, gtitle, gintro in GROUPS:
    items = [p for p in P if p[0] == gid]
    out.append(f'<h3 id="must-group-{gid.lower()}">{gid} · {esc(gtitle)}</h3>')
    out.append(f'<p>{esc(gintro)}</p>')
    out.append(f'<ol class="papers" start="{n + 1}">')
    for (g, url, cit, short, why, supports, rd, rp, sk, sp, tier) in items:
        n += 1
        mins = minutes(rp, sp)
        tier_min[tier] += mins
        sup = ", ".join(f'<a href="{slug[s]}.html">{s} {esc(SHORT[s])}</a>' for s in supports)
        cnt = cites[url]
        out.append(
            f'<li id="must-{n:02d}"><a href="{url}">{esc(cit)}</a> <span class="why">Why read it: {esc(lc(why))}</span><br>'
            f'<span class="small muted"><b>Supports:</b> {sup}. <b>Read:</b> {esc(rd)}. <b>Skim:</b> {esc(sk)}. '
            f'<b>Time:</b> about {mins} min ({fmt_pages(rp)} pages read, {fmt_pages(sp)} skimmed) · Tier {tier} · linked on {cnt} course page{"s" if cnt != 1 else ""}.</span></li>'
        )
    out.append("</ol>")
open("/tmp/w37/must.html", "w", encoding="utf-8").write("\n".join(out) + "\n")

# reading budgets table
TIERDESC = {1: ("Tier 1", "Read before any interview loop", "The 12 papers whose content is asked about most directly: attention, kernels, scaling, the reference recipe, memory accounting, the RLHF and RL objectives, MLA and the DeepSeek-V3 design, and DeepSeek-V4 for long-context efficiency."),
            2: ("Tier 2", "Read for depth", "Papers that sharpen an answer you already have: position encoding, GQA, tensor parallelism, MoE design, RL at scale, sparse attention and context extension."),
            3: ("Tier 3", "Read when it is your target or your research", "The newest KV-compression design, the linear-attention layer used in hybrids, and two open model families that made different choices from DeepSeek.")}
rows = []
tot = 0
for t in (1, 2, 3):
    names = [p[3] for p in P if p[10] == t]
    mins = sum(minutes(p[7], p[9]) for p in P if p[10] == t)
    tot += mins
    rows.append(f'<tr><td><b>{TIERDESC[t][0]}</b>: {esc(TIERDESC[t][1])}<br><span class="small muted">{len(names)} papers, {mins} min (about {mins / 60:.1f} h)</span></td><td>{esc("; ".join(names))}. <span class="small muted">{esc(TIERDESC[t][2])}</span></td></tr>')
rows.append(f'<tr class="hl"><td><b>All 25</b><br><span class="small muted">{tot} min (about {tot / 60:.0f} h)</span></td><td>One working weekend of reading.</td></tr>')
budgets = ('<div class="table-wrap"><table class="data"><thead><tr><th>Tier</th><th>Which papers, and why</th></tr></thead><tbody>'
           + "".join(rows) + '</tbody></table></div>')
open("/tmp/w37/budgets.html", "w", encoding="utf-8").write(budgets + "\n")
path = ["Attention Is All You Need", "Chinchilla", "PPO", "DPO", "DeepSeek-V2"]
path_min = sum(minutes(p[7], p[9]) for p in P if p[3] in path)
assert len([p for p in P if p[3] in path]) == 5
json.dump({"tier_min": tier_min, "n": n, "path_min": path_min}, open("/tmp/w37/must_stats.json", "w"))
print(n, tier_min, sum(tier_min.values()))
