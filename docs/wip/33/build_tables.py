import sys, html, math, json
sys.path.insert(0, '/tmp/w33')
from atlas_data import *
from derive import derive
from rows import T1, T2, T3

HL = {'llama31_8b', 'dsv3'}          # the course's two reference models

def esc(s):
    return s  # cells already contain intended HTML entities/tags

def link(lbl, url):
    return f'<a href="{url}">{html.escape(lbl)}</a>'

def fmt_params(x):
    if x >= 1e12:
        return f'{x/1e12:.3f}T'
    return f'{x/1e9:.2f}B'

def comma(n):
    return f'{int(round(n)):,}'

DER = {m['key']: derive(m) for m in MODELS}

# ---- checkpoint comparison text for Table 4
CHECK = {
    'gpt3': 'hand count (no config file)¶',
    'llama2_70b': 'exact (file: +5,120 buffer elements)',
    'mistral7b': 'exact',
    'llama31_8b': 'exact', 'llama31_70b': 'exact', 'llama31_405b': 'exact',
    'mixtral8x22': 'exact',
    'gemma2_27b': 'exact',
    'qwen25_72b': 'exact', 'olmo2_32b': 'exact',
    'gemma3_27b': 'text only; report Tab. 1: 27.02B',
    'qwen3_32b': 'exact',
    'gemma4_31b': 'text only; report Tab. 1: 30.70B',
    'dsv2': 'exact',
    'dsv3': 'file 684.53B adds MTP module and FP8 scales',
    'maverick': 'text tensors exact (file adds vision)',
    'qwen3_235b': 'exact',
    'kimik2': 'file +62.5M FP8 scale factors',
    'glm45': '+ MTP layer 3.99B = 355.2B (report: 355B)',
    'gptoss120': 'file +39.8M: expert and router biases, sinks',
    'minimax_m2': 'file +15,872: routing biases',
    'gemma4_26b': 'text only; report Tab. 1: 25.24B',
}

def kv_cells(m, d):
    """(bytes per token, KiB string, GiB at 128K string)"""
    k = m['key']
    if k == 'v4pro':
        return '≈ 4,940†', '–'
    if k == 'v4flash':
        return '≈ 3,471†', '–'
    if k == 'v41flash':
        return '890†', '–'
    kv = d['kv_tok']
    ctx_ok = m['key'] not in ('gpt3', 'llama2_70b', 'mistral7b', 'mixtral8x22', 'gemma2_27b', 'olmo2_32b')
    g = f"{d['kv128k']/2**30:.2f}" if ctx_ok else '–'
    kvs = comma(kv) if kv else '0 (all windowed)'
    return kvs, g

def n_used(m, d):
    """(total N used for D/N, active N used) as reported, precise where the report is."""
    return m['rep_total'], m['rep_active']

rows4 = []
stats = {}
for m in MODELS:
    d = DER[m['key']]
    kvs, g = kv_cells(m, d)
    tot = act = None
    if m['key'] == 'gpt3':
        tot = 174_604_259_328
        rc = f'{tot/1e9:.2f}B'
        actr = '–'
    elif 'total' in d and d['how'] != 'hand':
        rc = fmt_params(d['total'])
        if m['kind'] == 'moe':
            actr = f"{d['act_none']/1e9:.2f}–{d['act_both']/1e9:.2f}B"
        else:
            actr = '–'
    else:
        rc = 'n/a‡'
        actr = 'n/a‡'
    if m['key'] == 'gpt3':
        # hand: 12 L d^2 + head (tied) matmul params; attention from the drill
        matmul = 12 * 96 * 12288**2 + 50257 * 12288
        fl = (2 * matmul + d['attn8k']) / 1e9
        fls = f'{fl:.1f}'
    elif 'flops8k' in d and d['how'] != 'hand':
        fls = f"{d['flops8k']/1e9:.1f}"
    else:
        fls = 'n/a‡'
    N_act = m['rep_active']
    D = m['tokens']
    if D:
        tpp = D / N_act
        tpp_s = f'{tpp:,.1f}' if tpp < 100 else f'{tpp:,.0f}'
        c6 = 6 * N_act * D
        c6_s = f'{c6:.1e}'.replace('e+', '×10<sup>') + '</sup>'
    else:
        tpp_s = 'n/d'; c6_s = 'n/d'
    stats[m['key']] = dict(tpp=(D / N_act if D else None), kvtok=d.get('kv_tok'), d=d)
    cls = ' class="hl"' if m['key'] in HL else ''
    rows4.append(f'<tr{cls}><td><strong>{m["name"]}</strong></td><td class="num">{rc}</td><td class="num">{actr}</td>'
                 f'<td>{CHECK.get(m["key"], "n/a")}</td><td class="num">{kvs}</td><td class="num">{g}</td>'
                 f'<td class="num">{fls}</td><td class="num">{tpp_s}</td><td class="num">{c6_s}</td></tr>')

def tbl(head, rows, minw):
    return (f'<div class="table-wrap"><table class="data" style="min-width:{minw}px">\n<thead><tr>{head}</tr></thead>\n<tbody>\n'
            + '\n'.join(rows) + '\n</tbody></table></div>')

# ---- Table 1
rows1 = []
for m in MODELS:
    t = T1[m['key']]
    src = '; '.join(link(l, u) for l, u in m['sources'])
    cls = ' class="hl"' if m['key'] in HL else ''
    rows1.append(f'<tr{cls}><td><strong>{m["name"]}</strong></td><td>{m["date"]}</td><td class="num">{t["params"]}</td>'
                 f'<td class="num">{t["L"]}</td><td class="num">{t["d"]}</td><td class="num">{t["V"]}</td>'
                 f'<td>{m["ctx"]}</td><td class="num">{m["tokens_s"]}</td><td class="small">{src}</td></tr>')
t1 = tbl('<th>Model</th><th>Released</th><th class="num">Parameters (total / activated)</th><th class="num">L</th><th class="num">d</th>'
         '<th class="num">Vocab (rows)</th><th>Context</th><th class="num">Train tokens</th><th>Sources</th>', rows1, 1180)

rows2 = []
for m in MODELS:
    t = T2[m['key']]
    cls = ' class="hl"' if m['key'] in HL else ''
    rows2.append(f'<tr{cls}><td><strong>{m["name"]}</strong></td><td>{t["att"]}</td><td class="num">{t["h"]}</td>'
                 f'<td>{t["pos"]}</td><td>{t["norm"]}</td></tr>')
t2 = tbl('<th>Model</th><th>Attention</th><th class="num">Heads: query / KV × head dim</th><th>Positions</th><th>Norm and stability</th>', rows2, 1100)

rows3 = []
for m in MODELS:
    t = T3[m['key']]
    cls = ' class="hl"' if m['key'] in HL else ''
    rows3.append(f'<tr{cls}><td><strong>{m["name"]}</strong></td><td>{t["ffn"]}</td><td>{t["moe"]}</td>'
                 f'<td class="num">{t["sp"]}</td><td>{t["route"]}</td></tr>')
t3 = tbl('<th>Model</th><th>FFN (hidden width, ratio to d)</th><th>Experts: routed / active / shared × width</th>'
         '<th class="num">Sparsity</th><th>Routing and balancing</th>', rows3, 1060)

t4 = tbl('<th>Model</th><th class="num">Recount, total</th><th class="num">Recount, activated (non-embedding – both matrices)</th>'
         '<th>Recount versus released file</th><th class="num">KV bytes / token (bf16)</th><th class="num">KV GiB, 1 seq. at 128K</th>'
         '<th class="num">GFLOPs / token at 8K</th><th class="num">Tokens per activated param</th><th class="num">6·N<sub>act</sub>·D (FLOPs)</th>',
         rows4, 1320)

open('/tmp/w33/frag_t1.html', 'w').write(t1)
open('/tmp/w33/frag_t2.html', 'w').write(t2)
open('/tmp/w33/frag_t3.html', 'w').write(t3)
open('/tmp/w33/frag_t4.html', 'w').write(t4)
json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'd'} for k, v in stats.items()}, open('/tmp/w33/stats.json', 'w'), indent=1)
print('ok', len(rows1))
