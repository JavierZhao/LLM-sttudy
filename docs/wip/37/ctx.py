import json,re,html,sys,os
ROOT='/home/user/LLM-sttudy'
sys.path.insert(0,ROOT+'/tools')
import manifest
slug={p['n']:p['slug'] for part in manifest.load()['parts'] for p in part['pages']}
d=json.load(open('refs.json'))
def strip(s): return html.unescape(re.sub(r'\s+',' ',re.sub(r'<[^>]+>',' ',s))).strip()
out=[]
for i,t in enumerate(d['terms']):
    n=t['pages'][0]
    src=open(f"{ROOT}/pages/{slug[n]}.html",encoding='utf-8').read()
    m=re.search(r'<span class="zh">'+re.escape(t['zh'])+'</span>',src)
    # find enclosing paragraph/li/td
    s=m.start()
    a=max(src.rfind('<p',0,s),src.rfind('<li',0,s),src.rfind('<td',0,s),src.rfind('<dd',0,s),src.rfind('<summary',0,s))
    e=len(src)
    for tag in ('</p>','</li>','</td>','</dd>','</summary>'):
        k=src.find(tag,s)
        if k!=-1: e=min(e,k)
    seg=src[a:e]
    zpos=seg.find('<span class="zh">'+t['zh'])
    txt_before=strip(seg[:zpos]); txt_after=strip(seg[zpos:])
    ctx=(txt_before[-260:]+' ⟦'+t['zh']+'⟧ '+txt_after[len(t['zh']):][:380])
    out.append(f"#{i} {t['zh']} | {t['english']} | p{','.join(t['pages'])}\n   {ctx}\n")
open('ctx.txt','w').write('\n'.join(out))
print(len(out))
