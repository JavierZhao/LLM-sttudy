import json,re,sys,os
ROOT='/home/user/LLM-sttudy'
sys.path.insert(0,ROOT+'/tools')
import manifest
slug={p['n']:p['slug'] for part in manifest.load()['parts'] for p in part['pages'] if p['status']=='written'}
cites={}
for n,s in slug.items():
    if n in ('36','37'): continue
    src=open(f"{ROOT}/pages/{s}.html",encoding='utf-8').read()
    for url in set(re.findall(r'href="(https?://[^"]+)"',src)):
        cites.setdefault(url.split('#')[0].rstrip('/'),set()).add(n)
json.dump({k:sorted(v) for k,v in cites.items()},open('cites.json','w'))
