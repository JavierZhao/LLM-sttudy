import fitz,re,sys,json
ids="1706.03762 2001.08361 2203.15556 2407.21783 2104.09864 2305.13245 2205.14135 2309.00071 1909.08053 1910.02054 2203.02155 2305.18290 1707.06347 2402.03300 2501.12948 2503.14476 2401.06066 2405.04434 2412.19437 2512.02556 2505.09388 2507.20534 2606.19348 2609.19969 2412.06464".split()
res={}
for id in ids:
    d=fitz.open(f'pdfs/{id}.pdf')
    n=d.page_count
    toc=d.get_toc()
    heads=[]
    refpage=None
    for pi in range(n):
        txt=d[pi].get_text()
        for line in txt.split('\n'):
            l=line.strip()
            if re.match(r'^(\d+(\.\d+){0,2}|[A-Z](\.\d+)*)\.?\s+[A-Z][^.]{2,80}$',l) and len(l)<90:
                heads.append((pi+1,l))
            if refpage is None and re.match(r'^(References|REFERENCES|Bibliography)\s*$',l):
                refpage=pi+1
    res[id]={'pages':n,'ref':refpage,'toc':[(t[0],t[1],t[2]) for t in toc][:80],'heads':heads}
    print(id,'pages',n,'refs@',refpage,'toc entries',len(toc))
json.dump(res,open('toc.json','w'))
