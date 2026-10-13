import fitz,re,sys
id=sys.argv[1]
d=fitz.open(f'pdfs/{id}.pdf')
pat=re.compile(r'^(\d+(\.\d+){0,3}|[A-Z](\.\d+){0,3})\.?\s+\S.{2,90}$')
for pi in range(d.page_count):
    for l in d[pi].get_text().split('\n'):
        l=l.strip()
        if pat.match(l) and not re.match(r'^\d+(\.\d+)?\s+(and|of|the|in|to)\b',l) and len(l)<95 and not re.search(r'[=+]|\d{4}',l) :
            print(pi+1,l)
