import fitz,re,sys,json
fitz.TOOLS.mupdf_display_errors(False)
def numbered(id, maxlevel=3):
    d=fitz.open(f'pdfs/{id}.pdf')
    out=[]
    for lvl,title,pg in d.get_toc():
        if lvl>maxlevel: continue
        lines=[l.strip() for l in d[pg-1].get_text().split('\n')]
        key=re.sub(r'\s+',' ',title).strip()[:30].lower()
        num=None
        for i,l in enumerate(lines):
            if key and (l.lower().startswith(key) or l.lower().endswith(key) or key in l.lower()) and len(l)<len(title)+12:
                m=re.match(r'^((?:\d+|[A-Z])(?:\.\d+)*)\.?\s+',l)
                if m: num=m.group(1); break
                if i>0 and re.match(r'^((?:\d+|[A-Z])(?:\.\d+)*)\.?$',lines[i-1]): num=lines[i-1].rstrip('.'); break
        out.append((lvl,num,title,pg))
    return d.page_count,out
if __name__=='__main__':
    n,o=numbered(sys.argv[1], int(sys.argv[2]) if len(sys.argv)>2 else 3)
    print(sys.argv[1],'pages',n)
    for lvl,num,t,pg in o:
        print('  '*(lvl-1)+f'[{num or "?"}] {t[:80]} (p{pg})')
