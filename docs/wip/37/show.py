import sys,re
t=open('ctx.txt').read().split('\n\n')
a,b=int(sys.argv[1]),int(sys.argv[2])
for blk in t[a:b]:
    head,_,ctx=blk.partition('\n')
    i=ctx.find('⟦')
    j=ctx.find('⟧')
    pre=ctx[max(0,i-170):i].strip(); post=ctx[j+1:j+330]
    print(head); print('   ',pre,'⟦'+ctx[i+1:j]+'⟧',post); print()
