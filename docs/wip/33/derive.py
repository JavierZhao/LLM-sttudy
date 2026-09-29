import json, sys
sys.path.insert(0,'/tmp/w33')
from atlas_data import *

def spec_for(m):
    k=m['key']
    if k=='gpt3':
        return z.Spec(n_layers=96,d=12288,vocab=50257,tie_embeddings=True,n_heads=96,n_kv_heads=96,head_dim=128,d_ff=49152), 'hand'
    if k=='minimax_m1':
        return z.Spec(n_layers=80,d=6144,vocab=200064,tie_embeddings=False,n_heads=64,n_kv_heads=8,head_dim=128,d_ff=0,n_global=10), 'hand'
    if k=='k3':
        return z.Spec(n_layers=93,d=7168,vocab=163840,tie_embeddings=False,n_heads=96,n_kv_heads=96,head_dim=128,q_lora_rank=1536,kv_lora_rank=512,qk_nope_dim=128,qk_rope_dim=64,v_dim=128,n_global=24), 'hand'
    if m['kind']=='special':
        return None, 'special'
    return z.load_config(cfg(m['cfgname'])), 'config'

def derive(m):
    s,how=spec_for(m)
    out=dict(key=m['key'],how=how)
    if s is None:
        return out
    try:
        tot,act=z.params(s)
        out['total']=tot; out['act_both']=act
        out['act_head']=z.params(s,'head')[1]; out['act_none']=z.params(s,'none')[1]
        out['flops8k']=z.flops_per_token(s,8192)
    except NotImplementedError:
        pass
    out['kv_tok']=z.kv_bytes_per_token(s)
    out['kv128k']=z.kv_cache_bytes(s,131072)
    out['attn8k']=z.attention_flops_per_token(s,8192)
    out['elems_global']=z._cached_elems(s,True)
    out['n_global']=z._global_layers(s); out['n_local']=s.n_local; out['window']=s.window
    return out

if __name__=='__main__':
    for m in MODELS:
        d=derive(m)
        print(m['key'], {k:(round(v,3) if isinstance(v,float) else v) for k,v in d.items() if k!='key'})
