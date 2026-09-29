import json,sys
keys=None
for f in sys.argv[1:]:
    c=json.load(open(f))
    tc=c.get('text_config',c)
    print('=====',f, c.get('architectures'), c.get('model_type'))
    for k,v in tc.items():
        if k in ('architectures','auto_map','torch_dtype','transformers_version','use_cache','initializer_range','bos_token_id','eos_token_id','pad_token_id','attention_dropout','_name_or_path','quantization_config','vision_config','audio_config'): continue
        s=json.dumps(v)
        if len(s)>300: s=s[:300]+'...'
        print(' ',k,'=',s)
