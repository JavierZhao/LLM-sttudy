"""Atlas data: one record per model. Numbers come from released config.json files (/tmp/w33/cfg), papers, or finished pages.
Every row lists its sources. 'rep_total'/'rep_active' are the numbers as reported by the source (floats, parameters)."""
import json, sys
sys.path.insert(0, '/home/user/LLM-sttudy/exercises')
from solutions import config_zoo as z

CFG = '/tmp/w33/cfg/'
HF = 'https://huggingface.co/'


def cfg(name):
    return json.load(open(CFG + name + '.json'))


def hf(repo, label='config'):
    return (label, HF + repo + '/blob/main/config.json')


def arx(aid, label):
    return (label, 'https://arxiv.org/abs/' + aid)


B = 1e9
T = 1e12

# order = table order. cfgname None means a hand-built Spec or reported-only row.
MODELS = [
    dict(key='gpt3', name='GPT-3 175B', date='May 2020', cfgname=None, kind='dense',
         rep_total=175.0 * B, rep_active=175.0 * B, tokens=300 * B, tokens_s='300B',
         L=96, d=12288, vocab=50257, ctx='2,048',
         sources=[arx('2005.14165', 'GPT-3 §2.1, Tab. 2.1'), hf('openai-community/gpt2', 'GPT-2 config (vocab)')]),
    dict(key='llama2_70b', name='Llama 2 70B', date='Jul 2023', cfgname='llama2_70b', kind='dense',
         rep_total=70e9, rep_active=70e9, tokens=2.0 * T, tokens_s='2.0T', ctx='4,096',
         sources=[arx('2307.09288', 'Llama 2, Tab. 1'), hf('NousResearch/Llama-2-70b-hf', 'config (mirror)')]),
    dict(key='mistral7b', name='Mistral 7B', date='Sep 2023', cfgname='mistral7b', kind='dense',
         rep_total=7.3e9, rep_active=7.3e9, tokens=None, tokens_s='n/d', ctx='8,192 (paper); 32,768 (config)',
         sources=[arx('2310.06825', 'Mistral 7B, Tab. 1'), hf('mistralai/Mistral-7B-v0.1')]),
    dict(key='llama31_8b', name='Llama 3.1 8B', date='Jul 2024', cfgname='llama31_8b', kind='dense',
         rep_total=8.03e9, rep_active=8.03e9, tokens=15 * T, tokens_s='~15T', ctx='131,072',
         sources=[arx('2407.21783', 'Llama 3, Tab. 3'), hf('unsloth/Llama-3.1-8B', 'config (mirror)')]),
    dict(key='llama31_70b', name='Llama 3.1 70B', date='Jul 2024', cfgname='llama31_70b', kind='dense',
         rep_total=70.55e9, rep_active=70.55e9, tokens=15 * T, tokens_s='~15T', ctx='131,072',
         sources=[arx('2407.21783', 'Llama 3, Tab. 3'), hf('unsloth/Llama-3.1-70B', 'config (mirror)')]),
    dict(key='llama31_405b', name='Llama 3.1 405B', date='Jul 2024', cfgname='llama31_405b_fp8', kind='dense',
         rep_total=405.85e9, rep_active=405.85e9, tokens=15.6 * T, tokens_s='15.6T', ctx='131,072',
         sources=[arx('2407.21783', 'Llama 3, Tab. 3'), hf('nvidia/Llama-3.1-405B-Instruct-FP8', 'config (FP8 derivative of Meta\'s file)')]),
    dict(key='mixtral8x22', name='Mixtral 8x22B', date='Apr 2024', cfgname='mixtral8x22', kind='moe',
         rep_total=141e9, rep_active=39e9, tokens=None, tokens_s='n/d', ctx='65,536',
         sources=[('announcement', 'https://mistral.ai/news/mixtral-8x22b'), hf('mistralai/Mixtral-8x22B-v0.1')]),
    dict(key='gemma2_27b', name='Gemma 2 27B', date='Jun 2024', cfgname='gemma2_27b', kind='dense',
         rep_total=27.2e9, rep_active=27.2e9, tokens=13 * T, tokens_s='13T', ctx='8,192',
         sources=[arx('2408.00118', 'Gemma 2, Tab. 1-2'), hf('unsloth/gemma-2-27b', 'config (mirror)')]),
    dict(key='qwen25_72b', name='Qwen2.5-72B', date='Sep 2024', cfgname='qwen25_72b', kind='dense',
         rep_total=72.7e9, rep_active=72.7e9, tokens=18 * T, tokens_s='18T', ctx='131,072',
         sources=[arx('2412.15115', 'Qwen2.5, Tab. 1'), hf('Qwen/Qwen2.5-72B')]),
    dict(key='olmo2_32b', name='OLMo 2 32B', date='Mar 2025', cfgname='olmo2_32b', kind='dense',
         rep_total=32.2e9, rep_active=32.2e9, tokens=6.6 * T, tokens_s='6.6T', ctx='4,096',
         sources=[arx('2501.00656', 'OLMo 2, §2.1, Tab. 3'), hf('allenai/OLMo-2-0325-32B')]),
    dict(key='gemma3_27b', name='Gemma 3 27B', date='Mar 2025', cfgname='gemma3_27b', kind='dense',
         rep_total=27.0e9, rep_active=27.0e9, tokens=14 * T, tokens_s='14T', ctx='131,072',
         sources=[arx('2503.19786', 'Gemma 3, §2, Tab. 1'), hf('unsloth/gemma-3-27b-pt', 'config (mirror)')]),
    dict(key='qwen3_32b', name='Qwen3-32B', date='Apr 2025', cfgname='qwen3_32b', kind='dense',
         rep_total=32.8e9, rep_active=32.8e9, tokens=36 * T, tokens_s='36T', ctx='32,768 (128K with YaRN)',
         sources=[arx('2505.09388', 'Qwen3, §2, Tab. 1'), hf('Qwen/Qwen3-32B')]),
    dict(key='gemma4_31b', name='Gemma 4 31B', date='Apr 2026', cfgname='g4_gemma-4-31B', kind='dense',
         rep_total=30.7e9, rep_active=30.7e9, tokens=None, tokens_s='n/d', ctx='262,144',
         sources=[arx('2607.02770', 'Gemma 4, §2, Tab. 1'), hf('google/gemma-4-31B')]),
    # ---- MoE and hybrids
    dict(key='dsv2', name='DeepSeek-V2', date='May 2024', cfgname='dsv2', kind='moe',
         rep_total=236e9, rep_active=21e9, tokens=8.1 * T, tokens_s='8.1T', ctx='128K (config: 163,840)',
         sources=[arx('2405.04434', 'V2, §3.1.2'), hf('deepseek-ai/DeepSeek-V2')]),
    dict(key='dsv3', name='DeepSeek-V3 / V3.2', date='Dec 2024 / Dec 2025', cfgname='dsv3', kind='moe',
         rep_total=671e9, rep_active=37e9, tokens=14.8 * T, tokens_s='14.8T', ctx='128K (config: 163,840)',
         sources=[arx('2412.19437', 'V3, §4.2'), arx('2512.02556', 'V3.2'), hf('deepseek-ai/DeepSeek-V3'), hf('deepseek-ai/DeepSeek-V3.2', 'V3.2 config')]),
    dict(key='maverick', name='Llama 4 Maverick', date='Apr 2025', cfgname='llama4_mav', kind='moe',
         rep_total=400e9, rep_active=17e9, tokens=22 * T, tokens_s='~22T (card)', ctx='1,048,576',
         sources=[('Meta post', 'https://ai.meta.com/blog/llama-4-multimodal-intelligence/'), hf('unsloth/Llama-4-Maverick-17B-128E-Instruct', 'config (mirror)')]),
    dict(key='qwen3_235b', name='Qwen3-235B-A22B', date='Apr 2025', cfgname='qwen3_235b', kind='moe',
         rep_total=235e9, rep_active=22e9, tokens=36 * T, tokens_s='36T', ctx='32,768 (128K with YaRN)',
         sources=[arx('2505.09388', 'Qwen3, §2, Tab. 2'), hf('Qwen/Qwen3-235B-A22B')]),
    dict(key='kimik2', name='Kimi K2', date='Jul 2025', cfgname='kimik2', kind='moe',
         rep_total=1.04 * T, rep_active=32.6e9, tokens=15.5 * T, tokens_s='15.5T', ctx='131,072',
         sources=[arx('2507.20534', 'K2, §2.3, Tab. 2'), hf('moonshotai/Kimi-K2-Base')]),
    dict(key='glm45', name='GLM-4.5', date='Jul 2025', cfgname='glm45', kind='moe',
         rep_total=355e9, rep_active=32e9, tokens=23 * T, tokens_s='23T', ctx='131,072',
         sources=[arx('2508.06471', 'GLM-4.5, §2.1, Tab. 1'), hf('zai-org/GLM-4.5')]),
    dict(key='gptoss120', name='gpt-oss-120b', date='Aug 2025', cfgname='gptoss120', kind='moe',
         rep_total=116.83e9, rep_active=5.13e9, tokens=None, tokens_s='n/d', ctx='131,072',
         sources=[arx('2508.10925', 'gpt-oss model card, Tab. 1'), hf('openai/gpt-oss-120b')]),
    dict(key='minimax_m1', name='MiniMax-M1', date='Jun 2025', cfgname='minimax_m1', kind='hybrid',
         rep_total=456e9, rep_active=45.9e9, tokens=None, tokens_s='7.5T on top of Text-01', ctx='1M (config: 10.24M positions)',
         sources=[arx('2506.13585', 'M1, §1'), arx('2501.08313', 'MiniMax-01, §2'), hf('MiniMaxAI/MiniMax-M1-80k')]),
    dict(key='minimax_m2', name='MiniMax-M2', date='Oct 2025', cfgname='minimax_m2', kind='moe',
         rep_total=229.9e9, rep_active=9.8e9, tokens=29.2 * T, tokens_s='29.2T', ctx='196,608',
         sources=[arx('2605.26494', 'M2 report, §1-2'), hf('MiniMaxAI/MiniMax-M2')]),
    dict(key='qwen3_next', name='Qwen3-Next-80B-A3B', date='Sep 2025', cfgname='qwen3_next', kind='hybrid',
         rep_total=80e9, rep_active=3e9, tokens=15 * T, tokens_s='15T', ctx='262,144',
         sources=[('model card', HF + 'Qwen/Qwen3-Next-80B-A3B-Instruct'), hf('Qwen/Qwen3-Next-80B-A3B-Instruct')]),
    dict(key='qwen35', name='Qwen3.5-397B-A17B', date='Feb 2026', cfgname='qwen35', kind='hybrid',
         rep_total=397e9, rep_active=17e9, tokens=None, tokens_s='n/d', ctx='262,144',
         sources=[('model card', HF + 'Qwen/Qwen3.5-397B-A17B'), hf('Qwen/Qwen3.5-397B-A17B')]),
    dict(key='gemma4_26b', name='Gemma 4 26B-A4B', date='Apr 2026', cfgname='g4_gemma-4-26B-A4B', kind='moe',
         rep_total=26e9, rep_active=3.8e9, tokens=None, tokens_s='n/d', ctx='262,144',
         sources=[arx('2607.02770', 'Gemma 4, §2, Tab. 1'), hf('google/gemma-4-26B-A4B')]),
    dict(key='q38fn', name='Qwen3.8-Flash-Next', date='Aug 2026', cfgname='q38fn', kind='hybrid',
         rep_total=125e9, rep_active=6e9, tokens=None, tokens_s='n/d', ctx='262,144',
         sources=[('model card', HF + 'Qwen/Qwen3.8-Flash-Next'), arx('2608.30320', 'design paper'), hf('Qwen/Qwen3.8-Flash-Next')]),
    dict(key='v4flash', name='DeepSeek-V4-Flash', date='Apr 2026', cfgname='dsv4flash', kind='special',
         rep_total=284e9, rep_active=13e9, tokens=32 * T, tokens_s='32T', ctx='1,048,576',
         sources=[arx('2606.19348', 'V4, §4.2'), hf('deepseek-ai/DeepSeek-V4-Flash')]),
    dict(key='v4pro', name='DeepSeek-V4-Pro', date='Apr 2026', cfgname='dsv4pro', kind='special',
         rep_total=1.6 * T, rep_active=49e9, tokens=33 * T, tokens_s='33T', ctx='1,048,576',
         sources=[arx('2606.19348', 'V4, §4.2'), hf('deepseek-ai/DeepSeek-V4-Pro')]),
    dict(key='k3', name='Kimi K3', date='Jul 2026', cfgname='kimik3', kind='special',
         rep_total=2.8 * T, rep_active=104e9, tokens=None, tokens_s='n/d', ctx='1,048,576',
         sources=[('model card', HF + 'moonshotai/Kimi-K3'), hf('moonshotai/Kimi-K3')]),
    dict(key='v41flash', name='DeepSeek-V4.1-Flash', date='Sep 2026', cfgname='dsv41flash', kind='special',
         rep_total=552e9, rep_active=16e9, tokens=45 * T, tokens_s='45T', ctx='1,048,576',
         sources=[arx('2609.19969', 'V4.1-Flash, §1, §4.2'), hf('deepseek-ai/DeepSeek-V4.1-Flash')]),
]
