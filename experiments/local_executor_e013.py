"""Frozen small local executor diagnostic. No network/model code in prepare/analyze.

prepare/analyze use the existing pinned Pydantic runtime; freeze/run use the
isolated MLX runtime. Inputs/outputs are deliberately separate from prompts.
"""
import argparse
from collections import defaultdict, Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/local_executor_e013'
REPO = Path('/private/tmp/tau3-audit-20261027')
MODEL = Path('/Users/annagubareva/.cache/clarification-research/models/qwen3-4b-instruct-2507-4bit')
REVISION = '50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b'
MODEL_SHA = '2a73c6c248601ab904e035548abd8e6abb65ea27dcb5f342fb0a8910eb44173f'
Q = ['order_id', 'contents', 'total', 'count']
COST = .05
SEED = 20261027
SYSTEM = ('You help a user identify one pending order to cancel. Before the user\'s '
          'reply, each listed order is equally likely to be the target. Choose one '
          'plausible listed order using only the available information. If the '
          'information is insufficient, still choose a single plausible order. '
          'Do not ask another question or invent facts. Return only a JSON object '
          'with the single key "order" and one of the listed labels, for example '
          '{"order":"A"}.')
ASK = {'order_id': 'What is the order number of the order you want to cancel?',
       'contents': 'What items do you remember being in the order you want to cancel?',
       'total': 'What was the total price of the order you want to cancel?',
       'count': 'How many items were in the order you want to cancel?'}
IDK = {'order_id': "I don't know the order number.",
       'contents': "I don't remember any items in that order.",
       'total': "I don't know the total price.",
       'count': "I don't know how many items there were."}

def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()

def stable_best(options, value):
    best=options[0]
    for option in options[1:]:
        if value(option)>value(best)+1e-12: best=option
    return best

def reply(q, obs):
    if obs[0] == 'idk': return IDK[q]
    if q == 'order_id': return f'The order number is {obs[1]}.'
    if q == 'total': return f'The total was ${obs[1]//100}.{obs[1]%100:02d}.'
    if q == 'count': return f'There were {obs[1]} items.'
    return 'Items I remember: ' + '; '.join(obs[1:]) + '.'

def prepare():
    from retail_clarification_contract import feature, kernel
    assert not (OUT/'pack.json').exists(), 'Never silently replace a prepared pack'
    rawpath = REPO/'data/tau2/domains/retail/db.json'
    assert sha(rawpath) == '413a65160adbdb5fde0ffc0015c49b6d70250b10c18128de169b597af7766765'
    source = ROOT/'results/retail_clarification_contract/families.json'
    families = [f for f in json.loads(source.read_text()) if f['split_label']=='development'][:6]
    raw = json.loads(rawpath.read_text())
    byuser = defaultdict(list)
    for o in raw['orders'].values():
        if o['status']=='pending': byuser[digest(o['user_id'])].append(o)
    pack = []; family_data = []
    for family in families:
        fh = family['family_hash']
        fs = [feature(o) for o in sorted(byuser[fh], key=lambda o:o['order_id'])]
        n = len(fs); labels = [chr(65+i) for i in range(n)]
        assert n == family['n_orders']
        catalog = '\n'.join(f"{l}: order number {f['order_id']}; items: " +
            '; '.join(f['names']) + f"; item count: {f['count']}; " +
            f"total: ${f['total_cents']//100}.{f['total_cents']%100:02d}."
            for l,f in zip(labels,fs))
        base = [{'role':'system','content':SYSTEM},
                {'role':'user','content':'I need to cancel one of my pending orders.\n\nPending orders:\n'+catalog}]
        metrics = {}
        for q in ['no_ask'] + Q:
            joints = defaultdict(lambda:[0.]*n)
            if q == 'no_ask': joints[('no_ask',)] = [1/n]*n
            else:
                for i,f in enumerate(fs):
                    for y,p in kernel(f,q,'imperfect').items():
                        if q=='contents' and y[0]!='idk': y=('names',*y[1:])
                        joints[y][i] += p/n
            assert abs(sum(sum(j) for j in joints.values())-1)<1e-12
            entropy_after = 0.
            for y,joint in sorted(joints.items(), key=lambda x:repr(x[0])):
                mass = sum(joint)
                entropy_after += -sum(x*math.log2(x/mass) for x in joint if x>0)
                messages = base if q=='no_ask' else base + [
                    {'role':'assistant','content':ASK[q]}, {'role':'user','content':reply(q,y)}]
                pack.append({'id':digest(messages),'family_hash':fh,'question':q,
                             'observation':list(y),'labels':labels,'joint':joint,
                             'messages':messages})
            metrics[q] = {'accuracy':sum(max(j) for j in joints.values()),
                          'information_gain_bits':math.log2(n)-entropy_after}
        evsi = stable_best(['no_ask']+Q, lambda q:metrics[q]['accuracy']-COST*(q!='no_ask'))
        eq = stable_best(Q, lambda q:metrics[q]['information_gain_bits'])
        if metrics[eq]['information_gain_bits']<1e-12: eq='no_ask'
        family_data.append({'family_hash':fh,'features':fs,'labels':labels,
                            'symbolic_metrics':metrics,'symbolic_evsi':evsi,'entropy':eq})
    assert len(pack) == len({r['id'] for r in pack}) <= 400
    OUT.mkdir(parents=True,exist_ok=True)
    save(OUT/'pack.json',pack); save(OUT/'families.json',family_data)
    save(OUT/'preparation.json',{'protocol_sha256':sha(ROOT/'research/E013_PROTOCOL.md'),
         'source_families_sha256':sha(source),'database_sha256':sha(rawpath),
         'families':len(family_data),'unique_prompts':len(pack),
         'input_files':{str(p.relative_to(ROOT)):sha(p) for p in [
             ROOT/'experiments/retail_clarification_contract.py',
             ROOT/'experiments/tau3_retail_state_probe.py']}})
    print(json.dumps({'families':len(family_data),'unique_prompts':len(pack)}))

def freeze():
    from transformers import AutoTokenizer
    import importlib.metadata as md
    assert not (OUT/'manifest.json').exists()
    assert md.version('mlx-lm') == '0.32.2'
    assert sha(MODEL/'model.safetensors') == MODEL_SHA
    assert sha(MODEL/'tokenizer.json') == 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4'
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL),trust_remote_code=False,local_files_only=True)
    rendered=[]
    for r in json.loads((OUT/'pack.json').read_text()):
        prompt = tokenizer.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True)
        tokens = tokenizer.encode(prompt,add_special_tokens=False)
        assert len(tokens)<=4096
        rendered.append({'id':r['id'],'prompt':prompt,'input_tokens':tokens})
    save(OUT/'rendered.json',rendered)
    (OUT/'requirements.lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    paths=[ROOT/'research/E013_PROTOCOL.md', Path(__file__).resolve()] + [OUT/n for n in
        ['pack.json','families.json','preparation.json','rendered.json','requirements.lock.txt']]
    save(OUT/'manifest.json',{'files':{str(p.relative_to(ROOT)):sha(p) for p in paths},
         'model':'mlx-community/Qwen3-4B-Instruct-2507-4bit','revision':REVISION,
         'model_files':{p.name:sha(p) for p in sorted(MODEL.iterdir()) if p.is_file()},
         'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),
         'seed':SEED,'max_tokens':64,'temperature':0.,'budget_seconds':1800,
         'unique_prompts':len(rendered),'max_input_tokens':max(len(r['input_tokens']) for r in rendered)})
    print(json.dumps({'frozen':True,'unique_prompts':len(rendered),'max_input_tokens':max(len(r['input_tokens']) for r in rendered)}))

def verify_frozen():
    manifest=json.loads((OUT/'manifest.json').read_text())
    for name,h in manifest['files'].items(): assert sha(ROOT/name)==h, name
    return manifest

def run():
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    verify_frozen()
    assert not (OUT/'generations.jsonl').exists(), 'No implicit retries or replacement'
    assert sha(MODEL/'model.safetensors') == MODEL_SHA
    start=time.monotonic(); mx.random.seed(SEED)
    model,tokenizer=load(str(MODEL),trust_remote_code=False)
    rendered=json.loads((OUT/'rendered.json').read_text()); completed=0
    with (OUT/'generations.jsonl').open('x') as output, (OUT/'started.jsonl').open('x') as started:
        for i,r in enumerate(rendered):
            if time.monotonic()-start>=1800: break
            assert tokenizer.encode(r['prompt'],add_special_tokens=False)==r['input_tokens']
            started.write(json.dumps({'id':r['id'],'index':i,'elapsed':time.monotonic()-start})+'\n'); started.flush()
            before=time.monotonic(); chunks=[]; last=None; error=None
            try:
                for part in stream_generate(model,tokenizer,prompt=r['input_tokens'],max_tokens=64,sampler=make_sampler(temp=0.)):
                    chunks.append(part.text); last=part
                    if time.monotonic()-start>=1800 and part.finish_reason is None:
                        raise TimeoutError('Overall generation budget exhausted')
                if last is None or last.finish_reason not in ['stop','length']:
                    raise RuntimeError('No final generation status')
            except Exception as exc:
                error={'type':type(exc).__name__,'message':str(exc)}
            row={'id':r['id'],'raw_output':''.join(chunks),'error':error,
                 'elapsed_seconds':time.monotonic()-before,
                 'finish_reason':last.finish_reason if last else None,
                 'generated_tokens':last.generation_tokens if last else 0,
                 'prompt_tokens':last.prompt_tokens if last else len(r['input_tokens']),
                 'peak_memory_gb':last.peak_memory if last else None}
            output.write(json.dumps(row,ensure_ascii=False)+'\n'); output.flush(); completed+=1
            if (i+1)%10==0 or error: print(json.dumps({'completed':i+1,'total':len(rendered),'elapsed':round(time.monotonic()-start,1),'error':error}),flush=True)
            if isinstance(error,dict) and error['type']=='TimeoutError': break
            mx.clear_cache()
    save(OUT/'run_status.json',{'completed':completed,'total':len(rendered),
                              'elapsed_seconds':time.monotonic()-start,'complete':completed==len(rendered)})

def parse(raw,labels):
    text=raw.strip()
    fenced=re.fullmatch(r'```(?:json)?\s*\n([\s\S]*?)\n```',text)
    if fenced: text=fenced.group(1).strip()
    def unique_pairs(pairs):
        if len(pairs)!=len(dict(pairs)): raise ValueError('Duplicate JSON keys')
        return dict(pairs)
    try: obj=json.loads(text,object_pairs_hook=unique_pairs)
    except (ValueError,TypeError): return None
    if type(obj) is not dict or set(obj)!= {'order'} or obj['order'] not in labels: return None
    return obj['order']

def analyze():
    verify_frozen()
    pack=json.loads((OUT/'pack.json').read_text()); families=json.loads((OUT/'families.json').read_text())
    gens=[json.loads(l) for l in (OUT/'generations.jsonl').read_text().splitlines()]
    assert len(gens)==len({r['id'] for r in gens})
    byid={r['id']:r for r in gens}; details=[]; question=defaultdict(lambda:{'accuracy_lower':0.,'accuracy_upper':0.,'missing_mass':0.,'invalid_mass':0.,'length_mass':0.})
    chosen=set(); completed_all=True
    for row in pack:
        gen=byid.get(row['id']); joint=row['joint']; mass=sum(joint)
        f=next(f for f in families if f['family_hash']==row['family_hash'])
        label=None; missing=gen is None or gen['error'] is not None
        if not missing: label=parse(gen['raw_output'],row['labels'])
        lo=0. if label is None else joint[row['labels'].index(label)]
        upper=mass if missing else lo
        if missing: completed_all=False
        if label is not None: chosen.add(f['features'][row['labels'].index(label)]['order_id'])
        metrics=question[row['family_hash'],row['question']]
        for k,v in [('accuracy_lower',lo),('accuracy_upper',upper),('missing_mass',mass*missing),
                    ('invalid_mass',mass*(not missing and label is None)),
                    ('length_mass',mass*(gen is not None and gen['finish_reason']=='length'))]: metrics[k]+=v
        details.append({'id':row['id'],'family_hash':row['family_hash'],'question':row['question'],
            'observation':row['observation'],'selected_label':label,'missing':missing,
            'invalid':not missing and label is None,'mass':mass,'accuracy_contribution':lo,
            'bayes_contribution':max(joint),'regret_contribution':None if missing else max(joint)-lo})
    family_rows=[]; agg=defaultdict(lambda:defaultdict(float))
    for f in families:
        fh=f['family_hash']; qs={q:question[fh,q] for q in ['no_ask']+Q}
        for q,m in qs.items():
            m['utility_lower']=m['accuracy_lower']-COST*(q!='no_ask')
            m['utility_upper']=m['accuracy_upper']-COST*(q!='no_ask')
        selections={'no_ask':'no_ask','always_contents':'contents','entropy':f['entropy'],'symbolic_evsi':f['symbolic_evsi']}
        if completed_all: selections['executor_oracle']=stable_best(['no_ask']+Q,lambda q:qs[q]['utility_lower'])
        for method,q in selections.items():
            for k,v in qs[q].items(): agg[method][k]+=v/len(families)
            agg[method]['ask_rate']+=float(q!='no_ask')/len(families)
        family_rows.append({'family_hash':fh,'questions':qs,'selections':selections})
    from retail_clarification_contract import verify_execution
    from tau3_retail_state_probe import load_domain
    db_class,tools_class,_=load_domain(REPO)
    raw=json.loads((REPO/'data/tau2/domains/retail/db.json').read_text())
    old={e['order_hash']:e for e in json.loads((ROOT/'results/retail_clarification_contract/execution.json').read_text())}
    executions=[]
    for oid in sorted(chosen):
        e=verify_execution(db_class,tools_class,raw,oid)
        e['e011_state_matches']=e['after_state_hash']==old[e['order_hash']]['after_state_hash']
        assert e['success'] and e['e011_state_matches']
        executions.append(e)
    save(OUT/'details.json',details);save(OUT/'execution.json',executions)
    result={'complete':completed_all,'families':len(families),'unique_prompts':len(pack),
            'generated':len(gens),'invalid_outputs':sum(r['invalid'] for r in details),
            'length_outputs':sum(g['finish_reason']=='length' for g in gens),
            'runtime_errors':sum(g['error'] is not None for g in gens),
            'unique_selected_orders_replayed':len(executions),'methods':dict(agg),'per_family':family_rows}
    save(OUT/'summary.json',result)
    save(OUT/'analysis_manifest.json',{'files':{str(p.relative_to(ROOT)):sha(p) for p in [
        OUT/'manifest.json',OUT/'generations.jsonl',OUT/'details.json',OUT/'execution.json',OUT/'summary.json']}})
    print(json.dumps({k:v for k,v in result.items() if k!='per_family'}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','freeze','run','analyze'])
    globals()[p.parse_args().stage]()
