"""One-shot replay of fixed external probes. All metrics are realized, not EVSI."""
import argparse
from collections import Counter,defaultdict
import hashlib,json,math,platform,subprocess,sys
from pathlib import Path
import local_executor_e013 as runner

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/lavoir_executor_e014'
CACHE=Path('/private/tmp/lavoir-data-20261027')
Q=['no_ask','issue','product','money_gone','compensation','customer_name']
TEAMS=['cards','fraud','loans','accounts','complaints']
COST=.05
SYSTEM='''You route a banking customer to one team using these explicit rules in this exact priority order:
1. If the issue is an unrecognized charge or a lost card AND money has already left the account, choose fraud.
2. Otherwise, if the issue is an unrecognized charge or a lost card, choose cards.
3. Otherwise, if the issue is a fee question AND the customer asks for compensation, choose complaints.
4. Otherwise, if the product is a loan, choose loans.
5. Otherwise, if the product is a card, choose cards.
6. Otherwise choose accounts.
Possible issues: unrecognized charge, lost card, fee question, change request, failed payment. Possible products: card, current account, loan, savings account. The customer's name does not change the routing rules.
Use only facts in the conversation. An unmentioned fact is unknown, not automatically false. If the facts do not fully determine a team, still choose one plausible team. Do not ask another question or invent facts. Return only a JSON object with the single key "team" and a value from cards, fraud, loans, accounts, complaints, for example {"team":"cards"}.'''

def save(n,x):runner.save(OUT/n,x)
def read(n):return json.loads((OUT/n).read_text())
def case(r):return '/'.join(r['id'].split('/')[:3])

def worlds(schema):
    ws=[({},1.)]
    for k,s in schema['slots'].items():
        if s.get('decisive') is False:continue
        nxt=[]
        for w,p in ws:
            ps=s['cond'][w[s['depends_on']]] if 'depends_on' in s else s['values']
            for v,pv in ps.items():
                if pv:nxt.append(({**w,k:v},p*pv))
        ws=nxt
    def route(w):
        for rule in schema['rules']:
            if 'else' in rule:return rule['else']
            if all(w[k] in vs for k,vs in rule['if'].items()):return rule['unit']
        raise AssertionError('Unroutable')
    return [(w,p,route(w)) for w,p in ws],route

def posterior(ws,e):
    ps=dict.fromkeys(TEAMS,0.)
    for w,p,g in ws:
        if all(k not in w or w[k] in vs for k,vs in e.items()):ps[g]+=p
    z=sum(ps.values());assert z>0
    return {g:p/z for g,p in ps.items()}

def prepare():
    assert not (OUT/'pack.json').exists()
    sel=read('selection.json');rows=read('source_rows.json');review=read('semantic_review.json')
    assert runner.sha(OUT/'source_rows.json')==sel['rows_sha256']
    assert runner.sha(ROOT/'research/E014_PROTOCOL.md')==sel['protocol_sha256']
    assert review['before_generation'] and review['contradiction']==0
    schema=json.loads((CACHE/'schemas/banking_support.json').read_text())
    assert runner.sha(CACHE/'schemas/banking_support.json')=='625280cca355d1644bd604dd77837d03637c2e087d2f36bdf913b31a19082e1d'
    ws,route=worlds(schema);counts=Counter(case(r) for r in rows)
    prompts={};branches=[]
    for r in rows:
        assert route(r['profile'])==r['gold'];assert len(r['state'])==1
        base=[{'role':'system','content':SYSTEM},{'role':'user','content':r['state'][0]['text']}]
        group=runner.digest(base);p0=posterior(ws,r['evidence'])
        assert all(abs(p0[k]-r['target'][k])<2e-9 for k in TEAMS)
        assert list(r['probes'])==Q[1:]
        for q in Q:
            messages=list(base);ev={k:list(v) for k,v in r['evidence'].items()};kind='no_ask'
            if q!='no_ask':
                probe=r['probes'][q];kind=probe['kind']
                messages += [{'role':'assistant','content':probe['q']},{'role':'user','content':probe['a']}]
                for k,values in probe['reveal'].items():
                    assert r['profile'][k] in values
                    ev[k]=sorted(set(ev[k])&set(values)) if k in ev else list(values)
                    assert ev[k]
            ps=posterior(ws,ev);symbolic=runner.stable_best(TEAMS,lambda g:ps[g]);pid=runner.digest(messages)
            prompts.setdefault(pid,{'id':pid,'messages':messages})
            branches.append({'source_id':r['id'],'case_id':case(r),'visible_group':group,'question':q,
                'prompt_id':pid,'gold':r['gold'],'kind':kind,'weight':1/(24*counts[case(r)]),
                'semantic_posterior':ps,'semantic_choice':symbolic,'semantic_correct':int(symbolic==r['gold']),
                'sensitivity_excluded':case(r) in review['sensitivity_excluded_case_ids']})
    assert len(prompts)<=600
    assert abs(sum(b['weight'] for b in branches if b['question']=='no_ask')-1)<1e-12
    save('pack.json',list(prompts.values()));save('branches.json',branches)
    print(json.dumps({'unique_prompts':len(prompts),'branches':len(branches),'cases':len(counts),'profiles':len(rows)}))

def freeze():
    from transformers import AutoTokenizer
    import importlib.metadata as md
    assert not (OUT/'manifest.json').exists();assert md.version('mlx-lm')=='0.32.2'
    old=json.loads((ROOT/'results/local_executor_e013/manifest.json').read_text())
    for name,h in old['model_files'].items():assert runner.sha(runner.MODEL/name)==h
    tok=AutoTokenizer.from_pretrained(str(runner.MODEL),trust_remote_code=False,local_files_only=True)
    rendered=[]
    for r in read('pack.json'):
        prompt=tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True)
        ids=tok.encode(prompt,add_special_tokens=False);assert len(ids)<=4096
        rendered.append({'id':r['id'],'prompt':prompt,'input_tokens':ids})
    save('rendered.json',rendered)
    (OUT/'requirements.lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    paths=[ROOT/'research/E014_PROTOCOL.md',Path(__file__).resolve(),Path(runner.__file__).resolve()]+[OUT/n for n in
        ['source_rows.json','selection.json','review_units.json','semantic_review.json','pack.json','branches.json','rendered.json','requirements.lock.txt']]
    save('manifest.json',{'files':{str(p.relative_to(ROOT)):runner.sha(p) for p in paths},
         'model':old['model'],'revision':old['revision'],'model_files':old['model_files'],
         'seed':runner.SEED,'temperature':0,'max_tokens':64,'max_input_tokens':max(len(r['input_tokens']) for r in rendered),
         'unique_prompts':len(rendered),'budget_seconds':1800,'python':sys.version,'platform':platform.platform()})
    print(json.dumps({'frozen':True,'unique_prompts':len(rendered),'max_input_tokens':max(len(r['input_tokens']) for r in rendered)}))

def run():
    runner.OUT=OUT;runner.run()

def parse(raw):
    # Same strict JSON shape as E013, with a different key.
    import re
    text=raw.strip();match=re.fullmatch(r'```(?:json)?\s*\n([\s\S]*?)\n```',text)
    if match:text=match.group(1).strip()
    def unique(pairs):
        if len(pairs)!=len(dict(pairs)):raise ValueError('Duplicate keys')
        return dict(pairs)
    try:obj=json.loads(text,object_pairs_hook=unique)
    except (ValueError,TypeError):return None
    return obj['team'] if type(obj)is dict and set(obj)=={'team'} and obj['team'] in TEAMS else None

def aggregate(rows):
    scale=sum(b['weight'] for b in rows if b['question']=='no_ask')
    assert scale>0
    fixed={q:{k:0. for k in ['accuracy','utility','invalid_mass','missing_mass','upper_accuracy']} for q in Q}
    byprofile=defaultdict(dict);bygroup=defaultdict(list)
    for b in rows:
        w=b['weight']/scale;m=fixed[b['question']]
        for k,v in [('accuracy',b['correct']),('utility',b['correct']-COST*(b['question']!='no_ask')),
                    ('invalid_mass',b['invalid']),('missing_mass',b['missing']),
                    ('upper_accuracy',b['correct']+int(b['missing']))]:m[k]+=w*v
        byprofile[b['source_id']][b['question']]=b;bygroup[b['visible_group']].append(b)
    complete=not any(b['missing'] for b in rows)
    result={'cases':len(set(b['case_id'] for b in rows)),'profiles':len(byprofile),'fixed':fixed,
            'complete':complete,'uniform_random':{k:sum(fixed[q][k] for q in Q[1:])/5 for k in fixed['no_ask']}}
    if complete:
        best=runner.stable_best(Q,lambda q:fixed[q]['utility'])
        result['best_fixed_including_noask']={'question':best,**fixed[best]}
        policies={}
        for mode,groups in [('hidden_profile_oracle',list(byprofile.values())),('visible_input_oracle',list(bygroup.values()))]:
            totals={'accuracy':0.,'utility':0.,'ask_rate':0.};choices=[]
            for group in groups:
                bs=list(group.values()) if type(group)is dict else group
                utilities={q:sum(b['weight']*(b['correct']-COST*(q!='no_ask')) for b in bs if b['question']==q) for q in Q}
                choice=runner.stable_best(Q,lambda q:utilities[q]);selected=[b for b in bs if b['question']==choice]
                for b in selected:
                    w=b['weight']/scale;totals['accuracy']+=w*b['correct'];totals['utility']+=w*(b['correct']-COST*(choice!='no_ask'));totals['ask_rate']+=w*(choice!='no_ask')
                choices.append({'group':bs[0]['source_id'] if mode=='hidden_profile_oracle' else bs[0]['visible_group'],'question':choice})
            policies[mode]={**totals,'choices':choices}
        result['oracles']=policies
    return result

def analyze():
    runner.OUT=OUT;runner.verify_frozen()
    gens=[json.loads(l) for l in (OUT/'generations.jsonl').read_text().splitlines()]
    byid={g['id']:g for g in gens};assert len(byid)==len(gens)
    branches=read('branches.json');details=[]
    for b in branches:
        gen=byid.get(b['prompt_id']);missing=gen is None or gen['error'] is not None
        team=None if missing else parse(gen['raw_output']);correct=int(team==b['gold'])
        details.append({**b,'team':team,'correct':correct,'missing':missing,'invalid':not missing and team is None})
    full=aggregate(details);clean=aggregate([b for b in details if not b['sensitivity_excluded']])
    symbolic=[{**b,'correct':b['semantic_correct'],'missing':False,'invalid':False} for b in details]
    summary={'unique_generations':len(gens),'invalid_unique':sum(parse(g['raw_output'])is None and g['error']is None for g in gens),
             'length_unique':sum(g['finish_reason']=='length' for g in gens),'runtime_errors':sum(g['error']is not None for g in gens),
             'all_cases':full,'predeclared_sensitivity':clean,'privileged_symbolic':aggregate(symbolic)}
    save('details.json',details);save('summary.json',summary)
    save('analysis_manifest.json',{'files':{str((OUT/n).relative_to(ROOT)):runner.sha(OUT/n) for n in
        ['manifest.json','generations.jsonl','details.json','summary.json']}})
    print(json.dumps({'generated':len(gens),'invalid':summary['invalid_unique'],'all_fixed':full['fixed'],
                     'oracles':{k:{a:b for a,b in v.items() if a!='choices'} for k,v in full.get('oracles',{}).items()}}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','freeze','run','analyze']);globals()[p.parse_args().stage]()
