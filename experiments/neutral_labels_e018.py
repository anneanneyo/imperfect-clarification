"""Fixed neutral-label no-question diagnostic; development/train only."""
import argparse,json,re,sys
from collections import Counter,defaultdict
from pathlib import Path
from fractions import Fraction as F
import local_executor_e013 as runner
import lavoir_executor_e014 as e14

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/neutral_labels_e018'
MAPPING={'cards':'U1','fraud':'U2','loans':'U3','accounts':'U4','complaints':'U5'};REVERSE={v:k for k,v in MAPPING.items()}
CLEAN={'00037','00384','00210','00390','00406','00176','00348'}
def read(p):return json.loads(p.read_text())
def save(p,x):runner.save(p,x)
def verify():
    for p,h in read(OUT/'manifest.json')['files'].items():assert runner.sha(ROOT/p)==h,p
def prepare():
    assert not OUT.exists()
    from transformers import AutoTokenizer
    sm=read(ROOT/'results/heldout_value_e016/manifest.json')
    for n,h in sm['model_files'].items():assert runner.sha(runner.MODEL/n)==h
    system=re.sub(r'\b(cards|fraud|loans|accounts|complaints)\b',lambda m:MAPPING[m[0]],e14.SYSTEM)
    token=AutoTokenizer.from_pretrained(str(runner.MODEL),trust_remote_code=False,local_files_only=True)
    sources=[('E014_dev',ROOT/'results/lavoir_executor_e014','details.json'),('E016_train',ROOT/'results/heldout_value_e016','train_details.json')]
    prompts={};branches=[];paths=[ROOT/'research/E018_PROTOCOL.md',Path(__file__).resolve(),Path(runner.__file__).resolve(),Path(e14.__file__).resolve()]
    for label,folder,detailsname in sources:
        details=read(folder/detailsname);rows={r['id']:r for r in read(folder/'source_rows.json')}
        initial=[d for d in details if d['question']=='no_ask'];counts=Counter(d['case_id'] for d in initial);n=len(counts)
        assert n==(24 if label=='E014_dev' else 96)
        for d in initial:
            r=rows[d['source_id']];messages=[{'role':'system','content':system},{'role':'user','content':r['state'][0]['text']}];pid=runner.digest(messages)
            prompts.setdefault(pid,{'id':pid,'messages':messages})
            branches.append({'dataset':label,'case_id':d['case_id'],'source_id':r['id'],'prompt_id':pid,'gold':r['gold'],
                'old_team':d['team'],'old_correct':d['correct'],'weight_numerator':1,'weight_denominator':n*counts[d['case_id']],
                'initial_symbolic_certain':max(r['target'].values())>1-1e-12,'clean_known_error':label=='E016_train' and d['case_id'].split('/')[-1]in CLEAN})
        paths.extend([folder/detailsname,folder/'source_rows.json'])
    assert len(prompts)==120;OUT.mkdir();save(OUT/'pack.json',list(prompts.values()));save(OUT/'branches.json',branches);save(OUT/'mapping.json',MAPPING)
    rendered=[]
    for p in prompts.values():
        prompt=token.apply_chat_template(p['messages'],tokenize=False,add_generation_prompt=True);ids=token.encode(prompt,add_special_tokens=False);assert len(ids)<=4096
        rendered.append({'id':p['id'],'prompt':prompt,'input_tokens':ids})
    save(OUT/'rendered.json',rendered);paths.extend([OUT/n for n in ['pack.json','branches.json','mapping.json','rendered.json']])
    save(OUT/'manifest.json',{'files':{str(p.relative_to(ROOT)):runner.sha(p) for p in paths},'model':sm['model'],'revision':sm['revision'],
        'model_files':sm['model_files'],'python':sys.version,'seed':20261027,'temperature':0,'max_tokens':64,'unique_prompts':120,
        'max_input_tokens':max(len(r['input_tokens']) for r in rendered),'budget_seconds':1800,'E016_test_used':False})
    print(json.dumps({'frozen':True,'unique_prompts':120,'profile_branches':len(branches),'datasets':{'E014_dev':24,'E016_train':96}}))
def run():verify();runner.OUT=OUT;runner.run()
def parse(text):
    try:r=json.loads(text)
    except (json.JSONDecodeError,TypeError):return None
    if not isinstance(r,dict) or set(r)!={'team'} or not isinstance(r['team'],str):return None
    return REVERSE.get(r['team'])
def score():
    verify();assert read(OUT/'run_status.json')['complete'];assert not (OUT/'summary.json').exists()
    gens={g['id']:g for g in [json.loads(l) for l in (OUT/'generations.jsonl').read_text().splitlines()]};assert len(gens)==120 and all(g['error']is None for g in gens.values())
    details=[]
    for b in read(OUT/'branches.json'):
        g=gens[b['prompt_id']];team=parse(g['raw_output']);details.append({**b,'team':team,'correct':int(team==b['gold']),'invalid':team is None})
    summary={};case_results=[]
    for name in ['E014_dev','E016_train']:
        ds=[d for d in details if d['dataset']==name];w=lambda d:F(d['weight_numerator'],d['weight_denominator'])
        old=sum((w(d)*d['old_correct'] for d in ds),F(0));new=sum((w(d)*d['correct'] for d in ds),F(0));bc=defaultdict(list)
        for d in ds:bc[d['case_id']].append(d)
        diffs=[];clean_fixed=[]
        for c,rs in bc.items():
            a=F(sum(d['old_correct'] for d in rs),len(rs));b=F(sum(d['correct'] for d in rs),len(rs));diffs.append(b-a)
            case_results.append({'dataset':name,'case_id':c,'old_accuracy':float(a),'new_accuracy':float(b),'clean_known_error':rs[0]['clean_known_error']})
            if rs[0]['clean_known_error'] and b==1:clean_fixed.append(c)
        certain=[d for d in ds if d['initial_symbolic_certain']];mass=sum((w(d) for d in certain),F(0))
        summary[name]={'cases':len(bc),'old_accuracy':float(old),'new_accuracy':float(new),'difference':float(new-old),
            'improved_cases':sum(d>0 for d in diffs),'worse_cases':sum(d<0 for d in diffs),'tied_cases':sum(d==0 for d in diffs),
            'invalid_unique_outputs':len({d['prompt_id'] for d in ds if d['invalid']}),
            'initial_certain_case_mass':float(mass),'certain_old_accuracy':float(sum((w(d)*d['old_correct'] for d in certain),F(0))/mass),
            'certain_new_accuracy':float(sum((w(d)*d['correct'] for d in certain),F(0))/mass),'clean_known_errors_fixed':clean_fixed}
    tr=summary['E016_train'];summary['candidate_gate_passed']=len(tr['clean_known_errors_fixed'])>=4 and tr['new_accuracy']>=tr['old_accuracy']
    summary['run_status']=read(OUT/'run_status.json');save(OUT/'details.json',details);save(OUT/'case_results.json',case_results);save(OUT/'summary.json',summary)
    paths=[OUT/n for n in ['manifest.json','generations.jsonl','details.json','case_results.json','summary.json']]
    save(OUT/'score_manifest.json',{'files':{str(p.relative_to(ROOT)):runner.sha(p) for p in paths}});print(json.dumps(summary))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','score']);globals()[p.parse_args().stage]()
