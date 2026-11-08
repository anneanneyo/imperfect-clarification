"""Post-hoc control design, frozen before control outputs; no new user facts."""
import argparse,json,sys
from pathlib import Path
import local_executor_e013 as runner
import lavoir_executor_e014 as e14
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/lavoir_executor_e014'
OUT=ROOT/'results/e014_no_information_controls'
def read(p):return json.loads(p.read_text())
def save(n,x):runner.save(OUT/n,x)

def prepare():
    assert not OUT.exists();OUT.mkdir()
    original=read(BASE/'pack.json');byid={p['id']:p for p in original}
    bases=[b for b in read(BASE/'details.json') if b['question']=='no_ask']
    prompts={};branches=[]
    for b in bases:
        initial=byid[b['prompt_id']]['messages'];assert len(initial)==2
        for control,q,a in [('repeat','Could you repeat the information you already provided?',initial[1]['content']),
                            ('no_extra','Is there anything else you can tell me?',"I don't have any additional information to provide.")]:
            messages=initial+[{'role':'assistant','content':q},{'role':'user','content':a}]
            pid=runner.digest(messages);prompts.setdefault(pid,{'id':pid,'messages':messages})
            branches.append({'source_id':b['source_id'],'case_id':b['case_id'],'question':control,'prompt_id':pid,
                 'gold':b['gold'],'weight':b['weight'],'base_team':b['team'],'base_correct':b['correct'],
                 'schema_certain':max(b['semantic_posterior'].values())>1-1e-12,
                 'sensitivity_excluded':b['sensitivity_excluded']})
    assert len(prompts)==48 and len(branches)==130
    save('pack.json',list(prompts.values()));save('branches.json',branches)
    print(json.dumps({'prompts':len(prompts),'linked_branches':len(branches)}))

def freeze():
    from transformers import AutoTokenizer
    assert not (OUT/'manifest.json').exists()
    old=read(BASE/'manifest.json')
    for p,h in old['files'].items():assert runner.sha(ROOT/p)==h
    for n,h in old['model_files'].items():assert runner.sha(runner.MODEL/n)==h
    tok=AutoTokenizer.from_pretrained(str(runner.MODEL),trust_remote_code=False,local_files_only=True)
    rendered=[]
    for r in read(OUT/'pack.json'):
        prompt=tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True)
        ids=tok.encode(prompt,add_special_tokens=False);assert len(ids)<=4096
        rendered.append({'id':r['id'],'prompt':prompt,'input_tokens':ids})
    save('rendered.json',rendered)
    paths=[Path(__file__).resolve(),Path(runner.__file__).resolve(),Path(e14.__file__).resolve(),
           ROOT/'research/E014_NO_INFORMATION_CONTROLS.md',BASE/'manifest.json',BASE/'summary.json',BASE/'details.json',BASE/'pack.json',BASE/'requirements.lock.txt']+[OUT/n for n in ['pack.json','branches.json','rendered.json']]
    save('manifest.json',{'files':{str(p.relative_to(ROOT)):runner.sha(p) for p in paths},
         'model':old['model'],'revision':old['revision'],'model_files':old['model_files'],
         'seed':20261027,'temperature':0,'max_tokens':64,'max_input_tokens':max(len(r['input_tokens']) for r in rendered),
         'unique_prompts':len(rendered),'design':'post-hoc to E014; before control generations'})
    print(json.dumps({'frozen':True,'prompts':len(rendered)}))

def run():runner.OUT=OUT;runner.run()

def analyze():
    runner.OUT=OUT;runner.verify_frozen()
    gs=[json.loads(l) for l in (OUT/'generations.jsonl').read_text().splitlines()];byid={g['id']:g for g in gs};details=[]
    for b in read(OUT/'branches.json'):
        g=byid.get(b['prompt_id']);missing=g is None or g['error']is not None
        team=None if missing else e14.parse(g['raw_output']);correct=int(team==b['gold'])
        details.append({**b,'team':team,'correct':correct,'missing':missing,'invalid':not missing and team is None,
                       'label_changed':not missing and team!=b['base_team'],'gain':None if missing else correct-b['base_correct']})
    summaries={}
    for scope,rows in [('all_cases',details),('predeclared_sensitivity',[b for b in details if not b['sensitivity_excluded']])]:
        methods={}
        for q in ['repeat','no_extra']:
            rs=[b for b in rows if b['question']==q];z=sum(b['weight'] for b in rs)
            methods[q]={k:sum(b['weight']*fn(b) for b in rs)/z for k,fn in {
                'accuracy':lambda b:b['correct'],'upper_accuracy':lambda b:b['correct']+b['missing'],
                'base_accuracy':lambda b:b['base_correct'],'utility':lambda b:b['correct']-.05,
                'label_change_mass':lambda b:b['label_changed'],'improvement_mass':lambda b:not b['missing'] and b['gain']>0,
                'worsening_mass':lambda b:not b['missing'] and b['gain']<0,'missing_mass':lambda b:b['missing'],
                'invalid_mass':lambda b:b['invalid'],'certain_improvement_mass':lambda b:b['schema_certain'] and not b['missing'] and b['gain']>0}.items()}
            methods[q]['cases']=len(set(b['case_id'] for b in rs))
        summaries[scope]=methods
    summary={'unique_generations':len(gs),'length_outputs':sum(g['finish_reason']=='length' for g in gs),
             'runtime_errors':sum(g['error']is not None for g in gs),'results':summaries}
    save('details.json',details);save('summary.json',summary)
    save('analysis_manifest.json',{'files':{str((OUT/n).relative_to(ROOT)):runner.sha(OUT/n) for n in ['manifest.json','generations.jsonl','details.json','summary.json']}})
    print(json.dumps(summary))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','freeze','run','analyze']);globals()[p.parse_args().stage]()
