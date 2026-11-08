"""Frozen larger train/test replay. Test inference is gated on sealed predictions."""
import argparse
from collections import Counter,defaultdict
import hashlib,json,re,sys,time
from pathlib import Path
import numpy as np
import local_executor_e013 as runner
import lavoir_executor_e014 as e14
import text_value_e015 as learner

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/heldout_value_e016'
CACHE=Path('/private/tmp/lavoir-data-20261027');Q=e14.Q;COST=np.array([0,.05,.05,.05,.05,.05])
def read(p):return json.loads(p.read_text())
def save(p,x):runner.save(p,x)
def norm(t):return re.sub(r'\s+',' ',t.casefold()).strip()
def verify():
    for name,h in read(OUT/'manifest.json')['files'].items():assert runner.sha(ROOT/name)==h,name
def prepare():
    assert not OUT.exists()
    source=CACHE/'data/train/banking_support.jsonl'
    assert runner.sha(source)=='6641c847f59d539fc99de6dae24f92babeb0358c68699514528e23f3c8c71365'
    rows=[json.loads(l) for l in source.read_text().splitlines()];bycase=defaultdict(list)
    for r in rows:
        if r['kind']=='a':bycase[e14.case(r)].append(r)
    assert len(bycase)==485
    bytext=defaultdict(list)
    for c,rs in bycase.items():
        texts={norm(r['state'][0]['text']) for r in rs};assert len(texts)==1
        bytext[texts.pop()].append(c)
    dev=set(read(ROOT/'results/lavoir_executor_e014/selection.json')['cases']);components=[];removed=[]
    for text,cs in bytext.items():
        cs=sorted(cs);gid=hashlib.sha256(('E016-v1:'+'\n'.join(cs)).encode()).hexdigest()
        if set(cs)&dev:removed.extend(cs);continue
        components.append({'group_id':gid,'cases':cs,'text':text})
    components.sort(key=lambda g:g['group_id']);chosen=components[:144];assert len(chosen)==144
    schema=read(CACHE/'schemas/banking_support.json');worlds,route=e14.worlds(schema)
    prompts={'train':{},'test':{}};branches=[];selected_rows=[];selector_inputs=[]
    for i,g in enumerate(chosen):
        split='train' if i<96 else 'test';g['split']=split;selector_inputs.append({'group_id':g['group_id'],'split':split,'text':g['text']})
        for c in g['cases']:
            for r in bycase[c]:
                selected_rows.append(r);assert route(r['profile'])==r['gold']
                ps=e14.posterior(worlds,r['evidence']);assert all(abs(ps[k]-r['target'][k])<2e-9 for k in ps)
                base=[{'role':'system','content':e14.SYSTEM},{'role':'user','content':r['state'][0]['text']}]
                for q in Q:
                    messages=list(base)
                    if q!='no_ask':
                        pr=r['probes'][q]
                        assert all(r['profile'][k] in vs for k,vs in pr['reveal'].items())
                        messages += [{'role':'assistant','content':pr['q']},{'role':'user','content':pr['a']}]
                    pid=runner.digest(messages);prompts[split].setdefault(pid,{'id':pid,'messages':messages})
                    branches.append({'split':split,'group_id':g['group_id'],'case_id':c,'source_id':r['id'],
                        'question':q,'prompt_id':pid,'gold':r['gold'],'kind':'no_ask' if q=='no_ask' else r['probes'][q]['kind'],
                        'weight_within_group':1/(len(g['cases'])*len(bycase[c]))})
    assert set(prompts['train']).isdisjoint(prompts['test'])
    assert len(prompts['train'])<=1200 and len(prompts['test'])<=600,(len(prompts['train']),len(prompts['test']))
    OUT.mkdir();save(OUT/'selector_inputs.json',selector_inputs);save(OUT/'source_rows.json',selected_rows)
    save(OUT/'branches.json',branches);save(OUT/'components.json',chosen)
    save(OUT/'selection.json',{'eligible_initial_cases':len(bycase),'eligible_components_after_dev_exclusion':len(components),
         'excluded_dev_component_cases':sorted(removed),'train_components':96,'test_components':48,
         'source_sha256':runner.sha(source),'groups':chosen,'prompt_counts':{s:len(ps) for s,ps in prompts.items()}})
    shards=[]
    for split,ps in prompts.items():
        vals=list(ps.values())
        for offset in range(0,len(vals),600):
            shard=f'{split}_{offset//600:02d}';p=OUT/'shards'/shard;p.mkdir(parents=True)
            save(p/'pack.json',vals[offset:offset+600]);shards.append({'name':shard,'split':split,'prompts':len(vals[offset:offset+600])})
    assert len(shards)<=3
    save(OUT/'shards.json',shards)
    print(json.dumps({'components':144,'source_cases':len(set(b['case_id'] for b in branches)),'source_profiles':len(selected_rows),
                      'prompt_counts':{s:len(ps) for s,ps in prompts.items()},'shards':shards}))

def freeze():
    from transformers import AutoTokenizer
    import importlib.metadata as md
    assert not (OUT/'manifest.json').exists();assert md.version('mlx-lm')=='0.32.2'
    modelmeta=read(ROOT/'results/lavoir_executor_e014/manifest.json')
    for n,h in modelmeta['model_files'].items():assert runner.sha(runner.MODEL/n)==h
    tok=AutoTokenizer.from_pretrained(str(runner.MODEL),trust_remote_code=False,local_files_only=True)
    paths=[ROOT/'research/E016_PROTOCOL.md',Path(__file__).resolve(),Path(runner.__file__).resolve(),Path(e14.__file__).resolve(),Path(learner.__file__).resolve(),
        CACHE/'schemas/banking_support.json']+[OUT/n for n in ['selector_inputs.json','source_rows.json','branches.json','components.json','selection.json','shards.json']]
    # Persist the schema so later integrity checks do not depend on a mutable tmp path.
    save(OUT/'schema.json',read(CACHE/'schemas/banking_support.json'));paths.remove(CACHE/'schemas/banking_support.json');paths.append(OUT/'schema.json')
    max_input=0
    for shard in read(OUT/'shards.json'):
        folder=OUT/'shards'/shard['name'];rendered=[]
        for r in read(folder/'pack.json'):
            prompt=tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True);ids=tok.encode(prompt,add_special_tokens=False)
            assert len(ids)<=4096;max_input=max(max_input,len(ids));rendered.append({'id':r['id'],'prompt':prompt,'input_tokens':ids})
        save(folder/'rendered.json',rendered);paths +=[folder/'pack.json',folder/'rendered.json']
    import subprocess
    (OUT/'requirements.lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True));paths.append(OUT/'requirements.lock.txt')
    manifest={'files':{str(p.relative_to(ROOT)):runner.sha(p) for p in paths},'model':modelmeta['model'],'revision':modelmeta['revision'],
        'model_files':modelmeta['model_files'],'python':sys.version,'numpy':np.__version__,'seed':20261027,'temperature':0,
        'max_tokens':64,'max_input_tokens':max_input,'budget_seconds_per_shard':1800,'max_shards':3}
    save(OUT/'manifest.json',manifest)
    for shard in read(OUT/'shards.json'):
        sm={**manifest,'files':{**manifest['files'],str((OUT/'manifest.json').relative_to(ROOT)):runner.sha(OUT/'manifest.json')},'shard':shard}
        name='manifest.json' if shard['split']=='train' else 'manifest.pending.json'
        save(OUT/'shards'/shard['name']/name,sm)
    print(json.dumps({'frozen':True,'max_input_tokens':max_input,'test_manifest_state':'pending until predictions sealed'}))

def run_split(split):
    verify()
    if split=='test':
        fit=read(OUT/'fit_manifest.json')
        for p,h in fit['files'].items():assert runner.sha(ROOT/p)==h
        assert fit['test_generations_existed_at_seal'] is False
    for shard in read(OUT/'shards.json'):
        if shard['split']!=split:continue
        folder=OUT/'shards'/shard['name'];assert (folder/'manifest.json').exists()
        assert not (folder/'generations.jsonl').exists(),'Never silently resume or retry a started shard'
        print(json.dumps({'starting_shard':shard['name'],'prompts':shard['prompts']}),flush=True)
        runner.OUT=folder;runner.run()
        status=read(folder/'run_status.json');assert status['complete'],'Incomplete shard, stop rather than continue blindly'

def collect(split):
    byid={}
    for s in read(OUT/'shards.json'):
        if s['split']!=split:continue
        p=OUT/'shards'/s['name'];assert read(p/'run_status.json')['complete']
        gs=[json.loads(l) for l in (p/'generations.jsonl').read_text().splitlines()]
        assert len(gs)==s['prompts'] and all(g['error']is None for g in gs)
        for g in gs:assert g['id']not in byid;byid[g['id']]=g
    targets=defaultdict(lambda:np.zeros(6));details=[]
    for b in read(OUT/'branches.json'):
        if b['split']!=split:continue
        gen=byid[b['prompt_id']];team=e14.parse(gen['raw_output']);correct=int(team==b['gold'])
        targets[b['group_id']][Q.index(b['question'])]+=b['weight_within_group']*correct
        details.append({**b,'team':team,'correct':correct,'invalid':team is None})
    return {g:v.tolist() for g,v in targets.items()},details,byid

def fit():
    verify();assert not (OUT/'fit_manifest.json').exists()
    testshards=[OUT/'shards'/s['name'] for s in read(OUT/'shards.json') if s['split']=='test']
    assert all(not (p/'generations.jsonl').exists() for p in testshards)
    targets,details,gens=collect('train');inputs=read(OUT/'selector_inputs.json');texts={r['group_id']:r['text'] for r in inputs}
    trainids=[r['group_id'] for r in inputs if r['split']=='train'];testids=[r['group_id'] for r in inputs if r['split']=='test']
    assert set(targets)==set(trainids) and set(targets).isdisjoint(testids)
    predictions,model=learner.predict_fold(texts,targets,trainids,testids)
    save(OUT/'train_targets.json',targets);save(OUT/'train_details.json',details)
    meta={'train_ids':trainids,'test_ids':testids,'features':model.pop('vocab'),'feature_counts':model.pop('feature_counts'),
          'alpha':1.,'training_label_components':len(targets),'training_branch_labels':len(details),'training_unique_generations':len(gens)}
    save(OUT/'model_metadata.json',meta);np.savez_compressed(OUT/'model.npz',**model)
    save(OUT/'test_predictions.json',predictions)
    paths=[OUT/n for n in ['manifest.json','train_targets.json','train_details.json','model_metadata.json','model.npz','test_predictions.json']]
    paths +=[OUT/'shards'/s['name']/'generations.jsonl' for s in read(OUT/'shards.json') if s['split']=='train']
    fit_manifest={'files':{str(p.relative_to(ROOT)):runner.sha(p) for p in paths},'test_generations_existed_at_seal':False,
                  'train_components':len(trainids),'test_predictions':len(predictions),'timing':'Predictions sealed before any test inference'}
    save(OUT/'fit_manifest.json',fit_manifest)
    # The prepared test inputs stay immutable. Only now create executable manifests.
    for folder in testshards:
        assert not (folder/'manifest.json').exists();sm=read(folder/'manifest.pending.json')
        sm['files'].update(fit_manifest['files']);sm['files'][str((OUT/'fit_manifest.json').relative_to(ROOT))]=runner.sha(OUT/'fit_manifest.json')
        sm['files'][str((folder/'manifest.pending.json').relative_to(ROOT))]=runner.sha(folder/'manifest.pending.json')
        sm['predictions_sealed']=True;save(folder/'manifest.json',sm)
    print(json.dumps({'trained_components':len(trainids),'sealed_test_predictions':len(predictions),
                      'ridge_choices':dict(Counter(p['ridge_question'] for p in predictions)),
                      'constant_choices':dict(Counter(p['train_constant_question'] for p in predictions))}))

def score():
    verify();assert not (OUT/'summary.json').exists()
    for p,h in read(OUT/'fit_manifest.json')['files'].items():assert runner.sha(ROOT/p)==h
    targets,details,gens=collect('test');pred=read(OUT/'test_predictions.json');results=[]
    for p in pred:
        y=targets[p['case_id']];actions={'ridge':p['ridge_question'],'train_constant':p['train_constant_question'],**{q:q for q in Q}}
        metrics={}
        for m,q in actions.items():
            a=y[Q.index(q)];metrics[m]={'question':q,'accuracy':a,'utility':a-.05*(q!='no_ask'),'ask_rate':int(q!='no_ask')}
        a=sum(y[1:])/5;metrics['random_question']={'question':'uniform_random','accuracy':a,'utility':a-.05,'ask_rate':1}
        q=Q[learner.stable_best(np.array(y)-COST)];a=y[Q.index(q)];metrics['component_posthoc_oracle']={'question':q,'accuracy':a,'utility':a-.05*(q!='no_ask'),'ask_rate':int(q!='no_ask')}
        results.append({'group_id':p['case_id'],'target_accuracy':y,'metrics':metrics,
                        'ridge_mae':float(np.mean(np.abs(np.array(p['prediction'])-y))),
                        'constant_mae':float(np.mean(np.abs(np.array(p['constant_prediction'])-y)))})
    methods={}
    for m in results[0]['metrics']:
        ms=[r['metrics'][m] for r in results]
        methods[m]={k:sum(r[k] for r in ms)/len(ms) for k in ['accuracy','utility','ask_rate']}
        methods[m]['choices']=dict(Counter(r['question'] for r in ms))
    rng=np.random.default_rng(20261027);indices=rng.integers(0,len(results),size=(10000,len(results)));pairs={}
    for compare in ['train_constant','issue']:
        diff=np.array([r['metrics']['ridge']['utility']-r['metrics'][compare]['utility'] for r in results])
        boot=diff[indices].mean(axis=1);ci=np.quantile(boot,[.025,.975])
        pairs[compare]={'difference':float(diff.mean()),'percentile95_interval':ci.tolist(),'improved_components':int((diff>1e-12).sum()),
                        'worse_components':int((diff< -1e-12).sum()),'tied_components':int((abs(diff)<=1e-12).sum())}
    summary={'test_components':len(results),'test_source_profiles':len({d['source_id'] for d in details}),'test_unique_generations':len(gens),
             'invalid_unique_generations':sum(e14.parse(g['raw_output'])is None for g in gens.values()),
             'length_outputs':sum(g['finish_reason']=='length' for g in gens.values()),'methods':methods,'paired_utility_comparisons':pairs,
             'ridge_mae':sum(r['ridge_mae'] for r in results)/len(results),'constant_mae':sum(r['constant_mae'] for r in results)/len(results),
             'bootstrap':{'seed':20261027,'resamples':10000,'unit':'held-out component','interpretation':'descriptive component variability; excludes new response and model variation'}}
    save(OUT/'test_targets.json',targets);save(OUT/'test_details.json',details);save(OUT/'component_results.json',results);save(OUT/'summary.json',summary)
    paths=[OUT/n for n in ['manifest.json','fit_manifest.json','test_targets.json','test_details.json','component_results.json','summary.json']]
    paths +=[OUT/'shards'/s['name']/'generations.jsonl' for s in read(OUT/'shards.json') if s['split']=='test']
    save(OUT/'score_manifest.json',{'files':{str(p.relative_to(ROOT)):runner.sha(p) for p in paths}})
    print(json.dumps(summary))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','freeze','train','fit','test','score']);a=parser.parse_args()
    if a.stage in ['train','test']:run_split(a.stage)
    else:globals()[a.stage]()
