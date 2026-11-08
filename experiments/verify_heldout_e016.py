"""Independent split/prompt provenance, SVD predictions and rational test scoring."""
from collections import Counter,defaultdict
from fractions import Fraction as F
import hashlib,json,re
from pathlib import Path
import numpy as np
from verify_text_value_e015 import tf
from verify_lavoir_executor_e014 import parse
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/heldout_value_e016';CACHE=Path('/private/tmp/lavoir-data-20261027')
Q=['no_ask','issue','product','money_gone','compensation','customer_name'];C=np.array([0,.05,.05,.05,.05,.05])
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(t):return ' '.join(t.casefold().split())
def close(a,b):np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)
def best(v):
    m=max(v);return next(i for i,x in enumerate(v) if x>=m-1e-12)
def independent_targets(split,branches,weights):
    outputs={}
    for s in read(OUT/'shards.json'):
        if s['split']!=split:continue
        p=OUT/'shards'/s['name'];status=read(p/'run_status.json');assert status['complete']
        gs=[json.loads(l) for l in (p/'generations.jsonl').read_text().splitlines()]
        assert len(gs)==s['prompts']
        for g in gs:assert g['id']not in outputs and g['error']is None;outputs[g['id']]=g
    target=defaultdict(lambda:[F(0)]*6)
    for b in branches:
        if b['split']!=split:continue
        team=parse(outputs[b['prompt_id']]['raw_output'])
        target[b['group_id']][Q.index(b['question'])]+=weights[b['source_id']]*(team==b['gold'])
    return target

def main():
    for p,h in read(OUT/'manifest.json')['files'].items():assert sha(ROOT/p)==h,p
    sel=read(OUT/'selection.json');assert sha(CACHE/'data/train/banking_support.jsonl')==sel['source_sha256']
    assert sha(CACHE/'schemas/banking_support.json')=='625280cca355d1644bd604dd77837d03637c2e087d2f36bdf913b31a19082e1d'
    assert read(OUT/'schema.json')==read(CACHE/'schemas/banking_support.json')
    source=[json.loads(l) for l in (CACHE/'data/train/banking_support.jsonl').read_text().splitlines()]
    initial={r['id']:r for r in source if r['kind']=='a'};bytext=defaultdict(set);bycase=defaultdict(list)
    for r in initial.values():
        case='/'.join(r['id'].split('/')[:3]);bytext[norm(r['state'][0]['text'])].add(case);bycase[case].append(r)
    dev=set(read(ROOT/'results/lavoir_executor_e014/selection.json')['cases']);expected=[]
    for text,cs in bytext.items():
        if cs&dev:continue
        cases=sorted(cs);gid=hashlib.sha256(('E016-v1:'+'\n'.join(cases)).encode()).hexdigest()
        expected.append((gid,cases,text))
    expected.sort();expected=expected[:144];components=read(OUT/'components.json');assert len(expected)==len(components)==144
    for i,(ex,g) in enumerate(zip(expected,components)):
        assert (g['group_id'],g['cases'],g['text'])==ex and g['split']==('train' if i<96 else 'test')
    groups={g['group_id']:g for g in components};inputs=read(OUT/'selector_inputs.json')
    assert inputs==[{'group_id':g['group_id'],'split':g['split'],'text':g['text']} for g in components]
    selected=read(OUT/'source_rows.json');assert all(r==initial[r['id']] for r in selected)
    assert set(r['id'] for r in selected)=={r['id'] for g in components for c in g['cases'] for r in bycase[c]}
    pack={};promptsplit={};shards=read(OUT/'shards.json')
    for s in shards:
        folder=OUT/'shards'/s['name'];ps=read(folder/'pack.json');assert len(ps)==s['prompts']<=600
        for p in ps:assert p['id']not in pack;pack[p['id']]=p;promptsplit[p['id']]=s['split']
    branches=read(OUT/'branches.json');weights={};totals=defaultdict(F);seen=set();system=None
    for b in branches:
        g=groups[b['group_id']];r=initial[b['source_id']];c='/'.join(r['id'].split('/')[:3]);q=b['question']
        assert b['split']==g['split']==promptsplit[b['prompt_id']] and b['case_id']==c and c in g['cases']
        key=(r['id'],q);assert key not in seen;seen.add(key)
        w=F(1,len(g['cases'])*len(bycase[c]));weights[r['id']]=w;close(float(w),b['weight_within_group']);totals[g['group_id'],q]+=w
        assert b['gold']==r['gold'];m=pack[b['prompt_id']]['messages']
        if system is None:system=m[0]
        assert m[0]==system and m[1]=={'role':'user','content':r['state'][0]['text']}
        if q=='no_ask':assert len(m)==2
        else:assert m[2:]==[{'role':'assistant','content':r['probes'][q]['q']},{'role':'user','content':r['probes'][q]['a']}]
    assert all(v==1 for v in totals.values()) and len(totals)==144*6 and len(branches)==len(selected)*6
    assert set(pack)=={b['prompt_id'] for b in branches}
    result={'verified':True,'phase':'inputs verified','train_components':96,'test_components':48,'profiles':len(selected),
            'branch_joins':len(branches),'unique_prompts':len(pack),'exact_group_weight_sums':len(totals),
            'train_test_text_disjoint':True,'excluded_development_cases':len(sel['excluded_dev_component_cases'])}
    if (OUT/'fit_manifest.json').exists():
        fm=read(OUT/'fit_manifest.json')
        for p,h in fm['files'].items():assert sha(ROOT/p)==h
        assert fm['test_generations_existed_at_seal'] is False
        targets=independent_targets('train',branches,weights);meta=read(OUT/'model_metadata.json')
        train=meta['train_ids'];test=meta['test_ids'];assert set(targets)==set(train) and set(train).isdisjoint(test)
        assert train==[g['group_id'] for g in components if g['split']=='train'] and test==[g['group_id'] for g in components if g['split']=='test']
        for gid,v in targets.items():close([float(x) for x in v],read(OUT/'train_targets.json')[gid])
        text={g['group_id']:g['text'] for g in components};cs=[tf(text[i]) for i in train];total=Counter()
        for c in cs:total.update(c)
        vocab=sorted(total,key=lambda k:(-total[k],k))[:2000];assert vocab==meta['features'] and dict(total)==meta['feature_counts']
        df=np.array([sum(t in c for c in cs) for t in vocab]);idf=np.log((1+len(train))/(1+df))+1
        def matrix(ids):
            cs=[tf(text[i]) for i in ids];X=np.array([[c[t]*idf[j] for j,t in enumerate(vocab)] for c in cs]);n=np.linalg.norm(X,axis=1,keepdims=True)
            return np.divide(X,n,out=np.zeros_like(X),where=n!=0)
        X=matrix(train);Xt=matrix(test);Y=np.array([[float(t) for t in targets[i]] for i in train]);xm=X.mean(0);ym=Y.mean(0)
        U,s,Vt=np.linalg.svd(X-xm,full_matrices=False);coef=(Vt.T*(s/(s*s+1)))@U.T@(Y-ym);raw=ym+(Xt-xm)@coef;pred=np.clip(raw,0,1)
        saved=np.load(OUT/'model.npz');close(saved['training_labels'],Y);close(saved['xmean'],xm);close(saved['ymean'],ym);close(saved['Xc'],X-xm)
        close(saved['Xc'].T@saved['dual'],coef)
        predictions=read(OUT/'test_predictions.json');assert [p['case_id'] for p in predictions]==test
        err=0.
        for p,r,v in zip(predictions,raw,pred):
            close(r,p['raw_prediction']);close(v,p['prediction']);close(ym,p['constant_prediction']);err=max(err,float(max(abs(r-np.array(p['raw_prediction'])))))
            assert p['ridge_question']==Q[best(v-C)] and p['train_constant_question']==Q[best(ym-C)]
        for s in shards:
            if s['split']=='test':
                m=read(OUT/'shards'/s['name']/'manifest.json')
                assert m['predictions_sealed'] and m['files'][str((OUT/'test_predictions.json').relative_to(ROOT))]==sha(OUT/'test_predictions.json')
        result.update(phase='sealed predictions verified',training_targets=96*6,test_predictions=48,max_svd_prediction_error=err)
    if (OUT/'summary.json').exists():
        truth=independent_targets('test',branches,weights);summary=read(OUT/'summary.json');rs=read(OUT/'component_results.json')
        assert len(rs)==48 and {r['group_id'] for r in rs}==set(truth)
        predbyid={p['case_id']:p for p in predictions};saved_targets=read(OUT/'test_targets.json')
        for r in rs:
            ys=truth[r['group_id']];p=predbyid[r['group_id']]
            close([float(t) for t in ys],r['target_accuracy']);close([float(t) for t in ys],saved_targets[r['group_id']])
            assert r['metrics']['ridge']['question']==p['ridge_question']
            assert r['metrics']['train_constant']['question']==p['train_constant_question']
            for q in Q:assert r['metrics'][q]['question']==q
            assert r['metrics']['random_question']['question']=='uniform_random'
            assert r['metrics']['component_posthoc_oracle']['question']==Q[best(np.array([float(t) for t in ys])-C)]
            close(np.mean(abs(np.array(p['prediction'])-np.array([float(t) for t in ys]))),r['ridge_mae'])
            close(np.mean(abs(np.array(p['constant_prediction'])-np.array([float(t) for t in ys]))),r['constant_mae'])
        close(np.mean([r['ridge_mae'] for r in rs]),summary['ridge_mae']);close(np.mean([r['constant_mae'] for r in rs]),summary['constant_mae'])
        for method in rs[0]['metrics']:
            acc=F(0);ask=F(0)
            for r in rs:
                q=r['metrics'][method]['question'];ys=truth[r['group_id']]
                a=sum(ys[1:])/5 if q=='uniform_random' else ys[Q.index(q)]
                close(float(a),r['metrics'][method]['accuracy']);close(float(a-F(1,20)*(q!='no_ask')),r['metrics'][method]['utility']);assert r['metrics'][method]['ask_rate']==int(q!='no_ask');acc+=a;ask+=q!='no_ask'
            acc/=48;ask/=48;close(float(acc),summary['methods'][method]['accuracy']);close(float(ask),summary['methods'][method]['ask_rate']);close(float(acc-F(1,20)*ask),summary['methods'][method]['utility'])
            assert summary['methods'][method]['choices']==dict(Counter(r['metrics'][method]['question'] for r in rs))
        rng=np.random.default_rng(20261027);indices=rng.integers(0,48,size=(10000,48))
        for m in ['train_constant','issue']:
            differences=np.array([r['metrics']['ridge']['utility']-r['metrics'][m]['utility'] for r in rs])
            boot=np.array([np.mean(differences[row]) for row in indices])
            close(np.quantile(boot,[.025,.975]),summary['paired_utility_comparisons'][m]['percentile95_interval'])
            close(differences.mean(),summary['paired_utility_comparisons'][m]['difference'])
        for p,h in read(OUT/'score_manifest.json')['files'].items():assert sha(ROOT/p)==h
        result.update(phase='completed',rational_metrics_and_bootstrap_verified=True)
    result['verifier_sha256']=sha(Path(__file__).resolve());savepath=OUT/('verification.json' if result['phase']=='completed' else 'input_verification.json' if result['phase']=='inputs verified' else 'prediction_verification.json')
    savepath.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
