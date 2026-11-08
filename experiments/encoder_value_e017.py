"""Frozen last-token representation baseline, sealed before E016 test inference."""
import argparse,hashlib,json,sys,time
from collections import Counter
from pathlib import Path
import numpy as np
import text_value_e015 as baseline

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/encoder_value_e017';SOURCE=ROOT/'results/heldout_value_e016'
MODEL=Path('/Users/annagubareva/.cache/clarification-research/models/qwen3-4b-instruct-2507-4bit')
Q=baseline.Q;C=baseline.COST
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def check_files(m):
    for p,h in m['files'].items():assert sha(ROOT/p)==h,p
def unopened():
    for s in read(SOURCE/'shards.json'):
        if s['split']=='test':
            p=SOURCE/'shards'/s['name'];assert not (p/'generations.jsonl').exists() and not (p/'started.jsonl').exists()
def verify():check_files(read(OUT/'manifest.json'))
def freeze():
    from transformers import AutoTokenizer
    import mlx_lm.models.qwen3 as qwen3
    import importlib.metadata as md
    unopened();assert not OUT.exists();check_files(read(SOURCE/'manifest.json'))
    sm=read(SOURCE/'manifest.json')
    for n,h in sm['model_files'].items():assert sha(MODEL/n)==h
    tok=AutoTokenizer.from_pretrained(str(MODEL),trust_remote_code=False,local_files_only=True)
    records=[]
    for r in read(SOURCE/'selector_inputs.json'):
        ids=tok.encode(r['text'],add_special_tokens=False);assert 0<len(ids)<=512
        records.append({**r,'input_tokens':ids})
    OUT.mkdir();save(OUT/'inputs.json',records)
    paths=[ROOT/'research/E017_PROTOCOL.md',Path(__file__).resolve(),Path(baseline.__file__).resolve(),SOURCE/'manifest.json',SOURCE/'selector_inputs.json',OUT/'inputs.json']
    save(OUT/'manifest.json',{'files':{str(p.relative_to(ROOT)):sha(p) for p in paths},
        'model':sm['model'],'revision':sm['revision'],'model_files':sm['model_files'],
        'qwen3_module':str(Path(qwen3.__file__).resolve()),'qwen3_module_sha256':sha(Path(qwen3.__file__)),
        'mlx_lm_version':md.version('mlx-lm'),'python':sys.version,'numpy':np.__version__,
        'pooling':'last token after final normalization; raw normalized text; no chat template',
        'dimension':2560,'alpha':1.,'cost':C.tolist(),'seed':20261027,'records':len(records),
        'input_tokens':sum(len(r['input_tokens']) for r in records),'max_input_tokens':max(len(r['input_tokens']) for r in records),
        'test_generations_existed_at_freeze':False})
    print(json.dumps({'frozen':True,'records':len(records),'max_tokens':max(len(r['input_tokens']) for r in records)}))

def extract():
    import mlx.core as mx
    from mlx_lm import load
    import mlx_lm.models.qwen3 as qwen3
    verify();unopened();m=read(OUT/'manifest.json')
    assert sha(Path(qwen3.__file__))==m['qwen3_module_sha256']
    for n,h in m['model_files'].items():assert sha(MODEL/n)==h
    assert not (OUT/'features.jsonl').exists() and not (OUT/'extraction_started.json').exists()
    save(OUT/'extraction_started.json',{'time_unix':time.time(),'test_generations_exist':False})
    start=time.monotonic();mx.random.seed(20261027)
    model,tok=load(str(MODEL),tokenizer_config={'trust_remote_code':False});model.eval()
    vectors=[];records=read(OUT/'inputs.json');error=None
    with (OUT/'features.jsonl').open('w') as f:
        for i,r in enumerate(records):
            try:
                if time.monotonic()-start>600:raise TimeoutError('Frozen extraction time budget reached')
                assert tok.encode(r['text'],add_special_tokens=False)==r['input_tokens']
                hidden=model.model(mx.array([r['input_tokens']]))
                vector=hidden[0,-1,:].astype(mx.float32);mx.eval(vector)
                arr=np.array(vector,dtype=np.float32);assert arr.shape==(2560,) and np.isfinite(arr).all() and np.linalg.norm(arr)>0
                f.write(json.dumps({'group_id':r['group_id'],'vector':arr.tolist()})+'\n');f.flush();vectors.append(arr)
                del hidden,vector;mx.clear_cache()
                if (i+1)%12==0:print(json.dumps({'extracted':i+1,'total':len(records),'elapsed':round(time.monotonic()-start,2)}),flush=True)
            except Exception as e:
                error=repr(e);break
    complete=len(vectors)==len(records)
    save(OUT/'extraction_status.json',{'complete':complete,'vectors':len(vectors),'error':error,
        'elapsed_seconds':time.monotonic()-start,'peak_memory_gb':mx.get_peak_memory()/1e9,
        'input_tokens_completed':sum(len(r['input_tokens']) for r in records[:len(vectors)]),'new_decoded_tokens':0})
    assert complete,error
    np.savez_compressed(OUT/'features.npz',vectors=np.array(vectors))
    paths=[OUT/n for n in ['manifest.json','features.jsonl','features.npz','extraction_started.json','extraction_status.json']]
    save(OUT/'feature_manifest.json',{'files':{str(p.relative_to(ROOT)):sha(p) for p in paths}})
    print(json.dumps(read(OUT/'extraction_status.json')))

def fit():
    verify();unopened();check_files(read(OUT/'feature_manifest.json'));check_files(read(SOURCE/'fit_manifest.json'))
    assert not (OUT/'fit_manifest.json').exists()
    records=read(OUT/'inputs.json');train=[i for i,r in enumerate(records) if r['split']=='train'];test=[i for i,r in enumerate(records) if r['split']=='test']
    target=read(SOURCE/'train_targets.json');assert set(target)=={records[i]['group_id'] for i in train}
    features=np.load(OUT/'features.npz')['vectors'].astype(np.float64);features/=np.linalg.norm(features,axis=1,keepdims=True)
    X=features[train];Xt=features[test];Y=np.array([target[records[i]['group_id']] for i in train])
    xm=X.mean(0);ym=Y.mean(0);Xc=X-xm;dual=np.linalg.solve(Xc@Xc.T+np.eye(len(train)),Y-ym)
    raw=ym+(Xt-xm)@Xc.T@dual;clipped=np.clip(raw,0,1)
    ps=[{'group_id':records[i]['group_id'],'raw_prediction':r.tolist(),'prediction':p.tolist(),'question':Q[baseline.stable_best(p-C)]} for i,r,p in zip(test,raw,clipped)]
    save(OUT/'predictions.json',ps);save(OUT/'model_metadata.json',{'train_ids':[records[i]['group_id'] for i in train],'test_ids':[records[i]['group_id'] for i in test],'alpha':1.,'feature_dimension':2560})
    np.savez_compressed(OUT/'model.npz',Xc=Xc,xmean=xm,ymean=ym,dual=dual,training_labels=Y)
    paths=[OUT/n for n in ['manifest.json','feature_manifest.json','predictions.json','model.npz','model_metadata.json']]+[SOURCE/'train_targets.json',SOURCE/'fit_manifest.json']
    save(OUT/'fit_manifest.json',{'files':{str(p.relative_to(ROOT)):sha(p) for p in paths},'test_generations_existed_at_seal':False,'sealed_unix_time':time.time()})
    print(json.dumps({'sealed_predictions':len(ps),'choices':dict(Counter(p['question'] for p in ps))}))

def score():
    verify();check_files(read(OUT/'feature_manifest.json'));check_files(read(OUT/'fit_manifest.json'));check_files(read(SOURCE/'score_manifest.json'))
    assert not (OUT/'summary.json').exists()
    source={r['group_id']:r for r in read(SOURCE/'component_results.json')};pred=read(OUT/'predictions.json');rows=[]
    for p in pred:
        r=source[p['group_id']];ys=r['target_accuracy'];q=p['question'];acc=ys[Q.index(q)]
        rows.append({'group_id':p['group_id'],'question':q,'accuracy':acc,'utility':acc-.05*(q!='no_ask'),
            'ask_rate':int(q!='no_ask'),'mae':float(np.abs(np.array(p['prediction'])-ys).mean()),
            'comparators':{m:r['metrics'][m] for m in ['ridge','train_constant','issue']}})
    summary={k:sum(r[k] for r in rows)/len(rows) for k in ['accuracy','utility','ask_rate','mae']}
    summary.update(components=len(rows),choices=dict(Counter(r['question'] for r in rows)),additional_encoder_cost=read(OUT/'extraction_status.json'))
    rng=np.random.default_rng(20261027);ix=rng.integers(0,len(rows),size=(10000,len(rows)));pairs={}
    for m in ['ridge','train_constant','issue']:
        diff=np.array([r['utility']-r['comparators'][m]['utility'] for r in rows]);ci=np.quantile(diff[ix].mean(1),[.025,.975])
        pairs[m]={'utility_difference':float(diff.mean()),'percentile95_interval':ci.tolist(),'improved':int((diff>1e-12).sum()),'worse':int((diff< -1e-12).sum()),'tied':int((abs(diff)<=1e-12).sum())}
    summary['paired_comparisons']=pairs;save(OUT/'component_results.json',rows);save(OUT/'summary.json',summary)
    paths=[OUT/n for n in ['manifest.json','feature_manifest.json','fit_manifest.json','component_results.json','summary.json']]+[SOURCE/'score_manifest.json']
    save(OUT/'score_manifest.json',{'files':{str(p.relative_to(ROOT)):sha(p) for p in paths}})
    print(json.dumps(summary))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','extract','fit','score']);globals()[p.parse_args().stage]()
