"""Independent E017 provenance, SVD, choices, and exact-weight scoring."""
import hashlib,json
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/encoder_value_e017';SOURCE=ROOT/'results/heldout_value_e016'
Q=['no_ask','issue','product','money_gone','compensation','customer_name'];C=np.array([0,.05,.05,.05,.05,.05])
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(m):
    for p,h in m['files'].items():assert sha(ROOT/p)==h,p
def close(a,b):np.testing.assert_allclose(a,b,atol=1e-12,rtol=0)
def choose(v):
    maximum=max(v);return next(i for i,x in enumerate(v) if maximum-x<=1e-12)
def main():
    m=read(OUT/'manifest.json');check(m);original=read(SOURCE/'selector_inputs.json');inputs=read(OUT/'inputs.json')
    assert [{k:r[k] for k in ['group_id','split','text']} for r in inputs]==original
    assert len(inputs)==144 and all(0<len(r['input_tokens'])<=512 for r in inputs)
    assert m['test_generations_existed_at_freeze'] is False
    result={'verified':True,'phase':'inputs','input_components':144,'no_hidden_labels_in_inputs':all(set(r)=={'group_id','split','text','input_tokens'} for r in inputs)}
    assert result['no_hidden_labels_in_inputs']
    if (OUT/'feature_manifest.json').exists():
        check(read(OUT/'feature_manifest.json'));status=read(OUT/'extraction_status.json');assert status['complete'] and status['error']is None
        rows=[json.loads(l) for l in (OUT/'features.jsonl').read_text().splitlines()]
        assert [r['group_id'] for r in rows]==[r['group_id'] for r in inputs]
        vectors=np.array([r['vector'] for r in rows],dtype=np.float32);assert vectors.shape==(144,2560) and np.isfinite(vectors).all()
        assert np.array_equal(vectors,np.load(OUT/'features.npz')['vectors'])
        result.update(phase='features',feature_dimension=2560)
    if (OUT/'fit_manifest.json').exists():
        fit=read(OUT/'fit_manifest.json');check(fit);assert fit['test_generations_existed_at_seal'] is False
        tr=[i for i,r in enumerate(inputs) if r['split']=='train'];te=[i for i,r in enumerate(inputs) if r['split']=='test']
        meta=read(OUT/'model_metadata.json');assert meta['train_ids']==[inputs[i]['group_id'] for i in tr] and meta['test_ids']==[inputs[i]['group_id'] for i in te]
        targets=read(SOURCE/'train_targets.json');assert set(targets)==set(meta['train_ids']) and set(meta['train_ids']).isdisjoint(meta['test_ids'])
        Xall=vectors.astype(np.float64);norm=np.sqrt(np.sum(Xall*Xall,axis=1));assert (norm>0).all();Xall=np.stack([v/n for v,n in zip(Xall,norm)])
        X=Xall[tr];Xt=Xall[te];Y=np.array([targets[inputs[i]['group_id']] for i in tr]);xm=X.mean(0);ym=Y.mean(0)
        u,s,vt=np.linalg.svd(X-xm,full_matrices=False);beta=(vt.T*(s/(s*s+1)))@u.T@(Y-ym)
        saved=np.load(OUT/'model.npz');close(saved['training_labels'],Y);close(saved['xmean'],xm);close(saved['ymean'],ym);close(saved['Xc'],X-xm);close(saved['Xc'].T@saved['dual'],beta)
        raw=ym+(Xt-xm)@beta;ps=read(OUT/'predictions.json');assert len(ps)==48
        err=0.
        for i,p,r in zip(te,ps,raw):
            assert p['group_id']==inputs[i]['group_id'];close(r,p['raw_prediction']);v=np.clip(r,0,1);close(v,p['prediction']);assert p['question']==Q[choose(v-C)]
            err=max(err,float(max(abs(r-np.array(p['raw_prediction'])))))
        result.update(phase='predictions',train_components=96,test_predictions=48,max_svd_prediction_error=err)
    if (OUT/'test_start_witness.json').exists():
        witness=read(OUT/'test_start_witness.json');check(witness)
        assert witness['test_generations_existed_before_start'] is False and witness['started_unix_time']>=fit['sealed_unix_time']
        result['sealed_before_test_start_witness']=True
    if (OUT/'summary.json').exists():
        check(read(OUT/'score_manifest.json'));assert read(SOURCE/'verification.json')['verified']
        source={r['group_id']:r for r in read(SOURCE/'component_results.json')};rows=read(OUT/'component_results.json');summary=read(OUT/'summary.json')
        assert len(rows)==48 and [r['group_id'] for r in rows]==[p['group_id'] for p in ps]
        accuracies=[];utilities=[];asks=[];maes=[]
        for r,p in zip(rows,ps):
            sr=source[r['group_id']];ys=[F(y).limit_denominator(10000) for y in sr['target_accuracy']];q=p['question'];assert r['question']==q
            a=ys[Q.index(q)];ask=int(q!='no_ask');utility=a-F(1,20)*ask
            close(float(a),r['accuracy']);close(float(utility),r['utility']);assert r['ask_rate']==ask
            mae=np.mean(abs(np.array(p['prediction'])-np.array([float(y) for y in ys])));close(mae,r['mae'])
            for method in ['ridge','train_constant','issue']:assert r['comparators'][method]==sr['metrics'][method]
            accuracies.append(a);utilities.append(utility);asks.append(ask);maes.append(mae)
        for field,values in [('accuracy',accuracies),('utility',utilities),('ask_rate',asks),('mae',maes)]:close(float(sum(values)/48),summary[field])
        assert summary['choices']==dict(Counter(p['question'] for p in ps))
        rng=np.random.default_rng(20261027);ix=rng.integers(0,48,size=(10000,48))
        for method in ['ridge','train_constant','issue']:
            diff=np.array([float(u)-r['comparators'][method]['utility'] for u,r in zip(utilities,rows)]);boots=np.array([diff[i].mean() for i in ix]);pair=summary['paired_comparisons'][method]
            close(diff.mean(),pair['utility_difference']);close(np.percentile(boots,[2.5,97.5]),pair['percentile95_interval'])
            assert pair['improved']==sum(diff>1e-12) and pair['worse']==sum(diff< -1e-12) and pair['tied']==sum(abs(diff)<=1e-12)
        result.update(phase='completed',metrics_and_bootstrap_verified=True)
    result['verifier_sha256']=sha(Path(__file__).resolve());name='verification.json' if result['phase']=='completed' else 'prediction_verification.json' if result['phase']=='predictions' else 'input_verification.json'
    (OUT/name).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
