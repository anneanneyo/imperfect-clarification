"""Independent train-only features, SVD ridge and rational scoring checks."""
from collections import Counter
from fractions import Fraction as F
import hashlib,json,re
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/text_value_e015';SRC=ROOT/'results/lavoir_executor_e014'
Q=['no_ask','issue','product','money_gone','compensation','customer_name'];C=np.array([0,.05,.05,.05,.05,.05])
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)
def tf(text):
    words=[m.group() for m in re.finditer('[a-z]+|[0-9]+',text.lower())];out=Counter()
    for index,w in enumerate(words):
        out['u:'+w]+=1
        if index:out['b:'+words[index-1]+' '+w]+=1
    return out
def best(v):
    m=max(v)
    return next(i for i,x in enumerate(v) if x>=m-1e-12)

def main():
    for p,h in read(OUT/'manifest.json')['files'].items():assert sha(ROOT/p)==h,p
    texts=read(OUT/'texts.json');targets=read(OUT/'targets.json');groups=read(OUT/'groups.json')
    ds=read(SRC/'details.json');truth={}
    for case in texts:
        truth[case]=[]
        for q in Q:
            bs=[b for b in ds if b['case_id']==case and b['question']==q]
            truth[case].append(F(sum(b['correct'] for b in bs),len(bs)))
        close([float(t) for t in truth[case]],targets[case])
    predictions=read(OUT/'predictions.json');bycase={p['case_id']:p for p in predictions}
    assert set(bycase)==set(texts) and len(bycase)==len(predictions)
    fit=read(OUT/'fit_manifest.json');assert sha(OUT/'predictions.json')==fit['predictions_sha256']
    largest_error=0.;feature_checks=0;mutation_checks=0
    # Primary implementation is used only for the explicit test-label contamination challenge.
    import text_value_e015 as primary
    for fm in fit['folds']:
        folder=OUT/'folds'/f"{fm['fold']:02d}"
        assert sha(folder/'metadata.json')==fm['metadata_sha256'] and sha(folder/'model.npz')==fm['model_sha256']
        meta=read(folder/'metadata.json');train=meta['train_ids'];test=meta['test_ids']
        assert set(train).isdisjoint(test) and set(train)|set(test)==set(texts)
        assert set(test)==set(groups[meta['group']])
        assert set(texts[i] for i in train).isdisjoint(texts[i] for i in test)
        cs=[tf(texts[i]) for i in train];total=Counter()
        for c in cs:total+=c
        vocab=[k for k,v in sorted(total.items(),key=lambda kv:(-kv[1],kv[0]))[:2000]]
        assert vocab==meta['features'] and dict(total)==meta['feature_counts']
        df=np.array([sum(c[t]>0 for c in cs) for t in vocab]);idf=np.log((1+len(train))/(1+df))+1
        def design(ids):
            mat=np.array([[tf(texts[i])[t]*idf[k] for k,t in enumerate(vocab)] for i in ids])
            norm=np.sqrt(np.sum(mat**2,axis=1,keepdims=True));return np.divide(mat,norm,out=np.zeros_like(mat),where=norm!=0)
        X=design(train);Xt=design(test);Y=np.array([[float(t) for t in truth[i]] for i in train])
        xm=X.mean(0);ym=Y.mean(0);xc=X-xm;yc=Y-ym
        U,s,Vt=np.linalg.svd(xc,full_matrices=False)
        coef=(Vt.T*(s/(s*s+1.)))@U.T@yc
        raw=ym+(Xt-xm)@coef;pred=np.clip(raw,0,1)
        model=np.load(folder/'model.npz')
        for key,val in [('df',df),('idf',idf),('Xc',xc),('xmean',xm),('ymean',ym),('Xtest',Xt),('training_labels',Y)]:close(model[key],val)
        for i,r,p in zip(test,raw,pred):
            recorded=bycase[i];err=max(abs(r-np.array(recorded['raw_prediction'])));largest_error=max(largest_error,float(err));close(p,recorded['prediction'])
            assert recorded['ridge_question']==Q[best(p-C)]
            assert recorded['train_constant_question']==Q[best(ym-C)];close(recorded['constant_prediction'],ym)
        feature_checks+=len(vocab)
        corrupt={k:list(v) for k,v in targets.items()}
        for i in test:corrupt[i]=[99,-99,88,-88,77,-77]
        mutated,_=primary.predict_fold(texts,corrupt,train,test)
        for p in mutated:
            close(p['prediction'],bycase[p['case_id']]['prediction']);assert p['ridge_question']==bycase[p['case_id']]['ridge_question'];mutation_checks+=1
    result={'verified':True,'phase':'predictions verified','folds':len(fit['folds']),'source_target_values':len(truth)*6,
            'vocabulary_entries_checked':feature_checks,'max_svd_prediction_error':largest_error,'heldout_label_mutation_checks':mutation_checks,
            'train_only_preprocessing':True,'groups_disjoint':True}
    if (OUT/'summary.json').exists():
        summary=read(OUT/'summary.json');cases=read(OUT/'case_results.json');excluded=set(read(OUT/'exclusions.json'))
        for scope,cs in [('all_cases',cases),('predeclared_sensitivity',[r for r in cases if r['case_id']not in excluded])]:
            for method in cs[0]['metrics']:
                acc=F(0);ask=F(0)
                for case in cs:
                    q=case['metrics'][method]['question'];t=truth[case['case_id']]
                    a=sum(t[1:])/5 if q=='uniform_random' else t[Q.index(q)]
                    close(float(a),case['metrics'][method]['accuracy']);acc+=a;ask+=q!='no_ask'
                acc/=len(cs);ask/=len(cs)
                close(float(acc),summary[scope]['methods'][method]['accuracy'])
                close(float(acc-F(1,20)*ask),summary[scope]['methods'][method]['utility'])
            for method,pkey in [('ridge','prediction'),('constant','constant_prediction')]:
                mae=sum(sum(abs(a-float(t)) for a,t in zip(bycase[c['case_id']][pkey],truth[c['case_id']]))/6 for c in cs)/len(cs)
                close(mae,summary[scope][method+'_mae'])
        for p,h in read(OUT/'score_manifest.json')['files'].items():assert sha(ROOT/p)==h
        result.update(phase='completed',rational_metric_checks=True)
    result['verifier_sha256']=sha(Path(__file__).resolve())
    savepath=OUT/('verification.json' if result['phase']=='completed' else 'prediction_verification.json')
    savepath.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
