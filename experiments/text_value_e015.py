"""Visible-text multi-output ridge baseline; fit/predict and scoring are stages."""
import argparse
from collections import Counter,defaultdict
import hashlib,json,math,re,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/text_value_e015';SOURCE=ROOT/'results/lavoir_executor_e014'
Q=['no_ask','issue','product','money_gone','compensation','customer_name'];COST=np.array([0,.05,.05,.05,.05,.05]);ALPHA=1.
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def read(p):return json.loads(p.read_text())
def text_hash(t):return hashlib.sha256(t.encode()).hexdigest()
def stable_best(values):
    best=0
    for i,x in enumerate(values[1:],1):
        if x>values[best]+1e-12:best=i
    return best
def counts(text):
    tokens=re.findall(r'[a-z]+|[0-9]+',text.lower())
    return Counter(['u:'+t for t in tokens]+['b:'+a+' '+b for a,b in zip(tokens,tokens[1:])])
def vectorize(counter,vocab,idf):
    x=np.array([counter.get(term,0) for term in vocab],dtype=float)*idf
    norm=np.linalg.norm(x)
    return x/norm if norm else x

def predict_fold(texts,targets,train_ids,test_ids):
    # Test target values are never indexed, passed to the model, or used for preprocessing.
    train_counts=[counts(texts[i]) for i in train_ids];totals=Counter()
    for c in train_counts:totals.update(c)
    vocab=sorted(totals,key=lambda t:(-totals[t],t))[:2000]
    df=np.array([sum(t in c for c in train_counts) for t in vocab])
    idf=np.log((1+len(train_ids))/(1+df))+1
    X=np.array([vectorize(c,vocab,idf) for c in train_counts])
    Y=np.array([targets[i] for i in train_ids],dtype=float)
    xmean=X.mean(axis=0);ymean=Y.mean(axis=0);Xc=X-xmean;Yc=Y-ymean
    dual=np.linalg.solve(Xc@Xc.T+ALPHA*np.eye(len(train_ids)),Yc)
    Xt=np.array([vectorize(counts(texts[i]),vocab,idf) for i in test_ids])
    raw=ymean+(Xt-xmean)@Xc.T@dual;clipped=np.clip(raw,0,1)
    constant=stable_best(ymean-COST)
    predictions=[{'case_id':cid,'raw_prediction':r.tolist(),'prediction':p.tolist(),
                  'ridge_question':Q[stable_best(p-COST)],'train_constant_question':Q[constant],
                  'constant_prediction':ymean.tolist()} for cid,r,p in zip(test_ids,raw,clipped)]
    return predictions,{'vocab':vocab,'df':df,'idf':idf,'Xc':Xc,'xmean':xmean,'ymean':ymean,'dual':dual,'Xtest':Xt,
                        'training_labels':Y,'Yc':Yc,'feature_counts':dict(totals)}

def prepare():
    assert not OUT.exists();OUT.mkdir()
    for path,h in read(SOURCE/'analysis_manifest.json')['files'].items():assert sha(ROOT/path)==h
    rows=read(SOURCE/'source_rows.json');details=read(SOURCE/'details.json');selection=read(SOURCE/'selection.json')
    texts={};targets={};groups=defaultdict(list);profiles=defaultdict(list)
    for r in rows:profiles['/'.join(r['id'].split('/')[:3])].append(r)
    for case in selection['cases']:
        ps=profiles[case];ts={p['state'][0]['text'] for p in ps};assert len(ts)==1
        texts[case]=ts.pop();groups[text_hash(texts[case])].append(case)
        target=[]
        for q in Q:
            ds=[d for d in details if d['case_id']==case and d['question']==q]
            assert len(ds)==len(ps) and all(not d['missing'] for d in ds)
            target.append(sum(d['correct'] for d in ds)/len(ps))
        targets[case]=target
    save(OUT/'texts.json',texts);save(OUT/'targets.json',targets);save(OUT/'groups.json',dict(groups))
    exclusions=read(SOURCE/'semantic_review.json')['sensitivity_excluded_case_ids'];save(OUT/'exclusions.json',exclusions)
    paths=[ROOT/'research/E015_PROTOCOL.md',Path(__file__).resolve(),SOURCE/'details.json',SOURCE/'source_rows.json',SOURCE/'semantic_review.json']+[OUT/n for n in ['texts.json','targets.json','groups.json','exclusions.json']]
    save(OUT/'manifest.json',{'files':{str(p.relative_to(ROOT)):sha(p) for p in paths},'python':sys.version,'numpy':np.__version__,
         'alpha':ALPHA,'cost':COST.tolist(),'questions':Q,'feature_limit':2000,'timing':'Frozen before E015 fit; E014 results already inspected.',
         'cases':len(texts),'outer_groups':len(groups),'branch_labels':len(details),'new_model_generations':0})
    print(json.dumps({'cases':len(texts),'outer_groups':len(groups),'branch_labels':len(details),'frozen':True}))

def verify_manifest():
    for p,h in read(OUT/'manifest.json')['files'].items():assert sha(ROOT/p)==h,p

def fit():
    verify_manifest();assert not (OUT/'predictions.json').exists();start=time.monotonic()
    texts=read(OUT/'texts.json');targets=read(OUT/'targets.json');groups=read(OUT/'groups.json');allids=list(texts)
    (OUT/'folds').mkdir();predictions=[];folds=[]
    for index,(group,testids) in enumerate(groups.items()):
        trainids=[i for i in allids if i not in testids];assert not set(trainids)&set(testids)
        pred,model=predict_fold(texts,targets,trainids,testids)
        meta={'fold':index,'group':group,'train_ids':trainids,'test_ids':testids,'features':model.pop('vocab'),
              'feature_counts':model.pop('feature_counts'),'train_cases':len(trainids)}
        fdir=OUT/'folds'/f'{index:02d}';fdir.mkdir()
        save(fdir/'metadata.json',meta);np.savez_compressed(fdir/'model.npz',**model)
        for p in pred:p['fold']=index
        predictions.extend(pred);folds.append({'fold':index,'metadata_sha256':sha(fdir/'metadata.json'),'model_sha256':sha(fdir/'model.npz')})
    # No test scoring happens in this stage.
    save(OUT/'predictions.json',predictions)
    save(OUT/'fit_manifest.json',{'folds':folds,'predictions_sha256':sha(OUT/'predictions.json'),'elapsed_seconds':time.monotonic()-start,
                                'stage':'training and predictions only; no held-out scoring performed'})
    print(json.dumps({'folds':len(folds),'predictions':len(predictions),'elapsed_seconds':time.monotonic()-start}))

def score():
    verify_manifest();assert not (OUT/'summary.json').exists()
    predictions=read(OUT/'predictions.json');assert sha(OUT/'predictions.json')==read(OUT/'fit_manifest.json')['predictions_sha256']
    targets=read(OUT/'targets.json');excluded=set(read(OUT/'exclusions.json'));results=[]
    for p in predictions:
        ys=targets[p['case_id']];actions={'ridge':p['ridge_question'],'train_constant':p['train_constant_question'],**{q:q for q in Q}}
        metrics={}
        for method,q in actions.items():
            a=ys[Q.index(q)];metrics[method]={'question':q,'accuracy':a,'utility':a-.05*(q!='no_ask'),'ask_rate':int(q!='no_ask')}
        metrics['random_question']={'question':'uniform_random','accuracy':sum(ys[1:])/5,'utility':sum(ys[1:])/5-.05,'ask_rate':1}
        oracle=Q[stable_best(np.array(ys)-COST)];a=ys[Q.index(oracle)]
        metrics['case_posthoc_oracle']={'question':oracle,'accuracy':a,'utility':a-.05*(oracle!='no_ask'),'ask_rate':int(oracle!='no_ask')}
        results.append({'case_id':p['case_id'],'fold':p['fold'],'target_accuracy':ys,'metrics':metrics,
                        'ridge_mae':float(np.mean(np.abs(np.array(p['prediction'])-ys))),
                        'constant_mae':float(np.mean(np.abs(np.array(p['constant_prediction'])-ys))),
                        'sensitivity_excluded':p['case_id'] in excluded})
    summary={}
    for name,rows in [('all_cases',results),('predeclared_sensitivity',[r for r in results if not r['sensitivity_excluded']])]:
        methods={}
        for method in rows[0]['metrics']:
            ms=[r['metrics'][method] for r in rows]
            methods[method]={key:sum(m[key] for m in ms)/len(ms) for key in ['accuracy','utility','ask_rate']}
            methods[method]['choices']=dict(Counter(m['question'] for m in ms))
        diffs=[r['metrics']['ridge']['utility']-r['metrics']['train_constant']['utility'] for r in rows]
        summary[name]={'cases':len(rows),'methods':methods,'ridge_mae':sum(r['ridge_mae'] for r in rows)/len(rows),
                       'constant_mae':sum(r['constant_mae'] for r in rows)/len(rows),
                       'ridge_minus_train_constant_utility':sum(diffs)/len(diffs),
                       'paired_improved_cases':sum(d>1e-12 for d in diffs),'paired_worse_cases':sum(d< -1e-12 for d in diffs),
                       'paired_tied_cases':sum(abs(d)<=1e-12 for d in diffs)}
    save(OUT/'case_results.json',results);save(OUT/'summary.json',summary)
    save(OUT/'score_manifest.json',{'files':{str((OUT/n).relative_to(ROOT)):sha(OUT/n) for n in ['manifest.json','fit_manifest.json','predictions.json','case_results.json','summary.json']}})
    print(json.dumps(summary))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','fit','score']);globals()[p.parse_args().stage]()
