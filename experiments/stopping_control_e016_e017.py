"""Exploratory question-choice versus stopping decomposition, sealed before scores."""
import argparse,hashlib,json,time
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'results/heldout_value_e016';B=ROOT/'results/encoder_value_e017';OUT=ROOT/'results/e016_e017_stopping_control'
Q=['no_ask','issue','product','money_gone','compensation','customer_name']
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze():
    assert not (A/'summary.json').exists() and not (B/'summary.json').exists() and not OUT.exists()
    a=read(A/'test_predictions.json');b={p['group_id']:p for p in read(B/'predictions.json')};rows=[]
    for p in a:
        gid=p['case_id'];constant=p['train_constant_question'];assert constant!='no_ask'
        for method,chosen,pred in [('lexical',p['ridge_question'],p['prediction']),('encoder',b[gid]['question'],b[gid]['prediction'])]:
            v=pred[1:];maximum=max(v);best=next(i for i,x in enumerate(v) if maximum-x<=1e-12)
            rows.append({'group_id':gid,'method':method,'full':chosen,'constant':constant,
                'matched_stop_constant':'no_ask' if chosen=='no_ask' else constant,'always_ask':Q[1+best]})
    OUT.mkdir();save(OUT/'policies.json',rows)
    paths=[Path(__file__).resolve(),A/'test_predictions.json',B/'predictions.json',OUT/'policies.json']
    save(OUT/'manifest.json',{'files':{str(p.relative_to(ROOT)):sha(p) for p in paths},'sealed_unix_time':time.time(),
        'test_generations_had_started':True,'primary_scores_existed':False,'status':'Exploratory diagnostic added during test inference, before aggregate outcomes were inspected; not an original confirmatory contrast.',
        'scope':'Same stopping decisions and ask rate as each frozen selector, replace every asked question with the training-chosen constant. Also compare forced best predicted question.'})
    print(json.dumps({'frozen_policies':len(rows),'test_in_progress':True,'scores_exist':False}))
def score():
    m=read(OUT/'manifest.json')
    for p,h in m['files'].items():assert sha(ROOT/p)==h
    assert read(A/'verification.json')['verified'] and read(B/'verification.json')['verified'];assert not (OUT/'summary.json').exists()
    truth=read(A/'test_targets.json');rows=read(OUT/'policies.json');results={};details=[]
    for method in ['lexical','encoder']:
        selected=[r for r in rows if r['method']==method];assert len(selected)==48;metrics={}
        for policy in ['full','constant','matched_stop_constant','always_ask']:
            acc=[];ask=[]
            for r in selected:
                a=F(truth[r['group_id']][Q.index(r[policy])]).limit_denominator(10000);asking=int(r[policy]!='no_ask');acc.append(a);ask.append(asking)
                details.append({'group_id':r['group_id'],'method':method,'policy':policy,'question':r[policy],'accuracy':float(a),'utility':float(a-F(1,20)*asking),'ask_rate':asking})
            metrics[policy]={'accuracy':float(sum(acc)/48),'utility':float((sum(acc)-F(1,20)*sum(ask))/48),'ask_rate':sum(ask)/48,
                'choices':dict(Counter(r[policy] for r in selected))}
        assert metrics['full']['ask_rate']==metrics['matched_stop_constant']['ask_rate']
        expected=read(A/'summary.json')['methods']['ridge'] if method=='lexical' else read(B/'summary.json')
        for field in ['accuracy','utility','ask_rate']:assert abs(metrics['full'][field]-expected[field])<1e-12
        gain=metrics['full']['utility']-metrics['constant']['utility'];stop=metrics['matched_stop_constant']['utility']-metrics['constant']['utility'];choice=metrics['full']['utility']-metrics['matched_stop_constant']['utility'];assert abs(gain-stop-choice)<1e-12
        rng=np.random.default_rng(20261027);ix=rng.integers(0,48,size=(10000,48));pairs={}
        lookup={(d['group_id'],d['policy']):d for d in details if d['method']==method}
        for alternative in ['matched_stop_constant','always_ask']:
            d=np.array([lookup[r['group_id'],'full']['utility']-lookup[r['group_id'],alternative]['utility'] for r in selected]);ci=np.quantile(d[ix].mean(axis=1),[.025,.975])
            pairs[alternative]={'difference':float(d.mean()),'percentile95_interval':ci.tolist(),'improved':int((d>1e-12).sum()),'worse':int((d< -1e-12).sum()),'tied':int((abs(d)<=1e-12).sum())}
        results[method]={'metrics':metrics,'utility_gain_over_constant':gain,'stopping_component':stop,'question_choice_component':choice,'paired':pairs}
    save(OUT/'details.json',details);save(OUT/'summary.json',{'interpretation':m['status'],'methods':results,'same_ask_rate_and_gain_decomposition_verified':True})
    paths=[OUT/n for n in ['manifest.json','policies.json','details.json','summary.json']]+[A/'score_manifest.json',B/'score_manifest.json']
    save(OUT/'score_manifest.json',{'files':{str(p.relative_to(ROOT)):sha(p) for p in paths}});print(json.dumps(results))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','score']);globals()[p.parse_args().stage]()
