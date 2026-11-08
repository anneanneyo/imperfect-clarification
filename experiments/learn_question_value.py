"""E002: supervised question value in a structured partial-answer simulator.
Requires numpy. No LLM calls. See research/E002_PROTOCOL.md.
"""
import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
COMMIT='27466a7832d5aafff82017a659e08942e18b01ae'
INDICES=[7,6,5,0,1,2,3,4]
BASE=np.array([.4,.7,.5,.75,.6,.6,.6,.6])


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def catalog_id(options):
    return digest(sorted(options))


def response_channel(signs, exact, ambiguous):
    # Columns: -, 0, +, {-0}, {-+}, {0+}, unknown.
    c=np.zeros((len(signs),7))
    c[np.arange(len(signs)),signs+1]=exact
    for j,pair in enumerate([(-1,0),(-1,1),(0,1)]):
        c[:,3+j]=np.isin(signs,pair)*ambiguous/2
    c[:,6]=1-exact-ambiguous
    assert np.all(c>=-1e-12) and np.allclose(c.sum(axis=1),1)
    return c


def gain(scores, channel):
    # Equal prior over profile records; sum joint utility, then choose per response.
    return (channel.T @ scores).max(axis=1).sum()/len(scores)-scores.mean(axis=0).max()


def features(ideal, span, gap, spread, q):
    local=np.array([1,ideal,span,ideal*span,ideal*span*span,gap,spread,ideal*gap])
    x=np.zeros(8*len(local));x[q*len(local):(q+1)*len(local)]=local
    return x


def make_task(record, profiles):
    options=np.array(record['options'])
    scores=profiles @ options.T
    means=scores.mean(axis=0)
    initial=means.max()
    risk=scores.max(axis=1).mean()-initial
    ideal=[];target=[];discard=[];xs=[];rates=[]
    span=np.ptp(options,axis=0)
    margin=np.sort(means)[-1]-np.sort(means)[-2]
    for q,index in enumerate(INDICES):
        signs=np.sign(profiles[:,index]).astype(int)
        full=gain(scores,response_channel(signs,1.,0.))
        exact=BASE[q]-.15*span[index];ambiguous=.2+.1*span[index]
        actual=gain(scores,response_channel(signs,exact,ambiguous))
        c=response_channel(signs,exact,ambiguous)
        c[:,6]+=c[:,3:6].sum(axis=1);c[:,3:6]=0
        dropped=gain(scores,c)
        assert -1e-10 <= dropped <= actual+1e-10 <= full+2e-10 <= risk+3e-10
        ideal.append(max(0.,full));target.append(max(0.,actual));discard.append(max(0.,dropped))
        xs.append(features(max(0.,full),span[index],margin,float(np.linalg.norm(span)),q))
        rates.append([float(exact),float(ambiguous)])
    return {'id':record['id'],'catalog_hash':catalog_id(record['options']),'risk':float(risk),
            'ideal':ideal,'target':target,'discard':discard,'x':np.array(xs),'rates':rates}


def tests():
    scores=np.eye(3);signs=np.array([-1,0,1])
    assert np.isclose(gain(scores,response_channel(signs,1,0)),2/3)
    assert np.isclose(gain(scores,response_channel(signs,0,1)),1/6)
    assert np.isclose(gain(scores,response_channel(signs,0,0)),0)
    c=response_channel(signs,.4,.3)
    assert np.isclose(gain(scores,c),.4*2/3+.3/6)
    assert np.isclose(gain(scores,c),gain(scores[:,[2,0,1]],c))


def choose_loss(tasks,predictions,cost=0., actual_key='target'):
    values=np.array([t[actual_key] for t in tasks]);q=predictions.argmax(axis=1)
    ask=np.ones(len(tasks),dtype=bool) if cost==0 else predictions.max(axis=1)>cost
    return np.array([t['risk'] for t in tasks])-values[np.arange(len(tasks)),q]*ask+cost*ask, ask


def ci(values):
    rng=np.random.default_rng(20261019)
    estimates=values[rng.integers(0,len(values),size=(4000,len(values)))].mean(axis=1)
    return [float(v) for v in np.quantile(estimates,[.025,.975])]


def main():
    tests()
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);args=p.parse_args()
    repo=args.repo.resolve()
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==COMMIT
    base=repo/'flight_rec/data'
    train=json.loads((base/'train.json').read_text());evaluation=json.loads((base/'eval.json').read_text())
    keys=sorted({tuple(g['reward_function']) for g in train});random.Random(20261019).shuffle(keys)
    fit_keys=set(keys[:int(.8*len(keys))])
    groups={'fit':[],'validation':[],'holdout':[]};seen=set();excluded=[]
    # Deduplicate across all partitions in fit -> validation -> holdout priority.
    for split in groups:
        records=train if split!='holdout' else evaluation
        for i,g in enumerate(records):
            if split!='holdout' and ((tuple(g['reward_function']) in fit_keys)!=(split=='fit')):continue
            options=g['rounds'][-1]['options'];key=catalog_id(options)
            if key in seen:excluded.append({'split':split,'record':i});continue
            seen.add(key);groups[split].append({'id':i,'options':options})
    profile_records=[i for i,g in enumerate(train) if tuple(g['reward_function']) in fit_keys]
    profiles=np.array([train[i]['reward_function'] for i in profile_records])
    tasks={s:[make_task(g,profiles) for g in records] for s,records in groups.items()}
    def arrays(split):
        ts=tasks[split]
        return np.concatenate([t['x'] for t in ts]),np.array([t['target'] for t in ts]),np.array([t['ideal'] for t in ts])
    X,y,ideal=arrays('fit');V,vy,vi=arrays('validation');H,hy,hi=arrays('holdout')
    center=X.mean(axis=0);scale=X.std(axis=0);scale[scale<1e-12]=1
    Z=(X-center)/scale;Z=np.column_stack([np.ones(len(Z)),Z])
    def transform(x):return np.column_stack([np.ones(len(x)),(x-center)/scale])
    def predict(x,weights,upper):return np.clip((transform(x)@weights).reshape(-1,8),0,upper)
    gram=Z.T@Z;rhs=Z.T@y.ravel();penalty=np.eye(Z.shape[1]);penalty[0,0]=0
    trials=[];weights=[]
    for alpha in [.01,.1,1.,10.,100.]:
        w=np.linalg.solve(gram+alpha*penalty,rhs);weights.append(w)
        pred=predict(V,w,vi);loss,_=choose_loss(tasks['validation'],pred)
        trials.append({'alpha':alpha,'validation_loss':float(loss.mean())})
    selected=min(range(len(trials)),key=lambda i:trials[i]['validation_loss']);w=weights[selected]
    slopes=np.clip((ideal*y).sum(axis=0)/np.maximum((ideal*ideal).sum(axis=0),1e-15),0,1)
    predictions={'ideal':hi,'calibrated_simple':hi*slopes,'ridge':predict(H,w,hi),
                 'oracle':hy,'oracle_discard_ambiguity':np.array([t['discard'] for t in tasks['holdout']])}
    results=[];per_task=[];contrasts=[]
    for cost in [0.,.02]:
        losses={}
        for name,pred in predictions.items():
            loss,ask=choose_loss(tasks['holdout'],pred,cost,'discard' if name=='oracle_discard_ambiguity' else 'target')
            losses[name]=loss
            results.append({'cost':cost,'method':name,'loss':float(loss.mean()),'ask_rate':float(ask.mean()),
                            'value_mse':float(((pred-hy)**2).mean())})
            for task,pr,l,a in zip(tasks['holdout'],pred,loss,ask):
                per_task.append({'task':task['id'],'method':name,'cost':cost,'question':int(pr.argmax()) if a else None,'loss':float(l)})
        risks=np.array([t['risk'] for t in tasks['holdout']])
        for name,loss,ask in [('no_ask',risks,0.),('random',risks-hy.mean(axis=1)+cost,1.)]:
            losses[name]=loss;results.append({'cost':cost,'method':name,'loss':float(loss.mean()),'ask_rate':ask})
        for name,loss in losses.items():
            assert np.all(loss>=losses['oracle']-1e-10)
        for other in ['ideal','calibrated_simple','oracle']:
            delta=losses['ridge']-losses[other]
            contrasts.append({'cost':cost,'contrast':'ridge minus '+other,'mean':float(delta.mean()),'ci95':ci(delta)})
    out=ROOT/'results/learn_question_value';out.mkdir(parents=True,exist_ok=True)
    np.savez(out/'model.npz',weights=w,center=center,scale=scale,slopes=slopes)
    result={'status':'trained_structured_baseline_not_llm_or_novel_method','checks':'passed','commit':COMMIT,
            'split_counts':{s:len(t) for s,t in tasks.items()},'population_records':len(profiles),
            'excluded_duplicates':excluded,'selected_alpha':trials[selected]['alpha'],'validation_trials':trials,
            'metrics':results,'paired_contrasts':contrasts,'numpy_version':np.__version__,
            'input_sha256':{name:hashlib.sha256((base/name).read_bytes()).hexdigest() for name in ['train.json','eval.json']},
            'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    (out/'splits.json').write_text(json.dumps({'tasks':{s:[{'record':t['id'],'hash':t['catalog_hash']} for t in ts] for s,ts in tasks.items()},'prior_fit_record_ids':profile_records},indent=2))
    (out/'per_task.json').write_text(json.dumps(per_task,indent=2))
    dataset={s:[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in t.items()} for t in ts] for s,ts in tasks.items()}
    (out/'training_data.json').write_text(json.dumps(dataset))
    assert np.allclose(predict(H,np.load(out/'model.npz')['weights'],hi),predictions['ridge'])
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
