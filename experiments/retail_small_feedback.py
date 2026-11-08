"""E012: learn only observable response-type frequencies; no neural claim."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from retail_clarification_contract import feature, kernel, QUESTIONS, digest

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL_SHA='2507321fe44eca547634be23500377a2435a585c790a30c7c432a98143976d1b'
BUDGETS=[0,16,64,256,1024]
COST=.05
CATEGORIES={q:['answer','idk'] for q in QUESTIONS}
CATEGORIES['contents']=['full','partial','idk']

def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def category(reply):
    return {'idk':'idk','full_names':'full','one_name':'partial'}.get(reply[0],'answer')

def coefficients(features):
    """Known observation shapes and uniform prior; no response-type rates."""
    n=len(features)
    names=[set(f['names']) for f in features]
    partial=sum(max(1/len(s) if name in s else 0 for s in names) for name in set().union(*names))/n
    return {
        'order_id':{'answer':1.,'idk':1/n},
        'total':{'answer':len({f['total_cents'] for f in features})/n,'idk':1/n},
        'count':{'answer':len({f['count'] for f in features})/n,'idk':1/n},
        'contents':{'full':len({tuple(f['names']) for f in features})/n,'partial':partial,'idk':1/n}}

def fit(observations):
    counts={q:Counter({k:1 for k in ks}) for q,ks in CATEGORIES.items()}
    for r in observations:
        assert set(r)=={'step','family_hash','question','question_propensity','reply'}
        q=r['question'];k=category(r['reply']);assert k in counts[q]
        counts[q][k]+=1
    return {q:{k:c[k]/sum(c.values()) for k in CATEGORIES[q]} for q,c in counts.items()}

def predicted_values(coef, estimated):
    return {q:sum(estimated[q][k]*coef[q][k] for k in CATEGORIES[q]) for q in QUESTIONS}

def choose(values, base):
    choice='no_ask';best=base
    for q in QUESTIONS:
        v=values[q]-COST
        if v>best+1e-12:choice,best=q,v
    return choice

def sample_categorical(rng, dist):
    u=rng.random();total=0
    for value,p in dist.items():
        total+=p
        if u<total:return value
    return next(reversed(dist))

def evaluate(test, choices):
    accuracy=sum(f['base'] if q=='no_ask' else f['truth'][q] for f,q in zip(test,choices))/len(test)
    ask=sum(q!='no_ask' for q in choices)/len(test)
    return {'accuracy':accuracy,'utility':accuracy-COST*ask,'ask_rate':ask,'question_counts':dict(Counter(choices))}

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'results/retail_small_feedback');a=p.parse_args()
    assert hashlib.sha256((ROOT/'research/E012_PROTOCOL.md').read_bytes()).hexdigest()==PROTOCOL_SHA
    a.output.mkdir(exist_ok=True,parents=True)
    old=json.loads((ROOT/'results/retail_clarification_contract/families.json').read_text())
    raw_path=a.repo/'data/tau2/domains/retail/db.json'
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest()=='413a65160adbdb5fde0ffc0015c49b6d70250b10c18128de169b597af7766765'
    raw=json.loads(raw_path.read_text());grouped=defaultdict(list)
    for order in raw['orders'].values():
        if order['status']=='pending':grouped[digest(order['user_id'])].append(order)
    families=[]
    for r in old:
        fs=[feature(o) for o in sorted(grouped[r['family_hash']],key=lambda x:x['order_id'])]
        coef=coefficients(fs)
        truth={q:r['question_results']['imperfect'][q]['accuracy'] for q in QUESTIONS}
        families.append({'family_hash':r['family_hash'],'split':r['split_label'],'features':fs,'coefficients':coef,'base':1/len(fs),'truth':truth})
    train=[f for f in families if f['split']=='development']
    test=[f for f in families if f['split']=='reserved']
    assert len(train)==60 and len(test)==20 and not ({f['family_hash'] for f in train}&{f['family_hash'] for f in test})
    # Evaluator-only rates for validating the algebra, never passed to fit.
    truth_rates={'order_id':{'answer':.35,'idk':.65},'total':{'answer':.6,'idk':.4},
                 'count':{'answer':.9,'idk':.1},'contents':{'full':.45,'partial':.45,'idk':.1}}
    decomposition_errors=[]
    for f in families:
        computed=predicted_values(f['coefficients'],truth_rates)
        decomposition_errors.extend(abs(computed[q]-f['truth'][q]) for q in QUESTIONS)
    assert max(decomposition_errors)<1e-12
    controls={m:evaluate(test,choices) for m,choices in {
        'no_ask':['no_ask']*len(test),'always_contents':['contents']*len(test),
        'oracle':[choose(f['truth'],f['base']) for f in test]}.items()}
    rows=[];log_hashes={};goal_exposures=0
    for seed in range(20261100,20261130):
        rng=random.Random(seed);log=[]
        for i in range(1024):
            f=rng.choice(train);g=rng.randrange(len(f['features']));q=rng.choice(QUESTIONS)
            reply=sample_categorical(rng,kernel(f['features'][g],q,'imperfect'))
            log.append({'step':i,'family_hash':f['family_hash'],'question':q,'question_propensity':.25,'reply':list(reply)})
        filename=f'log_{seed}.jsonl';path=a.output/filename
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in log))
        log_hashes[filename]=hashlib.sha256(path.read_bytes()).hexdigest()
        for budget in BUDGETS:
            estimated=fit(log[:budget])
            assert all(abs(sum(d.values())-1)<1e-12 for d in estimated.values())
            preds=[predicted_values(f['coefficients'],estimated) for f in test]
            contextual=[choose(v,f['base']) for v,f in zip(preds,test)]
            train_preds=[predicted_values(f['coefficients'],estimated) for f in train]
            fixed=choose({q:sum(v[q] for v in train_preds)/len(train) for q in QUESTIONS},sum(f['base'] for f in train)/len(train))
            mae=sum(abs(v[q]-f['truth'][q]) for f,v in zip(test,preds) for q in QUESTIONS)/(len(test)*len(QUESTIONS))
            for method,choices in [('learned_contextual',contextual),('learned_fixed',[fixed]*len(test))]:
                score=evaluate(test,choices)
                assert score['utility']<=controls['oracle']['utility']+1e-12
                rows.append({'seed':seed,'budget':budget,'method':method,'fitted_probabilities':estimated,'value_mae':mae,
                             'oracle_gap':controls['oracle']['utility']-score['utility'],**score})
    aggregate=[]
    for budget in BUDGETS:
        for method in ['learned_contextual','learned_fixed']:
            rs=[r for r in rows if r['budget']==budget and r['method']==method]
            aggregate.append({'budget':budget,'method':method,'seeds':len(rs),
                              **{key:{'mean':statistics.mean(r[key] for r in rs),'sd':statistics.stdev(r[key] for r in rs),
                                      'min':min(r[key] for r in rs),'max':max(r[key] for r in rs)}
                                 for key in ['accuracy','utility','oracle_gap','value_mae']}})
    summary={'protocol_sha256':PROTOCOL_SHA,'train_families':60,'evaluation_families':20,'seeds':30,
             'observations_per_full_log':1024,'free_channel_parameters':5,'decomposition_cases':len(decomposition_errors),
             'max_decomposition_error':max(decomposition_errors),'controls':controls,'aggregate':aggregate,
             'feedback':'Only observed question and reply category; no latent goal, counterfactual or terminal reward supplied to fit.',
             'known_privileges':['Uniform prior','All candidate order features','Exact within-category response shape R','Decision execution contract'],
             'limitations':['Artificial fixed channel','Previously analyzed development evaluation families','Not an H3 or neural architecture comparison',
                            'Variation across seeds measures only training-log sampling','No full dialogue or human validation'],
             'new_external_model_calls':0}
    write(a.output/'results.json',rows);write(a.output/'summary.json',summary)
    write(a.output/'manifest.json',{'protocol_sha256':PROTOCOL_SHA,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'source_db_sha256':hashlib.sha256(raw_path.read_bytes()).hexdigest(),
          'e011_families_sha256':hashlib.sha256((ROOT/'results/retail_clarification_contract/families.json').read_bytes()).hexdigest(),
          'log_hashes':log_hashes,'output_hashes':{n:hashlib.sha256((a.output/n).read_bytes()).hexdigest() for n in ['results.json','summary.json']}})
    print(json.dumps({'controls':controls,'aggregate':aggregate,'decomposition_error':max(decomposition_errors)},indent=2))

if __name__=='__main__':main()
