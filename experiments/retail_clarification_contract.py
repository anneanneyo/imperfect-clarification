"""E011: exact one-turn calculations and isolated retail cancellation checks.

This is a constructed task on a pinned external database, not an official tau
benchmark, a human-response model, or a trained selector evaluation.
"""
import argparse
from collections import Counter, defaultdict
import copy
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
from pathlib import Path

from tau3_retail_state_probe import load_domain, digest, COMMIT, HASHES

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA = 'fe08fb60590527f5a5c146712ca654f33751c83412cbfa2a71ade7973348ef95'
QUESTIONS = ['order_id', 'contents', 'total', 'count']
COSTS = [0.0, 0.02, 0.05, 0.10]
IDK = ('idk',)

def save(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def entropy(p):
    return -sum(x * math.log2(x) for x in p if x > 0)

def feature(order):
    total = sum((Decimal(str(i['price'])) for i in order['items']), Decimal(0))
    return {'order_id': order['order_id'],
            'names': sorted(i['name'] for i in order['items']),
            'total_cents': int((total * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP)),
            'count': len(order['items'])}

def kernel(f, q, mode):
    if q == 'contents':
        full = ('full_names', *f['names'])
        if mode == 'perfect':
            return {full: 1.0}
        names = sorted(set(f['names']))
        assert names, 'No names available for partial response'
        return {full: 0.45, IDK: 0.10,
                **{('one_name', name): 0.45 / len(names) for name in names}}
    key, value, rate = {
        'order_id': ('order_id', f['order_id'], 0.35),
        'total': ('total_cents', f['total_cents'], 0.60),
        'count': ('count', f['count'], 0.90),
    }[q]
    if mode == 'perfect':
        return {(key, value): 1.0}
    return {(key, value): rate, IDK: 1.0 - rate}

def evaluate_question(features, question, mode):
    n = len(features)
    distributions = [kernel(f, question, mode) for f in features]
    for d in distributions:
        assert all(p >= 0 for p in d.values()) and abs(sum(d.values()) - 1) < 1e-12
    observations = sorted(set().union(*(d.keys() for d in distributions)), key=repr)
    expected_accuracy = conditional_entropy = naive_accuracy = 0.0
    diagnostic = []
    for y in observations:
        joint = [d.get(y, 0.0) / n for d in distributions]
        mass = sum(joint)
        assert mass > 0
        posterior = [j / mass for j in joint]
        # The observation likelihood is checked against the revealed facts.
        for f, j in zip(features, joint):
            if j == 0 or y == IDK:
                continue
            if y[0] == 'one_name': assert y[1] in f['names']
            if y[0] == 'full_names': assert tuple(f['names']) == y[1:]
            if y[0] == 'order_id': assert f['order_id'] == y[1]
            if y[0] == 'count': assert f['count'] == y[1]
            if y[0] == 'total_cents': assert f['total_cents'] == y[1]
        best = max(range(n), key=lambda i: posterior[i])
        expected_accuracy += joint[best]
        conditional_entropy += mass * entropy(posterior)
        if y[0] == 'one_name':
            compatible = [i for i, j in enumerate(joint) if j > 0]
            naive = compatible[0]
            naive_posterior = [1 / len(compatible) if i in compatible else 0 for i in range(n)]
            error = max(abs(a-b) for a,b in zip(posterior, naive_posterior))
            if error > 1e-12:
                diagnostic.append({'observation': list(y), 'mass':mass, 'posterior':posterior,
                                   'uniform_support_posterior':naive_posterior,
                                   'max_probability_error':error, 'bayes_choice':best,
                                   'naive_choice':naive, 'accuracy_loss_contribution':joint[best]-joint[naive]})
        else:
            naive = best
        naive_accuracy += joint[naive]
    assert -1e-12 <= expected_accuracy <= 1+1e-12
    return {'accuracy':expected_accuracy, 'information_gain_bits':math.log2(n)-conditional_entropy,
            'uniform_partial_accuracy':naive_accuracy, 'partial_diagnostics':diagnostic}

def select_metrics(question_results, n, cost):
    noask = 1/n
    evsi = 'no_ask'
    best = noask
    for q in QUESTIONS:
        v = question_results[q]['accuracy']-cost
        if v > best+1e-12:
            evsi, best = q, v
    hq = max(QUESTIONS, key=lambda q: question_results[q]['information_gain_bits'])
    if question_results[hq]['information_gain_bits'] < 1e-12:
        hq = 'no_ask'
    selected = {'no_ask':'no_ask', 'always_id':'order_id', 'entropy':hq, 'oracle_evsi':evsi}
    rows = {}
    for method,q in selected.items():
        accuracy = noask if q == 'no_ask' else question_results[q]['accuracy']
        rows[method] = {'question':q,'accuracy':accuracy,'ask_rate':float(q!='no_ask'),
                        'utility':accuracy-cost*(q!='no_ask')}
    a = sum(question_results[q]['accuracy'] for q in QUESTIONS)/len(QUESTIONS)
    rows['random_question'] = {'question':'uniform_random','accuracy':a,'ask_rate':1.0,'utility':a-cost}
    assert all(rows['oracle_evsi']['utility']+1e-12 >= r['utility'] for r in rows.values())
    return rows

def verify_execution(db_class, tools_class, raw, order_id):
    db = db_class.model_validate(copy.deepcopy(raw))
    before = db.model_dump()
    expected = copy.deepcopy(before)
    target = expected['orders'][order_id]
    checks = {'source_pending':target['status']=='pending',
              'only_original_payments':all(p['transaction_type']=='payment' for p in target['payment_history'])}
    original = copy.deepcopy(target['payment_history'])
    target['status'] = 'cancelled'
    target['cancel_reason'] = 'no longer needed'
    target['payment_history'].extend({**p, 'transaction_type':'refund'} for p in original)
    user = expected['users'][target['user_id']]
    deltas = defaultdict(Decimal)
    for payment in original:
        method = user['payment_methods'][payment['payment_method_id']]
        if method['source']=='gift_card':
            deltas[payment['payment_method_id']] += Decimal(str(payment['amount']))
    for method_id, amount in deltas.items():
        method = user['payment_methods'][method_id]
        method['balance'] = float((Decimal(str(method['balance']))+amount).quantize(Decimal('.01'), rounding=ROUND_HALF_UP))
    error = None
    try:
        tools_class(db).use_tool('cancel_pending_order', order_id=order_id, reason='no longer needed')
    except Exception as e:
        error = {'type':type(e).__name__, 'message':str(e)}
    after = db.model_dump()
    checks['independent_full_state_matches'] = after == expected
    checks['target_changed'] = before['orders'][order_id] != after['orders'][order_id]
    checks['all_other_orders_unchanged'] = all(v==after['orders'][k] for k,v in before['orders'].items() if k!=order_id)
    checks['products_unchanged'] = before['products']==after['products']
    return {'order_hash':digest(order_id),'checks':checks, 'error':error,
            'success':error is None and all(checks.values()),'after_state_hash':digest(after),
            'gift_card_refund':bool(deltas)}

def analytical_checks():
    fs = [{'order_id':'a','names':['alpha','beta'],'total_cents':200,'count':2},
          {'order_id':'b','names':['alpha'],'total_cents':100,'count':1}]
    r = evaluate_question(fs,'contents','imperfect')
    # Hand-derived: full .45 + partial-alpha .225 + partial-beta .1125 + IDK .05.
    assert math.isclose(r['accuracy'], .8375, abs_tol=1e-12)
    assert math.isclose(r['uniform_partial_accuracy'], .725, abs_tol=1e-12)
    d = next(d for d in r['partial_diagnostics'] if d['observation']==['one_name','alpha'])
    assert all(math.isclose(x,y,abs_tol=1e-12) for x,y in zip(d['posterior'],[1/3,2/3]))
    return {'known_example_accuracy':r['accuracy'],'known_example_naive_accuracy':r['uniform_partial_accuracy'],
            'posterior_for_alpha':d['posterior']}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'results/retail_clarification_contract')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    assert hashlib.sha256((ROOT/'research/E011_PROTOCOL.md').read_bytes()).hexdigest()==PROTOCOL_SHA
    tests=analytical_checks()
    db_class,tools_class,extracted=load_domain(args.repo)
    raw=json.loads((args.repo/'data/tau2/domains/retail/db.json').read_text())
    byuser=defaultdict(list)
    for oid,order in raw['orders'].items():
        if order['status']=='pending':byuser[order['user_id']].append(oid)
    users=sorted((u for u,ids in byuser.items() if len(ids)>=2),
                 key=lambda u:hashlib.sha256(('E011-v1:'+u).encode()).hexdigest())
    assert len(users)==100
    records=[];execution=[];examples=[]
    for index,u in enumerate(users):
        ids=sorted(byuser[u]);fs=[feature(raw['orders'][i]) for i in ids]
        family=digest(u);n=len(ids)
        bymode={mode:{q:evaluate_question(fs,q,mode) for q in QUESTIONS} for mode in ['perfect','imperfect']}
        assert math.isclose(bymode['perfect']['order_id']['accuracy'],1,abs_tol=1e-12)
        for q in QUESTIONS:
            assert bymode['imperfect'][q]['accuracy'] <= bymode['perfect'][q]['accuracy']+1e-12
        record={'family_hash':family,'n_orders':n,'split_label':['development','validation','reserved'][0 if index<60 else 1 if index<80 else 2],
                'question_results':bymode,'policy_results':[]}
        for mode,qr in bymode.items():
            for cost in COSTS:
                record['policy_results'].append({'mode':mode,'cost':cost,'methods':select_metrics(qr,n,cost)})
        records.append(record)
        for oid in ids:execution.append({'family_hash':family,**verify_execution(db_class,tools_class,raw,oid)})
        ds=bymode['imperfect']['contents']['partial_diagnostics']
        if ds:
            examples.append({'family_hash':family,'orders':[{'local_index':i,'names':f['names'],'count':f['count']} for i,f in enumerate(fs)],
                             'diagnostics':ds})
        if (index+1)%20==0:print(f'Completed {index+1}/{len(users)} families',flush=True)
    tables=[]
    for mode in ['perfect','imperfect']:
        for cost in COSTS:
            vals=[next(r for r in f['policy_results'] if r['mode']==mode and r['cost']==cost)['methods'] for f in records]
            for method in ['no_ask','always_id','random_question','entropy','oracle_evsi']:
                tables.append({'mode':mode,'cost':cost,'method':method,
                               **{k:sum(v[method][k] for v in vals)/len(vals) for k in ['accuracy','utility','ask_rate']},
                               'question_counts':dict(Counter(v[method]['question'] for v in vals))})
    ds=[d for f in records for d in f['question_results']['imperfect']['contents']['partial_diagnostics']]
    summary={'source_commit':COMMIT,'protocol_sha256':PROTOCOL_SHA,'families':len(records),
             'hidden_goals':len(execution),'all_execution_postconditions_pass':all(r['success'] for r in execution),
             'execution_failures':sum(not r['success'] for r in execution),'gift_card_refund_goals':sum(r['gift_card_refund'] for r in execution),
             'macro_weighting':'Equal families, uniform goals within each family','analytical_checks':tests,'table':tables,
             'partial_support_diagnostics':{'families_nonuniform_posterior':len(examples),'observations_nonuniform_posterior':len(ds),
                 'max_posterior_error':max([d['max_probability_error'] for d in ds],default=0),
                 'families_action_loss':sum(any(d['accuracy_loss_contribution']>1e-12 for d in f['question_results']['imperfect']['contents']['partial_diagnostics']) for f in records),
                 'macro_contents_accuracy_loss_uniform_partial':sum(f['question_results']['imperfect']['contents']['accuracy']-f['question_results']['imperfect']['contents']['uniform_partial_accuracy'] for f in records)/len(records)},
             'limitations':['Constructed task and artificial known response kernel, no empirical human or LLM response rates.',
                 'Oracle EVSI is not learned. No neural method or H3 evaluation.',
                 'Only decision and state component; mandatory dialogue confirmation and official tau task rewards not evaluated.',
                 'Reserved families were included in this engineering run and are not a sealed final test.'],
             'new_external_model_calls':0}
    save(args.output/'families.json',records);save(args.output/'execution.json',execution)
    save(args.output/'partial_examples.json',examples);save(args.output/'summary.json',summary)
    save(args.output/'manifest.json',{'commit':COMMIT,'source_hashes':HASHES,'extracted_ast':extracted,
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'protocol_sha256':PROTOCOL_SHA,
        'outputs':{name:hashlib.sha256((args.output/name).read_bytes()).hexdigest() for name in ['families.json','execution.json','partial_examples.json','summary.json']}})
    print(json.dumps({k:v for k,v in summary.items() if k!='table'},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
