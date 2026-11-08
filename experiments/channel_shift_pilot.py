"""E001a: exact finite-population diagnostic on external flight catalogs.

No LLM, no access to eval user weights, no external code imported.
Run: python3 experiments/channel_shift_pilot.py --repo /path/to/VOI_communication
"""
import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean

COMMIT = '27466a7832d5aafff82017a659e08942e18b01ae'
FEATURES = [(7,), (6,), (5,), (0,), (1, 2, 3, 4)]
NAMES = ['price', 'stops', 'layover', 'arrival', 'airline_signs']
SOURCE = [.9] * 5
REGIMES = {'unchanged': SOURCE, 'selective': [.2,.9,.5,.8,.9], 'uniform_drop': [.3]*5}
METHODS = ['no_ask', 'random', 'ideal', 'ig_source', 'evsi_source', 'evsi_pooled', 'evsi_adapt', 'ig_oracle', 'evsi_oracle']
ROOT = Path(__file__).resolve().parents[1]


def task_values(options, profiles):
    scores = [[sum(x*w for x,w in zip(option, theta)) for option in options] for theta in profiles]
    n = len(scores)
    best = mean(max(row) for row in scores)
    initial = max(mean(row[a] for row in scores) for a in range(len(options)))
    benefits, information = [], []
    for feature in FEATURES:
        groups = defaultdict(list)
        for theta, row in zip(profiles, scores):
            label = tuple((theta[j] > 0) - (theta[j] < 0) for j in feature)
            groups[label].append(row)
        posterior_value = sum(max(sum(row[a] for row in group) for a in range(len(options))) for group in groups.values())/n
        benefits.append(max(0., posterior_value-initial))
        information.append(-sum((len(group)/n)*math.log2(len(group)/n) for group in groups.values()))
    return best-initial, benefits, information


def observations(rates, n, seed):
    rng = random.Random(seed)
    return [int(rng.random() < rates[i % len(rates)]) for i in range(n)]


def estimate(labels, additional=()):
    k = len(FEATURES)
    return [(1 + sum(labels[q::k]) + sum(additional[q::k]))/
            (2 + len(labels[q::k]) + len(additional[q::k])) for q in range(k)]


def decision(benefits, information, rates, cost, use_ig=False):
    assert len(benefits) == len(information) == len(rates)
    values = [p*v for p,v in zip(rates, benefits)]
    ranking = [p*v for p,v in zip(rates, information)] if use_ig else values
    q = max(range(len(benefits)), key=lambda i: ranking[i])
    return None if cost > 0 and values[q] <= cost else q


def measured_loss(base, benefits, actual_rates, question, cost):
    return base if question is None else base-actual_rates[question]*benefits[question]+cost


def checks():
    # Independent two-bit construction: revealing important/minor has value .45/.05.
    profiles = [[0,0,0,0,0,0, minor, important] for important in [-.9,.9] for minor in [-.1,.1]]
    options = [[0,0,0,0,0,0, minor, important] for important in [0,1] for minor in [0,1]]
    base, benefits, _ = task_values(options, profiles)
    assert math.isclose(base, .5)
    assert math.isclose(benefits[0], .45) and math.isclose(benefits[1], .05)
    assert math.isclose(measured_loss(base, benefits, [.8,1,0,0,0], 0, 0), .14)
    assert measured_loss(base, benefits, [0]*5, 0, 0) == base
    assert decision(benefits, [1]*len(FEATURES), [0]*len(FEATURES), .02) is None
    assert all(0 < p < 1 for p in estimate([1]*50))
    assert estimate([], [1]*50) == estimate([1]*50)
    flat = task_values([[0]*8, [0]*8], profiles)
    assert flat[0] == 0 and flat[1] == [0]*len(FEATURES)


def bootstrap(values):
    rng = random.Random(1589)
    samples = sorted(mean(rng.choices(values, k=len(values))) for _ in range(2000))
    return [samples[50], samples[1949]]


def main():
    global FEATURES, NAMES, SOURCE, REGIMES
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--question-set', choices=['bundled','scalar'], default='bundled')
    args = parser.parse_args()
    if args.question_set == 'scalar':
        FEATURES = [(i,) for i in [7,6,5,0,1,2,3,4]]
        NAMES = ['price','stops','layover','arrival','american','delta','jetblue','southwest']
        SOURCE = [.9]*8
        REGIMES = {'unchanged':SOURCE,'selective':[.2,.9,.5,.8,.9,.9,.9,.9], 'uniform_drop':[.3]*8}
    checks()
    k = len(FEATURES)
    repo = args.repo.resolve()
    commit = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True).strip()
    assert commit == COMMIT
    data_path = repo/'flight_rec/data'
    train = json.loads((data_path/'train.json').read_text())
    evaluation = json.loads((data_path/'eval.json').read_text())
    profiles = [g['reward_function'] for g in train]
    tasks = []
    for i,g in enumerate(evaluation):
        if len(g['rounds']) < 6:
            continue
        options = g['rounds'][5]['options']
        base, benefits, info = task_values(options, profiles)
        assert all(-1e-10 <= v <= base+1e-10 for v in benefits)
        tasks.append({'record': i, 'catalog_sha256': hashlib.sha256(json.dumps(options,sort_keys=True).encode()).hexdigest(),
                      'base': base, 'benefits': benefits, 'information': info})
    assert len({t['catalog_sha256'] for t in tasks}) == len(tasks), 'Group duplicate catalogs before inference'
    records, feedback = [], []
    for regime, actual in REGIMES.items():
        for seed in range(30):
            source_labels = observations(SOURCE, 500, seed+10000)
            target_labels = observations(actual, 200, seed+20000)
            source_estimate = estimate(source_labels)
            feedback.append({'regime': regime, 'seed': seed, 'source': source_labels, 'target': target_labels})
            for budget in [0,50,200]:
                labels = target_labels[:budget]
                adapted = estimate(labels) if budget else source_estimate
                pooled = estimate(source_labels, labels)
                for cost in [0.,.02]:
                    for task in tasks:
                        base, benefits, info = task['base'], task['benefits'], task['information']
                        rate_sets = {'ideal':[1]*k, 'ig_source':source_estimate, 'evsi_source':source_estimate,
                                     'evsi_pooled':pooled, 'evsi_adapt':adapted, 'ig_oracle':actual, 'evsi_oracle':actual}
                        oracle_q = decision(benefits, info, actual, cost)
                        reference = measured_loss(base, benefits, actual, oracle_q, cost)
                        for method in METHODS:
                            if method == 'random':
                                loss = mean(measured_loss(base, benefits, actual, q, cost) for q in range(k)); ask=1; q='uniform'
                            else:
                                q = None if method == 'no_ask' else decision(benefits, info, rate_sets[method], cost, method.startswith('ig_'))
                                loss = measured_loss(base, benefits, actual, q, cost); ask=int(q is not None)
                            assert loss >= reference-1e-10
                            records.append({'regime':regime,'seed':seed,'budget':budget,'cost':cost,'task':task['record'],
                                            'method':method,'loss':loss,'gap':loss-reference,'ask':ask,'question':q})
    out = ROOT/('results/channel_shift_pilot' if args.question_set=='bundled' else 'results/channel_shift_pilot_scalar')
    out.mkdir(parents=True, exist_ok=True)
    with (out/'per_task.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    grouped = defaultdict(list)
    for r in records:
        grouped[(r['regime'],r['budget'],r['cost'],r['method'])].append(r)
    summary = []
    for key, rows in grouped.items():
        by_seed,by_task=defaultdict(list),defaultdict(list)
        for r in rows:
            by_seed[r['seed']].append(r['gap']);by_task[r['task']].append(r['gap'])
        seed_gaps = [mean(v) for v in by_seed.values()]
        summary.append(dict(zip(['regime','budget','cost','method'],key), loss=mean(r['loss'] for r in rows),
                            gap=mean(r['gap'] for r in rows),ask_rate=mean(r['ask'] for r in rows),
                            gap_seed_minmax=[min(seed_gaps),max(seed_gaps)],
                            gap_task_bootstrap_ci95=bootstrap([mean(v) for v in by_task.values()])))
    result={'status':'synthetic_channel_exact_executor_external_catalogs_not_llm', 'checks':'passed',
            'source_commit':commit,'question_set':args.question_set,'profiles':len(profiles),'catalogs':len(tasks),'regimes':REGIMES,
            'questions':NAMES,'summary':summary,'tasks':tasks,
            'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [data_path/'train.json',data_path/'eval.json']},
            'limitations':['Known empirical prior and exact executor','Artificial goal-independent erasures',
                           'No eval user weights used','Adaptation cost reported in interactions, not amortized into task loss',
                           'Bootstrap conditions on empirical profile population; seeds are not independent tasks',
                           'No direct value head or factorized neural model tested']}
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    (out/'feedback.json').write_text(json.dumps(feedback))
    print('Checks passed; catalogs:',len(tasks),'profiles:',len(profiles),'rows:',len(records))
    for r in summary:
        if r['regime']=='selective' and r['cost']==0 and r['method'] in ['no_ask','ideal','evsi_source','evsi_adapt','evsi_pooled','evsi_oracle']:
            print(r['budget'],r['method'],'loss',round(r['loss'],6),'gap',round(r['gap'],6))


if __name__=='__main__':
    main()
