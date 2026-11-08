"""Exhaustive pre-inference validation of the finite E019 rule contract."""
from collections import Counter
from fractions import Fraction
import itertools
import json
from pathlib import Path
import structured_facts_e019 as exp

ROOT = exp.ROOT
OUT = ROOT / 'results/e019_contract_audit'
schema = exp.read(ROOT / 'results/heldout_value_e016/schema.json')

def generic_route(w):
    for rule in schema['rules']:
        if 'else' in rule: return rule['else']
        if all(w[k] in allowed for k, allowed in rule['if'].items()): return rule['unit']
    raise AssertionError('No matching route')

def prior(w):
    p = Fraction(1)
    for key in exp.DOMAINS:
        spec = schema['slots'][key]
        probs = spec['values'] if 'values' in spec else spec['cond'][w[spec['depends_on']]]
        p *= Fraction(str(probs.get(w[key], 0)))
    return p

def nonempty_subsets(a):
    return [list(c) for n in range(1, len(a) + 1) for c in itertools.combinations(a, n)]

assert len(exp.WORLDS) == 80
assert sum(map(prior, exp.WORLDS)) == 1
assert all(exp.route(w) == generic_route(w) for w in exp.WORLDS)
all_fact_states = 0
for values in itertools.product(*(nonempty_subsets(v) for v in exp.DOMAINS.values())):
    facts = dict(zip(exp.DOMAINS, values))
    reference = sorted({generic_route(dict(zip(facts, world))) for world in itertools.product(*values)})
    assert exp.routes(facts) == reference
    assert exp.parse(json.dumps(facts)) == exp.normalized(facts)
    all_fact_states += 1
assert all_fact_states == 4185
valid = json.dumps(exp.DOMAINS)
assert exp.parse(valid) == exp.DOMAINS
invalids = ['', 'null', '[]', '```json\n' + valid + '\n```', valid + ' extra',
    valid[:-1] + ',"issue":["lost_card"]}', json.dumps({**exp.DOMAINS, 'extra': []})]
for key in exp.DOMAINS:
    for value in [[], 'yes', None, [True], ['not_a_value'], [exp.DOMAINS[key][0]] * 2]:
        invalids.append(json.dumps({**exp.DOMAINS, key: value}))
for raw in invalids: assert exp.parse(raw) is None, raw

cases = exp.cases(); details = []; mismatches = []
for c in cases:
    worlds = [w for w in exp.WORLDS if all(w[k] in c['facts'][k] for k in exp.DOMAINS)]
    prior_routes = sorted({generic_route(w) for w in worlds if prior(w) > 0})
    assert c['posterior_support'] == prior_routes
    assert set(prior_routes) <= set(c['routes'])
    assert all(g in c['routes'] for g in c['gold'])
    if prior_routes != c['routes']: mismatches.append(c['case_id'])
    details.append({'dataset': c['dataset'], 'case_id': c['case_id'], 'facts': c['facts'],
                    'all80_routes': c['routes'], 'positive_prior_routes': prior_routes})
summary = {'full_worlds': 80, 'positive_prior_worlds': sum(prior(w) > 0 for w in exp.WORLDS),
           'prior_mass_exact': '1', 'all_partial_fact_states_verified': all_fact_states,
           'invalid_parser_cases_verified': len(invalids), 'source_cases': len(cases),
           'all80_vs_source_support_mismatches': mismatches, 'datasets': {}}
for label in ['E014_dev', 'E016_train']:
    cs = [c for c in cases if c['dataset'] == label]
    summary['datasets'][label] = {'cases': len(cs), 'all80_singleton_routes': sum(len(c['routes']) == 1 for c in cs),
        'source_prior_singleton_routes': sum(len(c['posterior_support']) == 1 for c in cs),
        'constrained_slots': sum(len(c['facts'][k]) < len(exp.DOMAINS[k]) for c in cs for k in exp.DOMAINS),
        'partial_slots': sum(1 < len(c['facts'][k]) < len(exp.DOMAINS[k]) for c in cs for k in exp.DOMAINS),
        'unknown_slots': sum(len(c['facts'][k]) == len(exp.DOMAINS[k]) for c in cs for k in exp.DOMAINS),
        'support_size_histogram': dict(Counter(len(c['routes']) for c in cs))}
assert not OUT.exists(), 'Contract audit is not overwritten'
OUT.mkdir(); exp.save(OUT / 'details.json', details); exp.save(OUT / 'summary.json', summary)
paths = [Path(__file__).resolve(), Path(exp.__file__).resolve(), ROOT / 'results/heldout_value_e016/schema.json', OUT / 'details.json', OUT / 'summary.json']
exp.save(OUT / 'manifest.json', {'files': {str(p.relative_to(ROOT)): exp.base.sha(p) for p in paths}})
print(json.dumps(summary))
