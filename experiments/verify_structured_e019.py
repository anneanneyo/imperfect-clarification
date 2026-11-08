"""Independent verification of E019 source joins, outputs and finite-set scores."""
import ast
from collections import defaultdict
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/structured_facts_e019'
DOMAINS = {'issue': ['unrecognized_charge', 'lost_card', 'fee_question', 'change_request', 'failed_payment'],
           'product': ['card', 'current_account', 'loan', 'savings'], 'money_gone': ['yes', 'no'], 'compensation': ['yes', 'no']}

def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def verify_files(m):
    for name, h in m['files'].items(): assert sha(ROOT / name) == h, name

def unique_object(pairs):
    assert len(pairs) == len(dict(pairs))
    return dict(pairs)

def decode(raw):
    try:
        obj = json.loads(raw, object_pairs_hook=unique_object)
        assert isinstance(obj, dict) and set(obj) == set(DOMAINS)
        for k, choices in DOMAINS.items():
            assert isinstance(obj[k], list) and len(obj[k]) > 0
            assert all(isinstance(v, str) and v in choices for v in obj[k])
            assert len(obj[k]) == len(set(obj[k]))
        return {k: [v for v in choices if v in obj[k]] for k, choices in DOMAINS.items()}
    except (AssertionError, ValueError, TypeError): return None

schema = read(ROOT / 'results/heldout_value_e016/schema.json')
def route(w):
    for rule in schema['rules']:
        if 'else' in rule: return rule['else']
        if all(w[k] in choices for k, choices in rule['if'].items()): return rule['unit']
    raise AssertionError('No rule')

def route_set(facts):
    return sorted({route(dict(zip(DOMAINS, row))) for row in itertools.product(*(facts[k] for k in DOMAINS))})

manifest = read(OUT / 'manifest.json'); verify_files(manifest)
verify_files(read(ROOT / 'results/e019_contract_audit/manifest.json'))
assert manifest['E016_test_used'] is False and manifest['unique_prompts'] == 120
assert manifest['max_tokens'] == 192 and manifest['budget_seconds'] == 1800
assert manifest['gate'] == {'train_invalid_max': 0, 'train_slot_exact_min': .9, 'train_singleton_route_correct_min': .9}
module = ast.parse((ROOT / 'experiments/structured_facts_e019.py').read_text())
system = next(ast.literal_eval(n.value) for n in module.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'SYSTEM' for t in n.targets))
cs = read(OUT / 'cases.json'); ps = read(OUT / 'pack.json'); prompts = {p['id']: p['messages'] for p in ps}
assert len(cs) == len(ps) == len(prompts) == 120
assert len({c['case_id'] for c in cs}) == 120
expected = {}
for label, folder, filename, count in [('E014_dev', 'lavoir_executor_e014', 'details.json', 24),
                                      ('E016_train', 'heldout_value_e016', 'train_details.json', 96)]:
    base = ROOT / 'results' / folder
    rows = {r['id']: r for r in read(base / 'source_rows.json')}
    grouped = defaultdict(list)
    for d in read(base / filename):
        if d['question'] == 'no_ask': grouped[d['case_id']].append(d)
    assert len(grouped) == count
    for cid, ds in grouped.items():
        rs = [rows[d['source_id']] for d in ds]
        facts = {k: [v for v in choices if v in rs[0]['evidence'].get(k, choices)] for k, choices in DOMAINS.items()}
        assert all({k: [v for v in choices if v in r['evidence'].get(k, choices)] for k, choices in DOMAINS.items()} == facts for r in rs)
        assert len({r['state'][0]['text'] for r in rs}) == 1
        expected[cid] = {'dataset': label, 'case_id': cid, 'text': rs[0]['state'][0]['text'],
                         'source_ids': [r['id'] for r in rs], 'facts': facts, 'routes': route_set(facts),
                         'posterior_support': sorted(k for k, p in rs[0]['target'].items() if p > 0),
                         'old_teams': [d['team'] for d in ds], 'gold': [r['gold'] for r in rs]}
assert {c['case_id'] for c in cs} == set(expected)
for c in cs:
    assert {k: v for k, v in c.items() if k != 'prompt_id'} == expected[c['case_id']]
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': c['text']}]
    assert prompts[c['prompt_id']] == messages
    digest = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert digest == c['prompt_id']

completed = (OUT / 'summary.json').exists()
extras = {}
if completed:
    verify_files(read(OUT / 'score_manifest.json'))
    assert read(OUT / 'run_status.json')['complete']
    raw = [json.loads(l) for l in (OUT / 'generations.jsonl').read_text().splitlines()]
    gs = {g['id']: g for g in raw}; assert len(raw) == len(gs) == 120 and set(gs) == set(prompts)
    ds = read(OUT / 'details.json'); assert len(ds) == 120
    summary = read(OUT / 'summary.json')
    for c, d in zip(cs, ds):
        assert all(d[k] == v for k, v in c.items())
        g = gs[c['prompt_id']]; f = decode(g['raw_output']) if g['error'] is None else None
        r = route_set(f) if f is not None else []
        assert d['predicted_facts'] == f and d['predicted_routes'] == r and d['invalid'] == (f is None)
        for k in DOMAINS:
            a = set(c['facts'][k]); b = set(f[k]) if f else None
            assert d['fields'][k] == {'exact': a == b, 'unsupported_exclusion': b is not None and not a <= b,
                'missed_constraint': b is not None and not b <= a, 'invalid': b is None}
        assert d['all_fields_exact'] == (f == c['facts'])
        assert d['route_set_exact'] == (r == c['routes'])
        singleton = len(c['routes']) == 1
        assert d['reference_singleton'] == singleton
        assert d['singleton_correct'] == (singleton and r == c['routes'])
        assert d['singleton_wrong'] == (singleton and len(r) == 1 and r != c['routes'])
        assert d['singleton_ambiguous'] == (singleton and len(r) > 1)
        assert d['excludes_reference_route'] == (bool(set(c['routes']) - set(r)) if f is not None else False)
    for label in ['E014_dev', 'E016_train']:
        subset = [d for d in ds if d['dataset'] == label]; n = len(subset)
        certain = [d for d in subset if len(d['routes']) == 1]
        s = summary[label]
        values = {'cases': n, 'invalid_outputs': sum(d['invalid'] for d in subset),
            'all_fields_exact_count': sum(d['all_fields_exact'] for d in subset),
            'route_set_exact_count': sum(d['route_set_exact'] for d in subset),
            'reference_singleton_cases': len(certain),
            'singleton_correct': sum(d['singleton_correct'] for d in certain),
            'singleton_wrong': sum(d['singleton_wrong'] for d in certain),
            'singleton_ambiguous': sum(d['singleton_ambiguous'] for d in certain),
            'singleton_invalid': sum(d['invalid'] for d in certain)}
        per_field = {k: {metric: sum(d['fields'][k][metric] for d in subset)
                         for metric in ['exact', 'unsupported_exclusion', 'missed_constraint', 'invalid']} for k in DOMAINS}
        assert s['per_field'] == per_field
        values['unsupported_exclusion_slots'] = sum(v['unsupported_exclusion'] for v in per_field.values())
        values['missed_constraint_slots'] = sum(v['missed_constraint'] for v in per_field.values())
        values['all_fields_exact_rate'] = float(Fraction(values['all_fields_exact_count'], n))
        values['route_set_exact_rate'] = float(Fraction(values['route_set_exact_count'], n))
        values['slot_exact_rate'] = float(Fraction(sum(v['exact'] for v in per_field.values()), n * 4))
        values['singleton_correct_rate'] = float(Fraction(values['singleton_correct'], len(certain)))
        assert s == {**values, 'per_field': per_field}
        extras[label] = {}
        for kind in ['constrained', 'unknown']:
            fs = [d['fields'][k] for d in subset for k in DOMAINS
                  if (len(d['facts'][k]) < len(DOMAINS[k])) == (kind == 'constrained')]
            extras[label][kind] = {'slots': len(fs), 'exact': sum(f['exact'] for f in fs),
                                  'exact_rate': float(Fraction(sum(f['exact'] for f in fs), len(fs)))}
    tr = summary['E016_train']
    gate = tr['invalid_outputs'] == 0 and sum(v['exact'] for v in tr['per_field'].values()) >= 346 and tr['singleton_correct'] >= 35
    assert summary['candidate_gate_passed'] == gate
    assert summary['accounting'] == {'unique_generations': 120, 'runtime_errors': sum(g['error'] is not None for g in raw),
        'length_finishes': sum(g['finish_reason'] == 'length' for g in raw), 'prompt_tokens': sum(g['prompt_tokens'] for g in raw),
        'generated_tokens': sum(g['generated_tokens'] for g in raw), 'peak_memory_gb': max(g['peak_memory_gb'] or 0 for g in raw),
        'elapsed_seconds': read(OUT / 'run_status.json')['elapsed_seconds']}
report = {'passed': True, 'completed': completed, 'source_cases': 120, 'prompt_inputs_verified': True,
          'generic_rule_interpreter_used': True, 'verifier_sha256': sha(Path(__file__)), 'manifest_sha256': sha(OUT / 'manifest.json')}
if completed: report['constrained_unknown_diagnostic'] = extras
(OUT / ('verification.json' if completed else 'input_verification.json')).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
