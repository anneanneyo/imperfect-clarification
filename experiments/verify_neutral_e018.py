"""Independent source joins, prompt substitution and exact score verification for E018."""
import ast
import hashlib
import json
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/neutral_labels_e018'
LABELS = {'cards': 'U1', 'fraud': 'U2', 'loans': 'U3', 'accounts': 'U4', 'complaints': 'U5'}
CLEAN = {'00037', '00384', '00210', '00390', '00406', '00176', '00348'}

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_hashes(manifest):
    for path, expected in manifest['files'].items():
        assert sha(ROOT / path) == expected, path

def close(a, b):
    assert abs(float(a) - float(b)) < 1e-12, (a, b)

manifest = read(OUT / 'manifest.json')
verify_hashes(manifest)
assert manifest['E016_test_used'] is False
assert read(OUT / 'mapping.json') == LABELS
tree = ast.parse((ROOT / 'experiments/lavoir_executor_e014.py').read_text())
original = next(ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SYSTEM' for t in n.targets))
expected_system = re.sub(r'\b\w+\b', lambda m: LABELS.get(m[0], m[0]), original)
pack = read(OUT / 'pack.json')
prompts = {p['id']: p['messages'] for p in pack}
assert len(prompts) == len(pack) == 120
branches = read(OUT / 'branches.json')
assert len(branches) == 339
assert len({(b['dataset'], b['source_id']) for b in branches}) == 339
expected_keys = set()
case_groups = {}
for label, folder, filename, count in [
    ('E014_dev', 'lavoir_executor_e014', 'details.json', 24),
    ('E016_train', 'heldout_value_e016', 'train_details.json', 96),
]:
    base = ROOT / 'results' / folder
    old = {d['source_id']: d for d in read(base / filename) if d['question'] == 'no_ask'}
    source = {r['id']: r for r in read(base / 'source_rows.json')}
    counts = Counter(d['case_id'] for d in old.values())
    assert len(counts) == count
    case_groups[label] = counts
    expected_keys.update((label, k) for k in old)
    for b in [b for b in branches if b['dataset'] == label]:
        d, r = old[b['source_id']], source[b['source_id']]
        assert b['case_id'] == d['case_id']
        assert b['gold'] == r['gold']
        assert b['old_team'] == d['team']
        assert b['old_correct'] == d['correct'] == int(d['team'] == r['gold'])
        assert Fraction(b['weight_numerator'], b['weight_denominator']) == Fraction(1, count * counts[b['case_id']])
        assert b['initial_symbolic_certain'] == (max(r['target'].values()) > 1 - 1e-12)
        assert b['clean_known_error'] == (label == 'E016_train' and b['case_id'].split('/')[-1] in CLEAN)
        assert prompts[b['prompt_id']] == [
            {'role': 'system', 'content': expected_system},
            {'role': 'user', 'content': r['state'][0]['text']},
        ]
assert expected_keys == {(b['dataset'], b['source_id']) for b in branches}
assert set(case_groups['E014_dev']).isdisjoint(case_groups['E016_train'])
assert set(prompts) == {b['prompt_id'] for b in branches}
completed = (OUT / 'summary.json').exists()
if completed:
    verify_hashes(read(OUT / 'score_manifest.json'))
    assert read(OUT / 'run_status.json')['complete']
    raw = [json.loads(l) for l in (OUT / 'generations.jsonl').read_text().splitlines()]
    assert len(raw) == len({g['id'] for g in raw}) == 120
    assert {g['id'] for g in raw} == set(prompts)
    decoded = {}
    for g in raw:
        assert g['error'] is None
        try:
            obj = json.loads(g['raw_output'])
        except (json.JSONDecodeError, TypeError):
            obj = None
        team = None
        if isinstance(obj, dict) and list(obj) == ['team'] and isinstance(obj['team'], str):
            team = next((k for k, v in LABELS.items() if v == obj['team']), None)
        decoded[g['id']] = team
    details = read(OUT / 'details.json')
    assert len(details) == len(branches)
    for b, d in zip(branches, details):
        assert all(d[k] == v for k, v in b.items())
        team = decoded[b['prompt_id']]
        assert d['team'] == team and d['correct'] == int(team == b['gold']) and d['invalid'] == (team is None)
    summary = read(OUT / 'summary.json')
    saved_cases = {(d['dataset'], d['case_id']): d for d in read(OUT / 'case_results.json')}
    for label, counts in case_groups.items():
        ds = [d for d in details if d['dataset'] == label]
        old = sum((Fraction(d['old_correct'], len(counts) * counts[d['case_id']]) for d in ds), Fraction())
        new = sum((Fraction(d['correct'], len(counts) * counts[d['case_id']]) for d in ds), Fraction())
        s = summary[label]
        close(s['old_accuracy'], old); close(s['new_accuracy'], new); close(s['difference'], new - old)
        assert s['cases'] == len(counts)
        diffs, fixed = [], []
        for c, n in counts.items():
            rows = [d for d in ds if d['case_id'] == c]
            a = Fraction(sum(d['old_correct'] for d in rows), n)
            b = Fraction(sum(d['correct'] for d in rows), n)
            diffs.append(b - a)
            cr = saved_cases[(label, c)]
            close(cr['old_accuracy'], a); close(cr['new_accuracy'], b)
            assert cr['clean_known_error'] == rows[0]['clean_known_error']
            if rows[0]['clean_known_error'] and b == 1:
                fixed.append(c)
        assert sorted(s['clean_known_errors_fixed']) == sorted(fixed)
        assert (s['improved_cases'], s['worse_cases'], s['tied_cases']) == (sum(d > 0 for d in diffs), sum(d < 0 for d in diffs), sum(d == 0 for d in diffs))
        assert s['invalid_unique_outputs'] == len({d['prompt_id'] for d in ds if d['invalid']})
        certain = [d for d in ds if d['initial_symbolic_certain']]
        mass = sum((Fraction(1, len(counts) * counts[d['case_id']]) for d in certain), Fraction())
        close(s['initial_certain_case_mass'], mass)
        for key in ['old', 'new']:
            field = 'old_correct' if key == 'old' else 'correct'
            value = sum((Fraction(d[field], len(counts) * counts[d['case_id']]) for d in certain), Fraction()) / mass
            close(s['certain_' + key + '_accuracy'], value)
    tr = summary['E016_train']
    assert summary['candidate_gate_passed'] == (len(tr['clean_known_errors_fixed']) >= 4 and tr['new_accuracy'] >= tr['old_accuracy'])
report = {'passed': True, 'completed': completed, 'prompts': 120, 'branches': 339,
          'verifier_sha256': sha(Path(__file__)), 'manifest_sha256': sha(OUT / 'manifest.json')}
(OUT / ('verification.json' if completed else 'pre_score_audit.json')).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
