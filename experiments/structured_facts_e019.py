"""E019: fixed language-to-facts diagnostic, not a learned question policy."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction as F
import itertools
import json
from pathlib import Path
import sys
import time

import local_executor_e013 as base

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/structured_facts_e019'
DOMAINS = {
    'issue': ['unrecognized_charge', 'lost_card', 'fee_question', 'change_request', 'failed_payment'],
    'product': ['card', 'current_account', 'loan', 'savings'],
    'money_gone': ['yes', 'no'],
    'compensation': ['yes', 'no'],
}
SYSTEM = '''Extract only the banking facts stated in the customer's message. Do not choose a department or ask questions.
Return a JSON object with exactly four keys. Each value must be a nonempty array of possible values from its domain:
issue: ["unrecognized_charge", "lost_card", "fee_question", "change_request", "failed_payment"]
product: ["card", "current_account", "loan", "savings"]
money_gone: ["yes", "no"]
compensation: ["yes", "no"]
If a fact is not stated or is unknown, include ALL values of that domain. Unknown does not mean no. If the customer explicitly gives alternatives, include all those alternatives. If a fact is clear, include just that value. Do not infer unmentioned facts from typical banking correlations.
The issue is the type of problem: an unrecognized charge, a lost card, a fee question, a change request, or a failed payment. A failed payment is not automatically an unrecognized charge. Product refers to the product the issue concerns; a generic mention of an account does not specify a current account. Use savings for an explicitly mentioned savings account.
money_gone asks whether money has already left the customer's account. compensation asks whether the customer requests money back as compensation. A payment or fee being mentioned does not itself establish a request for compensation. Respect explicit negations separately for each field.
For a message containing no banking facts, the format is {"issue":["unrecognized_charge","lost_card","fee_question","change_request","failed_payment"],"product":["card","current_account","loan","savings"],"money_gone":["yes","no"],"compensation":["yes","no"]}.
Return only the JSON object. Do not add explanations or any other keys.'''

def read(p): return json.loads(p.read_text())
def save(p, x): base.save(p, x)

def route(w):
    if w['issue'] in ['unrecognized_charge', 'lost_card']:
        return 'fraud' if w['money_gone'] == 'yes' else 'cards'
    if w['issue'] == 'fee_question' and w['compensation'] == 'yes': return 'complaints'
    if w['product'] == 'loan': return 'loans'
    if w['product'] == 'card': return 'cards'
    return 'accounts'

WORLDS = [dict(zip(DOMAINS, v)) for v in itertools.product(*DOMAINS.values())]

def normalized(evidence):
    return {k: [v for v in values if v in evidence.get(k, values)] for k, values in DOMAINS.items()}

def routes(facts):
    return sorted({route(w) for w in WORLDS if all(w[k] in facts[k] for k in DOMAINS)})

def parse(raw):
    def unique(pairs):
        obj = dict(pairs)
        if len(obj) != len(pairs): raise ValueError('duplicate key')
        return obj
    try: obj = json.loads(raw, object_pairs_hook=unique)
    except (ValueError, TypeError): return None
    if type(obj) is not dict or set(obj) != set(DOMAINS): return None
    for k, domain in DOMAINS.items():
        a = obj[k]
        if type(a) is not list or not a or any(type(v) is not str or v not in domain for v in a): return None
        if len(set(a)) != len(a): return None
    return normalized(obj)

def cases():
    result = []
    for name, folder, detail in [
        ('E014_dev', 'lavoir_executor_e014', 'details.json'),
        ('E016_train', 'heldout_value_e016', 'train_details.json'),
    ]:
        path = ROOT / 'results' / folder
        rows = {r['id']: r for r in read(path / 'source_rows.json')}
        bycase = defaultdict(list)
        for d in read(path / detail):
            if d['question'] == 'no_ask': bycase[d['case_id']].append(d)
        assert len(bycase) == (24 if name == 'E014_dev' else 96)
        for case_id, ds in bycase.items():
            rs = [rows[d['source_id']] for d in ds]
            assert len({r['state'][0]['text'] for r in rs}) == 1
            facts = normalized(rs[0]['evidence'])
            assert all(normalized(r['evidence']) == facts for r in rs)
            assert all(all(facts[k] for k in DOMAINS) for r in rs)
            assert all(all(r['profile'][k] in facts[k] for k in DOMAINS) for r in rs)
            result.append({'dataset': name, 'case_id': case_id, 'text': rs[0]['state'][0]['text'],
                           'source_ids': [r['id'] for r in rs], 'facts': facts,
                           'routes': routes(facts), 'posterior_support': sorted(k for k, v in rs[0]['target'].items() if v > 0),
                           'old_teams': [d['team'] for d in ds], 'gold': [r['gold'] for r in rs]})
    assert len(result) == 120 and len({c['case_id'] for c in result}) == 120
    return result

def prepare():
    assert not OUT.exists(), 'Never overwrite frozen inputs'
    from transformers import AutoTokenizer
    previous = read(ROOT / 'results/neutral_labels_e018/manifest.json')
    for n, h in previous['model_files'].items(): assert base.sha(base.MODEL / n) == h
    tokenizer = AutoTokenizer.from_pretrained(str(base.MODEL), trust_remote_code=False, local_files_only=True)
    cs = cases(); pack = []; rendered = []
    for c in cs:
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': c['text']}]
        pid = base.digest(messages); c['prompt_id'] = pid
        pack.append({'id': pid, 'messages': messages})
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(prompt, add_special_tokens=False); assert len(ids) <= 4096
        rendered.append({'id': pid, 'prompt': prompt, 'input_tokens': ids})
    assert len({p['id'] for p in pack}) == 120
    OUT.mkdir(); save(OUT / 'cases.json', cs); save(OUT / 'pack.json', pack); save(OUT / 'rendered.json', rendered)
    paths = [Path(__file__).resolve(), Path(base.__file__).resolve(), ROOT / 'research/E019_PROTOCOL.md',
             ROOT / 'results/e019_contract_audit/summary.json', ROOT / 'results/e019_contract_audit/manifest.json']
    for f in ['lavoir_executor_e014/details.json', 'lavoir_executor_e014/source_rows.json',
              'heldout_value_e016/train_details.json', 'heldout_value_e016/source_rows.json', 'heldout_value_e016/schema.json']:
        paths.append(ROOT / 'results' / f)
    paths.extend(OUT / n for n in ['cases.json', 'pack.json', 'rendered.json'])
    save(OUT / 'manifest.json', {'files': {str(p.relative_to(ROOT)): base.sha(p) for p in paths},
        'model': previous['model'], 'revision': previous['revision'], 'model_files': previous['model_files'],
        'python': sys.version, 'seed': 20261027, 'temperature': 0, 'max_tokens': 192, 'budget_seconds': 1800,
        'unique_prompts': 120, 'E016_test_used': False,
        'gate': {'train_invalid_max': 0, 'train_slot_exact_min': .90, 'train_singleton_route_correct_min': .90}})
    print(json.dumps({'frozen': True, 'prompts': 120, 'max_input_tokens': max(len(r['input_tokens']) for r in rendered)}))

def verify():
    m = read(OUT / 'manifest.json')
    for p, h in m['files'].items(): assert base.sha(ROOT / p) == h, p
    return m

def run():
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    m = verify()
    assert not (OUT / 'generations.jsonl').exists()
    for n, h in m['model_files'].items(): assert base.sha(base.MODEL / n) == h
    mx.random.seed(m['seed']); start = time.monotonic()
    model, tokenizer = load(str(base.MODEL), trust_remote_code=False)
    rendered = read(OUT / 'rendered.json'); count = 0
    with (OUT / 'generations.jsonl').open('x') as out, (OUT / 'started.jsonl').open('x') as started:
        for i, r in enumerate(rendered):
            if time.monotonic() - start >= m['budget_seconds']: break
            assert tokenizer.encode(r['prompt'], add_special_tokens=False) == r['input_tokens']
            started.write(json.dumps({'id': r['id'], 'index': i, 'elapsed': time.monotonic() - start}) + '\n'); started.flush()
            before = time.monotonic(); last = None; chunks = []; error = None
            try:
                for part in stream_generate(model, tokenizer, prompt=r['input_tokens'], max_tokens=m['max_tokens'], sampler=make_sampler(temp=0.)):
                    chunks.append(part.text); last = part
                    if time.monotonic() - start >= m['budget_seconds'] and part.finish_reason is None:
                        raise TimeoutError('Overall generation budget exhausted')
                if last is None or last.finish_reason not in ['stop', 'length']: raise RuntimeError('No final status')
            except Exception as exc:
                error = {'type': type(exc).__name__, 'message': str(exc)}
            row = {'id': r['id'], 'raw_output': ''.join(chunks), 'error': error, 'elapsed_seconds': time.monotonic() - before,
                   'finish_reason': last.finish_reason if last else None, 'generated_tokens': last.generation_tokens if last else 0,
                   'prompt_tokens': last.prompt_tokens if last else len(r['input_tokens']), 'peak_memory_gb': last.peak_memory if last else None}
            out.write(json.dumps(row, ensure_ascii=False) + '\n'); out.flush(); count += 1
            if count % 10 == 0 or error: print(json.dumps({'completed': count, 'total': 120, 'elapsed': round(time.monotonic() - start, 1), 'error': error}), flush=True)
            if error and error['type'] == 'TimeoutError': break
            mx.clear_cache()
    save(OUT / 'run_status.json', {'completed': count, 'total': 120, 'complete': count == 120, 'elapsed_seconds': time.monotonic() - start})

def score():
    m = verify(); assert read(OUT / 'run_status.json')['complete']; assert not (OUT / 'summary.json').exists()
    raw = [json.loads(l) for l in (OUT / 'generations.jsonl').read_text().splitlines()]
    gens = {g['id']: g for g in raw}; assert len(raw) == len(gens) == 120
    details = []
    for c in read(OUT / 'cases.json'):
        g = gens[c['prompt_id']]; facts = parse(g['raw_output']) if g['error'] is None else None
        predicted = routes(facts) if facts is not None else []
        fields = {}
        for k in DOMAINS:
            a = set(c['facts'][k]); b = set(facts[k]) if facts else None
            fields[k] = {'exact': b == a, 'unsupported_exclusion': b is not None and bool(a - b),
                         'missed_constraint': b is not None and bool(b - a), 'invalid': b is None}
        single = len(c['routes']) == 1
        details.append({**c, 'predicted_facts': facts, 'predicted_routes': predicted, 'invalid': facts is None,
                        'fields': fields, 'all_fields_exact': all(v['exact'] for v in fields.values()),
                        'route_set_exact': predicted == c['routes'], 'reference_singleton': single,
                        'singleton_correct': single and predicted == c['routes'],
                        'singleton_wrong': single and len(predicted) == 1 and predicted != c['routes'],
                        'singleton_ambiguous': single and len(predicted) > 1,
                        'excludes_reference_route': bool(set(c['routes']) - set(predicted)) if facts else False})
    summary = {}
    for name in ['E014_dev', 'E016_train']:
        ds = [d for d in details if d['dataset'] == name]; n = len(ds); certain = [d for d in ds if d['reference_singleton']]
        count = lambda key: sum(d[key] for d in ds)
        summary[name] = {'cases': n, 'invalid_outputs': count('invalid'), 'all_fields_exact_count': count('all_fields_exact'),
            'all_fields_exact_rate': float(F(count('all_fields_exact'), n)),
            'slot_exact_rate': float(F(sum(v['exact'] for d in ds for v in d['fields'].values()), 4 * n)),
            'unsupported_exclusion_slots': sum(v['unsupported_exclusion'] for d in ds for v in d['fields'].values()),
            'missed_constraint_slots': sum(v['missed_constraint'] for d in ds for v in d['fields'].values()),
            'route_set_exact_count': count('route_set_exact'), 'route_set_exact_rate': float(F(count('route_set_exact'), n)),
            'reference_singleton_cases': len(certain), 'singleton_correct': count('singleton_correct'),
            'singleton_wrong': count('singleton_wrong'), 'singleton_ambiguous': count('singleton_ambiguous'),
            'singleton_invalid': sum(d['invalid'] for d in certain),
            'singleton_correct_rate': float(F(count('singleton_correct'), len(certain))) if certain else None,
            'per_field': {k: {metric: sum(d['fields'][k][metric] for d in ds) for metric in ['exact', 'unsupported_exclusion', 'missed_constraint', 'invalid']} for k in DOMAINS}}
    tr = summary['E016_train']; gate = m['gate']
    summary['candidate_gate_passed'] = tr['invalid_outputs'] <= gate['train_invalid_max'] and tr['slot_exact_rate'] >= gate['train_slot_exact_min'] and tr['singleton_correct_rate'] is not None and tr['singleton_correct_rate'] >= gate['train_singleton_route_correct_min']
    summary['accounting'] = {'unique_generations': len(raw), 'runtime_errors': sum(g['error'] is not None for g in raw),
        'length_finishes': sum(g['finish_reason'] == 'length' for g in raw), 'prompt_tokens': sum(g['prompt_tokens'] for g in raw),
        'generated_tokens': sum(g['generated_tokens'] for g in raw), 'peak_memory_gb': max(g['peak_memory_gb'] or 0 for g in raw),
        'elapsed_seconds': read(OUT / 'run_status.json')['elapsed_seconds']}
    save(OUT / 'details.json', details); save(OUT / 'summary.json', summary)
    paths = [OUT / n for n in ['manifest.json', 'generations.jsonl', 'details.json', 'summary.json', 'run_status.json']]
    save(OUT / 'score_manifest.json', {'files': {str(p.relative_to(ROOT)): base.sha(p) for p in paths}})
    print(json.dumps(summary))

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare', 'run', 'score']); globals()[p.parse_args().stage]()
