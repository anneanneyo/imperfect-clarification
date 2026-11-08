"""E006: execute reviewed ClarifyBench metric definitions on synthetic call logs.

No benchmark tools, agents, model providers or package imports are executed.
Upstream source/data remain in the external checkout; outputs contain metadata.
"""
import argparse
import ast
import builtins
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import typing

ROOT = Path(__file__).resolve().parents[1]
COMMIT = 'a85d4f9df1fc87d05c713fd408f8a57d72bcc348'
EVALUATOR_SHA = 'c63ab42c76eb83680a1d42378c461894af41cf43081ae5352cd8e0131a2c09f8'
PROTOCOL_SHA = 'e6ab1d72ff6a24321eaba5f4719f5a4f9bf3525b2d5e4f0c5e890e56f3a09b15'
METHODS = {
    '_values_match', '_params_match', 'evaluate_simulation',
    '_calculate_correctness_metrics_for_final_calls', '_check_exact_match',
    '_calculate_conversation_metrics', '_calculate_coverage_success_for_final_calls',
    '_calculate_attempt_based_metrics',
}
VARIANTS = ['gold', 'delete_first', 'duplicate_first', 'extra_action',
            'wrong_parameter', 'reverse_order', 'failed_attempt',
            'drop_repeated_identical', 'replace_duplicate_with_extra']
METRIC_PATHS = {
    'exact_match': ('correctness', 'exact_match'),
    'tool_match_rate': ('correctness', 'tool_match_rate'),
    'param_match_rate': ('correctness', 'param_match_rate'),
    'coverage_rate': ('coverage', 'coverage_rate'),
    'coverage_success': ('coverage', 'coverage_success'),
    'execution_success': ('execution', 'execution_success'),
    'success': ('success',),
}
EXTRA_TOOL = {
    'DocumentPlugin': ['delete_page', 'delete_page_range', 'redact_text', 'add_comment'],
    'VehicleControlAPI': ['startEngine', 'setCruiseControl', 'setHeadlights', 'lockDoors', 'set_navigation', 'fillFuelTank', 'activateParkingBrake'],
    'TradingBot': ['place_order', 'cancel_order', 'make_transaction', 'fund_account', 'update_market_status'],
    'TravelAPI': ['cancel_booking', 'book_flight', 'purchase_insurance', 'set_budget_limit', 'register_credit_card'],
    'GorillaFileSystem': ['mv', 'cp', 'touch', 'mkdir', 'echo'],
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reviewed_evaluator(path):
    assert file_hash(path) == EVALUATOR_SHA, 'Review required: evaluator source changed'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SimulationEvaluator')
    methods = [deepcopy(n) for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in METHODS]
    assert len(methods) == len(METHODS)
    extracted = ast.Module(body=[ast.ClassDef(name='SimulationEvaluator', bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    ast.fix_missing_locations(extracted)
    allowed_calls = {'isinstance', 'len', 'all', 'any', 'str', 'float', 'abs', 'list', 'set', 'sum', 'zip'}
    allowed_attrs = METHODS | {'get', 'lower', 'strip', 'append', 'keys', 'values', 'items', 'intersection', 'update'}
    for node in ast.walk(extracted):
        assert not isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith)), type(node).__name__
        if isinstance(node, ast.Attribute):
            assert node.attr in allowed_attrs, node.attr
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id in allowed_calls, node.func.id
            else:
                assert isinstance(node.func, ast.Attribute) and node.func.attr in allowed_attrs
    safe_names = allowed_calls | {'__build_class__', 'bool', 'dict', 'ValueError', 'TypeError'}
    safe_builtins = {name: getattr(builtins, name) for name in safe_names}
    namespace = {'__builtins__': safe_builtins, '__name__': 'reviewed_clarifybench_metrics'}
    namespace.update({name: getattr(typing, name) for name in ['Dict', 'List', 'Any', 'Tuple', 'Optional']})
    exec(compile(extracted, str(path), 'exec'), namespace)
    return namespace['SimulationEvaluator'](), digest(ast.dump(extracted, include_attributes=False))


def normalized(call):
    return {'tool_name': call.get('name', call.get('tool_name', '')),
            'parameters': call.get('args', call.get('arguments', call.get('parameters', {})))}


def trace(calls, failed=False):
    final = [{'tool_name': c['tool_name'], 'arguments': deepcopy(c['parameters'])} for c in calls]
    attempts = [{'tool_call': deepcopy(c), 'was_executed': True, 'success': True, 'reason': 'synthetic_success'} for c in final]
    if failed:
        attempts.insert(0, {'tool_call': {'tool_name': '__audit_failed_attempt__', 'arguments': {}},
                            'was_executed': True, 'success': False, 'reason': 'synthetic_failure'})
    return {'final_tool_calls': final, 'all_tool_call_attempts': attempts, 'conversation': []}


def repeated_identical_index(calls):
    seen = set()
    for i, call in enumerate(calls):
        key = canonical(call)
        if key in seen:
            return i
        seen.add(key)
    return None


def mutated(gold, variant, extra):
    calls = deepcopy(gold)
    if variant == 'gold':
        return calls, False, None
    if variant == 'failed_attempt':
        return calls, True, None
    if not calls:
        return None, False, 'empty_gold'
    if variant == 'delete_first':
        del calls[0]
    elif variant == 'duplicate_first':
        calls.append(deepcopy(calls[0]))
    elif variant == 'extra_action':
        if extra is None:
            return None, False, 'no_distinct_same_domain_donor'
        calls.append(deepcopy(extra))
    elif variant == 'wrong_parameter':
        indices = [i for i, c in enumerate(calls) if c['parameters']]
        if not indices:
            return None, False, 'no_parameters'
        i = indices[0]
        key = sorted(calls[i]['parameters'])[0]
        marker = '__E006_WRONG_PARAMETER__'
        while any(marker == value for c in gold for value in c['parameters'].values()):
            marker += '_'
        calls[i]['parameters'][key] = marker
    elif variant == 'reverse_order':
        calls.reverse()
        if calls == gold:
            return None, False, 'reversal_does_not_change_sequence'
    elif variant in ['drop_repeated_identical', 'replace_duplicate_with_extra']:
        i = repeated_identical_index(calls)
        if i is None:
            return None, False, 'no_identical_repeat'
        if variant == 'drop_repeated_identical':
            del calls[i]
        else:
            if extra is None:
                return None, False, 'no_distinct_same_domain_donor'
            calls[i] = deepcopy(extra)
    else:
        raise ValueError(variant)
    return calls, False, None


def flat_metrics(metrics):
    out = {}
    for name, path in METRIC_PATHS.items():
        item = metrics
        for key in path:
            item = item.get(key) if isinstance(item, dict) else None
        out[name] = item
    return out


def score(evaluator, gold, calls, failed):
    result = evaluator.evaluate_simulation({'ground_truth_tool_calls': gold}, trace(calls, failed))
    return {'metrics': flat_metrics(result), 'ordered_equal': gold == calls,
            'multiset_equal': Counter(map(canonical, gold)) == Counter(map(canonical, calls)),
            'actual_call_count': len(calls), 'actual_hash': digest(calls)}


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['subset'], row['variant'])].append(row)
        groups[('ALL', row['variant'])].append(row)
    result = []
    for (subset, variant), group in sorted(groups.items()):
        done = [r for r in group if r['status'] == 'completed']
        record = {'subset': subset, 'variant': variant, 'planned': len(group), 'completed': len(done),
                  'not_applicable': sum(r['status'] == 'not_applicable' for r in group),
                  'errors': sum(r['status'] == 'error' for r in group),
                  'not_applicable_reasons': dict(Counter(r['reason'] for r in group if r['status'] == 'not_applicable'))}
        for name in METRIC_PATHS:
            values = [r['metrics'][name] for r in done if r['metrics'][name] is not None]
            record[name] = {'present': len(values), 'missing': len(done)-len(values),
                            'mean': sum(values)/len(values) if values else None,
                            'equals_one_count': sum(v == 1 for v in values)}
        record['ordered_equal_count'] = sum(r['ordered_equal'] for r in done)
        record['multiset_equal_count'] = sum(r['multiset_equal'] for r in done)
        record['accepted_non_multiset_count'] = sum(r['metrics']['exact_match'] is True and not r['multiset_equal'] for r in done)
        record['rejected_equal_multiset_count'] = sum(r['metrics']['exact_match'] is False and r['multiset_equal'] for r in done)
        result.append(record)
    return result


def own_examples(evaluator):
    def set_call(value):
        return {'tool_name': 'set_value', 'parameters': {'value': value}}
    delete = {'tool_name': 'delete_value', 'parameters': {}}
    cases = [
        ('repeat_tool_different_arguments', [set_call(1), set_call(2)], [set_call(1), set_call(2)], False),
        ('repeat_tool_identical_gold', [set_call(1), set_call(1)], [set_call(1), set_call(1)], False),
        ('missing_repeat_replaced_by_destructive_call', [set_call(1), set_call(1)], [set_call(1), delete], False),
        ('extra_destructive_call', [set_call(1)], [set_call(1), delete], False),
        ('reverse_dependent_actions', [set_call(1), set_call(2)], [set_call(2), set_call(1)], False),
        ('reverse_distinct_dependent_actions', [set_call(1), {'tool_name': 'add_value', 'parameters': {'value': 2}}],
         [{'tool_name': 'add_value', 'parameters': {'value': 2}}, set_call(1)], False),
        ('correct_final_after_failed_attempt', [set_call(1)], [set_call(1)], True),
        ('drop_all_calls', [set_call(1)], [], False),
    ]
    def final_state(calls):
        state = None
        for call in calls:
            if call['tool_name'] == 'set_value':
                state = call['parameters']['value']
            elif call['tool_name'] == 'add_value':
                state = (state or 0) + call['parameters']['value']
            else:
                state = None
        return state
    return [{'case': name, 'gold': gold, 'actual': actual, 'failed_attempt': failed,
             'own_toy_gold_state': final_state(gold), 'own_toy_actual_state': final_state(actual),
             'own_toy_state_equal': final_state(gold) == final_state(actual),
             **score(evaluator, gold, actual, failed)} for name, gold, actual, failed in cases]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'results/clarifybench_evaluator_audit')
    args = parser.parse_args()
    repo = args.repo.resolve()
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
    protocol = ROOT/'research/E006_PROTOCOL.md'
    assert file_hash(protocol) == PROTOCOL_SHA, 'Protocol changed since pre-execution freeze'
    source = repo/'simulation/evaluation.py'
    evaluator, ast_hash = load_reviewed_evaluator(source)
    files = sorted((repo/'ClarifyBench').glob('ClarifyBench_*/*.json'))
    hashes_before = {str(p.relative_to(repo)): file_hash(p) for p in files}
    records = []; invalid = []; donors = defaultdict(list)
    for path in files:
        row = json.loads(path.read_text())
        if not row or any(k not in row for k in ['user_query', 'ground_truth_tool_calls', 'primary_api']):
            invalid.append(str(path.relative_to(repo)))
            continue
        gold = [normalized(c) for c in row['ground_truth_tool_calls']]
        record = {'path': str(path.relative_to(repo)), 'id': path.stem, 'subset': path.parent.name,
                  'domain': row['primary_api'], 'gold': gold}
        records.append(record)
        for call in gold:
            if call['tool_name'] in EXTRA_TOOL[row['primary_api']]:
                donors[row['primary_api']].append((record['path'], call))
    assert len(records) == 602 and len(invalid) == 2
    rows = []
    for record in records:
        gold = record['gold']
        candidates = [(path, c) for path, c in donors[record['domain']] if c not in gold]
        donor_path, extra = candidates[0] if candidates else (None, None)
        for variant in VARIANTS:
            output = {k: record[k] for k in ['path', 'id', 'subset', 'domain']}
            output.update({'variant': variant, 'gold_hash': digest(gold), 'gold_call_count': len(gold)})
            calls, failed, reason = mutated(gold, variant, extra)
            if reason:
                output.update({'status': 'not_applicable', 'reason': reason})
            else:
                try:
                    output.update(score(evaluator, gold, calls, failed))
                    output['status'] = 'completed'
                    if variant in ['extra_action', 'replace_duplicate_with_extra']:
                        output['extra_call_source'] = donor_path
                        output['extra_tool_name'] = extra['tool_name']
                except Exception as error:
                    output.update({'status': 'error', 'error_class': type(error).__name__, 'error': str(error)})
            rows.append(output)
    # Fixtures selected deterministically from metadata, not model performance.
    chosen = []
    for domain in sorted(EXTRA_TOOL):
        chosen += [r['path'] for r in records if r['domain'] == domain and r['subset'] == 'ClarifyBench_A'][:2]
    repeated_records = [r['path'] for r in records if len({c['tool_name'] for c in r['gold']}) < len(r['gold'])]
    for path in repeated_records:
        if sum(p in repeated_records for p in chosen) >= 3 and len(chosen) >= 12:
            break
        if path not in chosen:
            chosen.append(path)
    assert len(chosen) >= 12 and sum(p in repeated_records for p in chosen) >= 3
    toy = own_examples(evaluator)
    completed = [r for r in rows if r['status'] == 'completed']
    assert all(r['ordered_equal'] and r['multiset_equal'] for r in completed if r['variant'] == 'gold')
    assert all(not r['multiset_equal'] for r in completed if r['variant'] in ['delete_first', 'duplicate_first', 'extra_action', 'wrong_parameter', 'drop_repeated_identical', 'replace_duplicate_with_extra'])
    assert all(not r['ordered_equal'] and r['multiset_equal'] for r in completed if r['variant'] == 'reverse_order')
    assert all(not r['metrics']['execution_success'] for r in completed if r['variant'] == 'failed_attempt')
    examples = {r['case']: r for r in toy}
    assert examples['repeat_tool_different_arguments']['ordered_equal']
    assert not examples['repeat_tool_different_arguments']['metrics']['exact_match']
    assert examples['missing_repeat_replaced_by_destructive_call']['metrics']['success']
    assert not examples['missing_repeat_replaced_by_destructive_call']['own_toy_state_equal']
    assert examples['reverse_distinct_dependent_actions']['metrics']['success']
    assert not examples['reverse_distinct_dependent_actions']['own_toy_state_equal']
    assert all(hashes_before[str(p.relative_to(repo))] == file_hash(p) for p in files)
    assert file_hash(source) == EVALUATOR_SHA
    summary = {'date': '2026-10-27', 'status': 'completed_metric_audit_not_agent_evaluation',
               'valid_scenarios': len(records), 'invalid_json_objects': invalid,
               'planned': len(records)*len(VARIANTS),
               'completed': sum(r['status'] == 'completed' for r in rows),
               'not_applicable': sum(r['status'] == 'not_applicable' for r in rows),
               'errors': sum(r['status'] == 'error' for r in rows),
               'metrics': aggregate(rows), 'fixture_paths': chosen,
               'fixture_repeated_tools_count': sum(p in repeated_records for p in chosen),
               'own_control_cases': len(toy),
               'checks': ['AST_allowlist', 'source_and_data_hashes_unchanged', 'mutation_reference_contracts', 'gold_reference_identity', 'failed_attempt_execution_flag', 'own_toy_counterexamples'],
               'gold_exact_failures': [r['path'] for r in rows if r['variant'] == 'gold' and r['status'] == 'completed' and not r['metrics']['exact_match']],
               'data_and_evaluator_unchanged': True, 'model_calls': 0, 'benchmark_tool_executions': 0}
    manifest = {'commit': COMMIT, 'repository_url': 'https://github.com/MananSuri27/ClarifyBench',
                'python': platform.python_version(), 'code_sha256': file_hash(Path(__file__)),
                'protocol_sha256': file_hash(protocol), 'evaluator_sha256': EVALUATOR_SHA,
                'extracted_ast_sha256': ast_hash, 'extracted_methods': sorted(METHODS),
                'data_file_sha256': hashes_before, 'data_manifest_sha256': digest(hashes_before),
                'model_calls': 0, 'benchmark_tool_executions': 0,
                'execution_note': 'Reviewed metric methods only, extracted unchanged by AST; no upstream imports. All attempt status flags are synthetic.'}
    args.output.mkdir(exist_ok=True, parents=True)
    for name, data in [('summary', summary), ('manifest', manifest), ('per_case', rows), ('own_examples', toy)]:
        (args.output/(name+'.json')).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: summary[k] for k in ['status', 'valid_scenarios', 'planned', 'completed', 'not_applicable', 'errors']}, indent=2))
    for item in summary['metrics']:
        if item['subset'] == 'ALL':
            print(item['variant'], 'exact=',item['exact_match']['equals_one_count'],
                  'coverage=',item['coverage_success']['equals_one_count'],
                  'success=',item['success']['equals_one_count'], '/',item['completed'])
    assert summary['errors'] == 0, 'Read per_case.json errors before relying on results'


if __name__ == '__main__':
    main()
