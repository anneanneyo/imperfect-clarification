"""Offline GFS replay/checker feasibility audit; NOT an official benchmark score.

Loads the pinned upstream plugin read-only. It mutates an in-memory filesystem,
not the host's files. Outputs only derived IDs/counts/hashes and our own fixtures.
No LLM, API, package installs, or copying upstream code/data into this repository.
"""
import argparse
from collections import Counter
import copy
import hashlib
import importlib
import json
import logging
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/clarifybench_state_probe'
COMMIT = 'a85d4f9df1fc87d05c713fd408f8a57d72bcc348'
READS = {'pwd', 'ls', 'cat', 'find', 'wc', 'sort', 'grep', 'du', 'tail', 'diff'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def call(name, **parameters):
    return {'tool_name': name, 'parameters': parameters}


def snapshot(plugin):
    fs = plugin.file_system
    cwd = []
    alias_paths = []
    seen = {}
    bad_parents = []

    def walk(node, path, parent, ancestors=frozenset()):
        if node is fs._current_dir:
            cwd.append(path)
        if id(node) in seen:
            alias_paths.append([seen[id(node)], path])
        else:
            seen[id(node)] = path
        if hasattr(node, 'content'):
            return ['file', node.name, digest(node.content)]
        if node.parent is not parent:
            bad_parents.append(path)
        # Runtime fixtures have no cycles. Guard the audit itself against them.
        if id(node) in ancestors:
            return ['cycle', node.name]
        ancestors = ancestors | {id(node)}
        return ['directory', node.name,
                [[name, walk(child, path+[name], node, ancestors)] for name, child in node.contents.items()]]

    tree = walk(fs.root, [], None)

    def sorted_tree(node):
        if node[0] != 'directory':
            return node
        return node[:2]+[[[name, sorted_tree(child)] for name, child in sorted(node[2])]]

    return {'operational_hash': digest({'tree_in_insertion_order': tree, 'cwd_paths': cwd}),
            'content_hash': digest(sorted_tree(tree)), 'tree': tree,
            'cwd_paths': cwd, 'alias_paths': alias_paths, 'bad_parent_paths': bad_parents}


def replay(cls, config, calls, initial_cwd=()):
    p = cls()
    assert p.initialize_from_config(copy.deepcopy(config))
    if initial_cwd:
        node = p.file_system.root
        for part in initial_cwd:
            node = node.contents[part]
        p.file_system._current_dir = node
    initial = snapshot(p)
    results = []
    observations = Counter()
    for index, tc in enumerate(calls):
        name = tc['tool_name']; params = copy.deepcopy(tc.get('parameters', {}))
        valid, error = p.validate_tool_call_with_virtual_support(name, params)
        if not valid:
            result = {'success': False, 'error': 'PREVALIDATION_FAILED', 'message': error}
        else:
            result = p.execute_tool_with_virtual_support(name, {k: v for k, v in params.items() if v != '<UNK>'})
        if result.get('success') and (name in READS or (name == 'echo' and not params.get('file_name'))):
            observations[digest(result.get('output', {}))] += 1
        results.append({'index': index, 'tool': name, 'success': bool(result.get('success')),
                        'error': result.get('error'), 'output_hash': digest(result.get('output', {})),
                        'message_hash': digest(result.get('message', ''))})
        if not result.get('success'):
            break
    state = snapshot(p)
    return {'all_success': len(results) == len(calls) and all(r['success'] for r in results),
            'attempted': len(results), 'planned': len(calls), 'steps': results,
            'initial_operational_hash': initial['operational_hash'],
            'initial_content_hash': initial['content_hash'],
            'state': state, 'read_observations': dict(observations)}


def compact(result):
    return {k: v for k, v in result.items() if k not in ['state', 'read_observations']} | {
        'state_hash': result['state']['operational_hash'],
        'content_hash': result['state']['content_hash'],
        'distinct_read_observations': len(result['read_observations']),
        'read_observation_count': sum(result['read_observations'].values()),
        'alias_count': len(result['state']['alias_paths']),
        'bad_parent_count': len(result['state']['bad_parent_paths'])}


def comparisons(actual, reference):
    state_equal = actual['state']['operational_hash'] == reference['state']['operational_hash']
    reads_cover = all(actual['read_observations'].get(k, 0) >= v for k, v in reference['read_observations'].items())
    return {'all_success': actual['all_success'], 'same_operational_state': state_equal,
            'same_content_state': actual['state']['content_hash'] == reference['state']['content_hash'],
            'covers_reference_read_outputs': reads_cover,
            'state_plus_read_reference_pass': actual['all_success'] and state_equal and reads_cover}


def directories(config):
    root = next(iter(config['GorillaFileSystem']['root'].values()))
    paths = []

    def walk(node, path):
        for name, child in node.get('contents', {}).items():
            if child.get('type') == 'directory':
                paths.append(path+[name]); walk(child, path+[name])

    walk(root, [])
    return paths


def mutations(calls):
    yield 'empty', [], None
    yield 'extra_read_ls', [call('ls')]+copy.deepcopy(calls), None
    yield 'extra_state_change_touch', copy.deepcopy(calls)+[call('touch', file_name='__state_probe_unexpected__')], None
    yield 'reverse', list(reversed(copy.deepcopy(calls))), None
    if calls:
        yield 'duplicate_last', copy.deepcopy(calls)+[copy.deepcopy(calls[-1])], len(calls)-1
    for i in range(len(calls)):
        yield 'drop_one', copy.deepcopy(calls[:i]+calls[i+1:]), i
    for i, tc in enumerate(calls):
        if tc.get('parameters'):
            variant = copy.deepcopy(calls)
            key = next(iter(tc['parameters']))
            del variant[i]['parameters'][key]
            yield 'missing_parameter', variant, i
            variant = copy.deepcopy(calls)
            value = variant[i]['parameters'][key]
            variant[i]['parameters'][key] = value+'__wrong__' if isinstance(value, str) else '__wrong__'
            yield 'wrong_parameter', variant, i
            break
    for i, tc in enumerate(calls):
        if tc['tool_name'] == 'touch':
            variant = copy.deepcopy(calls)
            variant[i] = call('echo', file_name=tc['parameters']['file_name'], content='')
            yield 'touch_to_echo_empty', variant, i
            break


def own_fixtures(cls):
    # Self-authored fixtures isolate observable semantics, not benchmark tasks.
    config = {'GorillaFileSystem': {'root': {'workspace': {'type': 'directory', 'contents': {
        'keep.txt': {'type': 'file', 'content': 'preserve me'}}}}}}
    gold = [call('mkdir', dir_name='notes'), call('cd', folder='notes'),
            call('touch', file_name='answer.txt'), call('echo', file_name='answer.txt', content='correct')]
    reference = replay(cls, config, gold)
    variants = {
        'gold': gold,
        'omit_redundant_touch': gold[:2]+gold[3:],
        'extra_navigation_back_to_root': gold+[call('cd', folder='..')],
        'extra_read': [call('ls')]+gold,
        'missing_write': gold[:-1],
        'wrong_content': gold[:-1]+[call('echo', file_name='answer.txt', content='wrong')],
        'extra_file': gold+[call('touch', file_name='unexpected.txt')],
        'wrong_order': [gold[0], gold[2], gold[1], gold[3]],
    }
    rows = []
    for name, actions in variants.items():
        result = replay(cls, config, actions)
        # The authored task explicitly asks only for final file contents/tree,
        # not cwd or output order. This predicate is NOT extrapolated to benchmark.
        goal = result['all_success'] and result['state']['content_hash'] == reference['state']['content_hash']
        rows.append({'variant': name, 'calls': len(actions), 'authored_goal_pass': goal, **comparisons(result, reference)})
    assert {r['variant'] for r in rows if r['authored_goal_pass']} == {'gold', 'omit_redundant_touch', 'extra_navigation_back_to_root', 'extra_read'}

    bugconfig = {'GorillaFileSystem': {'root': {'workspace': {'type': 'directory', 'contents': {
        'source': {'type': 'directory', 'contents': {'item.txt': {'type': 'file', 'content': 'before'},
                                                   'nested': {'type': 'directory', 'contents': {}}}},
        'a.txt': {'type': 'file', 'content': 'same'}, 'b.txt': {'type': 'file', 'content': 'same\nextra'}}}}}}
    p = cls(); p.initialize_from_config(copy.deepcopy(bugconfig))
    assert p.execute_tool('cp', {'source': 'source', 'destination': 'backup'})['success']
    assert p.execute_tool('cd', {'folder': 'backup'})['success']
    assert p.execute_tool('echo', {'file_name': 'item.txt', 'content': 'changed'})['success']
    original_changed = p.file_system.root.contents['source'].contents['item.txt'].content == 'changed'
    p.execute_tool('cd', {'folder': 'nested'}); p.execute_tool('cd', {'folder': '..'})
    wrong_parent = p.file_system._current_dir is p.file_system.root.contents['source']
    p.execute_tool('cd', {'folder': '/'})
    diff = p.execute_tool('diff', {'file_name1': 'a.txt', 'file_name2': 'b.txt'})
    ignored_extra_line = diff.get('output', {}).get('diff_lines') == ''
    assert original_changed and wrong_parent and ignored_extra_line
    return {'author_defined_goal_variants': rows,
            'plugin_semantic_counterexamples': {'directory_copy_shares_file_objects': original_changed,
                'copied_subdirectory_parent_points_to_original': wrong_parent,
                'diff_ignores_unmatched_trailing_line': ignored_extra_line}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--repo', type=Path, required=True); args = ap.parse_args()
    repo = args.repo.resolve()
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repo))
    module = importlib.import_module('plugins.gfs_plugin')
    cls = module.GFSPlugin
    logging.disable(logging.CRITICAL)
    primary = []; mutated = []; inputs = {}
    for path in sorted((repo/'ClarifyBench').glob('ClarifyBench_*/*.json')):
        data = json.loads(path.read_text())
        if set(data.get('initial_config', {})) != {'GorillaFileSystem'}:
            continue
        task_id = str(path.relative_to(repo/'ClarifyBench'))
        inputs[task_id] = hashlib.sha256(path.read_bytes()).hexdigest()
        calls = data['ground_truth_tool_calls']; config = data['initial_config']
        gold = replay(cls, config, calls)
        again = replay(cls, config, calls)
        assert compact(gold) == compact(again)
        row = {'task': task_id, 'split': path.parent.name, 'gold': compact(gold),
               'gold_content_state_unchanged': gold['state']['content_hash'] == gold['initial_content_hash'],
               'gold_operational_state_unchanged': gold['state']['operational_hash'] == gold['initial_operational_hash'],
               'has_DEFAULT_argument': any('DEFAULT' in c.get('parameters', {}) for c in calls),
               'successful_alternate_initial_cwd_count': 0}
        if not gold['all_success']:
            # Diagnostic search only: never silently repair initial cwd for scoring.
            row['successful_alternate_initial_cwd_count'] = sum(replay(cls, config, calls, cwd)['all_success'] for cwd in directories(config))
        else:
            for name, variant, index in mutations(calls):
                actual = replay(cls, config, variant)
                mutated.append({'task': task_id, 'mutation': name, 'call_index': index,
                                'planned_calls': len(variant), **comparisons(actual, gold)})
        primary.append(row)
    by_split = []
    for split in sorted({r['split'] for r in primary}):
        rows = [r for r in primary if r['split'] == split]
        by_split.append({'split': split, 'tasks': len(rows), 'gold_all_success': sum(r['gold']['all_success'] for r in rows),
                         'failed_gold_rescued_by_some_initial_cwd': sum(r['successful_alternate_initial_cwd_count'] > 0 for r in rows),
                         'successful_gold_content_unchanged': sum(r['gold']['all_success'] and r['gold_content_state_unchanged'] for r in rows)})
    mutation_summary = []
    for name in sorted({r['mutation'] for r in mutated}):
        rows = [r for r in mutated if r['mutation'] == name]
        mutation_summary.append({'mutation': name, 'variants': len(rows), **{k: sum(r[k] for r in rows) for k in
                    ['all_success', 'same_operational_state', 'same_content_state', 'covers_reference_read_outputs', 'state_plus_read_reference_pass']}})
    failure = Counter(r['gold']['steps'][-1]['tool']+':'+str(r['gold']['steps'][-1]['error']) for r in primary if not r['gold']['all_success'])
    fixtures = own_fixtures(cls)
    out = {'status': 'offline_state_checker_feasibility_not_official_benchmark_score', 'date': '2026-10-27',
           'commit': COMMIT, 'tasks': len(primary), 'by_split': by_split,
           'first_failure_tool_and_code': dict(failure), 'mutations': mutation_summary,
           'reference_state_and_read_checker_is_authored_diagnostic': True,
           'no_user_intent_semantic_checker_available': True, 'fixtures': fixtures,
           'hashes': {'code': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'upstream_gfs': hashlib.sha256((repo/'plugins/gfs_plugin.py').read_bytes()).hexdigest(),
                      'inputs': inputs},
           'checks': {'repeat_gold_deterministic': True, 'network_or_model_calls': 0,
                      'host_filesystem_operations_from_gfs': False, 'source_dataset_redistributed': False}}
    OUT.mkdir(exist_ok=True, parents=True)
    (OUT/'summary.json').write_text(json.dumps(out, indent=2))
    (OUT/'per_task.json').write_text(json.dumps(primary, indent=2))
    (OUT/'mutations.json').write_text(json.dumps(mutated, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != 'hashes'}, indent=2))


if __name__ == '__main__':
    main()
