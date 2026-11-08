"""E008: join published BAG logs; no network, models, or upstream code execution."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parents[1]
REVISION = '58b2215ce80c74731d6d8dd3a843d2767943f569'
PROTOCOL_SHA = '0fb416eae1d2eb23383b043f806be3dee53d48f80e7c5dfabaeb527069dbda4e'
PILOT_ID = '955636187401991675'
FILES = {
    'user': 'dev/user_qwen3-8b_gemini-2.5-flash_50_belief.jsonl',
    'final': 'dev/final_qwen3-8b_gemini-2.5-flash_50_belief_prompt.jsonl',
    'judge': 'dev/judge_final_qwen3-8b_gemini-2.5-flash_gemini-2.5-flash_50_belief_prompt.jsonl',
    'clarify': 'dev/clarify_qwen3-8b_50_belief.jsonl',
}
CLASSES = ['apparently_complete', 'partial', 'ambiguous', 'explicit_unknown_or_refusal', 'nonresponsive', 'unclear']


def audit_policy(logs):
    """Read saved policy fields; do not execute upstream code or infer semantic acts."""
    if 'clarify' not in logs:
        return {'status': 'clarify_log_not_yet_available_null_rows_not_interpreted_as_noask'}
    counts = Counter(); rows = []; issues = []
    for sid, policy in logs['clarify'].items():
        g = policy.get('generations') or {}; ctx = policy.get('context') or {}
        strategy = g.get('strategy'); counts[str(strategy)] += 1
        # Independently inspect explicit raw labels, rather than treating every
        # non-clarification row as an error-free no-ask action.
        raw_labels = re.findall(r'^STRATEGY:\s*([^\n]+)', g.get('raw_response') or '', re.M | re.I)
        normalized = [s.strip().lower().replace(' ', '_') for s in raw_labels]
        checks = {'valid_strategy': strategy in ['direct_answer', 'abstain', 'clarification_question'],
                  'single_raw_label_matches': normalized == [strategy],
                  'nonempty_policy_response': isinstance(g.get('response'), str) and bool(g['response'].strip())}
        expected_active = bool(strategy and 'clarification' in strategy and g.get('response'))
        user = logs['user'].get(sid)
        checks['user_id_present'] = user is not None
        active = user is not None and nonempty_response(user)
        checks['routing_matches_response_presence'] = active == expected_active
        row = {'source_id': sid, 'strategy': strategy, 'response_present': active, 'checks': checks}
        if active:
            uc = user.get('context') or {}; ds = ctx.get('disambiguations') or []
            chosen = random.Random(str(sid)).choice(ds) if ds else None
            expected_refs = [chosen['answer']] if ds else ctx.get('references')
            checks.update({'same_original_question': ctx.get('question') == uc.get('question'),
                           'same_clarification': g.get('response') == uc.get('clarification_q'),
                           'selected_disambiguation_matches_source_rule': chosen == uc.get('disambiguation'),
                           'reference_matches_source_rule': expected_refs == uc.get('reference')})
            row['explicit_disambiguation_count'] = len(ds)
            row['user_prompt_secret'] = 'disambiguated_question' if ds else 'first_reference_surface_form'
        failures = [k for k,v in checks.items() if not v]
        if failures: issues.append({'source_id': sid, 'failed_checks': failures})
        rows.append(row)
    active_ids = {sid for sid, r in logs['user'].items() if nonempty_response(r)}
    policy_active_ids = {r['source_id'] for r in rows if r['strategy'] == 'clarification_question'}
    return {'status': 'saved_policy_routing_verified' if not issues and active_ids == policy_active_ids else 'policy_audit_has_issues',
            'strategy_counts': dict(counts), 'rows': len(rows), 'issues': issues,
            'active_ids_missing_from_policy_clarification': sorted(active_ids-policy_active_ids),
            'policy_clarification_without_user_response': sorted(policy_active_ids-active_ids),
            'interpretation': 'Operational branch labels, not a semantic count of all questions in response text; not propensities or causal controls.',
            'per_case': rows}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def read_log(path):
    rows = []; errors = []; by_id = {}
    for line, text in enumerate(path.read_text().splitlines(), 1):
        if not text.strip():
            continue
        try:
            row = json.loads(text)
            if not isinstance(row, dict) or not isinstance(row.get('id'), str):
                raise ValueError('missing_string_id')
            rows.append(row)
            by_id.setdefault(row['id'], []).append((line, row))
        except (ValueError, TypeError) as exc:
            errors.append({'line': line, 'error': type(exc).__name__, 'reason': str(exc)})
    duplicate_ids = {key: len(items) for key, items in by_id.items() if len(items) > 1}
    conflicting = [key for key, items in by_id.items() if len({digest(r) for _, r in items}) > 1]
    unique = {key: items[0][1] for key, items in by_id.items() if len(items) == 1}
    lines = {key: items[0][0] for key, items in by_id.items() if len(items) == 1}
    configs = Counter(digest(r.get('generation_config')) for r in rows)
    return unique, lines, {'rows': len(rows), 'unique_ids': len(by_id), 'unambiguous_ids': len(unique),
            'duplicate_ids': duplicate_ids, 'conflicting_duplicate_ids': conflicting, 'parse_errors': errors,
            'missing_generations_key': sum('generations' not in r for r in rows),
            'null_generations': sum(r.get('generations') is None for r in rows),
            'nonnull_generations': sum(r.get('generations') is not None for r in rows),
            'generation_types': dict(Counter(type(r.get('generations')).__name__ for r in rows)),
            'generation_config_hash_counts': dict(configs)}


def nonempty_response(row):
    g = row.get('generations')
    return isinstance(g, dict) and isinstance(g.get('response'), str) and bool(g['response'].strip())


def populated_generation(stage, row):
    g = row.get('generations')
    # Judge writes a container even when both underlying verdicts are null.
    if stage == 'judge':
        return isinstance(g, dict) and any(g.get(k) is not None for k in ['final', 'final_anyref'])
    return g is not None


def class_agreement(left, right, field):
    pairs = [(a[field], b[field]) for a, b in zip(left, right)]
    n = len(pairs); observed = sum(a == b for a, b in pairs)/n
    a_counts = Counter(a for a, _ in pairs); b_counts = Counter(b for _, b in pairs)
    expected = sum(a_counts[k]*b_counts[k] for k in set(a_counts) | set(b_counts))/n**2
    return {'n': n, 'agreed': sum(a == b for a, b in pairs), 'agreement': observed,
            'cohens_kappa': (observed-expected)/(1-expected) if expected < 1 else None,
            'disagreement_ids': [a['blind_id'] for a, b in zip(left, right) if a[field] != b[field]],
            'confusion': [{'lead': a, 'reviewer': b, 'n': count} for (a, b), count in Counter(pairs).items()]}


def annotations(out, pack, outcomes):
    paths = {k: out/('annotations_'+k+'.json') for k in ['lead', 'reviewer']}
    if not all(p.exists() for p in paths.values()):
        return {'status': 'pending_two_independent_annotations', 'present': {k: p.exists() for k, p in paths.items()}}
    frozen_hashes = {k: sha(p) for k, p in paths.items()}
    freeze_path = out/'annotation_freeze.json'
    if freeze_path.exists():
        assert json.loads(freeze_path.read_text())['files_sha256'] == frozen_hashes, 'Independent labels changed after freeze; keep adjudication separate'
    else:
        write(freeze_path, {'files_sha256': frozen_hashes, 'blind_pack_sha256': sha(out/'blind_pack.json'),
                            'freeze_note': 'Both authors declared blind coding complete before outcome unblinding. Original labels immutable; adjudication, if needed, separate.'})
    by_reviewer = {}
    for role, path in paths.items():
        rows = json.loads(path.read_text())
        if isinstance(rows, dict):
            rows = rows.get('annotations', rows.get('labels'))
        assert isinstance(rows, list), (role, 'expected list')
        mapping = {r['blind_id']: r for r in rows}
        assert len(mapping) == len(rows) == len(pack)
        assert set(mapping) == {r['blind_id'] for r in pack}
        for row in rows:
            assert row['surface_class'] in CLASSES
            assert isinstance(row['has_explicit_unknown'], bool)
            assert row['has_useful_constraint_on_surface'] in ['yes', 'no', 'unclear']
            assert row['annotation_status'] == 'assistant_provisional'
            assert row.get('rationale')
        by_reviewer[role] = [mapping[p['blind_id']] for p in pack]
    agreement = {field: class_agreement(by_reviewer['lead'], by_reviewer['reviewer'], field)
                 for field in ['surface_class', 'has_explicit_unknown', 'has_useful_constraint_on_surface']}
    tables = {}
    outcome_map = {r['blind_id']: r for r in outcomes}
    for role, rows in by_reviewer.items():
        tables[role] = {}
        for exclude_pilot in [False, True]:
            table = []
            for label in CLASSES:
                labeled = [r for r in rows if r['surface_class'] == label]
                selected = [outcome_map[r['blind_id']] for r in labeled if r['blind_id'] in outcome_map
                            and not (exclude_pilot and outcome_map[r['blind_id']]['pilot_seen_by_eda'])]
                table.append({'surface_class': label, 'n': len(selected),
                              'target_judge_positive': sum(r['target_verdict'] == 1 for r in selected),
                              'anyref_judge_positive': sum(r['anyref_verdict'] == 1 for r in selected)})
            tables[role]['excluding_pilot' if exclude_pilot else 'all'] = table
    return {'status': 'both_annotations_validated_unblinded_descriptive_only',
            'files_sha256': {k: sha(p) for k, p in paths.items()}, 'agreement': agreement,
            'outcome_tables': tables,
            'caution': 'Two exploratory assistant reviews, not independent human gold. No causal inference. No adjudicated labels substituted.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'results/bag_response_audit')
    args = parser.parse_args(); data = args.data; out = args.output
    protocol = ROOT/'research/E008_PROTOCOL.md'
    assert sha(protocol) == PROTOCOL_SHA
    source_manifest = json.loads((data/'manifest.json').read_text())
    assert source_manifest['commit'] == REVISION
    for spec in source_manifest['files']:
        assert sha(data/spec['path']) == spec['sha256'], spec['path']
    logs = {}; lines = {}; audits = {}; hashes = {}
    for name, relative in FILES.items():
        path = data/relative
        if not path.exists():
            if name != 'clarify':
                raise FileNotFoundError(path)
            continue
        assert any(s['path'] == relative for s in source_manifest['files']), 'Unpinned input: '+relative
        logs[name], lines[name], audits[name] = read_log(path)
        audits[name]['populated_generation_rows'] = sum(populated_generation(name, r) for r in logs[name].values())
        hashes[relative] = sha(path)
    policy_audit = audit_policy(logs)
    ids = set().union(*(set(x) for x in logs.values()))
    mismatches = {name: sorted(ids-set(rows)) for name, rows in logs.items()}
    selected = sorted((r for r in logs['user'].values() if nonempty_response(r)), key=lambda r:r['id'])
    random.Random(20261027).shuffle(selected)
    pack = []; mapping = []; semantics = []; outcomes = []; joins = []
    configs = {}
    if 'clarify' in logs:
        configs['clarify'] = {digest(r.get('generation_config')): r.get('generation_config')
                              for r in logs['clarify'].values()}
    for number, user in enumerate(selected, 1):
        sid = user['id']; bid = f'B{number:03d}'
        ctx = user.get('context') or {}; ug = user['generations']
        blind = {'blind_id': bid, 'original_query': ctx.get('question'),
                 'clarification_question': ctx.get('clarification_q'), 'user_answer': ug['response']}
        pack.append(blind)
        mapping.append({'blind_id': bid, 'source_id': sid, 'pilot_seen_by_eda': sid == PILOT_ID})
        join = {'source_id': sid, 'blind_id': bid, 'source_lines': {k:v.get(sid) for k,v in lines.items()},
                'field_hashes': {k:digest(v) for k,v in blind.items() if k!='blind_id'}, 'checks': {}}
        reasons = []
        if any(not isinstance(blind[k], str) or not blind[k].strip() for k in ['original_query','clarification_question','user_answer']):
            reasons.append('missing_blind_text')
        if any(sid not in logs[name] for name in ['final','judge']):
            reasons.append('unmatched_or_duplicate_id')
        else:
            final = logs['final'][sid]; judge = logs['judge'][sid]
            fc = final.get('context') or {}; jc = judge.get('context') or {}
            fg = final.get('generations'); jg = judge.get('generations') or {}
            prediction = fg[0] if isinstance(fg,list) and len(fg)==1 and isinstance(fg[0],str) else None
            checks = {
                'same_question_user_final': ctx.get('question') == fc.get('question'),
                'same_question_user_judge': ctx.get('question') == jc.get('question'),
                'same_clarification_user_final': ctx.get('clarification_q') == fc.get('clarification_q'),
                'same_response_user_final': ug['response'] == fc.get('user_answer'),
                'same_prediction_final_judge': prediction is not None and prediction == jc.get('generation_candidate'),
                'same_selected_references_user_final': ctx.get('reference') == fc.get('reference'),
                'same_disambiguation_user_final': ctx.get('disambiguation') == fc.get('disambiguation'),
                'judge_ref_present_in_user_refs': jc.get('final_ref') in (ctx.get('reference') or []),
            }
            join['checks'].update(checks)
            reasons.extend(k for k, passed in checks.items() if not passed)
            target = (jg.get('final') or {}).get('verdict')
            anyref = (jg.get('final_anyref') or {}).get('verdict')
            if type(target) is not int or target not in [0,1]:reasons.append('missing_or_invalid_target_verdict')
            if type(anyref) is not int or anyref not in [0,1]:reasons.append('missing_or_invalid_anyref_verdict')
            join['field_hashes'].update({'prediction': digest(prediction), 'selected_references': digest(ctx.get('reference')),
                                         'judge_target_reference': digest(jc.get('final_ref'))})
            semantics.append({**blind, 'selected_disambiguation': ctx.get('disambiguation'),
                              'simulator_reference': ctx.get('reference'), 'judge_target_reference': jc.get('final_ref'),
                              'all_reference_groups': jc.get('final_all_refs'),
                              'semantic_consistency': 'unannotated', 'rationale': None,
                              'note': 'No final prediction or judge outcome included. Target references are published annotations, not independent verification.'})
            outcomes.append({'blind_id': bid, 'source_id': sid, 'target_verdict': target, 'anyref_verdict': anyref,
                             'pilot_seen_by_eda': sid == PILOT_ID, 'selected_disambiguation_present': ctx.get('disambiguation') is not None,
                             'simulator_reference_group_count': len(ctx.get('reference') or []),
                             'judge_target_is_first_simulator_reference': bool(ctx.get('reference')) and jc.get('final_ref') == ctx['reference'][0],
                             'usable_join': not reasons})
            for stage,row in [('user',user),('final',final),('judge',judge)]:
                configs.setdefault(stage,{})[digest(row.get('generation_config'))] = row.get('generation_config')
        join['status'] = 'usable' if not reasons else 'excluded'
        join['exclusion_reasons'] = reasons
        joins.append(join)
    out.mkdir(parents=True,exist_ok=True)
    pack_bytes=(json.dumps(pack,ensure_ascii=False,indent=2)+'\n').encode()
    if (out/'blind_pack.json').exists():
        assert (out/'blind_pack.json').read_bytes()==pack_bytes, 'Blind pack changed; preserve existing annotations and audit before proceeding'
    write(out/'blind_pack.json',pack)
    (out/'blind_pack.txt').write_text('\n\n'.join(f"{x['blind_id']}\nOriginal query: {x['original_query']}\nClarification question: {x['clarification_question']}\nUser answer: {x['user_answer']}" for x in pack)+'\n')
    usable = [r for r in outcomes if r['usable_join']]
    user_active_ids = {r['id'] for r in selected}
    stage_active_ids = {name: {sid for sid, row in rows.items() if populated_generation(name, row)}
                        for name, rows in logs.items() if name != 'clarify'}
    for name, value in [('blind_mapping_internal',mapping),('semantics_pack',semantics),('outcomes_internal',outcomes),('join_audit',joins)]:
        write(out/(name+'.json'),value)
    summary = {'date':'2026-10-27','status':'published_logs_audit_no_new_models_not_causal_VOI',
               'source_revision':REVISION,'source_file_audits':audits,'unmatched_ids':mismatches,
               'unique_ids_union':len(ids),'response_cases':len(pack),'usable_join_cases':len(usable),
               'excluded_cases':len(pack)-len(usable),'exclusion_reasons':dict(Counter(reason for r in joins for reason in r['exclusion_reasons'])),
               'inactive_user_rows':sum(not nonempty_response(r) for r in logs['user'].values()),
               'nonnull_user_rows_without_nonempty_response':sum(r.get('generations') is not None and not nonempty_response(r) for r in logs['user'].values()),
               'active_ids_vs_response_cases':{name:{'missing':sorted(user_active_ids-active),
                                                     'extra':sorted(active-user_active_ids)}
                                               for name,active in stage_active_ids.items()},
               'policy_selection_status':policy_audit['status'],
               'policy_summary':{k:v for k,v in policy_audit.items() if k != 'per_case'},
               'target_judge_positive':sum(r['target_verdict']==1 for r in usable),
               'anyref_judge_positive':sum(r['anyref_verdict']==1 for r in usable),
               'judge_pairs':[{"target":a,"anyref":b,"n":n} for (a,b),n in Counter((r['target_verdict'],r['anyref_verdict']) for r in usable).items()],
               'selected_disambiguation_present':sum(r['selected_disambiguation_present'] for r in usable),
               'selected_disambiguation_missing':sum(not r['selected_disambiguation_present'] for r in usable),
               'reference_group_counts_by_goal_presence':[
                   {'selected_disambiguation_present':goal,'simulator_reference_group_count':count,'n':n}
                   for (goal,count),n in sorted(Counter((r['selected_disambiguation_present'],r['simulator_reference_group_count']) for r in usable).items())],
               'judge_target_is_first_simulator_reference':sum(r['judge_target_is_first_simulator_reference'] for r in usable),
               'judge_counts_by_goal_presence':[
                   {'selected_disambiguation_present':goal,'n':len(part),
                    'target_judge_positive':sum(r['target_verdict']==1 for r in part),
                    'anyref_judge_positive':sum(r['anyref_verdict']==1 for r in part)}
                   for goal in [True,False] for part in [[r for r in usable if r['selected_disambiguation_present']==goal]]],
               'pilot_seen_by_eda':sum(r['pilot_seen_by_eda'] for r in usable),
               'model_calls':0,'source_data_copied_in_full':False}
    manifest = {'revision':REVISION,'source':'https://huggingface.co/datasets/jsbaan/bag-paper-data',
                'license':'CC BY-SA 3.0 per source data card; inherited from AmbigQA',
                'attribution':'jsbaan/bag-paper-data; Clarify, Abstain or Answer? Strategising in Conversation with Belief-Augmented Generation; source card credits AmbigQA (Min et al., 2020).',
                'protocol_sha256':sha(protocol),'code_sha256':sha(Path(__file__)),'source_manifest_sha256':sha(data/'manifest.json'),
                'source_file_sha256':hashes,'generation_configs':configs,'blind_pack_sha256':sha(out/'blind_pack.json'),
                'blind_pack_seed':20261027,'pilot_seen_id':PILOT_ID,'model_calls':0,
                'notes':['Only selected query/question/answer extracts saved; full original logs stay outside repository.',
                         'Surface and semantic packs exclude model prediction, judge outcome and simulator reasoning.',
                         'Generation model/config strings are recorded; exact weights/API snapshot not independently established.']}
    write(out/'summary.json',summary);write(out/'manifest.json',manifest)
    write(out/'policy_audit.json',policy_audit)
    write(out/'annotation_analysis.json',annotations(out,pack,usable))
    print(json.dumps({k:summary[k] for k in ['status','unique_ids_union','response_cases','usable_join_cases','excluded_cases','policy_selection_status']},indent=2))


if __name__=='__main__':
    main()
