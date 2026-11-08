"""Independent arithmetic audit of saved E027 choices and question rankings.

Does not import the original analyzer or regenerate model outputs. Heuristic
choice distributions and provisional semantic labels are inputs to this audit.
"""
import json
from collections import Counter
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/userbench_executor_e027'

def read(path):
    return json.loads(path.read_text())

def key(row):
    return row['case_id'], row['aspect'], row['mask'], row['policy']

def mean(rows):
    return {metric: sum(r[metric] for r in rows) / len(rows) for metric in rows[0]}

def check(actual, saved):
    assert actual == {k: F(v) for k, v in saved.items()}, (actual, saved)

def main():
    truth = {(r['case_id'], r['aspect'], r['option_id']): r for r in read(ROOT / 'results/userbench_semantics_e026/reviewed_option_evaluations.json')}
    gold = {r['case_id']: r for r in read(ROOT / 'results/userbench_train_e025/private_cases.json')}
    raw = {r['id']: r for r in map(json.loads, (DATA / 'generations.jsonl').read_text().splitlines())}
    choices = {key(r): r for r in read(DATA / 'executor_choices.json')}
    assert len(choices) == 576 and len(raw) == 192
    scores = {}
    for k, row in choices.items():
        case, aspect, mask, policy = k
        dist = {o: F(v) for o, v in row['distribution'].items()}
        assert all(v >= 0 for v in dist.values()) and sum(dist.values()) == 1
        if policy == 'qwen3_4b':
            generation = raw[row['id']]
            assert generation['error'] is None
            pairs = json.loads(generation['raw_output'], object_pairs_hook=list)
            assert len(pairs) == 1 and pairs[0][0] == 'id'
            selected = pairs[0][1]
            assert selected in dist and dist[selected] == 1
            assert row['model_chosen_id'] == selected
        if policy == 'uniform':
            assert set(dist.values()) == {F(1, 18)}
        labels = gold[case]['aspects'][aspect]
        values = {'valid': sum(dist.values()), 'source_best': dist[labels['best_id']],
                  'source_acceptable': sum(dist[o] for o in labels['acceptable_ids'])}
        values['semantic_lower'] = sum(v for o, v in dist.items() if truth[case, aspect, o]['conjunction'] == 'T')
        values['semantic_upper'] = sum(v for o, v in dist.items() if truth[case, aspect, o]['conjunction'] != 'F')
        scores[k] = values
    for row in read(DATA / 'executor_scores.json'):
        check(scores[key(row)], {m: row[m] for m in scores[key(row)]})
    questions = {}
    for row in read(DATA / 'question_outcomes.json'):
        case, aspect, mask, policy = key(row)
        aspects = sorted(gold[case]['aspects'])
        before = mean([scores[case, a, 'no_information', policy] for a in aspects])
        after = mean([scores[case, a, mask if a == aspect else 'no_information', policy] for a in aspects])
        check(before, row['baseline'])
        check(after, row['after_one_question'])
        for metric in ['source_best', 'source_acceptable']:
            assert after[metric] - before[metric] == F(row[metric + '_gain'])
        lower = upper = F(0)
        for option, value in choices[case, aspect, mask, policy]['distribution'].items():
            delta = (F(value) - F(choices[case, aspect, 'no_information', policy]['distribution'][option])) / 4
            label = truth[case, aspect, option]['conjunction']
            if label == 'T':
                lower += delta
                upper += delta
            elif label == 'U':
                lower += min(delta, 0)
                upper += max(delta, 0)
        check({'lower': lower, 'upper': upper}, row['semantic_gain'])
        questions[case, policy, mask, row['question']] = (lower, upper, after['source_acceptable'] - before['source_acceptable'])
    for row in read(DATA / 'ranking_details.json'):
        case, policy = row['case_id'], row['policy']
        winners = {}
        by_mask = {}
        for mask in ['no_information', 'full', 'first_only', 'second_only']:
            candidates = {q: v for (c, p, m, q), v in questions.items() if (c, p, m) == (case, policy, mask)}
            by_mask[mask] = candidates
            robust = [q for q, v in candidates.items() if all(v[0] > other[1] for name, other in candidates.items() if name != q)]
            winners[mask] = robust[0] if robust else None
        assert winners == row['robust_semantic_winners']
        full = by_mask['full']
        maximizers = sorted(q for q, v in full.items() if v[2] == max(x[2] for x in full.values()))
        for mask in ['first_only', 'second_only']:
            partial = by_mask[mask]
            gap = max(v[2] for v in partial.values()) - max(partial[q][2] for q in maximizers)
            saved = row['source_acceptable_tie_controlled_gaps'][mask]
            assert gap == F(saved['gap']) and maximizers == saved['full_maximizers']
            reversal = winners['full'] is not None and winners[mask] is not None and winners['full'] != winners[mask]
            assert reversal == row['robust_semantic_reversals'][mask]
    summary = read(DATA / 'summary.json')
    for policy, modes in summary['all_aspects_same_mask_diagnostic'].items():
        for mask, metrics in modes.items():
            check(mean([v for (c, a, m, p), v in scores.items() if (m, p) == (mask, policy)]), metrics)
    saved_questions = read(DATA / 'question_outcomes.json')
    for policy, modes in summary['constant_one_question'].items():
        for mask, constants in modes.items():
            for question, metrics in constants.items():
                rows = [r for r in saved_questions if (r['policy'], r['mask'], r['question']) == (policy, mask, question)]
                check(mean([{k: F(v) for k, v in r['after_one_question'].items()} for r in rows]), metrics)
    for policy, modes in summary['information_transitions'].items():
        for mask, saved in modes.items():
            rows = [r for r in saved_questions if (r['policy'], r['mask']) == (policy, mask)]
            actual = {'aspect_transitions': len(rows),
                'source_acceptable_harms': sum(F(r['source_acceptable_gain']) < 0 for r in rows),
                'source_acceptable_helps': sum(F(r['source_acceptable_gain']) > 0 for r in rows),
                'semantic_certified_harms': sum(F(r['semantic_gain']['upper']) < 0 for r in rows),
                'semantic_certified_helps': sum(F(r['semantic_gain']['lower']) > 0 for r in rows)}
            assert actual == saved
    errors = []
    for k, row in choices.items():
        case, aspect, mask, policy = k
        if policy != 'qwen3_4b':
            continue
        chosen = row['model_chosen_id']
        evaluation = truth[case, aspect, chosen]
        revealed = {'full': [0, 1], 'first_only': [0], 'second_only': [1], 'no_information': []}[mask]
        errors.append({'case': case, 'aspect': aspect, 'mask': mask, 'chosen': chosen,
                       'source_acceptable': bool(scores[k]['source_acceptable']),
                       'semantic': evaluation['conjunction'], 'criterion_values': evaluation['values'],
                       'violates_revealed_rule': any(evaluation['values'][i] == 'F' for i in revealed)})
    report = {'verified': {'choice_scores': len(scores), 'question_outcomes': len(questions), 'rankings': 36},
              'boundary': 'Saved heuristic distributions and provisional semantic labels are not independently reconstructed.',
              'by_mask': {}, 'selected_options': errors}
    for mask in ['no_information', 'full', 'first_only', 'second_only']:
        rows = [r for r in errors if r['mask'] == mask]
        report['by_mask'][mask] = {'semantic_counts': dict(Counter(r['semantic'] for r in rows)),
            'violates_revealed_rule': sum(r['violates_revealed_rule'] for r in rows),
            'source_acceptable': sum(r['source_acceptable'] for r in rows)}
    destination = ROOT / 'results/e027_audit.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'selected_options'}, indent=2))

if __name__ == '__main__':
    main()
