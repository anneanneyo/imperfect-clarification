"""Offline audit of a pinned external repository; no API calls or third-party copies.

python3 experiments/audit_flight_logs.py --repo /path/to/VOI_communication
External code is imported only from the inspected constants/utils/evaluate modules.
Bootstrap units are source dataset records, not turns or repeated sampled episodes.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

COMMIT = '27466a7832d5aafff82017a659e08942e18b01ae'
ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def interval(values, groups):
    """Percentile cluster bootstrap of episode-weighted mean; descriptive only."""
    buckets = defaultdict(list)
    for value, group in zip(values, groups):
        buckets[group].append(value)
    ids = sorted(buckets)
    rng = random.Random(20261015)
    estimates = []
    for _ in range(4000):
        sample = [v for key in rng.choices(ids, k=len(ids)) for v in buckets[key]]
        estimates.append(mean(sample))
    estimates.sort()
    return {'mean': mean(values), 'ci95': [estimates[100], estimates[3899]]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == COMMIT, 'Unexpected revision; inspect before running'
    base = repo / 'flight_rec'
    sys.path.insert(0, str(base))
    utils = load_module('flight_utils', base / 'utils.py')
    evaluator = load_module('flight_evaluator', base / 'evaluate.py')
    output = ROOT / 'results' / 'flight_audit'
    output.mkdir(parents=True, exist_ok=True)
    replay = subprocess.check_output([sys.executable, str(base / 'evaluate.py'), 'openai/gpt-5.4'], text=True)
    (output / 'upstream_table.txt').write_text(replay)

    datasets = {name: json.loads((base / f'data/{name}.json').read_text()) for name in ['train', 'eval']}
    games = datasets['eval']
    logs = {}
    fingerprints = {}
    anomalies = []
    for strategy in ['voi', 'confidence']:
        paths = sorted((base / f'results/{strategy}_openai-gpt-5.4').glob('*.jsonl'))
        assert len(paths) == 100
        logs[strategy] = [[json.loads(line) for line in p.read_text().splitlines() if line.strip()] for p in paths]
        for p in paths:
            fingerprints[str(p.relative_to(repo))] = hashlib.sha256(p.read_bytes()).hexdigest()
        for i, turns in enumerate(logs[strategy]):
            assert 1 <= len(turns) <= 4
            assert [t['turn'] for t in turns] == list(range(len(turns))), (strategy, i)
            if len(turns) < 4:
                assert turns[-1]['question_next'] is None, (strategy, i, 'unexplained truncation')
            for t in turns:
                assert t['game_idx'] == i and t['questions_asked'] == t['turn']
                assert math.isclose(t['best_possible_reward'], max(t['ground_truth_scores']), abs_tol=1e-4)
                assert math.isclose(t['predicted_reward'], t['ground_truth_scores'][t['predicted_index']], abs_tol=1e-4)
                expected_regret = t['best_possible_reward'] - t['predicted_reward']
                expected_correct = t['predicted_index'] == t['ground_truth_index']
                if not math.isclose(t['reward_regret'], expected_regret, abs_tol=1.1e-4) or t['prediction_correct'] != expected_correct:
                    anomalies.append({'strategy': strategy, 'episode': i, 'turn': t['turn'],
                                      'stored_regret': t['reward_regret'], 'recomputed_regret': expected_regret,
                                      'stored_correct': t['prediction_correct'], 'recomputed_correct': expected_correct})

    # Reconstruct source identities from documented defaults, then verify against both logs.
    groups = []
    rng = random.Random(8)
    for i in range(100):
        game = utils.choose_game(games, 5, rng)
        groups.append(games.index(game))
        _, query, _ = utils.build_cot_prompt(game, 5, False, rng)
        _, scores = utils.gold_from_theta(game, query)
        for strategy in logs:
            for t in logs[strategy][i]:
                assert all(abs(a-b) < 1.1e-4 for a, b in zip(scores, t['ground_truth_scores']))

    def select(turns, method, parameter):
        if method == 'NoQuestion':
            return turns[0]
        if method == 'Fixed':
            return turns[min(parameter, len(turns)-1)]
        field = 'voi_next' if method == 'VOI' else 'confidence'
        index = 0
        while index < len(turns)-1:
            value = turns[index][field]
            ask = value is not None and (value >= parameter if method == 'VOI' else value < parameter)
            if not ask:
                break
            index += 1
        return turns[index]

    settings = [('NoQuestion', 0)] + [('Fixed', n) for n in [1, 2, 3]] + [('Confidence', p) for p in [.5, .7, .9]]
    rows = []
    for cost in [.01, .05, .1]:
        selections = {}
        for method, parameter in settings + [('VOI', cost)]:
            source = logs['voi' if method == 'VOI' else 'confidence']
            stops = [select(t, method, parameter) for t in source]
            q, reward = mean(t['questions_asked'] for t in stops), mean(t['predicted_reward'] for t in stops)
            functions = {'NoQuestion': lambda: evaluator.noquestion_metrics(source),
                         'Fixed': lambda: evaluator.fixed_metrics(source, parameter),
                         'Confidence': lambda: evaluator.confidence_metrics(source, parameter),
                         'VOI': lambda: evaluator.voi_metrics(source, parameter)}
            original = functions[method]()
            assert math.isclose(q, original[0], abs_tol=1e-10) and math.isclose(reward, original[1], abs_tol=1e-10)
            values = [t['predicted_reward'] - cost*t['questions_asked'] for t in stops]
            key = f'{method}:{parameter}'
            selections[key] = values
            rows.append({'cost': cost, 'method': method, 'parameter': parameter, 'questions': q,
                         'reward': reward, 'utility': mean(values),
                         'accuracy_recomputed': mean(t['predicted_index'] == t['ground_truth_index'] for t in stops)})
        # Named descriptive contrasts; no selection of the best baseline on evaluation data.
        selections['VOI_own_turn0'] = [t[0]['predicted_reward'] for t in logs['voi']]
        for baseline in ['NoQuestion:0', 'Fixed:1', 'VOI_own_turn0']:
            delta = [a-b for a,b in zip(selections[f'VOI:{cost}'], selections[baseline])]
            rows.append({'cost': cost, 'contrast': f'VOI minus {baseline}', **interval(delta, groups)})

    transitions = {}
    for strategy, episodes in logs.items():
        delta = [episode[t+1]['predicted_reward']-episode[t]['predicted_reward'] for episode in episodes for t in range(len(episode)-1)]
        transitions[strategy] = {'transitions': len(delta), 'improved': sum(d > 1e-8 for d in delta),
                                 'worsened': sum(d < -1e-8 for d in delta), 'unchanged': sum(abs(d) <= 1e-8 for d in delta)}

    counts = Counter(groups)
    high_cost_cases = []
    for i, trajectory in enumerate(logs['voi']):
        stop = select(trajectory, 'VOI', .1)
        if stop['questions_asked']:
            high_cost_cases.append({'episode': i, 'questions': stop['questions_asked'],
                                    'reward_change': stop['predicted_reward']-trajectory[0]['predicted_reward']})
    metadata = {}
    for name, data in datasets.items():
        fingerprint = lambda g: json.dumps(g, sort_keys=True)
        metadata[name] = {'records': len(data), 'eligible_support5': sum(len(g['rounds']) >= 6 for g in data),
                          'unique_records': len({fingerprint(g) for g in data})}
    train_hashes = {json.dumps(g, sort_keys=True) for g in datasets['train']}
    metadata['exact_train_eval_overlap'] = sum(json.dumps(g, sort_keys=True) in train_hashes for g in games)
    hashes = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [base / 'evaluate.py', base / 'utils.py', base / 'constants.py', base / 'flight_rec.py',
                        base / 'voi_utils.py', base / 'run_experiments.sh', base / 'data/train.json', base / 'data/eval.json']}
    result = {'status': 'offline_reanalysis_of_author_logs_not_new_llm_rollouts', 'commit': revision,
              'python': sys.version, 'checks': 'replay_and_reward_checks_passed; auxiliary_anomalies_recorded',
              'auxiliary_field_anomalies': anomalies, 'datasets': metadata,
              'source_records_in_100_episodes': len(counts), 'source_record_multiplicities': dict(counts),
              'voi_own_turn0_mean_reward': mean(t[0]['predicted_reward'] for t in logs['voi']),
              'high_cost_question_cases': high_cost_cases,
              'group_definition': 'reconstructed eval record; shared underlying users across records unknown',
              'rows': rows, 'transitions_descriptive_only': transitions,
              'artifact_sha256': hashes, 'transcript_sha256': fingerprints,
              'limitations': ['No response-channel shift or adaptation tested', 'Different prediction prompts across strategies',
                              'CIs conditional on these logs, not full model reruns', 'No new LLM calls',
                              'No counterfactual answers to unasked questions', 'Transition counts are correlated, not independent samples']}
    (output / 'audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ['artifact_sha256','transcript_sha256','python','source_record_multiplicities']}, indent=2))


if __name__ == '__main__':
    main()
