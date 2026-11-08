"""E004: exact, post-hoc diagnostics; no LLM, no fitting, no new dependencies."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from learn_question_value import COMMIT, INDICES, catalog_id, response_channel, gain

ROOT = Path(__file__).resolve().parents[1]


def policy_value(scores, actual, assumed):
    actions = (assumed.T @ scores).argmax(axis=1)
    return float(sum(np.mean(actual[:, r] * scores[:, a])
                     for r, a in enumerate(actions)))


def quantiles(a):
    return np.quantile(a, [0, .25, .5, .75, 1]).tolist()


def examples():
    # Every pair contains the true state; neither world uses false answers.
    pairs = [{0, 1}, {1, 2}, {0, 2}]
    a = np.eye(3)
    b = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    assert all(z in pairs[r] for c in [a, b] for z, r in zip(*np.where(c > 0)))
    scores = np.eye(3)
    # Interior channels avoid reliance on deterministic extreme mechanisms.
    ca = .8*a + .2*b
    cb = .2*a + .8*b
    va = policy_value(scores, ca, ca)
    vb = policy_value(scores, cb, ca)
    assert np.allclose(ca.mean(0), cb.mean(0))
    assert np.isclose(va, .8) and np.isclose(vb, .2)
    # A randomized alternative executor selecting within each reported pair.
    symmetric_value = .5
    assert np.isclose(policy_value(scores, cb, cb), .8)
    # Within a fixed positive sign, response propensity reveals magnitude.
    magnitude_scores = np.column_stack([np.full(4, .5), [.1, .2, 2., 3.]])
    same_sign = np.ones((4, 1))
    endogenous = np.array([[0., 1.], [0., 1.], [1., 0.], [1., 0.]])
    vi = gain(magnitude_scores, same_sign)
    ve = gain(magnitude_scores, endogenous)
    assert np.isclose(vi, 0) and np.isclose(ve, .175)
    # General nonnegativity, upper bound by FULL state, and action permutations.
    rng = np.random.default_rng(20261020)
    for _ in range(100):
        s = rng.normal(size=(5, 4))
        c = rng.dirichlet(np.ones(3), size=5)
        v = gain(s, c)
        assert -1e-12 <= v <= gain(s, np.eye(5)) + 1e-12
        assert np.isclose(v, gain(s[:, [3, 1, 0, 2]], c))
        assert np.isclose(policy_value(s, c, c)-s.mean(0).max(), v)
    return {
        'truthful_ambiguous_nonidentification': {
            'states': [0, 1, 2], 'pairs': [sorted(p) for p in pairs],
            'prior': [1/3]*3, 'channel_A': ca.tolist(), 'channel_B': cb.tolist(),
            'response_marginal_both': ca.mean(0).tolist(),
            'no_ask_reward': 1/3, 'fixed_A_executor_reward_in_A': va,
            'fixed_A_executor_reward_in_B': vb,
            'fixed_executor_gain_A': va-1/3, 'fixed_executor_gain_B': vb-1/3,
            'correct_B_executor_reward_in_B': .8,
            'pair_uniform_executor_reward_both': symmetric_value,
            'scope': 'one answer per independent user; no revealed state or task reward; fixed executor'},
        'exact_sign_is_not_full_information': {
            'scores': magnitude_scores.tolist(), 'channel': endogenous.tolist(),
            'exact_sign_gain': float(vi), 'endogenous_gain': float(ve)},
        'randomized_finite_checks': 100,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    assert subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
    base = args.repo/'flight_rec/data'
    train = json.loads((base/'train.json').read_text())
    evaluation = json.loads((base/'eval.json').read_text())
    old = ROOT/'results/learn_question_value'
    splits = json.loads((old/'splits.json').read_text())
    saved = json.loads((old/'training_data.json').read_text())
    profiles = np.array([train[i]['reward_function'] for i in splits['prior_fit_record_ids']])
    slopes = np.load(old/'model.npz')['slopes']
    decompositions = {}
    stress = {s: [] for s in [-1., -.5, 0., .5, 1.]}
    detailed = []
    for split, tasks in saved.items():
        errors = []; ratios = []; targets = []; ideals = []; risks = []
        for t in tasks:
            record = (evaluation if split == 'holdout' else train)[t['id']]
            options = record['rounds'][-1]['options']
            assert catalog_id(options) == t['catalog_hash']
            scores = profiles @ np.array(options).T
            initial = scores.mean(0).max()
            risks.append(t['risk']); target=[]; ideal=[]
            candidate_stress = {s: [] for s in stress}
            for q, index in enumerate(INDICES):
                sign = np.sign(profiles[:, index]).astype(int)
                e, a = t['rates'][q]
                vf = gain(scores, response_channel(sign, 1., 0.))
                vp = gain(scores, response_channel(sign, 0., 1.))
                v = gain(scores, response_channel(sign, e, a))
                errors.append(abs(v - (e*vf+a*vp)))
                assert np.isclose(v, t['target'][q], atol=1e-12)
                assert np.isclose(vf, t['ideal'][q], atol=1e-12)
                if vf > 1e-9: ratios.append(vp/vf)
                target.append(v); ideal.append(vf)
                if split != 'holdout': continue
                magnitude = np.abs(profiles[:, index])
                centered = magnitude-magnitude.mean()
                z = centered/max(float(np.abs(centered).max()), 1e-12)
                assumed = response_channel(sign, e, a)
                for strength in stress:
                    exact = e + strength*min(e, 1-e-a)*z
                    actual = response_channel(sign, exact, a)
                    assert np.isclose(exact.mean(), e)
                    assert np.isclose(actual[:, 6].mean(), 1-e-a)
                    true = gain(scores, actual)
                    wrong_executor = policy_value(scores, actual, assumed)-initial
                    assert wrong_executor <= true+1e-12
                    candidate_stress[strength].append((true, wrong_executor))
            targets.append(target); ideals.append(ideal)
            if split == 'holdout':
                for strength, pairs in candidate_stress.items():
                    stress[strength].append({'id':t['id'], 'values':pairs, 'risk':t['risk'],
                                            'pooled_prediction':target})
        v = np.array(targets); i = np.array(ideals); pred = i*slopes
        row = np.arange(len(v)); chosen = pred.argmax(1); best = v.argmax(1)
        err = np.abs(pred-v).max(1)
        margin = np.sort(v, axis=1)[:, -1]-np.sort(v, axis=1)[:, -2]
        certified = margin > 2*err
        assert np.all(chosen[certified] == best[certified])
        gap = v.max(1)-v[row, chosen]
        decompositions[split] = {
            'tasks': len(v), 'pairs': int(v.size), 'max_decomposition_error': max(errors),
            'partial_to_exact_ratio_quantiles': quantiles(ratios),
            'simple_value_mae': float(np.abs(pred-v).mean()),
            'simple_selection_disagreements': int((chosen != best).sum()),
            'simple_mean_selection_regret': float(gap.mean()),
            'simple_max_selection_regret': float(gap.max()),
            'selection_certified_by_margin': int(certified.sum()),
            'oracle_margin_quantiles': quantiles(margin),
            'simple_loss': float((np.array(risks)-v[row, chosen]).mean()),
            'oracle_loss': float((np.array(risks)-v.max(1)).mean()),
        }
        if split == 'holdout':
            detailed = [{'id':t['id'], 'oracle_q':int(b), 'simple_q':int(c),
                         'gap':float(g), 'certified':bool(cert)}
                        for t,b,c,g,cert in zip(tasks,best,chosen,gap,certified)]
    summary_stress = []
    for strength, records in stress.items():
        true = np.array([[v[0] for v in r['values']] for r in records])
        wrong = np.array([[v[1] for v in r['values']] for r in records])
        pooled = np.array([r['pooled_prediction'] for r in records])
        row = np.arange(len(true)); q = pooled.argmax(1); optimal = true.argmax(1)
        summary_stress.append({
            'strength': strength, 'tasks':len(true),
            'question_disagreements':int((q != optimal).sum()),
            'selection_gap_correct_executor':float((true.max(1)-true[row,q]).mean()),
            'executor_gap_fixed_pooled_question':float((true[row,q]-wrong[row,q]).mean()),
            'total_gap_pooled_system':float((true.max(1)-wrong[row,q]).mean()),
            'negative_gain_selected_questions':int((wrong[row,q] < -1e-10).sum()),
            'oracle_mean_gain':float(true.max(1).mean()),
        })
        if strength == 0:
            assert np.allclose(true, pooled) and np.allclose(true, wrong)
    out = ROOT/'results/question_value_diagnostics'; out.mkdir(exist_ok=True, parents=True)
    result = {'status':'posthoc_diagnostic_not_new_method', 'seed':20261020,
              'decomposition':decompositions, 'matched_rate_stress':summary_stress,
              'counterexamples':examples(), 'upstream_commit':COMMIT,
              'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [old/'training_data.json', old/'splits.json', old/'model.npz']},
              'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'summary.json').write_text(json.dumps(result, indent=2))
    (out/'per_task.json').write_text(json.dumps({'selection':detailed,'stress':stress}, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
