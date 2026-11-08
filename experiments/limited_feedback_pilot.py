"""E005: matched sampled feedback; structured diagnostic, not an LLM experiment.

See research/E005_PROTOCOL.md. Numpy + standard library only. No network calls.
The sampler owns hidden states. Learners receive Observations without these states.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

from learn_question_value import COMMIT, INDICES, BASE, catalog_id, response_channel, gain

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/limited_feedback_pilot'
SOURCE_N = 4096
ADAPT_N = [0, 50, 200]
SEEDS = list(range(20261027, 20261037))
COSTS = [0., .02, .10]
SHIFT = np.array([.35, 1., .6, .9, .5, .5, .5, .5])
PAIRS = [(-1, 0), (-1, 1), (0, 1)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def simplex(x):
    """Euclidean simplex projection, rowwise; not probability clipping alone."""
    u = np.sort(x, axis=-1)[..., ::-1]
    css = np.cumsum(u, axis=-1) - 1
    j = np.arange(1, x.shape[-1] + 1)
    rho = (u - css/j > 0).sum(axis=-1) - 1
    theta = np.take_along_axis(css, rho[..., None], axis=-1)[..., 0]/(rho+1)
    return np.maximum(x-theta[..., None], 0)


@dataclass
class Visible:
    ideal: np.ndarray
    partial: np.ndarray
    span: np.ndarray
    initial: np.ndarray
    answer_control: np.ndarray
    answer_denoised_gain: np.ndarray


@dataclass
class Observations:
    task: np.ndarray
    question: np.ndarray
    response: np.ndarray
    kind: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    expected_baseline_given_response: np.ndarray
    denoised_gain: np.ndarray

    def first(self, n):
        return Observations(**{k: v[:n] for k, v in vars(self).items()})


class FiniteEnvironment:
    """Simulator and exact checker; neither is passed into fitted estimators."""
    def __init__(self, records, profiles):
        self.profiles = profiles
        self.sign = np.sign(profiles[:, INDICES]).astype(int)
        self.scores = np.array([profiles @ np.array(r['options']).T for r in records])
        assert all(np.min(r['options']) >= 0 and np.max(r['options']) <= 1 for r in records)
        self.span = np.array([np.ptp(np.array(r['options']), axis=0)[INDICES] for r in records])
        self.initial_action = self.scores.mean(1).argmax(1)
        initial = self.scores.mean(1).max(1)
        self.risk = self.scores.max(2).mean(1)-initial
        n = len(records)
        self.actions = np.zeros((n, 8, 7), dtype=int)
        controls = np.zeros((n, 8, 7))
        denoised = np.zeros_like(controls)
        ideal = np.zeros((n, 8)); partial = np.zeros_like(ideal)
        # Known truth-preserving semantics and population prior, not sampled z.
        for t in range(n):
            for q in range(8):
                exact_c = response_channel(self.sign[:, q], 1., 0.)
                partial_c = response_channel(self.sign[:, q], 0., 1.)
                ideal[t, q] = gain(self.scores[t], exact_c)
                partial[t, q] = gain(self.scores[t], partial_c)
                c = response_channel(self.sign[:, q], .4, .3)
                for r in range(7):
                    posterior = c[:, r]/c[:, r].sum() if c[:, r].sum() else np.ones(len(profiles))/len(profiles)
                    reward = posterior @ self.scores[t]
                    a = reward.argmax()
                    self.actions[t, q, r] = a
                    controls[t, q, r] = reward[self.initial_action[t]]
                    denoised[t, q, r] = reward[a]-controls[t, q, r]
        self.visible = Visible(ideal, partial, self.span, initial, controls, denoised)
        assert np.all(denoised >= -1e-12)
        assert np.allclose(denoised[:, :, 6], 0)

    def rates(self, shifted):
        e = BASE[None, :]-.15*self.span
        if shifted:
            e = e*SHIFT
        b = .2+.1*self.span
        rates = np.stack([e, b, 1-e-b], axis=-1)
        assert np.min(rates) >= 0 and np.allclose(rates.sum(-1), 1)
        return rates

    def truth(self, shifted):
        p = self.rates(shifted)
        return p[:, :, 0]*self.visible.ideal+p[:, :, 1]*self.visible.partial

    def sample(self, rng, n, shifted):
        task = rng.integers(0, len(self.scores), n)
        q = rng.integers(0, 9, n)
        latent = rng.integers(0, len(self.profiles), n)
        kind = np.full(n, 3, dtype=int)  # no-ask, outside three answer categories
        response = np.full(n, 7, dtype=int)
        action = self.initial_action[task].copy()
        controls = self.visible.initial[task].copy()
        denoised = np.zeros(n)
        rows = np.flatnonzero(q < 8)
        rates = self.rates(shifted)[task[rows], q[rows]]
        kind[rows] = (rng.random((len(rows), 1)) > np.cumsum(rates, axis=1)).sum(1)
        sign = self.sign[latent[rows], q[rows]]
        response[rows] = 6
        exact = kind[rows] == 0
        response[rows[exact]] = sign[exact]+1
        for i in rows[kind[rows] == 1]:
            candidates = [j for j, pair in enumerate(PAIRS) if self.sign[latent[i], q[i]] in pair]
            response[i] = 3+candidates[rng.integers(0, 2)]
        action[rows] = self.actions[task[rows], q[rows], response[rows]]
        controls[rows] = self.visible.answer_control[task[rows], q[rows], response[rows]]
        denoised[rows] = self.visible.answer_denoised_gain[task[rows], q[rows], response[rows]]
        reward = self.scores[task, latent, action]
        # No latent state, full reward vector, or integrated true value crosses here.
        return Observations(task, q, response, kind, action, reward, controls, denoised)


def design(v, kind):
    if kind == 'factorized':
        return np.stack([np.ones_like(v.span), v.span], axis=-1)
    return np.stack([v.ideal, v.partial, v.ideal*v.span, v.partial*v.span], axis=-1)


class Regressor:
    def __init__(self, kind, fit_visible, log):
        self.kind = kind
        xall = design(fit_visible, kind)
        self.scale = np.maximum(np.sqrt((xall*xall).mean(0)), 1e-8)
        self.gram, self.rhs = self.statistics(fit_visible, log)
        d = self.scale.shape[-1]
        self.weights = np.linalg.solve(self.gram+np.eye(d), self.rhs)

    def statistics(self, visible, log):
        xall = design(visible, self.kind)/self.scale
        d = self.scale.shape[-1]
        outputs = 3 if self.kind == 'factorized' else 1
        gram = np.zeros((8, d, d)); rhs = np.zeros((8, d, outputs))
        for q in range(8):
            selected = log.question == q
            x = xall[log.task[selected], q]
            if self.kind == 'factorized':
                y = np.eye(3)[log.kind[selected]]
            elif self.kind == 'direct_raw':
                y = (log.reward[selected]-log.expected_baseline_given_response[selected])[:, None]
            else:
                y = log.denoised_gain[selected, None]
            gram[q] = x.T@x; rhs[q] = x.T@y
        return gram, rhs

    def adapted(self, visible, log, pooled):
        gram, rhs = self.statistics(visible, log)
        eye = np.eye(self.scale.shape[-1])
        if pooled:
            return np.linalg.solve(self.gram+gram+eye, self.rhs+rhs)
        prior = self.gram*(32/SOURCE_N)+eye
        return np.linalg.solve(gram+prior, rhs+prior@self.weights)

    def predict(self, visible, weights=None):
        w = self.weights if weights is None else weights
        raw = np.einsum('nqd,qdk->nqk', design(visible, self.kind)/self.scale, w)
        if self.kind == 'factorized':
            p = simplex(raw)
            return p[:, :, 0]*visible.ideal+p[:, :, 1]*visible.partial
        return np.clip(raw[:, :, 0], 0, visible.ideal)


class Simple:
    def __init__(self, log):
        self.counts = self.statistics(log)
        self.p = (self.counts+1)/(self.counts+1).sum(-1, keepdims=True)

    @staticmethod
    def statistics(log):
        counts = np.zeros((8, 3))
        valid = log.question < 8
        np.add.at(counts, (log.question[valid], log.kind[valid]), 1)
        return counts

    def adapted(self, log, pooled):
        c = self.statistics(log)
        counts = self.counts+c+1 if pooled else c+(32/9)*self.p
        return counts/counts.sum(-1, keepdims=True)

    def predict(self, visible, p=None):
        p = self.p if p is None else p
        return p[:, 0]*visible.ideal+p[:, 1]*visible.partial


def select(pred, cost):
    p = np.column_stack([pred-cost, np.zeros(len(pred))])
    # Prefer no-ask on ties exactly at zero; ask tie uses first question.
    chosen = p.argmax(1)
    chosen[p.max(1) <= 0] = 8
    return chosen


def metric(env, truth, pred, cost):
    chosen = select(pred, cost)
    all_values = np.column_stack([truth-cost, np.zeros(len(truth))])
    achieved = all_values[np.arange(len(chosen)), chosen]
    best = all_values.max(1)
    return {'loss': float((env.risk-achieved).mean()), 'gain_net': float(achieved.mean()),
            'oracle_gap': float((best-achieved).mean()), 'ask_rate': float((chosen < 8).mean()),
            'value_mse': float(np.mean((pred-truth)**2))}, chosen


def ope(env, log, pred, chosen, cost):
    p = np.column_stack([pred-cost, np.zeros(len(pred))])
    observed = log.reward-env.visible.initial[log.task]-cost*(log.question < 8)
    matched = chosen[log.task] == log.question
    ips = 9*matched*observed
    dr = p[log.task, chosen[log.task]]+9*matched*(observed-p[log.task, log.question])
    return {'ips_net_gain': float(ips.mean()), 'dr_net_gain': float(dr.mean()),
            'matched_rows': int(matched.sum()), 'evaluation_rows': len(log.task)}


def summarize(rows):
    out = []
    keys = sorted({(r['target'], r['budget'], r['cost'], r['method']) for r in rows})
    for target, budget, cost, method in keys:
        subset = [r for r in rows if (r['target'], r['budget'], r['cost'], r['method']) == (target, budget, cost, method)]
        record = {'target': target, 'budget': budget, 'cost': cost, 'method': method, 'seeds': len(subset)}
        for k in ['loss', 'gain_net', 'oracle_gap', 'ask_rate', 'value_mse', 'ips_net_gain', 'dr_net_gain']:
            if k not in subset[0]:
                continue
            a = np.array([r[k] for r in subset])
            record[k] = float(a.mean())
            record[k+'_seed_sd'] = float(a.std(ddof=1))
        out.append(record)
    return out


def tests(env):
    for shifted in [False, True]:
        for t in [0, len(env.scores)-1]:
            for q in range(8):
                e, b, _ = env.rates(shifted)[t, q]
                c = response_channel(env.sign[:, q], e, b)
                value = gain(env.scores[t], c)
                fixed = sum(np.mean(c[:, r]*env.scores[t, :, env.actions[t, q, r]]) for r in range(7))-env.visible.initial[t]
                assert np.isclose(value, env.truth(shifted)[t, q])
                assert np.isclose(fixed, value)
                # Expected denoised label equals true V, without per-person counterfactual.
                assert np.isclose(c.mean(0)@env.visible.answer_denoised_gain[t, q], value)
    s = simplex(np.array([[.2, .3, .5], [-.4, .3, 2.]]))
    assert np.all(s >= 0) and np.allclose(s.sum(-1), 1)
    assert set(Observations.__dataclass_fields__) == {'task', 'question', 'response', 'kind', 'action', 'reward', 'expected_baseline_given_response', 'denoised_gain'}


def sampling_audit(env):
    """Post-run numerical verification; never used to select a method or setting."""
    rows = []
    for shifted in [False, True]:
        log = env.sample(np.random.default_rng(764829+int(shifted)), 500000, shifted)
        truth = env.truth(shifted)
        chosen = select(truth, .02)
        observed = log.reward-env.visible.initial[log.task]-.02*(log.question < 8)
        weighted = 9*(chosen[log.task] == log.question)*observed
        expected = metric(env, truth, truth, .02)[0]['gain_net']
        se = weighted.std(ddof=1)/np.sqrt(len(weighted))
        assert abs(weighted.mean()-expected) < 5*se
        rows.append({'target': 'selective_shift' if shifted else 'no_shift', 'seed': 764829+int(shifted),
                     'rows': len(log.task), 'oracle_true_net_gain': expected,
                     'sampled_ips_net_gain': float(weighted.mean()), 'monte_carlo_se': float(se),
                     'used_for_training_or_selection': False})
    return rows


def plot(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    styles = {'simple': ('#177E89', 'Per-question type counts'),
              'factorized': ('#C45115', 'Contextual response model'),
              'direct_denoised': ('#7A57A5', 'Direct, denoised outcome'),
              'direct_raw': ('#8D939A', 'Direct, observed reward')}
    for ax, target, title in zip(axes, ['no_shift', 'selective_shift'],
                                ['Unchanged response channel', 'Changed response channel']):
        for method, (color, label) in styles.items():
            rows = [next(r for r in summary['metrics'] if
                         (r['target'], r['budget'], r['cost'], r['method']) == (target, n, .02, method))
                    for n in ADAPT_N]
            mu = np.array([r['loss'] for r in rows]); sd = np.array([r['loss_seed_sd'] for r in rows])
            ax.plot(ADAPT_N, mu, 'o-', color=color, label=label, lw=1.8, ms=4)
            ax.fill_between(ADAPT_N, mu-sd, mu+sd, color=color, alpha=.12)
        oracle = next(r['loss'] for r in summary['metrics'] if
                      (r['target'], r['budget'], r['cost'], r['method']) == (target, 0, .02, 'oracle'))
        ax.axhline(oracle, color='#222222', ls=':', label='Known-channel oracle')
        ax.set_title(title, fontsize=11); ax.set_xlabel('Target interactions for adaptation')
        ax.set_xticks(ADAPT_N); ax.grid(alpha=.2); ax.spines[['right', 'top']].set_visible(False)
    axes[0].set_ylabel('Expected regret + question cost (lower is better)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=8.5)
    fig.suptitle('E005 | Same sampled feedback, 10 random seeds', fontsize=13)
    fig.text(.5, .16, 'Shading: ±1 SD across logs, fixed 90 development catalogs. Question cost: 0.02.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .21, 1, .94])
    fig.savefig(OUT/'adaptation.png', dpi=180); fig.savefig(OUT/'adaptation.svg'); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True, type=Path)
    args = p.parse_args()
    assert subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
    base = args.repo/'flight_rec/data'
    train = json.loads((base/'train.json').read_text()); evaluation = json.loads((base/'eval.json').read_text())
    splitpath = ROOT/'results/learn_question_value/splits.json'
    splits = json.loads(splitpath.read_text())
    profiles = np.array([train[i]['reward_function'] for i in splits['prior_fit_record_ids']])
    catalogs = {}
    for name in ['fit', 'validation', 'holdout']:
        data = evaluation if name == 'holdout' else train
        records = []
        for spec in splits['tasks'][name]:
            options = data[spec['record']]['rounds'][-1]['options']
            assert catalog_id(options) == spec['hash']
            records.append({'id': spec['record'], 'hash': spec['hash'], 'options': options})
        catalogs[name] = records
    hashes = [{r['hash'] for r in catalogs[name]} for name in ['fit', 'validation', 'holdout']]
    assert all(not hashes[i] & hashes[j] for i in range(3) for j in range(i))
    fit = FiniteEnvironment(catalogs['fit'], profiles)
    adapt = FiniteEnvironment(catalogs['validation'][:72], profiles)
    test = FiniteEnvironment(catalogs['holdout'], profiles)
    tests(fit); tests(test)
    OUT.mkdir(exist_ok=True, parents=True)
    results = []; predictions = {}; models = {}; log_hashes = []; contrasts = []
    for seed in SEEDS:
        source = fit.sample(np.random.default_rng(seed), SOURCE_N, False)
        estimators = {'simple': Simple(source)}
        estimators.update({k: Regressor(k, fit.visible, source) for k in ['factorized', 'direct_raw', 'direct_denoised']})
        for name, obj in estimators.items():
            models[f'{seed}_{name}_source'] = obj.p if name == 'simple' else obj.weights
            if name != 'simple':
                models[f'{seed}_{name}_scale'] = obj.scale
        for shifted in [False, True]:
            target = 'selective_shift' if shifted else 'no_shift'
            adaptation = adapt.sample(np.random.default_rng(seed+10000+1000*int(shifted)), 200, shifted)
            evallog = test.sample(np.random.default_rng(seed+20000+1000*int(shifted)), 4096, shifted)
            log_hashes.append({'seed': seed, 'target': target,
                               'source': hashlib.sha256(b''.join(v.tobytes() for v in vars(source).values())).hexdigest(),
                               'adaptation': hashlib.sha256(b''.join(v.tobytes() for v in vars(adaptation).values())).hexdigest()})
            truth = test.truth(shifted)
            for budget in ADAPT_N:
                pred = {'ideal': test.visible.ideal, 'oracle': truth}
                log = adaptation.first(budget)
                for name, obj in estimators.items():
                    if budget == 0:
                        pred[name] = obj.predict(test.visible)
                        pred[name+'_pooled'] = pred[name]
                    else:
                        for pooled in [False, True]:
                            key = name+('_pooled' if pooled else '')
                            w = obj.adapted(log, pooled) if name == 'simple' else obj.adapted(adapt.visible, log, pooled)
                            pred[key] = obj.predict(test.visible, w)
                            models[f'{seed}_{target}_{budget}_{key}'] = w
                for name, values in pred.items():
                    predictions[f'{seed}_{target}_{budget}_{name}'] = values
                    for cost in COSTS:
                        m, chosen = metric(test, truth, values, cost)
                        o = ope(test, evallog, values, chosen, cost)
                        results.append({'seed': seed, 'target': target, 'budget': budget, 'cost': cost, 'method': name, **m, **o})
                for cost in COSTS:
                    best = np.maximum(truth.max(1)-cost, 0)
                    for name, value, ask in [('no_ask', np.zeros(len(truth)), 0.), ('random', truth.mean(1)-cost, 1.)]:
                        results.append({'seed': seed, 'target': target, 'budget': budget, 'cost': cost, 'method': name,
                                        'loss': float((test.risk-value).mean()), 'gain_net': float(value.mean()),
                                        'oracle_gap': float((best-value).mean()), 'ask_rate': ask})
        print('completed seed', seed, flush=True)
    for target in ['no_shift', 'selective_shift']:
        for budget in ADAPT_N:
            for cost in COSTS:
                chosen = [r for r in results if (r['target'], r['budget'], r['cost']) == (target, budget, cost)]
                for other in ['simple', 'direct_raw', 'direct_denoised', 'factorized_pooled']:
                    delta = np.array([next(r['loss'] for r in chosen if r['seed'] == seed and r['method'] == 'factorized')-
                                      next(r['loss'] for r in chosen if r['seed'] == seed and r['method'] == other) for seed in SEEDS])
                    # Student-t(9) interval only across repeated log samples, conditional catalogs.
                    radius = 2.262157*delta.std(ddof=1)/np.sqrt(len(delta))
                    contrasts.append({'target': target, 'budget': budget, 'cost': cost, 'contrast': 'factorized minus '+other,
                                      'mean_loss_delta': float(delta.mean()), 'seed_t95': [float(delta.mean()-radius), float(delta.mean()+radius)],
                                      'factorized_better_seeds': int((delta < 0).sum())})
    np.savez_compressed(OUT/'predictions.npz', **predictions)
    np.savez_compressed(OUT/'models.npz', **models)
    (OUT/'per_seed.json').write_text(json.dumps(results, indent=2))
    (OUT/'log_hashes.json').write_text(json.dumps(log_hashes, indent=2))
    verification = {'status': 'passed', 'checks': ['canonical_catalog_hashes_match_E002', 'splits_disjoint',
                    'response_probabilities_normalized', 'fixed_executor_equals_bayes_in_restricted_channel',
                    'decomposition_matches_exact_enumeration', 'denoised_target_unbiased_by_exact_enumeration',
                    'unknown_denoised_gain_zero', 'simplex_projection', 'learner_observations_exclude_hidden_state_and_counterfactual_rewards'],
                    'network_calls': 0, 'model_calls': 0,
                    'post_run_sampling_audit': sampling_audit(test)}
    (OUT/'verification.json').write_text(json.dumps(verification, indent=2))
    summary = {'status': 'completed_restricted_structured_diagnostic_not_full_H3', 'date': '2026-10-27',
               'seeds': SEEDS, 'source_interactions': SOURCE_N, 'adaptation_budgets': ADAPT_N,
               'costs_native_utility': COSTS, 'upstream_commit': COMMIT,
               'counts': {'fit': len(fit.scores), 'adaptation': len(adapt.scores), 'development_holdout': len(test.scores),
                          'prior_records': len(profiles), 'unique_prior_profiles': len(np.unique(profiles, axis=0))},
               'conservative_reward_absolute_bound_from_prior': float(np.abs(profiles).sum(1).max()),
               'metrics': summarize(results), 'paired_contrasts': contrasts,
               'hashes': {'code': sha(__file__), 'protocol': sha(ROOT/'research/E005_PROTOCOL.md'), 'splits': sha(splitpath),
                          'train': sha(base/'train.json'), 'eval': sha(base/'eval.json')}, 'checks': verification}
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2))
    plot(summary)
    print(json.dumps({'status': summary['status'], 'rows': len(results), 'artifacts': str(OUT)}, indent=2))


if __name__ == '__main__':
    main()
