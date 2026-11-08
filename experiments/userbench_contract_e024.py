"""Offline diagnostic of pinned UserBench functions; no benchmark data or LLM calls.

Only reviewed function ASTs are executed, without upstream imports/constructors.
Missing source files are fetched at the pinned revision and verified as Git blobs.
"""
import ast
import asyncio
import hashlib
import json
import random
import re
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/userbench_contract_e024'
REV = '80506d2ab484cab843e60a2401ff3e0290d05b87'
CACHE = Path.home() / '.cache/clarification-research/userbench-80506d2'
PIN = {
    'travelgym/env/prompts.py': 'e1b38832b94503c095c912f47b1df351d0da94ca',
    'travelgym/env/prompt_async.py': '75719f16339060688e6944eed19dac26b3d75cd4',
    'travelgym/env/travel_env.py': '677262784514490f85c1a0073e22e23546e8e312',
}

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def sources():
    found = {}
    for rel, expected in PIN.items():
        p = CACHE / rel
        if not p.exists():
            data = urlopen(f'https://raw.githubusercontent.com/SalesforceAIResearch/UserBench/{REV}/{rel}', timeout=30).read()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        data = p.read_bytes()
        assert hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() == expected, rel
        found[rel] = data.decode()
    return found

def extract(text, names, namespace):
    tree = ast.parse(text)
    functions = [n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    assert {n.name for n in functions} == set(names)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + functions, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, '<pinned-userbench-functions>', 'exec'), namespace)

def forbidden(*args, **kwargs):
    raise AssertionError('Unexpected model call')

def fixture(one_choice=True):
    config = SimpleNamespace(api_key=None, model_name='offline-fixture', temperature=0,
        max_tokens=0, timeout=0, search_correct_reward=.2, preference_correct_reward=.2,
        choice_correct_reward=.8, choice_best_reward=1., search_failure_interval=5,
        elicitation_interval=3, wrong_choice_penalty=0., one_choice_per_aspect=one_choice,
        reward_scale=1., step_penalty=0., normalize_rewards=False, max_steps=20, verbose=False)
    env = SimpleNamespace(config=config, episode_complete=False, step_count=0,
        action_history=[], conversation_history=[], elicited_preferences=[], total_reward=0.,
        remaining_preferences=[dict(id='P1', aspect='hotel', preference='quiet room',
                                    implicit_elicitation='I am sensitive to noise.', elicited=False)],
        current_task=dict(id='handmade-fixture', scenario='Book a hotel.', dimensions=['hotel']),
        state_list=dict(remaining_best_options=['H1'], remaining_correct_options=['H1', 'H2'],
                        choice_initials=[], active_elicited_preferences=0,
                        passive_elicited_preferences=0, nonpreference_times=0, search_times=0))
    env._calculate_elicitation_ratio = lambda: len(env.elicited_preferences) / max(1, len(env.elicited_preferences) + len(env.remaining_preferences))
    env._get_remaining_preferences_summary = lambda: 'fixture-only hidden preferences'
    return env

def compact(env, result):
    observation, reward, terminated, truncated, _ = result
    return dict(reward=reward, terminated=terminated, truncated=truncated,
        episode_complete=env.episode_complete, feedback=observation['feedback'],
        remaining_best=list(env.state_list['remaining_best_options']),
        choices=list(env.state_list['choice_initials']), remaining_preferences=len(env.remaining_preferences),
        elicited_preferences=len(env.elicited_preferences))

def run():
    assert not OUT.exists(), 'Frozen result already exists; do not overwrite.'
    code = sources()
    ns = dict(re=re, json=json, random=random, model_call=forbidden,
              async_model_call=forbidden, generate_judge_response=forbidden,
              async_generate_judge_response=forbidden)
    extract(code['travelgym/env/prompts.py'], ['evaluate_action'], ns)
    extract(code['travelgym/env/prompt_async.py'], ['async_evaluate_action'], ns)
    extract(code['travelgym/env/travel_env.py'], ['step', 'step_async'], ns)
    rows = []
    for one_choice in [True, False]:
        for answer in ['H1', 'H2', 'H9']:
            for mode in ['sync', 'async']:
                env = fixture(one_choice)
                def step(action):
                    return ns['step'](env, action) if mode == 'sync' else asyncio.run(ns['step_async'](env, action))
                row = dict(one_choice=one_choice, answer=answer, mode=mode, first=compact(env, step('[answer] ' + answer)))
                try:
                    row['retry'] = compact(env, step('[answer] H1'))
                except ValueError as exc:
                    row['retry_exception'] = str(exc)
                rows.append(row)
    # Inject a generator failure, not a genuine user sample. Routing is fixed to P1.
    ns.update(generate_judge_response=lambda *a: dict(type='2', preference_id='P1'),
              model_call=lambda *a: dict(response="I don't know."),
              build_history_into_prompt=lambda *a, **kw: '',
              RESPONSE_PREFERENCE_SYSTEM='fixture', RESPONSE_PREFERENCE_USER='{preference}')
    env = fixture()
    injected = compact(env, ns['step'](env, '[action] Do you prefer a quiet hotel?'))
    assert injected['reward'] == .2 and injected['elicited_preferences'] == 1
    assert injected['feedback'] == "I don't know."
    mismatches = []
    for answer in ['H1', 'H2', 'H9']:
        pair = [r for r in rows if r['one_choice'] and r['answer'] == answer]
        if pair[0]['first']['terminated'] != pair[1]['first']['terminated']:
            mismatches.append(answer)
    assert mismatches == ['H2', 'H9']
    assert all(r['first']['terminated'] == (r['answer'] == 'H1') for r in rows if not r['one_choice'])
    summary = dict(source_revision=REV, handmade_trajectories=len(rows),
        one_choice_termination_mismatches=mismatches,
        injected_unknown_reply=injected, llm_calls=0, benchmark_tasks_loaded=0,
        source_functions_unmodified=True, upstream_package_imported=False,
        scope='Handmade contract probe; not agent evaluation or measured frequency of user-simulator failures.')
    OUT.mkdir(parents=True)
    def save(name, obj): (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    save('trajectories.json', rows)
    save('summary.json', summary)
    save('manifest.json', dict(source_revision=REV, git_blobs=PIN,
        source_sha256={p: digest(CACHE / p) for p in PIN},
        files={str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__), ROOT/'research/E024_PROTOCOL.md', OUT/'trajectories.json', OUT/'summary.json']}))
    print(json.dumps(summary, ensure_ascii=False))

if __name__ == '__main__':
    run()
