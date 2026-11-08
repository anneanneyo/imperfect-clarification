"""E003 preparation/transport/scoring. Standard library, no SDK required.
prepare -> run (explicit model/key) -> score; selftest makes zero network calls.
Only prompt text enters the API payload. Evaluator rewards stay local.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/llm_executor_pilot'
COMMIT='27466a7832d5aafff82017a659e08942e18b01ae'
FEATURES=['arrival time before meeting','American Airlines','Delta Airlines','JetBlue Airlines',
          'Southwest Airlines','layover duration','number of stops','price']
SYSTEM='Recommend one flight for this user. Use the history and any clarification. Reply with exactly a JSON object {"choice":"A"}, choosing A, B, or C. No explanation.'


def sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def scores(options,theta):
    return [sum(v*w for v,w in zip(option,theta)) for option in options]


def render(options):
    return '\n'.join(f'{chr(65+i)}: '+', '.join(f'{name}={value:g}' for name,value in zip(FEATURES,option)) for i,option in enumerate(options))


def prepare(repo):
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==COMMIT
    path=repo/'flight_rec/data/eval.json';data=json.loads(path.read_text())
    eligible=[i for i,g in enumerate(data) if len(g['rounds'])>=6]
    ids=random.Random(20261020).sample(eligible,12)
    requests=[];truth={}
    for idx in ids:
        g=data[idx];theta=g['reward_function'];history=[]
        for r in g['rounds'][:5]:
            correct=max(range(3),key=lambda a:scores(r['options'],theta)[a])
            history.append(render(r['options'])+f'\nUser preferred {chr(65+correct)}.')
        options=g['rounds'][5]['options']
        prefix=('Each flight has normalized numeric attributes. Airline indicators are 0/1. '
                'Larger values mean more of the named attribute; arrival time means minutes BEFORE the meeting. '
                'User preferences stay fixed. Past choices:\n'+'\n\n'.join(history)+'\n\nNew options:\n'+render(options))
        utility=scores(options,theta);truth[str(idx)]={'scores':utility,'best':max(utility)}
        requests.append({'id':f'{idx}:none','task':idx,'question':None,'condition':'none','prompt':prefix})
        for q,name in enumerate(FEATURES):
            direction=(theta[q]>0)-(theta[q]<0)
            alternatives={-1:'lower',0:'neither lower nor higher',1:'higher'}
            # Truth-containing adjacent ambiguity, not a factually wrong answer.
            pair=(-1,0) if direction<0 else (0,1)
            if direction==0 and q%2==0:pair=(-1,0)
            answers={'exact':f'I prefer {alternatives[direction]} values for {name}.',
                     'partial':f'For {name}, either I prefer {alternatives[pair[0]]} values or I prefer {alternatives[pair[1]]} values; I cannot distinguish which.',
                     'unknown':'I do not know.'}
            for condition,answer in answers.items():
                requests.append({'id':f'{idx}:{q}:{condition}','task':idx,'question':q,'condition':condition,
                                 'prompt':prefix+f'\n\nAssistant: Do you prefer lower or higher values for {name}, or neither?\nUser: {answer}'})
    assert len(requests)==300 and len({r['id'] for r in requests})==300
    OUT.mkdir(parents=True,exist_ok=True)
    manifest={'status':'prepared_not_model_results','commit':COMMIT,'dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'selected_records':ids,'cases':len(requests),'case_hash':sha(requests),'system':SYSTEM,
              'smoke_calls':25,'full_calls':300,'max_output_tokens_per_call':512}
    for name,value in [('requests.json',requests),('evaluator.json',truth),('manifest.json',manifest)]:
        target=OUT/name
        if target.exists() and json.loads(target.read_text())!=value:raise ValueError('Existing preparation differs; preserve it before revising')
        target.write_text(json.dumps(value,indent=2))
    print(json.dumps(manifest,indent=2))


def payload(case,model):
    return {'model':model,'instructions':SYSTEM,'input':case['prompt'],'max_output_tokens':512,'store':False}


def decode(response):
    if response.get('status')!='completed':raise ValueError('Response not completed')
    text=''.join(c.get('text','') for item in response.get('output',[]) if item.get('type')=='message'
                 for c in item.get('content',[]) if c.get('type')=='output_text')
    answer=json.loads(text)
    if set(answer)!={'choice'} or answer['choice'] not in ['A','B','C']:raise ValueError('Invalid choice JSON')
    return answer['choice'],text


def run(model,limit,transport=None,log_path=None):
    if limit<1:raise ValueError('Positive call limit required')
    cases=json.loads((OUT/'requests.json').read_text());manifest=json.loads((OUT/'manifest.json').read_text())
    assert sha(cases)==manifest['case_hash']
    log_path=log_path or OUT/'responses.jsonl'
    existing=[json.loads(line) for line in log_path.read_text().splitlines()] if log_path.exists() else []
    signature=sha({'cases':manifest['case_hash'],'model':model,'system':SYSTEM,'max_tokens':512})
    if any(r['signature']!=signature for r in existing):raise ValueError('Run/model mismatch; use separate output')
    done={r['case_id'] for r in existing}
    if len(done)!=len(existing):raise ValueError('Duplicate log IDs')
    if transport is None:
        key=os.environ.get('OPENAI_API_KEY')
        if not key:raise RuntimeError('OPENAI_API_KEY unavailable; no network calls made')
        def transport(body):
            request=urllib.request.Request('https://api.openai.com/v1/responses',data=json.dumps(body).encode(),
                    headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
            with urllib.request.urlopen(request,timeout=60) as response:return json.load(response)
    count=0
    for case in cases:
        if case['id'] in done:continue
        if count>=limit:break
        count+=1
        row={'case_id':case['id'],'signature':signature,'requested_model':model,'status':'error'}
        try:
            response=transport(payload(case,model))
            row.update(returned_model=response.get('model'),usage=response.get('usage'),response_id=response.get('id'))
            choice,text=decode(response);row.update(status='ok',choice=choice,text=text)
        except Exception as error:
            # Do not log headers, secrets, or potentially sensitive error response bodies.
            row['error_type']=type(error).__name__
        with log_path.open('a') as f:f.write(json.dumps(row)+'\n')
        if row['status']!='ok':raise RuntimeError('Call failed; recorded attempt, stopped without retries')
    print(f'Completed {count} calls; prior attempts {len(existing)}')
    return count


def score(log_path=None):
    log_path=log_path or OUT/'responses.jsonl'
    if not log_path.exists():raise RuntimeError('No model responses; nothing to score')
    rows=[json.loads(line) for line in log_path.read_text().splitlines()]
    if len({r['case_id'] for r in rows})!=len(rows):raise ValueError('Duplicate response IDs')
    cases=json.loads((OUT/'requests.json').read_text());truth=json.loads((OUT/'evaluator.json').read_text())
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert sha(cases)==manifest['case_hash']
    valid_ids={c['id'] for c in cases}
    for row in rows:
        expected=sha({'cases':manifest['case_hash'],'model':row['requested_model'],'system':SYSTEM,'max_tokens':512})
        if row['signature']!=expected or row['case_id'] not in valid_ids:raise ValueError('Stale or unrelated log')
    if len({r['requested_model'] for r in rows})>1:raise ValueError('Mixed models in one run')
    indexed={r['case_id']:r for r in rows};metrics=[]
    for case in cases:
        row=indexed.get(case['id']);initial=indexed.get(f"{case['task']}:none")
        if not row or row['status']!='ok' or not initial or initial['status']!='ok':continue
        values=truth[str(case['task'])]['scores'];value=values[ord(row['choice'])-65]
        initial_value=values[ord(initial['choice'])-65]
        metrics.append({'case_id':case['id'],'task':case['task'],'condition':case['condition'],'question':case['question'],
                        'regret':max(values)-value,'gain_vs_initial':value-initial_value})
    return {'status':'single_replicate_diagnostic_not_expected_value_estimate','attempted':len(rows),
            'expected':len(cases),'successful':sum(r['status']=='ok' for r in rows),'scored':metrics}


def selftest():
    valid={'status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':'{"choice":"B"}'}]}]}
    assert decode(valid)[0]=='B'
    for bad in [{**valid,'status':'incomplete'},{'status':'completed','output':[]}]:
        try:decode(bad)
        except (ValueError,KeyError):pass
        else:raise AssertionError('Malformed response accepted')
    captured=[]
    def mock(body):captured.append(body);return valid
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'mock.jsonl'
        assert run('MOCK-NOT-LLM',25,mock,path)==25
        assert run('MOCK-NOT-LLM',1,mock,path)==1
        for body in captured:assert set(body)=={'model','instructions','input','max_output_tokens','store'}
        result=score(path);assert result['successful']==26
        assert all(m['gain_vs_initial']==0 for m in result['scored'])
        before=path.read_bytes()
        try:run('OTHER-MODEL',1,mock,path)
        except ValueError:pass
        else:raise AssertionError('Mixed model accepted')
        assert path.read_bytes()==before
        failed=Path(d)/'failed.jsonl'
        def broken(body):return {'status':'incomplete','output':[]}
        try:run('MOCK-NOT-LLM',25,broken,failed)
        except RuntimeError:pass
        else:raise AssertionError('Failure did not stop run')
        assert len(failed.read_text().splitlines())==1
        # Full pipeline plumbing is tested with a fixed-answer fake; not a model experiment.
    out={'status':'local_mock_checks_passed','network_calls':0,'checks':['decode','incomplete_rejected','call_limit','resume','payload_allowlist','scoring','constant_answer_zero_gain','model_mixing_rejected','stop_on_error']}
    (OUT/'selftest.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','score','selftest']);p.add_argument('--repo',type=Path)
    p.add_argument('--model');p.add_argument('--max-calls',type=int,default=25);a=p.parse_args()
    if a.action=='prepare':
        if not a.repo:p.error('--repo required')
        prepare(a.repo)
    elif a.action=='run':
        if not a.model:p.error('--model required; no silent default')
        run(a.model,a.max_calls)
    elif a.action=='selftest':selftest()
    else:(OUT/'scores.json').write_text(json.dumps(score(),indent=2))
