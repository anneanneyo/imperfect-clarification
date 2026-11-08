"""Independent exact control construction and weighted-score checks."""
import hashlib,json
from fractions import Fraction as F
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/e014_no_information_controls';BASE=ROOT/'results/lavoir_executor_e014'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):assert abs(float(a)-float(b))<1e-12
def main():
    for p,h in read(OUT/'manifest.json')['files'].items():assert sha(ROOT/p)==h
    pack={r['id']:r for r in read(OUT/'pack.json')};br=read(OUT/'branches.json')
    raw={r['id']:r for r in read(BASE/'source_rows.json')};counts=Counter('/'.join(r.split('/')[:3]) for r in raw)
    oldpack={r['id']:r for r in read(BASE/'pack.json')};old={r['source_id']:r for r in read(BASE/'details.json') if r['question']=='no_ask'}
    weights={};seen=set()
    for b in br:
        key=b['source_id'],b['question'];assert key not in seen;seen.add(key)
        original=old[b['source_id']];messages=pack[b['prompt_id']]['messages']
        assert messages[:2]==oldpack[original['prompt_id']]['messages'] and len(messages)==4
        if b['question']=='repeat':assert messages[2:]==[{'role':'assistant','content':'Could you repeat the information you already provided?'},{'role':'user','content':messages[1]['content']}]
        else:assert b['question']=='no_extra' and messages[2:]==[{'role':'assistant','content':'Is there anything else you can tell me?'},{'role':'user','content':"I don't have any additional information to provide."}]
        weights[b['source_id']]=F(1,24*counts[b['case_id']]);close(weights[b['source_id']],b['weight'])
        assert b['gold']==original['gold'] and b['base_team']==original['team'] and b['base_correct']==original['correct']
        assert b['sensitivity_excluded']==original['sensitivity_excluded']
    assert len(pack)==48 and len(br)==130 and sum(weights.values())==1
    result={'verified':True,'unique_prompts':48,'branches':130,'source_fact_copies_verified':True,'phase':'inputs verified'}
    if (OUT/'summary.json').exists():
        # Parser differs from primary implementation: require plain JSON or one explicit fence.
        gs=[json.loads(l) for l in (OUT/'generations.jsonl').read_text().splitlines()];gmap={g['id']:g for g in gs}
        assert len(gs)==len(gmap)==48 and all(g['error']is None for g in gs)
        details=read(OUT/'details.json');summary=read(OUT/'summary.json')
        for d in details:
            s=gmap[d['prompt_id']]['raw_output'].strip()
            if s.startswith('```'):s='\n'.join(s.splitlines()[1:-1]).strip()
            obj=json.loads(s);assert type(obj)is dict and set(obj)=={'team'} and obj['team']==d['team']
            assert d['correct']==int(d['team']==d['gold'])
        for scope,rows in [('all_cases',details),('predeclared_sensitivity',[d for d in details if not d['sensitivity_excluded']])]:
            for q in ['repeat','no_extra']:
                rs=[d for d in rows if d['question']==q];z=sum(weights[d['source_id']] for d in rs)
                acc=sum(weights[d['source_id']]*d['correct'] for d in rs)/z
                base=sum(weights[d['source_id']]*d['base_correct'] for d in rs)/z
                close(acc,summary['results'][scope][q]['accuracy']);close(base,summary['results'][scope][q]['base_accuracy'])
                close(acc-F(1,20),summary['results'][scope][q]['utility'])
        for p,h in read(OUT/'analysis_manifest.json')['files'].items():assert sha(ROOT/p)==h
        result.update(phase='completed',rational_accuracy_checks=True)
    result['verifier_sha256']=sha(Path(__file__).resolve())
    (OUT/('verification.json' if result['phase']=='completed' else 'pre_verification.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
