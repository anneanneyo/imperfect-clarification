"""Post-outcome decomposition; no new model calls or altered scoring."""
from collections import defaultdict,Counter
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/lavoir_executor_e014'
def main():
    d=json.loads((OUT/'details.json').read_text());s=json.loads((OUT/'summary.json').read_text())
    base={r['source_id']:r for r in d if r['question']=='no_ask'}
    perq=defaultdict(lambda:defaultdict(float));certain_errors=set();certain_weight=0.;uncertain_weight=0.
    for b in base.values():
        if max(b['semantic_posterior'].values())>1-1e-12:
            certain_weight+=b['weight']
            if not b['correct']:certain_errors.add(b['case_id'])
        else:uncertain_weight+=b['weight']
    choices={r['group']:r['question'] for r in s['all_cases']['oracles']['visible_input_oracle']['choices']}
    decomposition=defaultdict(float);examples=[]
    for r in d:
        if r['question']=='no_ask':continue
        b=base[r['source_id']];delta=r['correct']-b['correct'];w=r['weight']
        changed=max(abs(r['semantic_posterior'][t]-b['semantic_posterior'][t]) for t in r['semantic_posterior'])>1e-12
        group='route_posterior_changed' if changed else 'route_posterior_unchanged'
        perq[r['question']][group+'_mass']+=w
        perq[r['question']][group+'_accuracy_gain']+=w*delta
        perq[r['question']]['improvement_mass']+=w*(delta>0)
        perq[r['question']]['worsening_mass']+=w*(delta<0)
        if choices[r['visible_group']]==r['question']:
            decomposition[group+'_accuracy_gain']+=w*delta
            decomposition[group+'_selected_mass']+=w
            if delta>0 and not changed:examples.append({'source_id':r['source_id'],'question':r['question'],'before':b['team'],'after':r['team'],'gold':r['gold'],'weight':w})
    total=s['all_cases']['oracles']['visible_input_oracle']['accuracy']-s['all_cases']['fixed']['no_ask']['accuracy']
    assert abs(total-sum(v for k,v in decomposition.items() if k.endswith('accuracy_gain')))<1e-12
    result={'timing':'Post-hoc after E014 outputs; descriptive decomposition, not causal attribution or preregistered hypothesis test.',
            'certain_initial_case_weight':certain_weight,'uncertain_initial_case_weight':uncertain_weight,
            'certain_initial_error_cases':sorted(certain_errors),'per_question':dict(perq),
            'visible_oracle_vs_noask_gain':total,'visible_oracle_decomposition':dict(decomposition),
            'nonchanged_posterior_improvement_examples':examples,
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['per_question','nonchanged_posterior_improvement_examples']}))
if __name__=='__main__':main()
