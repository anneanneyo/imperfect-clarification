"""Read-only source/data gate for one Lavoir training workflow, no model calls.

Schema posterior reconstruction is ours. Does not execute external source or
claim to reconstruct the unpublished answer-channel generator/oracle_voi.
"""
from collections import Counter,defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/lavoir_source_gate'
CACHE=Path('/private/tmp/lavoir-data-20261027')
REPO=Path('/private/tmp/lavoir-audit-20261027')
COMMIT='af79b0a7a7f9864343223d3f7bd40a02747d0c1f'
def digest(o):return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,o):(OUT/name).write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')

def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==COMMIT
    source=json.loads((OUT/'source_manifest.json').read_text())
    for f in source['files']:assert sha(CACHE/f['path'])==f['sha256']
    schema=json.loads((CACHE/'schemas/banking_support.json').read_text())
    rows=[json.loads(l) for l in (CACHE/'data/train/banking_support.jsonl').read_text().splitlines()]
    assert all(r['split']=='train' and r['workflow']=='banking_support' for r in rows)
    slots={k:v for k,v in schema['slots'].items() if v.get('decisive',True)}
    worlds=[({},1.)]
    # This schema's one-parent ordering is checked, not presumed for other workflows.
    for k,s in slots.items():
        nextworlds=[]
        for w,p in worlds:
            probs=s['cond'][w[s['depends_on']]] if 'depends_on' in s else s['values']
            assert abs(sum(probs.values())-1)<1e-12
            nextworlds.extend(({**w,k:v},p*pv) for v,pv in probs.items() if pv>0)
        worlds=nextworlds
    assert abs(sum(p for w,p in worlds)-1)<1e-12
    def route(w):
        for rule in schema['rules']:
            if 'else' in rule:return rule['else']
            if all(w[k] in vs for k,vs in rule['if'].items()):return rule['unit']
        raise AssertionError('No route')
    def posterior(evidence):
        ps={k:0. for k in schema['units']}
        for w,p in worlds:
            if all(k not in slots or w[k] in vs for k,vs in evidence.items()):ps[route(w)]+=p
        mass=sum(ps.values());assert mass>0
        return {k:v/mass for k,v in ps.items()}
    errors=[];probe_kind=Counter(); visible=defaultdict(list); negatives=[]; empty_probes=0
    for r in rows:
        assert route(r['profile'])==r['gold']
        expected=posterior(r['evidence']);err=max(abs(expected[k]-r['target'][k]) for k in expected)
        if err>2e-9:errors.append({'id':r['id'],'max_error':err})
        assert set(r['slots'])==set(r['probes'])==set(r['oracle_voi'])
        assert not set(r['asked']) & set(r['slots'])
        for slot,probe in r['probes'].items():
            probe_kind[probe['kind']]+=1
            if not probe['a'].strip():empty_probes+=1
            for k,vs in probe['reveal'].items():assert r['profile'][k] in vs
        negatives.extend({'id':r['id'],'slot':k,'value':v} for k,v in r['oracle_voi'].items() if v < -1e-8)
        visible[digest({k:r[k] for k in ['state','question','slots']})].append(r)
    collisions=[]; best_compatible_agreements=0; eligible_agreements=0
    for h,group in visible.items():
        opts=list(group[0]['slots'])
        if not opts:continue
        vectors=[g['oracle_voi'] for g in group]
        spread=max(max(v[k] for v in vectors)-min(v[k] for v in vectors) for k in opts)
        tops=[{k for k in opts if v[k]>=max(v.values())-1e-8} for v in vectors]
        eligible_agreements+=len(group)
        best_compatible_agreements+=max(sum(k in top for top in tops) for k in opts)
        disjoint=not set.intersection(*tops)
        if spread>1e-8:
            collisions.append({'visible_hash':h,'rows':[g['id'] for g in group],
                 'golds':[g['gold'] for g in group],'oracle_voi':vectors,
                 'top_sets':[sorted(s) for s in tops],'no_common_optimal_question':disjoint,
                 'max_value_spread':spread})
    files=['README.md','MODEL_CARD.md','lavoir/targets.py','lavoir/model.py','lavoir/evaluate.py','lavoir/policy.py','lavoir/loss.py']
    result={'scope':'One training workflow, not a benchmark evaluation','rows':len(rows),
        'underlying_cases':len(set('/'.join(r['id'].split('/')[:3]) for r in rows)),
        'rows_by_kind':dict(Counter(r['kind'] for r in rows)),
        'positive_prior_worlds':len(worlds),'profile_gold_mismatches':0,
        'posterior_tolerance':2e-9,'posterior_mismatches':len(errors),
        'probe_count':sum(probe_kind.values()),'probe_kind_counts':dict(probe_kind),'empty_probe_answers':empty_probes,
        'negative_oracle_threshold':-1e-8,'negative_oracle_values':len(negatives),'min_oracle_value':min(v for r in rows for v in r['oracle_voi'].values()),
        'unique_visible_inputs':len(visible),'repeated_visible_groups':sum(len(g)>1 for g in visible.values()),
        'same_visible_different_oracle_groups':len(collisions),
        'same_visible_no_common_top_question_groups':sum(c['no_common_optimal_question'] for c in collisions),
        'max_same_input_top_question_agreement_numerator':best_compatible_agreements,
        'max_same_input_top_question_agreement_denominator':eligible_agreements,
        'max_same_input_top_question_agreement':best_compatible_agreements/eligible_agreements,
        'source_commit':COMMIT,'source_files':{p:sha(REPO/p) for p in files},
        'script_sha256':sha(Path(__file__).resolve()),
        'oracle_recomputed':False,'test_rows_read':0,'model_calls':0}
    save('audit.json',result);save('posterior_mismatches.json',errors);save('visible_collisions.json',collisions);save('negative_values.json',negatives)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
