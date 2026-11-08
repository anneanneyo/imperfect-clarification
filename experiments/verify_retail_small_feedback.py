"""Recount saved E012 response observations without importing its learner."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/retail_small_feedback'

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    manifest=read(OUT/'manifest.json');rows=read(OUT/'results.json')
    assert sha(ROOT/'research/E012_PROTOCOL.md')==manifest['protocol_sha256']
    assert sha(ROOT/'experiments/retail_small_feedback.py')==manifest['code_sha256']
    for n,h in manifest['output_hashes'].items():assert sha(OUT/n)==h
    old=read(ROOT/'results/retail_clarification_contract/families.json')
    allowed={r['family_hash'] for r in old if r['split_label']=='development'}
    categories={'order_id':['answer','idk'],'contents':['full','partial','idk'],
                'total':['answer','idk'],'count':['answer','idk']}
    checked=0;observations=0;unique_estimates=set()
    for filename,h in manifest['log_hashes'].items():
        assert sha(OUT/filename)==h
        seed=int(filename.removeprefix('log_').removesuffix('.jsonl'))
        log=[json.loads(s) for s in (OUT/filename).read_text().splitlines()]
        assert len(log)==1024 and [r['step'] for r in log]==list(range(1024))
        for r in log:
            assert set(r)=={'step','family_hash','question','question_propensity','reply'}
            assert r['family_hash'] in allowed and r['question_propensity']==.25
        observations+=len(log)
        for row in [r for r in rows if r['seed']==seed]:
            b=row['budget'];counts={q:Counter() for q in categories}
            for r in log[:b]:
                tag=r['reply'][0]
                k={'idk':'idk','full_names':'full','one_name':'partial'}.get(tag,'answer')
                assert k in categories[r['question']]
                counts[r['question']][k]+=1
            for q,ks in categories.items():
                denominator=sum(counts[q].values())+len(ks)
                for k in ks:
                    assert abs(row['fitted_probabilities'][q][k]-(counts[q][k]+1)/denominator)<1e-15
                    checked+=1
            unique_estimates.add((seed,b))
        for b in [0,16,64,256,1024]:
            pair=[r for r in rows if r['seed']==seed and r['budget']==b]
            assert len(pair)==2 and pair[0]['fitted_probabilities']==pair[1]['fitted_probabilities']
    # A hand-built positive-rate mixture verifies why labels matter.
    # Orders: {book} and {book,lamp}, uniform prior. Tagged partial book
    # has likelihoods p and p/2; untagged book has f+p and p/2.
    tagged=[];untagged=[]
    for f,p in [(.45,.45),(.10,.80)]:
        tagged.append(p/(p+p/2))
        untagged.append((f+p)/(f+p+p/2))
    assert abs(tagged[0]-tagged[1])<1e-15
    assert abs(untagged[0]-untagged[1])>.05
    kb=ROOT/'results/retail_knowledge_boundary'
    selection=read(kb/'selection.json');pack=read(kb/'source_pack.json');annotations=read(kb/'annotations.json')
    assert sha(kb/'source_pack.json')==selection['pack_sha256']
    mapping={r['task_id']:r['instructions'] for r in pack}
    spans=0
    for r in annotations:
        for a in r['knowledge_atoms']:
            assert a['source_span'] in mapping[r['task_id']][a['source_field']]
            spans+=1
    result={'log_files_verified':len(manifest['log_hashes']),'observations_recounted':observations,
            'unique_seed_budget_estimates':len(unique_estimates),'fitted_probability_values_verified':checked,
            'same_observations_and_estimates_for_both_learners':True,'hidden_goal_field_absent':True,
            'training_family_membership_verified':True,'knowledge_source_spans_verified':spans,
            'tagged_example_posteriors':tagged,'untagged_example_posteriors':untagged,
            'verification_script_sha256':sha(Path(__file__))}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
