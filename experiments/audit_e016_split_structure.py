"""Input-only description of semantic-pattern overlap; no outcome selection."""
import hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'results/heldout_value_e016';OUT=ROOT/'results/e016_split_structure'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=True);components=read(SOURCE/'components.json');source=read(SOURCE/'source_rows.json');bycase=defaultdict(list)
    for r in source:bycase['/'.join(r['id'].split('/')[:3])].append(r)
    rows=[]
    for g in components:
        patterns=set();styles=set()
        for c in g['cases']:
            for r in bycase[c]:
                evidence={k:sorted(v) for k,v in r['evidence'].items() if k!='customer_name'}
                patterns.add(json.dumps(evidence,sort_keys=True));styles.add(r['style'])
        assert len(patterns)==len(styles)==1
        rows.append({'group_id':g['group_id'],'split':g['split'],'decisive_evidence_pattern':patterns.pop(),'style':styles.pop()})
    train={r['decisive_evidence_pattern'] for r in rows if r['split']=='train'};test={r['decisive_evidence_pattern'] for r in rows if r['split']=='test'}
    for r in rows:r['pattern_present_in_train']=r['decisive_evidence_pattern']in train
    summary={'train_components':96,'test_components':48,'distinct_decisive_patterns_train':len(train),'distinct_decisive_patterns_test':len(test),
        'shared_patterns':len(train&test),'test_components_with_training_pattern':sum(r['split']=='test' and r['pattern_present_in_train'] for r in rows),
        'styles':{s:dict(Counter(r['style'] for r in rows if r['split']==s)) for s in ['train','test']},
        'test_generations_exist_at_audit':any((SOURCE/'shards'/s['name']/'generations.jsonl').exists() for s in read(SOURCE/'shards.json') if s['split']=='test'),
        'interpretation':'Exact texts are disjoint; this is not a held-out task-rule or semantic-evidence-pattern split. Privileged evidence used for reporting only; no exclusions or feature additions.'}
    for name,obj in [('groups.json',rows),('summary.json',summary)]: (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
    paths=[Path(__file__).resolve(),SOURCE/'components.json',SOURCE/'source_rows.json',OUT/'groups.json',OUT/'summary.json']
    (OUT/'manifest.json').write_text(json.dumps({'files':{str(p.relative_to(ROOT)):sha(p) for p in paths}},indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
