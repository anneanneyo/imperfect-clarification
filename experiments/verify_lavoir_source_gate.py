"""Second implementation: rational Cartesian prior and observable-label ceiling."""
from collections import Counter,defaultdict
from fractions import Fraction as F
import hashlib,itertools,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/lavoir_source_gate'
CACHE=Path('/private/tmp/lavoir-data-20261027')
def main():
    summary=json.loads((OUT/'audit.json').read_text())
    schema=json.loads((CACHE/'schemas/banking_support.json').read_text())
    rows=[json.loads(l) for l in (CACHE/'data/train/banking_support.jsonl').read_text().splitlines()]
    names=['issue','product','money_gone','compensation']; vals=[]
    for name in names:
        s=schema['slots'][name]
        vals.append(sorted(s['values']) if 'values' in s else sorted(set().union(*(p.keys() for p in s['cond'].values()))))
    worlds=[]
    for assignment in itertools.product(*vals):
        w=dict(zip(names,assignment));p=F(1)
        for name in names:
            s=schema['slots'][name]
            pr=s['values'] if 'values' in s else s['cond'][w[s['depends_on']]]
            p*=F(str(pr.get(w[name],0)))
        if not p: continue
        if w['issue'] in ['unrecognized_charge','lost_card']: g='fraud' if w['money_gone']=='yes' else 'cards'
        elif w['issue']=='fee_question' and w['compensation']=='yes':g='complaints'
        elif w['product']=='loan':g='loans'
        elif w['product']=='card':g='cards'
        else:g='accounts'
        worlds.append((w,p,g))
    assert len(worlds)==44 and sum(p for w,p,g in worlds)==1
    groups=defaultdict(list); comparisons=0
    for r in rows:
        target=defaultdict(F)
        for w,p,g in worlds:
            if all(k not in names or w[k] in v for k,v in r['evidence'].items()):target[g]+=p
        z=sum(target.values());assert z>0
        for g,v in r['target'].items():
            assert abs(float(target[g]/z)-v)<2e-9;comparisons+=1
        # Canonical visible input includes candidate descriptions but no IDs/gold/probes.
        key=json.dumps([r['state'],r['question'],r['slots']],sort_keys=True)
        top=max(r['oracle_voi'].values())
        groups[key].append({k for k,v in r['oracle_voi'].items() if top-v<=1e-8})
    ceiling=sum(max(Counter(k for s in sets for k in s).values()) for sets in groups.values())
    impossible=sum(not set.intersection(*sets) for sets in groups.values())
    assert ceiling==summary['max_same_input_top_question_agreement_numerator']==2397
    assert impossible==summary['same_visible_no_common_top_question_groups']==122
    result={'verified':True,'posterior_values_recomputed':comparisons,'rational_prior_worlds':len(worlds),
            'agreement_ceiling_count':ceiling,'total_rows':len(rows),'incompatible_groups':impossible,
            'audit_sha256':hashlib.sha256((OUT/'audit.json').read_bytes()).hexdigest(),
            'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
