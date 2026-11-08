"""Check E011 using separate closed-form rational calculations, not its solver."""
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/retail_clarification_contract'

def main():
    raw=json.loads(Path('/private/tmp/tau3-audit-20261027/data/tau2/domains/retail/db.json').read_text())
    summary=json.loads((OUT/'summary.json').read_text())
    rows=json.loads((OUT/'families.json').read_text())
    manifest=json.loads((OUT/'manifest.json').read_text())
    for name,h in manifest['outputs'].items():assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==h
    assert hashlib.sha256((ROOT/'research/E011_PROTOCOL.md').read_bytes()).hexdigest()==manifest['protocol_sha256']
    assert hashlib.sha256((ROOT/'experiments/retail_clarification_contract.py').read_bytes()).hexdigest()==manifest['code_sha256']
    grouped=defaultdict(list)
    for o in raw['orders'].values():
        if o['status']=='pending':grouped[o['user_id']].append(o)
    byhash={hashlib.sha256(json.dumps(u,ensure_ascii=False,separators=(',',':'),sort_keys=True).encode()).hexdigest():sorted(os,key=lambda o:o['order_id']) for u,os in grouped.items() if len(os)>=2}
    assert set(byhash)=={r['family_hash'] for r in rows}
    checked=0;recomputed_loss=Fraction(0)
    for r in rows:
        os=byhash[r['family_hash']];n=len(os)
        sets=[set(i['name'] for i in o['items']) for o in os]
        allnames=set().union(*sets)
        full=len({tuple(sorted(i['name'] for i in o['items'])) for o in os})
        part=sum((max(Fraction(1,n*len(s)) if name in s else Fraction(0) for s in sets) for name in allnames),Fraction(0))
        naive=sum((Fraction(1,n*len(next(s for s in sets if name in s))) for name in allnames),Fraction(0))
        recomputed_loss+=Fraction(45,100)*(part-naive)
        # All source item prices are at most two decimal places. Integer cents
        # avoid depending on the original Decimal conversion.
        totals=[]
        for o in os:
            cents=[Fraction(str(i['price']))*100 for i in o['items']]
            assert all(c.denominator==1 for c in cents)
            totals.append(sum(cents))
        distinct={'order_id':n,'contents':full,'total':len(set(totals)),
                  'count':len({len(o['items']) for o in os})}
        rates={'order_id':Fraction(35,100),'total':Fraction(60,100),'count':Fraction(90,100)}
        for mode in ['perfect','imperfect']:
            for q,d in distinct.items():
                if mode=='perfect':expected=Fraction(d,n)
                elif q=='contents':expected=Fraction(45,100)*Fraction(full,n)+Fraction(45,100)*part+Fraction(10,100*n)
                else:expected=rates[q]*Fraction(d,n)+(1-rates[q])*Fraction(1,n)
                assert abs(float(expected)-r['question_results'][mode][q]['accuracy'])<1e-12,(r['family_hash'],mode,q)
                checked+=1
    assert abs(float(recomputed_loss/len(rows))-summary['partial_support_diagnostics']['macro_contents_accuracy_loss_uniform_partial'])<1e-12
    execution=json.loads((OUT/'execution.json').read_text())
    assert len(execution)==len({x['order_hash'] for x in execution})==249
    assert all(x['success'] and x['error'] is None and all(x['checks'].values()) for x in execution)
    states=defaultdict(list)
    for x in execution:states[x['family_hash']].append(x['after_state_hash'])
    assert all(len(v)==len(set(v)) for v in states.values())
    identical=all(x['methods']['entropy']['question']==x['methods']['oracle_evsi']['question'] for r in rows for x in r['policy_results'])
    result={'families':len(rows),'closed_form_question_accuracy_checks':checked,
            'all_output_and_protocol_hashes_match':True,'execution_records_verified':len(execution),
            'all_families_have_distinct_target_states':True,
            'entropy_evsi_same_choice_all_modes_costs':identical,
            'macro_partial_support_loss_exact_fraction':str(recomputed_loss/len(rows)),
            'macro_partial_support_loss':float(recomputed_loss/len(rows)),
            'verification_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
