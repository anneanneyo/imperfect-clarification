"""Exact post-hoc E012 ablation: merge observable full/partial tags only."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/retail_small_feedback'
Q=['order_id','contents','total','count']

def score(joints,n):
    accuracy=sum(max(v) for v in joints.values())
    ce=0
    for v in joints.values():
        mass=sum(v);ps=[x/mass for x in v]
        ce+=mass*(-sum(p*math.log2(p) for p in ps if p>0))
    return accuracy,math.log2(n)-ce

def main():
    rawp=Path('/private/tmp/tau3-audit-20261027/data/tau2/domains/retail/db.json')
    assert hashlib.sha256(rawp.read_bytes()).hexdigest()=='413a65160adbdb5fde0ffc0015c49b6d70250b10c18128de169b597af7766765'
    raw=json.loads(rawp.read_text());byuser=defaultdict(list)
    for o in raw['orders'].values():
        if o['status']=='pending':
            h=hashlib.sha256(json.dumps(o['user_id'],ensure_ascii=False,separators=(',',':'),sort_keys=True).encode()).hexdigest()
            byuser[h].append(o)
    old=json.loads((ROOT/'results/retail_clarification_contract/families.json').read_text())
    rows=[]
    for f in old:
        orders=sorted(byuser[f['family_hash']],key=lambda o:o['order_id']);n=len(orders)
        tagged=defaultdict(lambda:[0.]*n)
        for i,o in enumerate(orders):
            names=tuple(sorted(x['name'] for x in o['items']));unique=set(names)
            tagged[('full',*names)][i]+=.45/n
            for name in unique:tagged[('partial',name)][i]+=.45/(n*len(unique))
            tagged[('idk',)][i]+=.10/n
        merged=defaultdict(lambda:[0.]*n)
        for y,v in tagged.items():
            new=y if y[0]=='idk' else ('names',*y[1:])
            for i,x in enumerate(v):merged[new][i]+=x
        a0,h0=score(tagged,n);a1,h1=score(merged,n)
        expected=f['question_results']['imperfect']['contents']
        assert abs(a0-expected['accuracy'])<1e-12 and abs(h0-expected['information_gain_bits'])<1e-12
        assert a1<=a0+1e-12 and h1<=h0+1e-12
        values={q:f['question_results']['imperfect'][q]['accuracy'] for q in Q};values['contents']=a1
        gains={q:f['question_results']['imperfect'][q]['information_gain_bits'] for q in Q};gains['contents']=h1
        eq=max(Q,key=lambda q:gains[q]);best='no_ask';value=1/n
        for q in Q:
            if values[q]-.05>value+1e-12:best,value=q,values[q]-.05
        entropy_accuracy=values[eq] if gains[eq]>1e-12 else 1/n
        oracle_accuracy=1/n if best=='no_ask' else values[best]
        rows.append({'family_hash':f['family_hash'],'contents_accuracy_before':a0,'contents_accuracy_after':a1,
                     'information_gain_before':h0,'information_gain_after':h1,'untagged_entropy_question':eq,
                     'untagged_evsi_question':best,'untagged_entropy_accuracy':entropy_accuracy,'untagged_oracle_accuracy':oracle_accuracy})
    result={'families':len(rows),'max_accuracy_loss':max(r['contents_accuracy_before']-r['contents_accuracy_after'] for r in rows),
            'families_with_positive_accuracy_loss':sum(r['contents_accuracy_before']-r['contents_accuracy_after']>1e-12 for r in rows),
            'families_information_gain_decreases':sum(r['information_gain_before']-r['information_gain_after']>1e-12 for r in rows),
            'entropy_evsi_question_disagreements':sum(r['untagged_entropy_question']!=r['untagged_evsi_question'] for r in rows),
            'untagged_entropy_accuracy':sum(r['untagged_entropy_accuracy'] for r in rows)/len(rows),
            'untagged_oracle_accuracy':sum(r['untagged_oracle_accuracy'] for r in rows)/len(rows),
            'protocol_sha256':hashlib.sha256((ROOT/'research/E012_TAG_VISIBILITY_CHECK.md').read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rows':rows}
    (OUT/'tag_visibility.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
