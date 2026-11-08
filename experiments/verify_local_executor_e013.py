"""Independent probability/metric reconstruction; does not import pilot code."""
from collections import defaultdict
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/local_executor_e013'
Q=['no_ask','order_id','contents','total','count']

def read(name): return json.loads((OUT/name).read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(o): return hashlib.sha256(json.dumps(o,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def close(a,b): assert abs(float(a)-float(b))<1e-12,(a,b)

def strict_parse(raw,labels):
    raw=raw.strip()
    if raw.startswith('```'):
        lines=raw.splitlines()
        if lines[0] not in ['```','```json'] or lines[-1]!='```': return None
        raw='\n'.join(lines[1:-1]).strip()
    if not raw.startswith('{'): return None
    try:
        pairs=json.loads(raw,object_pairs_hook=lambda p:p)
        if type(pairs) is not list or len(pairs)!=1 or pairs[0][0]!='order': return None
        label=pairs[0][1]
        return label if type(label) is str and label in labels else None
    except (ValueError,TypeError,IndexError): return None

def main():
    manifest=read('manifest.json')
    for path,h in manifest['files'].items(): assert sha(ROOT/path)==h,path
    rawp=Path('/private/tmp/tau3-audit-20261027/data/tau2/domains/retail/db.json')
    assert sha(rawp)==read('preparation.json')['database_sha256']
    raw=json.loads(rawp.read_text());pack=read('pack.json');families=read('families.json')
    old=json.loads((ROOT/'results/retail_clarification_contract/families.json').read_text())
    assert [f['family_hash'] for f in families]==[f['family_hash'] for f in old if f['split_label']=='development'][:6]
    assert len(pack)==len(set(r['id'] for r in pack))==129
    expected={}; rates={}; orders_byfamily={}; vector_checks=0
    for f in families:
        h=f['family_hash'];orders=sorted([o for o in raw['orders'].values() if o['status']=='pending' and digest(o['user_id'])==h],key=lambda o:o['order_id'])
        orders_byfamily[h]=orders;n=len(orders);joints=defaultdict(lambda:[Fraction(0)]*n)
        for i,o in enumerate(orders):
            assert f['features'][i]['order_id']==o['order_id']
            names=sorted(x['name'] for x in o['items']); nameset=sorted(set(names))
            cents=int(sum(Decimal(str(x['price'])) for x in o['items'])*100)
            outcomes=[('no_ask',('no_ask',),Fraction(1)),
                      ('order_id',('order_id',o['order_id']),Fraction(35,100)),
                      ('order_id',('idk',),Fraction(65,100)),
                      ('contents',('names',*names),Fraction(45,100)),
                      ('contents',('idk',),Fraction(1,10)),
                      ('total',('total_cents',cents),Fraction(6,10)),
                      ('total',('idk',),Fraction(4,10)),
                      ('count',('count',len(names)),Fraction(9,10)),
                      ('count',('idk',),Fraction(1,10))]
            outcomes += [('contents',('names',name),Fraction(45,100*len(nameset))) for name in nameset]
            for q,y,p in outcomes: joints[q,y][i]+=p/n
        for (q,y),j in joints.items(): expected[h,q,y]=j
        for q in Q:
            branch=[j for (bq,y),j in joints.items() if bq==q]
            assert sum(sum(j) for j in branch)==1
            rates[h,q]=sum(max(j) for j in branch)
            close(rates[h,q],f['symbolic_metrics'][q]['accuracy'])
    for r in pack:
        j=expected.pop((r['family_hash'],r['question'],tuple(r['observation'])))
        assert r['id']==digest(r['messages'])
        assert len(j)==len(r['joint'])
        for a,b in zip(j,r['joint']): close(a,b);vector_checks+=1
        text=json.dumps(r['messages'])
        assert r['family_hash'] not in text
        assert all(tag not in text for tag in ['full_names','one_name','joint','family_hash','alpha'])
        assert len(r['messages'])==(2 if r['question']=='no_ask' else 4)
    assert not expected
    rendered=read('rendered.json'); assert [r['id'] for r in rendered]==[r['id'] for r in pack]
    assert max(len(r['input_tokens']) for r in rendered)<=4096
    result={'status':'pre-generation verified','families':len(families),'unique_prompts':len(pack),
            'joint_values_recomputed':vector_checks,'source_and_manifest_hashes_valid':True,
            'verifier_sha256':sha(Path(__file__).resolve())}
    if (OUT/'summary.json').exists():
        summary=read('summary.json');details={d['id']:d for d in read('details.json')}
        gens=[json.loads(l) for l in (OUT/'generations.jsonl').read_text().splitlines()];byid={r['id']:r for r in gens}
        assert len(gens)==len(byid)==len(pack); assert all(g['error'] is None for g in gens)
        totals=defaultdict(Fraction); invalid=0; choices=defaultdict(set)
        for r in pack:
            g=byid[r['id']];label=strict_parse(g['raw_output'],r['labels'])
            assert label==details[r['id']]['selected_label'];invalid+=label is None
            j=[Fraction(str(x)) for x in r['joint']]
            score=0 if label is None else j[r['labels'].index(label)]
            totals[r['family_hash'],r['question']]+=score
            close(score,details[r['id']]['accuracy_contribution'])
            if label is not None: choices[r['family_hash']].add(digest(orders_byfamily[r['family_hash']][r['labels'].index(label)]['order_id']))
        methodmeans=defaultdict(float)
        for f in summary['per_family']:
            h=f['family_hash']
            for q in Q: close(totals[h,q],f['questions'][q]['accuracy_lower']);close(totals[h,q],f['questions'][q]['accuracy_upper'])
            best=Q[0]
            for q in Q[1:]:
                if float(totals[h,q])-.05 > float(totals[h,best])-.05*(best!='no_ask')+1e-12: best=q
            assert f['selections']['executor_oracle']==best
            for method,q in f['selections'].items(): methodmeans[method]+=float(totals[h,q])/6
        for method,v in methodmeans.items():close(v,summary['methods'][method]['accuracy_lower'])
        assert invalid==summary['invalid_outputs']
        executed=read('execution.json'); assert {e['order_hash'] for e in executed}==set.union(*choices.values())
        assert all(e['success'] and e['e011_state_matches'] for e in executed)
        for path,h in read('analysis_manifest.json')['files'].items(): assert sha(ROOT/path)==h
        result.update(status='complete verified',generated=len(gens),invalid=invalid,
                      selected_order_states=len(executed),metrics_recomputed=True)
    savepath=OUT/('verification.json' if (OUT/'summary.json').exists() else 'pre_verification.json')
    savepath.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
