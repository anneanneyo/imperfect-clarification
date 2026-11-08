"""Descriptive, case-clustered contrasts for the development-only E020 control."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/e020_diagnostics';SOURCE=ROOT/'results/evidence_facts_e020'
D={'issue':['unrecognized_charge','lost_card','fee_question','change_request','failed_payment'],'product':['card','current_account','loan','savings'],'money_gone':['yes','no'],'compensation':['yes','no']}
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert read(SOURCE/'verification.json')['completed'] and read(SOURCE/'verification.json')['passed']
for n,h in read(SOURCE/'score_manifest.json')['files'].items():assert sha(ROOT/n)==h
assert not OUT.exists();ds=read(SOURCE/'details.json');summary={};mismatch=[]
for label in ['E014_dev','E016_train']:
 rows=[d for d in ds if d['dataset']==label];n=len(rows);rng=np.random.default_rng(20261108);indices=rng.integers(0,n,size=(10000,n));summary[label]={}
 for a,b in [('gated','old'),('raw','old'),('gated','raw')]:
  summary[label][a+'_minus_'+b]={}
  for subset in ['all','known','unknown']:
   num=[];den=[]
   for d in rows:
    fields=[k for k in D if subset=='all' or (len(d['facts'][k])<len(D[k]))==(subset=='known')]
    num.append(sum(int(d[a]is not None and set(d[a][k])==set(d['facts'][k]))-int(d[b]is not None and set(d[b][k])==set(d['facts'][k])) for k in fields));den.append(len(fields))
   num=np.array(num);den=np.array(den);totals=den[indices].sum(1);assert np.all(totals>0)
   samples=num[indices].sum(1)/totals
   summary[label][a+'_minus_'+b][subset]={'difference':float(num.sum()/den.sum()),'case_cluster_percentile95':np.quantile(samples,[.025,.975]).tolist(),'improved_cases':int((num>0).sum()),'worse_cases':int((num<0).sum()),'ties':int((num==0).sum())}
 for d in rows:
  for k in D:
   pred=d['gated'][k] if d['gated']else None
   if pred!=d['facts'][k]:mismatch.append({'dataset':label,'case_id':d['case_id'],'text':d['text'],'field':k,'reference':d['facts'][k],'prediction':pred,'quote':d['quotes'][k]['evidence'] if d['quotes']else None,'quote_check':d['checks'].get(k)})
OUT.mkdir()
for n,x in [('paired_contrasts.json',summary),('remaining_mismatches.json',mismatch)]: (OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
paths=[Path(__file__).resolve(),SOURCE/'details.json',SOURCE/'verification.json',OUT/'paired_contrasts.json',OUT/'remaining_mismatches.json'];(OUT/'manifest.json').write_text(json.dumps({'files':{str(p.relative_to(ROOT)):sha(p) for p in paths},'interpretation':'Descriptive cluster bootstrap on development/training cases after adaptive experiment selection; not held-out confirmation.'},indent=2)+'\n');print(json.dumps({'training':summary['E016_train'],'mismatch_fields':len(mismatch)}))
