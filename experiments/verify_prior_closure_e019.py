"""Reproduce the source-prior-support diagnostic without importing an experiment scorer."""
import hashlib,itertools,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/e019_prior_closure'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for n,h in read(OUT/'manifest.json')['files'].items():assert sha(ROOT/n)==h,n
schema=read(ROOT/'results/heldout_value_e016/schema.json');slots=schema['slots']
domains={'issue':['unrecognized_charge','lost_card','fee_question','change_request','failed_payment'],'product':['card','current_account','loan','savings'],'money_gone':['yes','no'],'compensation':['yes','no']}
worlds=[]
for vals in itertools.product(*domains.values()):
 w=dict(zip(domains,vals));prob=1
 for k in domains:
  s=slots[k];prob*=s['values'].get(w[k],0) if 'values'in s else s['cond'][w[s['depends_on']]].get(w[k],0)
 if prob>0:worlds.append(w)
assert len(worlds)==44
details=[]
for c in read(ROOT/'results/structured_facts_e019/details.json'):
 possible=[w for w in worlds if all(w[k]in c['facts'][k] for k in domains)]
 assert possible
 if c['predicted_facts']is None:continue
 for k,domain in domains.items():
  pred=c['predicted_facts'][k]
  if len(c['facts'][k])!=len(domain) or len(pred)==len(domain):continue
  support=sorted(v for v in domain if any(w[k]==v for w in possible))
  details.append(dict(dataset=c['dataset'],case_id=c['case_id'],field=k,prediction=pred,conditional_positive_prior_support=support,excludes_positive_prior_value=bool(set(support)-set(pred))))
assert details==read(OUT/'details.json')
summary={}
for label in ['E014_dev','E016_train']:
 rows=[d for d in details if d['dataset']==label]
 summary[label]=dict(total_unknown_contractions=len(rows),excludes_possible_values=sum(d['excludes_positive_prior_value'] for d in rows),consistent_with_conditional_support=sum(not d['excludes_positive_prior_value'] for d in rows))
assert summary==read(OUT/'summary.json')
report=dict(passed=True,positive_prior_worlds=44,diagnostic_rows=len(details),verifier_sha256=sha(Path(__file__)),manifest_sha256=sha(OUT/'manifest.json'))
(OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
