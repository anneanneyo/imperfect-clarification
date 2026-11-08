"""Reconstruct the entire imperfect-reply inventory directly from E016 training probes."""
import hashlib,itertools,json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/imperfect_reply_inventory_e021'
D={'issue':['unrecognized_charge','lost_card','fee_question','change_request','failed_payment'],'product':['card','current_account','loan','savings'],'money_gone':['yes','no'],'compensation':['yes','no']}
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
for n,h in read(OUT/'manifest.json')['files'].items():assert sha(ROOT/n)==h,n
source={r['id']:r for r in read(ROOT/'results/heldout_value_e016/source_rows.json')}
training=read(ROOT/'results/heldout_value_e016/train_details.json')
schema=read(ROOT/'results/heldout_value_e016/schema.json')
def routes(f):
 out=set()
 for vals in itertools.product(*(f[k] for k in D)):
  w=dict(zip(D,vals))
  for rule in schema['rules']:
   if 'else'in rule:out.add(rule['else']);break
   if all(w[k]in v for k,v in rule['if'].items()):out.add(rule['unit']);break
 return sorted(out)
expected={};records=Counter()
for t in training:
 if t['kind']not in ['partial','unknown']:continue
 r=source[t['source_id']];p=r['probes'][t['question']];assert p['kind']==t['kind'];records[p['kind']]+=1
 assert len(r['state'])==1 and r['state'][0]['role']=='user'
 messages=[{'role':'user','content':r['state'][0]['text']},{'role':'assistant','content':p['q']},{'role':'user','content':p['a']}]
 before={k:[v for v in domain if v in r['evidence'].get(k,domain)] for k,domain in D.items()}
 after={k:[v for v in before[k] if v in p['reveal'].get(k,domain)] for k,domain in D.items()}
 assert all(after.values())
 key=digest(messages)
 item=dict(key=key,messages=messages,kind=p['kind'],case_id=t['case_id'],question=t['question'],facts_before=before,facts_after=after,source_ids=[r['id']])
 if key in expected:
  old=expected[key];assert all(item[k]==old[k] for k in item if k!='source_ids');assert r['id']not in old['source_ids'];old['source_ids'].append(r['id'])
 else:expected[key]=item
actual=read(OUT/'cases.json');assert len(actual)==len(expected)==len({c['key'] for c in actual})==len({digest(c['messages']) for c in actual})==125
for item in actual:
 ex=expected[digest(item['messages'])]
 assert set(item)==set(ex)
 for k in ex:
  if k=='key':continue
  assert (sorted(item[k])==sorted(ex[k]) if k=='source_ids' else item[k]==ex[k]),(item['key'],k)
info={}
for kind in ['partial','unknown']:
 rows=[c for c in actual if c['kind']==kind]
 info[kind]=dict(dialogues=len(rows),questions=dict(Counter(c['question'] for c in rows)),facts_changed=sum(c['facts_before']!=c['facts_after'] for c in rows),route_set_changed=sum(routes(c['facts_before'])!=routes(c['facts_after']) for c in rows),binary_questions=sum(c['question']in ['money_gone','compensation'] for c in rows))
assert read(OUT/'information_content.json')==info
s=read(OUT/'summary.json');assert s['source_records']==dict(records)
assert s['unique_dialogues']=={k:v['dialogues'] for k,v in info.items()}
assert s['unique_cases']==len({c['case_id'] for c in actual})==64
assert s['unknown_facts_unchanged']==all(c['facts_before']==c['facts_after'] for c in actual if c['kind']=='unknown')
assert s['cases_with_partial_routes_changed']==len({c['case_id'] for c in actual if c['kind']=='partial' and routes(c['facts_before'])!=routes(c['facts_after'])})
report=dict(passed=True,source_records=dict(records),unique_dialogues=125,case_groups=64,all_probes_reconstructed=True,independent_rule_interpreter=True,verifier_sha256=sha(Path(__file__)),manifest_sha256=sha(OUT/'manifest.json'))
(OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
