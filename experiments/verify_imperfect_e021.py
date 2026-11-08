"""Independent reconstruction of E021 prompts, parsing, scores and IDK stability."""
import ast,hashlib,itertools,json,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/imperfect_replies_e021'
D={'issue':['unrecognized_charge','lost_card','fee_question','change_request','failed_payment'],'product':['card','current_account','loan','savings'],'money_gone':['yes','no'],'compensation':['yes','no']}
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def hashes(p):
 for n,h in read(p)['files'].items():assert sha(ROOT/n)==h,n
def unique(ps):
 assert len(ps)==len(dict(ps));return dict(ps)
def parse(s):
 try:
  o=json.loads(s,object_pairs_hook=unique);assert type(o)is dict and set(o)==set(D)
  for k,domain in D.items():
   x=o[k];assert type(x)is dict and set(x)=={'values','evidence'}
   a=x['values'];assert type(a)is list and a and all(type(t)is str and t in domain for t in a) and len(a)==len(set(a))
   assert x['evidence']is None or type(x['evidence'])is str
  return o
 except (AssertionError,ValueError,TypeError):return None
schema=read(ROOT/'results/heldout_value_e016/schema.json')
def routes(f):
 if f is None:return []
 out=set()
 for vals in itertools.product(*(f[k] for k in D)):
  w=dict(zip(D,vals))
  for rule in schema['rules']:
   if 'else'in rule:out.add(rule['else']);break
   if all(w[k]in v for k,v in rule['if'].items()):out.add(rule['unit']);break
 return sorted(out)
def system_node(path):
 tree=ast.parse(path.read_text())
 return next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SYSTEM' for t in n.targets))
runpy.run_path(str(ROOT/'experiments/verify_imperfect_inventory_e021.py'))
hashes(OUT/'manifest.json');hashes(ROOT/'results/evidence_facts_e020/score_manifest.json')
assert read(ROOT/'results/evidence_facts_e020/summary.json')['candidate_gate_passed']
system=ast.literal_eval(system_node(ROOT/'experiments/evidence_facts_e020.py'))+ast.literal_eval(system_node(ROOT/'experiments/imperfect_replies_e021.py').right)
inventory={c['key']:c for c in read(ROOT/'results/imperfect_reply_inventory_e021/cases.json')}
initial={c['case_id']:c for c in read(ROOT/'results/evidence_facts_e020/details.json')}
cs=read(OUT/'cases.json');pack=read(OUT/'pack.json');ps={p['id']:p['messages'] for p in pack}
assert len(cs)==len(pack)==len(ps)==125 and {c['key'] for c in cs}==set(inventory)
for c in cs:
 old=inventory[c['key']];assert all(c[k]==v for k,v in old.items())
 assert c['facts']==c['facts_after'] and c['dataset']==c['kind'] and c['initial_prediction']==initial[c['case_id']]['gated']
 messages=[dict(role='system',content=system)]+c['messages'];assert ps[c['prompt_id']]==messages
 assert c['prompt_id']==hashlib.sha256(json.dumps(messages,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
done=(OUT/'summary.json').exists()
if done:
 hashes(OUT/'score_manifest.json');assert read(OUT/'run_status.json')['complete']
 gs=[json.loads(l) for l in (OUT/'generations.jsonl').read_text().splitlines()];gm={g['id']:g for g in gs};assert len(gs)==len(gm)==125 and set(gm)==set(ps)
 ds=read(OUT/'details.json');assert len(ds)==125;summary=read(OUT/'summary.json')
 for c,d in zip(cs,ds):
  assert all(d[k]==v for k,v in c.items());g=gm[c['prompt_id']];o=parse(g['raw_output']) if g['error']is None else None;assert d['quotes']==o
  if o is None:assert d['raw']is None and d['gated']is None and d['checks']=={};continue
  for k,domain in D.items():
   vals=[v for v in domain if v in o[k]['values']];q=o[k]['evidence'];narrow=len(vals)<len(domain)
   valid=type(q)is str and 0<len(q)<=160 and any(q in m['content'] for m in c['messages'] if m['role']=='user');reset=narrow and not valid
   assert d['raw'][k]==vals and d['gated'][k]==(domain if reset else vals)
   assert d['checks'][k]==dict(narrowed=narrow,quote_valid=valid,reset_to_unknown=reset)
 for kind in ['partial','unknown']:
  rows=[d for d in ds if d['kind']==kind];n=len(rows)
  for method in ['raw','gated']:
   fields=[(set(d['facts'][k]),set(d[method][k]) if d[method]is not None else None,len(D[k])) for d in rows for k in D]
   p=[routes(d[method]) for d in rows];r=[routes(d['facts']) for d in rows]
   counts=dict(cases=n,invalid=sum(d[method]is None for d in rows),slot_exact=sum(a==b for a,b,_ in fields),slot_total=4*n,known_exact=sum(a==b for a,b,z in fields if len(a)<z),known_total=sum(len(a)<z for a,b,z in fields),unknown_exact=sum(a==b for a,b,z in fields if len(a)==z),unknown_total=sum(len(a)==z for a,b,z in fields),all_exact=sum(d['facts']==d[method] for d in rows),route_set_exact=sum(a==b for a,b in zip(r,p)),singleton_cases=sum(len(a)==1 for a in r),singleton_correct=sum(a==b for a,b in zip(r,p) if len(a)==1),singleton_wrong=sum(len(b)==1 and a!=b for a,b in zip(r,p) if len(a)==1),singleton_uncertain=sum(len(b)>1 for a,b in zip(r,p) if len(a)==1))
   counts['slot_accuracy']=counts['slot_exact']/(4*n);assert summary[kind][method]==counts,(kind,method)
 expected=[]
 for d in ds:
  if d['kind']!='unknown':continue
  for k in D:
   a=d['initial_prediction'][k] if d['initial_prediction']is not None else None;b=d['gated'][k] if d['gated']is not None else None;ref=d['facts'][k]
   expected.append(dict(key=d['key'],case_id=d['case_id'],field=k,before=a,after=b,reference=ref,changed=a!=b,before_exact=a==ref,after_exact=b==ref,newly_excluded_values=sorted(set(a or [])-set(b or [])) if b is not None else None,restored_values=sorted(set(b or [])-set(a or [])) if a is not None else None))
 assert read(OUT/'unknown_stability.json')==expected
 assert summary['unknown_stability']==dict(fields=len(expected),changed=sum(d['changed'] for d in expected),newly_incorrect=sum(d['before_exact'] and not d['after_exact'] for d in expected),newly_correct=sum(d['after_exact'] and not d['before_exact'] for d in expected))
 gate=all(summary[k]['gated']['invalid']==0 and summary[k]['gated']['slot_exact']*10>=summary[k]['gated']['slot_total']*9 and summary[k]['gated']['known_exact']*100>=summary[k]['gated']['known_total']*95 for k in ['partial','unknown']);assert summary['candidate_gate_passed']==gate
 assert summary['accounting']==dict(generations=125,runtime_errors=sum(g['error']is not None for g in gs),length_finishes=sum(g['finish_reason']=='length' for g in gs),prompt_tokens=sum(g['prompt_tokens'] for g in gs),generated_tokens=sum(g['generated_tokens'] for g in gs),peak_memory_gb=max(g['peak_memory_gb'] or 0 for g in gs),elapsed_seconds=read(OUT/'run_status.json')['elapsed_seconds'])
report=dict(passed=True,completed=done,source_and_prompt_joins=125,independent_schema_rules=True,verifier_sha256=sha(Path(__file__)),manifest_sha256=sha(OUT/'manifest.json'))
(OUT/('verification.json' if done else 'input_verification.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
