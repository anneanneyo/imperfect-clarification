"""Independent source joins, strict output decoding and exact E020 metrics."""
import ast,hashlib,itertools,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/evidence_facts_e020'
D={'issue':['unrecognized_charge','lost_card','fee_question','change_request','failed_payment'],'product':['card','current_account','loan','savings'],'money_gone':['yes','no'],'compensation':['yes','no']}
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def hashes(m):
 for n,h in m['files'].items():assert sha(ROOT/n)==h,n
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
 out=set()
 for vals in itertools.product(*(f[k] for k in D)):
  w=dict(zip(D,vals))
  for rule in schema['rules']:
   if 'else'in rule:out.add(rule['else']);break
   if all(w[k]in v for k,v in rule['if'].items()):out.add(rule['unit']);break
 return sorted(out)
m=read(OUT/'manifest.json');hashes(m);hashes(read(ROOT/'results/e019_semantic_review/manifest.json'))
old={c['case_id']:c for c in read(ROOT/'results/structured_facts_e019/details.json')}
cs=read(OUT/'cases.json');pack=read(OUT/'pack.json');ps={p['id']:p['messages'] for p in pack}
tree=ast.parse((ROOT/'experiments/evidence_facts_e020.py').read_text());system=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SYSTEM' for t in n.targets))
assert len(cs)==len(pack)==len(ps)==120 and {c['case_id'] for c in cs}==set(old)
for c in cs:
 o=old[c['case_id']]
 for k in ['dataset','case_id','text','source_ids','facts','routes','posterior_support','old_teams','gold']:assert c[k]==o[k],k
 assert c['routes']==routes(c['facts'])
 expected=[{'role':'system','content':system},{'role':'user','content':c['text']}];assert ps[c['prompt_id']]==expected
 assert hashlib.sha256(json.dumps(expected,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()==c['prompt_id']
done=(OUT/'summary.json').exists()
if done:
 hashes(read(OUT/'score_manifest.json'));assert read(OUT/'run_status.json')['complete']
 gs=[json.loads(l) for l in (OUT/'generations.jsonl').read_text().splitlines()];gm={g['id']:g for g in gs};assert len(gs)==len(gm)==120 and set(gm)==set(ps)
 ds=read(OUT/'details.json');assert len(ds)==120;summary=read(OUT/'summary.json');resets=0
 for c,d in zip(cs,ds):
  assert all(d[k]==v for k,v in c.items());assert d['old']==old[c['case_id']]['predicted_facts']
  g=gm[c['prompt_id']];o=parse(g['raw_output']) if g['error']is None else None;assert d['quotes']==o
  if o is None:assert d['raw']is None and d['gated']is None and d['checks']=={};continue
  for k,domain in D.items():
   vals=[v for v in domain if v in o[k]['values']];q=o[k]['evidence'];narrow=len(vals)<len(domain)
   valid=type(q)is str and 0<len(q)<=160 and q in c['text'];reset=narrow and not valid
   assert d['raw'][k]==vals and d['gated'][k]==(domain if reset else vals)
   assert d['checks'][k]==dict(narrowed=narrow,quote_valid=valid,reset_to_unknown=reset);resets+=reset
 for label in ['E014_dev','E016_train']:
  rows=[d for d in ds if d['dataset']==label];n=len(rows)
  for method in ['old','raw','gated']:
   fields=[(set(d['facts'][k]),set(d[method][k]) if d[method]is not None else None,len(D[k])) for d in rows for k in D]
   predicted=[routes(d[method]) if d[method]is not None else [] for d in rows];reference=[routes(d['facts']) for d in rows]
   counts=dict(cases=n,invalid=sum(d[method]is None for d in rows),slot_exact=sum(a==b for a,b,_ in fields),slot_total=4*n,
    known_exact=sum(a==b for a,b,z in fields if len(a)<z),known_total=sum(len(a)<z for a,b,z in fields),
    unknown_exact=sum(a==b for a,b,z in fields if len(a)==z),unknown_total=sum(len(a)==z for a,b,z in fields),
    all_exact=sum(d['facts']==d[method] for d in rows),route_set_exact=sum(a==b for a,b in zip(reference,predicted)),
    singleton_cases=sum(len(a)==1 for a in reference),singleton_correct=sum(a==b for a,b in zip(reference,predicted) if len(a)==1),
    singleton_wrong=sum(len(b)==1 and a!=b for a,b in zip(reference,predicted) if len(a)==1),singleton_uncertain=sum(len(b)>1 for a,b in zip(reference,predicted) if len(a)==1))
   counts['slot_accuracy']=counts['slot_exact']/(4*n);assert summary[label][method]==counts,(label,method)
 t=summary['E016_train']['gated'];assert summary['candidate_gate_passed']==(t['invalid']==0 and t['slot_exact']>=346 and t['known_exact']>=192 and t['singleton_correct']>=35)
 assert summary['quote_gate_resets']==resets
 assert summary['accounting']==dict(generations=120,runtime_errors=sum(g['error']is not None for g in gs),length_finishes=sum(g['finish_reason']=='length' for g in gs),prompt_tokens=sum(g['prompt_tokens'] for g in gs),generated_tokens=sum(g['generated_tokens'] for g in gs),peak_memory_gb=max(g['peak_memory_gb'] or 0 for g in gs),elapsed_seconds=read(OUT/'run_status.json')['elapsed_seconds'])
report=dict(passed=True,completed=done,source_and_prompt_joins=120,independent_schema_rules=True,verifier_sha256=sha(Path(__file__)),manifest_sha256=sha(OUT/'manifest.json'))
(OUT/('verification.json' if done else 'input_verification.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
