"""Check E022 with integer joint masses and an explicit banking rule implementation."""
import hashlib,itertools,json
from collections import Counter,defaultdict
from decimal import Decimal
from fractions import Fraction as F
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/symbolic_headroom_e022';SRC=ROOT/'results/heldout_value_e016'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for manifest in ['manifest.json','score_manifest.json']:
 for n,h in read(OUT/manifest)['files'].items():assert sha(ROOT/n)==h,n
Q=read(OUT/'manifest.json')['questions'];schema=read(SRC/'schema.json');slots=schema['slots'];fields=Q[:-1];teams=sorted(schema['units'])
domains={k:list(slots[k]['values']) if 'values'in slots[k] else list(dict.fromkeys(v for vals in slots[k]['cond'].values() for v in vals)) for k in fields}
def route(w):
 if w['issue']in ['unrecognized_charge','lost_card']:return 'fraud' if w['money_gone']=='yes' else 'cards'
 if w['issue']=='fee_question' and w['compensation']=='yes':return 'complaints'
 if w['product']=='loan':return 'loans'
 if w['product']=='card':return 'cards'
 return 'accounts'
worlds=[]
for issue,p0 in slots['issue']['values'].items():
 for vals in itertools.product(*(slots[k]['cond'][issue].items() for k in fields[1:])):
  w={'issue':issue};mass=Decimal(str(p0))
  for k,(v,p) in zip(fields[1:],vals):w[k]=v;mass*=Decimal(str(p))
  integer=mass*100000000;assert integer==int(integer);worlds.append((w,int(integer),route(w)))
assert len(worlds)==44 and sum(w[1] for w in worlds)==100000000
def prediction(f):
 masses={t:sum(p for w,p,t0 in worlds if t0==t and all(w[k]in f[k] for k in fields)) for t in teams};assert sum(masses.values())
 action=sorted(teams,key=lambda t:(-masses[t],t))[0]
 return action,{t:F(p,sum(masses.values())) for t,p in masses.items()}
source={r['id']:r for r in read(SRC/'source_rows.json')};base={r['source_id']:r for r in read(SRC/'train_details.json') if r['question']=='no_ask'};ds=read(OUT/'details.json');index={(r['source_id'],r['question']):r for r in ds}
assert len(ds)==len(index)==1644 and {r['source_id'] for r in ds}==set(base)
for sid,t in base.items():
 r=source[sid];initial={k:set(r['evidence'].get(k,domains[k])) for k in fields};a0,p0=prediction(initial)
 assert route(r['profile'])==r['gold']==t['gold'];assert all(r['profile'][k]in initial[k] for k in fields)
 for q in ['no_ask']+Q:
  d=index[sid,q];assert d['group_id']==t['group_id'] and d['gold']==r['gold']
  after=initial if q=='no_ask' else {k:initial[k]&set(r['probes'][q]['reveal'].get(k,domains[k])) for k in fields}
  assert all(r['profile'][k]in after[k] for k in fields)
  assert {k:set(v) for k,v in d['facts'].items()}==after
  a,p=prediction(after);assert d['action']==a and d['correct']==int(a==r['gold']) and {k:F(v) for k,v in d['posterior'].items()}==p
  if q=='no_ask':assert d['kind']=='no_ask';continue
  assert d['kind']==r['probes'][q]['kind']
  full={k:({r['profile'][k]} if k==q else initial[k]) for k in fields};af,pf=prediction(full)
  assert {k:set(v) for k,v in d['complete_facts'].items()}==full and d['complete_action']==af and d['complete_correct']==int(af==r['gold']) and {k:F(v) for k,v in d['complete_posterior'].items()}==pf
  assert d['facts_changed']==(after!=initial) and d['action_changed']==(a!=a0) and d['correct_before']==int(a0==r['gold'])
groups=defaultdict(list)
for sid,t in base.items():groups[t['group_id']].append(sid)
components=read(OUT/'components.json');assert len(components)==len(groups)==96
obs={};comp={}
for gid,ids in groups.items():
 obs[gid]={q:F(sum(index[sid,q]['correct'] for sid in ids),len(ids)) for q in ['no_ask']+Q}
 comp[gid]={q:F(sum(index[sid,q]['complete_correct'] for sid in ids),len(ids)) for q in Q}
avg={q:sum(v[q] for v in obs.values())/96 for q in ['no_ask']+Q};fixed=max(Q,key=avg.get);aq={gid:max(Q,key=obs[gid].get) for gid in groups};nq={gid:max(Q,key=comp[gid].get) for gid in groups}
for c in components:
 gid=c['group_id'];opt=[q for q in Q if comp[gid][q]==max(comp[gid].values())]
 assert set(c['source_ids'])==set(groups[gid]) and {q:F(v) for q,v in c['published'].items()}==obs[gid] and {q:F(v) for q,v in c['complete'].items()}==comp[gid]
 assert c['aware_question']==aq[gid] and c['complete_question']==nq[gid] and c['complete_optimal_questions']==opt
 assert list(map(F,c['complete_optimal_published_range']))==[min(obs[gid][q] for q in opt),max(obs[gid][q] for q in opt)]
 assert c['question_accuracies_differ']==(len({obs[gid][q] for q in Q})>1)
 assert F(c['aware_minus_fixed'])==obs[gid][aq[gid]]-obs[gid][fixed]
def exact(x):return dict(fraction=str(x),value=float(x))
aware=sum(obs[g][aq[g]] for g in groups)/96;naive=sum(obs[g][nq[g]] for g in groups)/96
lo=sum(F(c['complete_optimal_published_range'][0]) for c in components)/96;hi=sum(F(c['complete_optimal_published_range'][1]) for c in components)/96
kinds={}
for kind in sorted({d['kind'] for d in ds}-{'no_ask'}):
 rows=[d for d in ds if d['kind']==kind]
 kinds[kind]=dict(probes=len(rows),facts_changed=sum(d['facts_changed'] for d in rows),action_changed=sum(d['action_changed'] for d in rows),correct_before=sum(d['correct_before'] for d in rows),correct_after=sum(d['correct'] for d in rows),improved=sum(d['correct']>d['correct_before'] for d in rows),worsened=sum(d['correct']<d['correct_before'] for d in rows))
expected=dict(groups=96,profiles=274,positive_prior_worlds=44,branch_rows=1644,constant_questions={q:exact(avg[q]) for q in Q},no_ask=exact(avg['no_ask']),best_constant=dict(question=fixed,accuracy=exact(avg[fixed])),published_aware_oracle=dict(accuracy=exact(aware),question_counts=dict(Counter(aq.values()))),complete_answer_selector_on_published=dict(accuracy=exact(naive),question_counts=dict(Counter(nq.values())),optimal_tie_range=[exact(lo),exact(hi)]),aware_minus_fixed=exact(aware-avg[fixed]),aware_minus_complete_selector=exact(aware-naive),groups_with_question_accuracy_difference=sum(c['question_accuracies_differ'] for c in components),groups_aware_better_than_fixed=sum(F(c['aware_minus_fixed'])>0 for c in components),groups_changed_choice=sum(aq[g]!=nq[g] for g in groups),probe_kinds_unweighted=kinds,trained_method=False,empirical_bayes_ceiling=False,new_model_calls=0,source_prior_equals_evaluation_weights=False)
assert read(OUT/'summary.json')==expected
report=dict(passed=True,branches_verified=1644,groups=96,integer_joint_mass=True,independent_banking_rules=True,all_metrics_recomputed=True,verifier_sha256=sha(Path(__file__)),manifest_sha256=sha(OUT/'manifest.json'))
(OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
