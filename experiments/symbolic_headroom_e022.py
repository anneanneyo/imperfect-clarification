"""Frozen, privileged E022 diagnostic; never an ordinary agent or empirical Bayes ceiling."""
import argparse,hashlib,itertools,json
from collections import Counter,defaultdict
from fractions import Fraction as F
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/symbolic_headroom_e022';SRC=ROOT/'results/heldout_value_e016'
Q=['issue','product','money_gone','compensation','customer_name']
D={'issue':['unrecognized_charge','lost_card','fee_question','change_request','failed_payment'],'product':['card','current_account','loan','savings'],'money_gone':['yes','no'],'compensation':['yes','no']}
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def exact(x):return {'fraction':str(x),'value':float(x)}
def prepare():
 assert not OUT.exists();OUT.mkdir()
 paths=[Path(__file__).resolve(),ROOT/'research/E022_PROTOCOL.md']+[SRC/n for n in ['source_rows.json','train_details.json','schema.json','score_manifest.json']]
 for n,h in read(SRC/'score_manifest.json')['files'].items():assert sha(ROOT/n)==h,n
 save(OUT/'manifest.json',{'files':{str(p.relative_to(ROOT)):sha(p) for p in paths},'groups':96,'profiles':274,'questions':Q,'test_used':False,'model_calls':0,'scope':'Privileged finite-training-population diagnostic under fixed source-prior MAP executor.'})
 print('E022 inputs/code/protocol frozen; no outcomes calculated')
def run():
 m=read(OUT/'manifest.json')
 for n,h in m['files'].items():assert sha(ROOT/n)==h,n
 assert not (OUT/'summary.json').exists()
 schema=read(SRC/'schema.json');slots=schema['slots'];teams=sorted(schema['units'])
 def route(w):
  for rule in schema['rules']:
   if 'else'in rule:return rule['else']
   if all(w[k]in v for k,v in rule['if'].items()):return rule['unit']
  raise AssertionError('No route')
 worlds=[]
 for vals in itertools.product(*D.values()):
  w=dict(zip(D,vals));p=F(1)
  for k in D:
   s=slots[k];v=s['values'].get(w[k],0) if 'values'in s else s['cond'][w[s['depends_on']]].get(w[k],0);p*=F(str(v))
  if p:worlds.append((w,p,route(w)))
 assert len(worlds)==44 and sum(p for _,p,_ in worlds)==1
 def action(f):
  masses={t:F(0) for t in teams}
  for w,p,t in worlds:
   if all(w[k]in f[k] for k in D):masses[t]+=p
  z=sum(masses.values());assert z>0
  return max(teams,key=lambda t:masses[t]),{t:str(v/z) for t,v in masses.items()}
 source={r['id']:r for r in read(SRC/'source_rows.json')};tr=[r for r in read(SRC/'train_details.json') if r['question']=='no_ask'];groups=defaultdict(list)
 for r in tr:groups[r['group_id']].append(r)
 assert len(groups)==96 and len(tr)==274 and len({r['source_id'] for r in tr})==274
 details=[];components=[];kind=defaultdict(Counter);overall={q:F(0) for q in ['no_ask']+Q};aware=naive=low=high=F(0);choice=Counter();naivechoice=Counter()
 for gid,rs in groups.items():
  assert len({json.dumps(source[r['source_id']]['evidence'],sort_keys=True) for r in rs})==1
  assert len({source[r['source_id']]['state'][0]['text'] for r in rs})==1
  observed={q:F(0) for q in ['no_ask']+Q};complete={q:F(0) for q in Q}
  for t in rs:
   r=source[t['source_id']];before={k:[v for v in dom if v in r['evidence'].get(k,dom)] for k,dom in D.items()};gold=r['gold'];assert route(r['profile'])==gold==t['gold'];assert all(r['profile'][k]in before[k] for k in D)
   a0,p0=action(before);correct0=int(a0==gold);observed['no_ask']+=F(correct0,len(rs))
   details.append(dict(group_id=gid,source_id=r['id'],question='no_ask',kind='no_ask',facts=before,action=a0,posterior=p0,gold=gold,correct=correct0))
   for q in Q:
    probe=r['probes'][q];after={k:[v for v in before[k] if v in probe['reveal'].get(k,dom)] for k,dom in D.items()};full={k:([r['profile'][k]] if k==q else before[k]) for k in D}
    assert all(r['profile'][k]in after[k] for k in D)
    a,p=action(after);af,pf=action(full);correct=int(a==gold);cf=int(af==gold);observed[q]+=F(correct,len(rs));complete[q]+=F(cf,len(rs))
    if before==after:assert a==a0 and p==p0
    if probe['kind']=='unknown':assert before==after
    k=kind[probe['kind']];k['probes']+=1;k['facts_changed']+=before!=after;k['action_changed']+=a!=a0;k['correct_before']+=correct0;k['correct_after']+=correct;k['improved']+=correct>correct0;k['worsened']+=correct<correct0
    details.append(dict(group_id=gid,source_id=r['id'],question=q,kind=probe['kind'],facts=after,action=a,posterior=p,gold=gold,correct=correct,complete_facts=full,complete_action=af,complete_posterior=pf,complete_correct=cf,facts_changed=before!=after,action_changed=a!=a0,correct_before=correct0))
  aq=max(Q,key=observed.get);nq=max(Q,key=complete.get);opt=[q for q in Q if complete[q]==max(complete.values())];choice[aq]+=1;naivechoice[nq]+=1
  for q in overall:overall[q]+=observed[q]/96
  aware+=observed[aq]/96;naive+=observed[nq]/96;lo=min(observed[q] for q in opt);hi=max(observed[q] for q in opt);low+=lo/96;high+=hi/96
  components.append(dict(group_id=gid,source_ids=[r['source_id'] for r in rs],published={q:str(v) for q,v in observed.items()},complete={q:str(v) for q,v in complete.items()},aware_question=aq,complete_question=nq,complete_optimal_questions=opt,complete_optimal_published_range=[str(lo),str(hi)],question_accuracies_differ=len({observed[q] for q in Q})>1))
 fixed=max(Q,key=overall.get)
 for c in components:c['aware_minus_fixed']=str(F(c['published'][c['aware_question']])-F(c['published'][fixed]))
 summary={'groups':96,'profiles':274,'positive_prior_worlds':44,'branch_rows':len(details),'constant_questions':{q:exact(overall[q]) for q in Q},'no_ask':exact(overall['no_ask']),'best_constant':{'question':fixed,'accuracy':exact(overall[fixed])},'published_aware_oracle':{'accuracy':exact(aware),'question_counts':dict(choice)},'complete_answer_selector_on_published':{'accuracy':exact(naive),'question_counts':dict(naivechoice),'optimal_tie_range':[exact(low),exact(high)]},'aware_minus_fixed':exact(aware-overall[fixed]),'aware_minus_complete_selector':exact(aware-naive),'groups_with_question_accuracy_difference':sum(c['question_accuracies_differ'] for c in components),'groups_aware_better_than_fixed':sum(F(c['aware_minus_fixed'])>0 for c in components),'groups_changed_choice':sum(c['aware_question']!=c['complete_question'] for c in components),'probe_kinds_unweighted':{k:dict(v) for k,v in kind.items()},'trained_method':False,'empirical_bayes_ceiling':False,'new_model_calls':0,'source_prior_equals_evaluation_weights':False}
 save(OUT/'details.json',details);save(OUT/'components.json',components);save(OUT/'summary.json',summary)
 paths=[OUT/n for n in ['manifest.json','details.json','components.json','summary.json']];save(OUT/'score_manifest.json',{'files':{str(p.relative_to(ROOT)):sha(p) for p in paths}});print(json.dumps(summary))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);globals()[p.parse_args().stage]()
