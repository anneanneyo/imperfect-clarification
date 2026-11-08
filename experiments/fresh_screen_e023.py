"""Development screening of disjoint initial-text components; reserve outcomes stay unscored."""
import argparse,hashlib,itertools,json,re
from collections import Counter,defaultdict
from fractions import Fraction as F
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/fresh_screen_e023'
Q=['issue','product','money_gone','compensation','customer_name'];CH=['published','complete','repair_partial','repair_unknown','repair_both','remove_extra']
D={'issue':['unrecognized_charge','lost_card','fee_question','change_request','failed_payment'],'product':['card','current_account','loan','savings'],'money_gone':['yes','no'],'compensation':['yes','no']}
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def salted(s):return hashlib.sha256(s.encode()).hexdigest()
def norm(s):return re.sub(r'\s+',' ',s.casefold()).strip()
def case(r):return '/'.join(r['id'].split('/')[:3])
def facts(r):return {k:[v for v in dom if v in r['evidence'].get(k,dom)] for k,dom in D.items()}
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def ex(v):return {'fraction':str(v),'value':float(v)}
def prepare():
 assert not OUT.exists()
 cache=Path(read(ROOT/'results/runtime_restoration_2026-11-08/data_manifest.json')['cache']);src=cache/'data/train/banking_support.jsonl';assert sha(src)=='6641c847f59d539fc99de6dae24f92babeb0358c68699514528e23f3c8c71365'
 rs=[json.loads(l) for l in src.read_text().splitlines()];bycase=defaultdict(list)
 for r in rs:
  if r['kind']=='a':bycase[case(r)].append(r)
 texts=defaultdict(list)
 for c,rows in bycase.items():
  ts={norm(r['state'][0]['text']) for r in rows};assert len(ts)==1;texts[ts.pop()].append(c)
 used=set(read(ROOT/'results/lavoir_executor_e014/selection.json')['cases'])
 for g in read(ROOT/'results/heldout_value_e016/components.json'):used.update(g['cases'])
 excluded=[];eligible=[];oldpatterns=set()
 for t,cs in texts.items():
  cs=sorted(cs);ids=sorted(r['id'] for c in cs for r in bycase[c]);patterns=sorted({digest(facts(r)) for c in cs for r in bycase[c]});gid=salted('E016-v1:'+'\n'.join(cs))
  item={'group_id':gid,'cases':cs,'source_ids':ids,'normalized_text_sha256':salted(t),'evidence_fingerprints':patterns,'evidence_conflict':len(patterns)>1}
  if set(cs)&used:excluded.append(item);oldpatterns.update(patterns)
  else:eligible.append(item)
 eligible.sort(key=lambda g:salted('E023-screen-v1:'+g['group_id']));assert len(eligible)>=96
 groups=eligible[:96];reserve=eligible[96:];selected=[r for g in groups for c in g['cases'] for r in bycase[c]];byid={r['id']:r for r in selected}
 for g in groups:g['rank_score']=salted('E023-screen-v1:'+g['group_id'])
 initial=[];refs=[]
 for g in sorted(groups,key=lambda g:salted('E023-initial-review-v1:'+g['group_id']))[:24]:
  sid=min(g['source_ids'],key=lambda x:salted('E023-initial-profile-v1:'+x));r=byid[sid];msgs=[{'role':'user','content':r['state'][0]['text']}];uid='I-'+digest(msgs)[:16]
  initial.append({'unit_id':uid,'messages':msgs});refs.append({'unit_id':uid,'type':'initial','source_ids':[sid],'group_id':g['group_id'],'facts':facts(r)})
 candidates={};gidmap={sid:g['group_id'] for g in groups for sid in g['source_ids']}
 for r in selected:
  for q in Q:
   p=r['probes'][q];msgs=[{'role':'user','content':r['state'][0]['text']},{'role':'assistant','content':p['q']},{'role':'user','content':p['a']}];h=digest(msgs);bf=facts(r);af={k:[v for v in bf[k] if v in p['reveal'].get(k,dom)] for k,dom in D.items()}
   item={'unit_id':'R-'+h[:16],'type':'reply','kind':p['kind'],'question':q,'source_ids':[r['id']],'group_id':gidmap[r['id']],'facts':af,'messages':msgs}
   if h in candidates:
    old=candidates[h];assert all(old[k]==item[k] for k in ['facts','kind','question']);old['source_ids'].append(r['id'])
   else:candidates[h]=item
 replies=[]
 for kind in ['clean','partial','unknown','extra']:
  pool=[v for v in candidates.values() if v['kind']==kind];assert len(pool)>=3
  pool.sort(key=lambda v:salted('E023-reply-review-v1:'+digest(v['messages'])))
  for v in pool[:3]:replies.append({'unit_id':v['unit_id'],'messages':v['messages']});refs.append({k:x for k,x in v.items() if k!='messages'})
 blind=initial+sorted(replies,key=lambda v:v['unit_id']);assert len(blind)==len({v['unit_id'] for v in blind})==36
 OUT.mkdir();save(OUT/'groups.json',groups);save(OUT/'reserved_unscored.json',reserve);save(OUT/'excluded.json',excluded);save(OUT/'source_rows.json',selected);save(OUT/'schema.json',read(cache/'schemas/banking_support.json'));save(OUT/'blind_audit.json',blind);save(OUT/'audit_references.json',refs)
 inventory={'initial_cases':len(bycase),'initial_text_components':len(texts),'excluded_components':len(excluded),'eligible_components':len(eligible),'screen_components':len(groups),'screen_cases':sum(len(g['cases']) for g in groups),'screen_profiles':len(selected),'reserved_unscored_components':len(reserve),'screen_evidence_patterns':len(set(p for g in groups for p in g['evidence_fingerprints'])),'screen_components_sharing_prior_evidence_pattern':sum(any(p in oldpatterns for p in g['evidence_fingerprints']) for g in groups),'screen_evidence_conflicts':sum(g['evidence_conflict'] for g in groups),'blind_initial_units':24,'blind_reply_units':12,'reserve_outcomes_computed':False};save(OUT/'inventory.json',inventory)
 paths=[Path(__file__).resolve(),ROOT/'research/E023_PROTOCOL.md',ROOT/'results/lavoir_executor_e014/selection.json',ROOT/'results/heldout_value_e016/components.json',ROOT/'results/runtime_restoration_2026-11-08/data_manifest.json']+[OUT/n for n in ['groups.json','reserved_unscored.json','excluded.json','source_rows.json','schema.json','blind_audit.json','audit_references.json','inventory.json']]
 save(OUT/'manifest.json',{'files':{str(p.relative_to(ROOT)):sha(p) for p in paths},'source_path':str(src),'source_sha256':sha(src),'screen_groups':96,'gate':{'strict_components_min':8,'mean_strict_regret_min':'1/100'},'official_test_used':False,'model_calls':0})
 print(json.dumps(inventory))
def run():
 m=read(OUT/'manifest.json')
 for n,h in m['files'].items():assert sha(ROOT/n)==h,n
 assert (OUT/'audit_freeze.json').exists(),'Freeze blind interpretations before opening source comparisons or screen outcomes'
 for n,h in read(OUT/'audit_freeze.json')['files'].items():assert sha(ROOT/n)==h,n
 assert not (OUT/'summary.json').exists()
 schema=read(OUT/'schema.json');teams=sorted(schema['units']);worlds=[]
 def route(w):
  for r in schema['rules']:
   if 'else'in r:return r['else']
   if all(w[k]in v for k,v in r['if'].items()):return r['unit']
  raise AssertionError('unrouted')
 for vals in itertools.product(*D.values()):
  w=dict(zip(D,vals));p=F(1)
  for k,s in schema['slots'].items():
   if k not in D:continue
   d=s['values'] if 'values'in s else s['cond'][w[s['depends_on']]];p*=F(str(d.get(w[k],0)))
  if p:worlds.append((w,p,route(w)))
 assert len(worlds)==44 and sum(p for _,p,_ in worlds)==1
 cache={}
 def act(f):
  key=digest(f)
  if key not in cache:
   mass={t:sum(p for w,p,u in worlds if u==t and all(w[k]in f[k] for k in D)) for t in teams};assert sum(mass.values())>0;cache[key]=max(teams,key=mass.get)
  return cache[key]
 rows=read(OUT/'source_rows.json');bycase=defaultdict(list)
 for r in rows:bycase[case(r)].append(r)
 groups=read(OUT/'groups.json');details=[];components=[];kinds=defaultdict(Counter)
 for g in groups:
  values={ch:{q:F(0) for q in Q} for ch in CH};noask=F(0)
  for cid in g['cases']:
   for r in bycase[cid]:
    weight=F(1,len(g['cases'])*len(bycase[cid]));bf=facts(r);assert all(r['profile'][k]in bf[k] for k in D);assert route(r['profile'])==r['gold'];a0=act(bf);c0=int(a0==r['gold']);noask+=weight*c0
    details.append(dict(group_id=g['group_id'],case_id=cid,source_id=r['id'],question='no_ask',weight=str(weight),facts=bf,action=a0,gold=r['gold'],correct=c0))
    for q in Q:
     p=r['probes'][q];af={k:[v for v in bf[k] if v in p['reveal'].get(k,dom)] for k,dom in D.items()};full={k:([r['profile'][k]] if k==q else bf[k]) for k in D};reduced={k:(af[k] if k==q else bf[k]) for k in D};assert all(r['profile'][k]in af[k] for k in D)
     cf={'published':af,'complete':full,'repair_partial':full if p['kind']=='partial' else af,'repair_unknown':full if p['kind']=='unknown' else af,'repair_both':full if p['kind']in ['partial','unknown'] else af,'remove_extra':reduced if p['kind']=='extra' else af};actions={ch:act(f) for ch,f in cf.items()};correct={ch:int(a==r['gold']) for ch,a in actions.items()}
     for ch in CH:values[ch][q]+=weight*correct[ch]
     if p['kind']=='unknown':assert bf==af and actions['published']==a0
     if bf==af:assert actions['published']==a0
     k=kinds[p['kind']];k['probes']+=1;k['facts_changed']+=bf!=af;k['action_changed']+=actions['published']!=a0;k['improved']+=correct['published']>c0;k['worsened']+=correct['published']<c0
     details.append(dict(group_id=g['group_id'],case_id=cid,source_id=r['id'],question=q,kind=p['kind'],weight=str(weight),facts_by_channel=cf,actions=actions,gold=r['gold'],correct=correct,correct_before=c0,facts_changed=bf!=af))
  selected={ch:max(Q,key=values[ch].get) for ch in CH};opt={ch:[q for q in Q if values[ch][q]==max(values[ch].values())] for ch in CH};actual=values['published'];best=max(actual.values());comparisons={}
  for ch in CH:
   lo=min(actual[q] for q in opt[ch]);hi=max(actual[q] for q in opt[ch]);comparisons[ch]={'selected_accuracy':str(actual[selected[ch]]),'optimal_tie_range':[str(lo),str(hi)],'strict_regret':str(best-hi),'fixed_order_regret':str(best-actual[selected[ch]])}
  components.append(dict(group_id=g['group_id'],cases=g['cases'],profiles=len(g['source_ids']),no_ask=str(noask),values={ch:{q:str(v) for q,v in vs.items()} for ch,vs in values.items()},selected=selected,optimal_sets=opt,comparisons=comparisons,question_values_differ=len(set(actual.values()))>1))
 def mean(xs):return sum(xs,F(0))/len(groups)
 constants={q:mean(F(c['values']['published'][q]) for c in components) for q in Q};fixed=max(Q,key=constants.get);aware=mean(F(c['comparisons']['published']['selected_accuracy']) for c in components);contrasts={}
 for ch in CH:
  contrasted=[c['comparisons'][ch] for c in components];sr=mean(F(x['strict_regret']) for x in contrasted)
  contrasts[ch]={'published_accuracy':ex(mean(F(x['selected_accuracy']) for x in contrasted)),'optimal_tie_range':[ex(mean(F(x['optimal_tie_range'][i]) for x in contrasted)) for i in [0,1]],'mean_strict_regret':ex(sr),'mean_fixed_order_regret':ex(mean(F(x['fixed_order_regret']) for x in contrasted)),'strict_components':sum(F(x['strict_regret'])>0 for x in contrasted),'fixed_order_improved_components':sum(F(x['fixed_order_regret'])>0 for x in contrasted),'changed_choice_components':sum(c['selected'][ch]!=c['selected']['published'] for c in components)}
 primary=contrasts['complete'];passed=primary['strict_components']>=8 and F(primary['mean_strict_regret']['fraction'])>=F(1,100)
 summary={'groups':96,'cases':sum(len(g['cases']) for g in groups),'profiles':len(rows),'branch_rows':len(details),'channels':CH,'no_ask':ex(mean(F(c['no_ask']) for c in components)),'constant_questions':{q:ex(v) for q,v in constants.items()},'best_constant':{'question':fixed,'accuracy':ex(constants[fixed])},'published_aware_oracle':ex(aware),'aware_minus_fixed':ex(aware-constants[fixed]),'groups_question_values_differ':sum(c['question_values_differ'] for c in components),'contrasts':contrasts,'probe_kinds_unweighted':{k:dict(v) for k,v in kinds.items()},'resource_gate_passed':passed,'reserve_outcomes_computed':False,'trained_method':False}
 save(OUT/'details.json',details);save(OUT/'components.json',components);save(OUT/'summary.json',summary)
 paths=[OUT/n for n in ['manifest.json','audit_freeze.json','details.json','components.json','summary.json']];save(OUT/'score_manifest.json',{'files':{str(p.relative_to(ROOT)):sha(p) for p in paths}});print(json.dumps(summary))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);globals()[p.parse_args().stage]()
