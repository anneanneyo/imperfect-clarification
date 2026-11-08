"""Independent E023 selection, integer-probability replay, channel and audit verification."""
import hashlib,itertools,json,re
from collections import Counter,defaultdict
from decimal import Decimal
from fractions import Fraction as F
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/fresh_screen_e023'
Q=['issue','product','money_gone','compensation','customer_name'];CH=['published','complete','repair_partial','repair_unknown','repair_both','remove_extra']
D={'issue':['unrecognized_charge','lost_card','fee_question','change_request','failed_payment'],'product':['card','current_account','loan','savings'],'money_gone':['yes','no'],'compensation':['yes','no']}
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def h(s):return hashlib.sha256(s.encode()).hexdigest()
def digest(x):return h(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')))
def fact(r):return {k:[v for v in dom if v in r['evidence'].get(k,dom)] for k,dom in D.items()}
for mn in ['manifest.json','audit_freeze.json','score_manifest.json','audit_score_manifest.json']:
 for n,s in read(OUT/mn)['files'].items():assert sha(ROOT/n)==s,n
m=read(OUT/'manifest.json');assert sha(Path(m['source_path']))==m['source_sha256']
allrows=[json.loads(l) for l in Path(m['source_path']).read_text().splitlines()];bycase=defaultdict(list)
for r in allrows:
 if r['kind']=='a':bycase['/'.join(r['id'].split('/')[:3])].append(r)
bytext=defaultdict(list)
for cid,rs in bycase.items():
 ts={' '.join(r['state'][0]['text'].casefold().split()) for r in rs};assert len(ts)==1;bytext[ts.pop()].append(cid)
used=set(read(ROOT/'results/lavoir_executor_e014/selection.json')['cases'])
for g in read(ROOT/'results/heldout_value_e016/components.json'):used.update(g['cases'])
excluded=[];eligible=[];priorpatterns=set()
for text,ids in bytext.items():
 ids=sorted(ids);fps=sorted({digest(fact(r)) for cid in ids for r in bycase[cid]});g=dict(group_id=h('E016-v1:'+'\n'.join(ids)),cases=ids,source_ids=sorted(r['id'] for cid in ids for r in bycase[cid]),normalized_text_sha256=h(text),evidence_fingerprints=fps,evidence_conflict=len(fps)>1)
 if set(ids)&used:excluded.append(g);priorpatterns.update(fps)
 else:eligible.append(g)
eligible.sort(key=lambda g:h('E023-screen-v1:'+g['group_id']));chosen=eligible[:96]
for g in chosen:g['rank_score']=h('E023-screen-v1:'+g['group_id'])
assert read(OUT/'groups.json')==chosen and read(OUT/'reserved_unscored.json')==eligible[96:] and read(OUT/'excluded.json')==excluded
rs=read(OUT/'source_rows.json');expected_rows=[r for g in chosen for cid in g['cases'] for r in bycase[cid]];assert rs==expected_rows;src={r['id']:r for r in rs};gid={sid:g['group_id'] for g in chosen for sid in g['source_ids']}
inventory=dict(initial_cases=len(bycase),initial_text_components=len(bytext),excluded_components=len(excluded),eligible_components=len(eligible),screen_components=96,screen_cases=sum(len(g['cases']) for g in chosen),screen_profiles=len(rs),reserved_unscored_components=len(eligible)-96,screen_evidence_patterns=len({p for g in chosen for p in g['evidence_fingerprints']}),screen_components_sharing_prior_evidence_pattern=sum(bool(set(g['evidence_fingerprints'])&priorpatterns) for g in chosen),screen_evidence_conflicts=sum(g['evidence_conflict'] for g in chosen),blind_initial_units=24,blind_reply_units=12,reserve_outcomes_computed=False);assert read(OUT/'inventory.json')==inventory
schema=read(OUT/'schema.json');slots=schema['slots'];teams=sorted(schema['units']);worlds=[]
def route(w):
 if w['issue']in ['lost_card','unrecognized_charge']:return 'fraud' if w['money_gone']=='yes' else 'cards'
 if w['issue']=='fee_question' and w['compensation']=='yes':return 'complaints'
 return 'loans' if w['product']=='loan' else 'cards' if w['product']=='card' else 'accounts'
for issue,p0 in slots['issue']['values'].items():
 for rest in itertools.product(*(slots[k]['cond'][issue].items() for k in list(D)[1:])):
  w={'issue':issue};p=Decimal(str(p0))
  for k,(v,p1) in zip(list(D)[1:],rest):w[k]=v;p*=Decimal(str(p1))
  scaled=p*100000000;assert scaled==int(scaled);worlds.append((w,int(scaled),route(w)))
assert len(worlds)==44 and sum(p for _,p,_ in worlds)==100000000
def act(f):
 ms={t:sum(p for w,p,u in worlds if u==t and all(w[k]in f[k] for k in D)) for t in teams};assert sum(ms.values())>0
 return sorted(teams,key=lambda t:(-ms[t],t))[0]
ds=read(OUT/'details.json');index={(d['source_id'],d['question']):d for d in ds};assert len(index)==len(ds)==len(rs)*6 and {d['source_id'] for d in ds}==set(src)
values={g['group_id']:{ch:{q:F(0) for q in Q} for ch in CH} for g in chosen};na={g['group_id']:F(0) for g in chosen};kinds=defaultdict(Counter)
for g in chosen:
 for cid in g['cases']:
  for r in bycase[cid]:
   weight=F(1,len(g['cases'])*len(bycase[cid]));before=fact(r);a0=act(before);correct0=int(a0==r['gold']);assert route(r['profile'])==r['gold'];na[g['group_id']]+=weight*correct0
   d=index[r['id'],'no_ask'];assert d==dict(group_id=g['group_id'],case_id=cid,source_id=r['id'],question='no_ask',weight=str(weight),facts=before,action=a0,gold=r['gold'],correct=correct0)
   for q in Q:
    p=r['probes'][q];after={k:[v for v in before[k] if v in p['reveal'].get(k,D[k])] for k in D};full={k:([r['profile'][k]] if k==q else before[k]) for k in D};minimal={k:after[k] if k==q else before[k] for k in D};cf={'published':after,'complete':full}
    for ch,kset in [('repair_partial',{'partial'}),('repair_unknown',{'unknown'}),('repair_both',{'partial','unknown'})]:cf[ch]=full if p['kind']in kset else after
    cf['remove_extra']=minimal if p['kind']=='extra' else after;actions={ch:act(f) for ch,f in cf.items()};correct={ch:int(actions[ch]==r['gold']) for ch in CH}
    assert all(r['profile'][k]in before[k] and r['profile'][k]in after[k] for k in D)
    for ch in CH:values[g['group_id']][ch][q]+=weight*correct[ch]
    d=index[r['id'],q];assert d==dict(group_id=g['group_id'],case_id=cid,source_id=r['id'],question=q,kind=p['kind'],weight=str(weight),facts_by_channel=cf,actions=actions,gold=r['gold'],correct=correct,correct_before=correct0,facts_changed=before!=after)
    if p['kind']=='unknown':assert before==after and actions['published']==a0
    k=kinds[p['kind']];k.update({'probes':1,'facts_changed':int(before!=after),'action_changed':int(actions['published']!=a0),'improved':int(correct['published']>correct0),'worsened':int(correct['published']<correct0)})
components=[]
for g in chosen:
 vs=values[g['group_id']];selected={ch:max(Q,key=vs[ch].get) for ch in CH};opt={ch:[q for q in Q if vs[ch][q]==max(vs[ch].values())] for ch in CH};actual=vs['published'];comparison={}
 for ch in CH:
  low=min(actual[q] for q in opt[ch]);high=max(actual[q] for q in opt[ch]);comparison[ch]=dict(selected_accuracy=str(actual[selected[ch]]),optimal_tie_range=[str(low),str(high)],strict_regret=str(max(actual.values())-high),fixed_order_regret=str(max(actual.values())-actual[selected[ch]]))
 components.append(dict(group_id=g['group_id'],cases=g['cases'],profiles=len(g['source_ids']),no_ask=str(na[g['group_id']]),values={ch:{q:str(v) for q,v in a.items()} for ch,a in vs.items()},selected=selected,optimal_sets=opt,comparisons=comparison,question_values_differ=len(set(actual.values()))>1))
assert components==read(OUT/'components.json')
def avg(vs):return sum(vs,F(0))/96
def ex(x):return dict(fraction=str(x),value=float(x))
constants={q:avg(v['published'][q] for v in values.values()) for q in Q};fixed=max(Q,key=constants.get);aware=avg(max(v['published'].values()) for v in values.values());contrasts={}
for ch in CH:
 cr=[c['comparisons'][ch] for c in components];contrasts[ch]=dict(published_accuracy=ex(avg(F(c['selected_accuracy']) for c in cr)),optimal_tie_range=[ex(avg(F(c['optimal_tie_range'][i]) for c in cr)) for i in [0,1]],mean_strict_regret=ex(avg(F(c['strict_regret']) for c in cr)),mean_fixed_order_regret=ex(avg(F(c['fixed_order_regret']) for c in cr)),strict_components=sum(F(c['strict_regret'])>0 for c in cr),fixed_order_improved_components=sum(F(c['fixed_order_regret'])>0 for c in cr),changed_choice_components=sum(c['selected'][ch]!=c['selected']['published'] for c in components))
gate=contrasts['complete']['strict_components']>=8 and F(contrasts['complete']['mean_strict_regret']['fraction'])>=F(1,100)
expected=dict(groups=96,cases=sum(len(g['cases']) for g in chosen),profiles=len(rs),branch_rows=len(ds),channels=CH,no_ask=ex(avg(na.values())),constant_questions={q:ex(v) for q,v in constants.items()},best_constant=dict(question=fixed,accuracy=ex(constants[fixed])),published_aware_oracle=ex(aware),aware_minus_fixed=ex(aware-constants[fixed]),groups_question_values_differ=sum(c['question_values_differ'] for c in components),contrasts=contrasts,probe_kinds_unweighted={k:dict(v) for k,v in kinds.items()},resource_gate_passed=gate,reserve_outcomes_computed=False,trained_method=False);assert read(OUT/'summary.json')==expected
# Reconstruct both blind sampling rules and source references separately.
blind=[];refs=[]
for g in sorted(chosen,key=lambda g:h('E023-initial-review-v1:'+g['group_id']))[:24]:
 sid=min(g['source_ids'],key=lambda s:h('E023-initial-profile-v1:'+s));r=src[sid];msgs=[dict(role='user',content=r['state'][0]['text'])];uid='I-'+digest(msgs)[:16];blind.append(dict(unit_id=uid,messages=msgs));refs.append(dict(unit_id=uid,type='initial',source_ids=[sid],group_id=g['group_id'],facts=fact(r)))
pool={}
for r in rs:
 for q in Q:
  p=r['probes'][q];msgs=[dict(role='user',content=r['state'][0]['text']),dict(role='assistant',content=p['q']),dict(role='user',content=p['a'])];key=digest(msgs);after=index[r['id'],q]['facts_by_channel']['published']
  if key in pool:assert pool[key]['facts']==after;pool[key]['source_ids'].append(r['id'])
  else:pool[key]=dict(unit_id='R-'+key[:16],type='reply',kind=p['kind'],question=q,source_ids=[r['id']],group_id=gid[r['id']],facts=after,messages=msgs)
replies=[]
for kind in ['clean','partial','unknown','extra']:
 candidates=sorted([v for v in pool.values() if v['kind']==kind],key=lambda v:h('E023-reply-review-v1:'+digest(v['messages'])))[:3]
 for v in candidates:replies.append(dict(unit_id=v['unit_id'],messages=v['messages']));refs.append({k:x for k,x in v.items() if k!='messages'})
assert read(OUT/'blind_audit.json')==blind+sorted(replies,key=lambda v:v['unit_id']) and read(OUT/'audit_references.json')==refs
labels={r['unit_id']:r for r in read(OUT/'blind_interpretations.json')};rr={r['unit_id']:r for r in refs};inputs={r['unit_id']:r for r in read(OUT/'blind_audit.json')};joined=read(OUT/'audit_comparison.json');assert len(joined)==len(labels)==36
for c in joined:
 l=labels[c['unit_id']];r=rr[c['unit_id']];assert all(c[k]==v for k,v in r.items());assert c['messages']==inputs[c['unit_id']]['messages'] and c['blind_facts']==l['facts'] and c['blind_ambiguous_fields']==l['ambiguous_fields'] and c['blind_rationale']==l['rationale'];assert c['disagreeing_fields']==[k for k in D if set(l['facts'][k])!=set(r['facts'][k])]
a=read(OUT/'audit_summary.json');assert a['groups']==len({c['group_id'] for c in joined})==30
for typ in ['initial','reply']:
 rows0=[c for c in joined if c['type']==typ];fields0=[(c,k) for c in rows0 for k in D];known=[(c,k) for c,k in fields0 if len(c['facts'][k])<len(D[k])];unknown=[(c,k) for c,k in fields0 if len(c['facts'][k])==len(D[k])]
 counts=dict(units=len(rows0),fields=len(fields0),exact_fields=sum(k not in c['disagreeing_fields'] for c,k in fields0),exact_units=sum(not c['disagreeing_fields'] for c in rows0),known_fields=len(known),known_exact=sum(k not in c['disagreeing_fields'] for c,k in known),unknown_fields=len(unknown),unknown_exact=sum(k not in c['disagreeing_fields'] for c,k in unknown),blind_ambiguous_fields=sum(len(c['blind_ambiguous_fields']) for c in rows0));assert a['type_results'][typ]==counts
assert a['reply_kinds']==dict(Counter(c['kind'] for c in joined if c['type']=='reply'))
report=dict(passed=True,selection_reconstructed=True,branch_rows=len(ds),counterfactual_channels=6,integer_world_weights=True,independent_banking_rules=True,blind_units_verified=36,reserved_groups_unscored=217,verifier_sha256=sha(Path(__file__)),manifest_sha256=sha(OUT/'manifest.json'))
(OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
