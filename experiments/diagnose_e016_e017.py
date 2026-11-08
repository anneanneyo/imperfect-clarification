"""Descriptive paired failure analysis; cannot change primary evaluation."""
from collections import Counter,defaultdict
from fractions import Fraction as F
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'results/heldout_value_e016';ENC=ROOT/'results/encoder_value_e017';OUT=ROOT/'results/e016_e017_diagnostics'
Q=['no_ask','issue','product','money_gone','compensation','customer_name']
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def worlds(schema):
    ws=[({},F(1))]
    for slot,definition in schema['slots'].items():
        if definition.get('decisive')is False:continue
        nxt=[]
        for w,p in ws:
            ps=definition['cond'][w[definition['depends_on']]] if 'depends_on'in definition else definition['values']
            for v,mass in ps.items():
                if mass:nxt.append(({**w,slot:v},p*F(str(mass))))
        ws=nxt
    def route(w):
        for rule in schema['rules']:
            if 'else'in rule:return rule['else']
            if all(w[k] in vals for k,vals in rule['if'].items()):return rule['unit']
        raise AssertionError()
    return [(w,p,route(w)) for w,p in ws]
def posterior(ws,evidence,teams):
    p=dict.fromkeys(teams,F(0))
    for world,mass,gold in ws:
        if all(k not in world or world[k]in vals for k,vals in evidence.items()):p[gold]+=mass
    z=sum(p.values());assert z>0
    return tuple(p[k]/z for k in teams)
def main():
    assert read(SOURCE/'verification.json')['verified'] and read(ENC/'verification.json')['verified']
    for folder in [SOURCE,ENC]:
        for p,h in read(folder/'score_manifest.json')['files'].items():assert sha(ROOT/p)==h
    OUT.mkdir(exist_ok=True);schema=read(SOURCE/'schema.json');ws=worlds(schema);teams=list(schema['units'])
    split_audit=ROOT/'results/e016_split_structure'
    for p,h in read(split_audit/'manifest.json')['files'].items():assert sha(ROOT/p)==h
    seen_pattern={g['group_id']:g['pattern_present_in_train'] for g in read(split_audit/'groups.json')}
    source={r['id']:r for r in read(SOURCE/'source_rows.json')};components={g['group_id']:g for g in read(SOURCE/'components.json') if g['split']=='test'}
    casecounts=Counter('/'.join(r['id'].split('/')[:3]) for r in source.values())
    ds=read(SOURCE/'test_details.json');bykey={(d['source_id'],d['question']):d for d in ds};assert len(bykey)==len(ds)
    detail=[]
    for d in ds:
        r=source[d['source_id']];q=d['question'];g=components[d['group_id']];w=F(1,len(components)*len(g['cases'])*casecounts[d['case_id']])
        before=bykey[d['source_id'],'no_ask'];ev={k:set(v) for k,v in r['evidence'].items()};p0=posterior(ws,ev,teams)
        if q!='no_ask':
            for slot,vals in r['probes'][q]['reveal'].items():
                assert r['profile'][slot]in vals;ev[slot]=ev[slot]&set(vals) if slot in ev else set(vals);assert ev[slot]
        p1=posterior(ws,ev,teams)
        if d['kind']=='unknown':assert p0==p1
        detail.append({'group_id':d['group_id'],'case_id':d['case_id'],'source_id':d['source_id'],'question':q,'kind':d['kind'],
            'weight_numerator':w.numerator,'weight_denominator':w.denominator,'base_correct':before['correct'],'correct':d['correct'],
            'gain':d['correct']-before['correct'],'action_changed':d['team']!=before['team'],
            'symbolic_route_posterior_changed':p0!=p1,'initial_symbolic_route_certain':max(p0)==1,
            'evidence_pattern_seen_in_train':seen_pattern[d['group_id']],
            'queried_slot_explicitly_known':q in r['evidence'] and len(r['evidence'][q])==1,
            'base_team':before['team'],'team':d['team'],'gold':d['gold']})
    lookup={(d['group_id'],d['source_id'],d['question']):d for d in detail}
    cr=read(SOURCE/'component_results.json');policies={m:{r['group_id']:r['metrics'][m]['question'] for r in cr} for m in ['ridge','train_constant','no_ask','issue','product','money_gone','compensation','customer_name','component_posthoc_oracle']}
    policies['encoder']={p['group_id']:p['question'] for p in read(ENC/'predictions.json')}
    summaries={}
    def aggregate(rows):
        total=sum((F(d['weight_numerator'],d['weight_denominator']) for d in rows),F(0))
        accuracy=sum((F(d['weight_numerator'],d['weight_denominator'])*d['correct'] for d in rows),F(0))
        gain=sum((F(d['weight_numerator'],d['weight_denominator'])*d['gain'] for d in rows),F(0))
        improved=sum((F(d['weight_numerator'],d['weight_denominator'])*(d['gain']>0) for d in rows),F(0))
        worsened=sum((F(d['weight_numerator'],d['weight_denominator'])*(d['gain']<0) for d in rows),F(0))
        assert gain==improved-worsened
        return {'profile_branch_records':len(rows),'components':len({d['group_id'] for d in rows}),'mass':float(total),
            'accuracy_contribution':float(accuracy),'conditional_accuracy':float(accuracy/total) if total else None,
            'gain_contribution_vs_matched_noask':float(gain),'conditional_gain':float(gain/total) if total else None,
            'improved_mass':float(improved),'worsened_mass':float(worsened)}
    for method,choices in policies.items():
        chosen=[d for d in detail if d['question']==choices[d['group_id']]];overall=aggregate(chosen);assert abs(overall['mass']-1)<1e-12
        bykind={kind:aggregate([d for d in chosen if d['kind']==kind]) for kind in sorted({d['kind'] for d in chosen})}
        bychange={str(ch):aggregate([d for d in chosen if d['symbolic_route_posterior_changed']==ch]) for ch in [False,True]}
        bycertainty={str(ch):aggregate([d for d in chosen if d['initial_symbolic_route_certain']==ch]) for ch in [False,True]}
        bypattern={str(ch):aggregate([d for d in chosen if d['evidence_pattern_seen_in_train']==ch]) for ch in [False,True]}
        summaries[method]={'overall':overall,'response_kind':bykind,'route_posterior_changed':bychange,'initial_route_certain':bycertainty,'pattern_seen_in_train':bypattern}
        primary=read(ENC/'summary.json')['accuracy'] if method=='encoder' else read(SOURCE/'summary.json')['methods'][method]['accuracy']
        assert abs(overall['accuracy_contribution']-primary)<1e-12
    baseline=summaries['no_ask']['overall']['accuracy_contribution']
    for method,s in summaries.items():
        assert abs(s['overall']['gain_contribution_vs_matched_noask']-(s['overall']['accuracy_contribution']-baseline))<1e-12
        for dimension in ['response_kind','route_posterior_changed','initial_route_certain','pattern_seen_in_train']:
            assert abs(sum(v['gain_contribution_vs_matched_noask'] for v in s[dimension].values())-s['overall']['gain_contribution_vs_matched_noask'])<1e-12
    allkinds=Counter(d['kind'] for d in detail if d['question']!='no_ask')
    summary={'interpretation':'Post-outcome descriptive decomposition. Source symbolic prior; not causal effects of answer kind. No primary exclusions or tuning.',
        'test_components':len(components),'test_profiles':len({d['source_id'] for d in detail}),'probe_kind_counts':dict(allkinds),
        'policies':summaries,'exact_weight_and_gain_reconciliation':True}
    save(OUT/'details.json',detail);save(OUT/'summary.json',summary)
    inputs=[Path(__file__).resolve(),split_audit/'manifest.json',SOURCE/'source_rows.json',SOURCE/'schema.json',SOURCE/'components.json',SOURCE/'test_details.json',SOURCE/'component_results.json',SOURCE/'summary.json',ENC/'predictions.json',ENC/'summary.json',OUT/'details.json',OUT/'summary.json']
    save(OUT/'manifest.json',{'files':{str(p.relative_to(ROOT)):sha(p) for p in inputs}})
    print(json.dumps({'probe_kinds':allkinds,'learned':{m:summaries[m] for m in ['ridge','encoder']}}))
if __name__=='__main__':main()
