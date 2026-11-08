"""Independent source/prompt joins, rational weights and utility verification."""
from collections import Counter,defaultdict
from fractions import Fraction as F
import hashlib,itertools,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/lavoir_executor_e014'
Q=['no_ask','issue','product','money_gone','compensation','customer_name']
TEAMS=['cards','fraud','loans','accounts','complaints']
def read(n):return json.loads((OUT/n).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(o):return hashlib.sha256(json.dumps(o,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def close(a,b):assert abs(float(a)-float(b))<1e-12,(a,b)
def parse(text):
    s=text.strip()
    if s.startswith('```'):
        lines=s.splitlines()
        if lines[0]not in ['```','```json'] or lines[-1]!='```':return None
        s='\n'.join(lines[1:-1]).strip()
    if not s.startswith('{'):return None
    try:pairs=json.loads(s,object_pairs_hook=lambda p:p)
    except ValueError:return None
    if type(pairs)is not list or len(pairs)!=1 or pairs[0][0]!='team':return None
    return pairs[0][1] if pairs[0][1] in TEAMS else None

def main():
    for p,h in read('manifest.json')['files'].items():assert sha(ROOT/p)==h,p
    source=Path('/private/tmp/lavoir-data-20261027/data/train/banking_support.jsonl')
    assert sha(source)==read('selection.json')['source_sha256']
    allrows=[json.loads(l) for l in source.read_text().splitlines()]
    case=lambda r:'/'.join(r['id'].split('/')[:3])
    cases=sorted({case(r) for r in allrows},key=lambda c:hashlib.sha256(('E014-v1:'+c).encode()).hexdigest())[:24]
    rows=read('source_rows.json');assert rows==[r for c in cases for r in allrows if case(r)==c and r['kind']=='a']
    src={r['id']:r for r in rows};counts=Counter(case(r) for r in rows)
    packs={r['id']:r for r in read('pack.json')};assert len(packs)==len(read('pack.json'))
    branches=read('branches.json');seen=set();weights={}
    schema=json.loads(Path('/private/tmp/lavoir-data-20261027/schemas/banking_support.json').read_text())
    attrs=['issue','product','money_gone','compensation'];values=[];worlds=[]
    for k in attrs:
        s=schema['slots'][k]
        values.append(sorted(s['values']) if 'values' in s else sorted(set().union(*(d.keys() for d in s['cond'].values()))))
    for combination in itertools.product(*values):
        w=dict(zip(attrs,combination));p=F(1)
        for k in attrs:
            s=schema['slots'][k];probs=s['values'] if 'values' in s else s['cond'][w[s['depends_on']]]
            p*=F(str(probs.get(w[k],0)))
        if not p:continue
        if w['issue'] in ['unrecognized_charge','lost_card']:g='fraud' if w['money_gone']=='yes' else 'cards'
        elif w['issue']=='fee_question' and w['compensation']=='yes':g='complaints'
        elif w['product']=='loan':g='loans'
        elif w['product']=='card':g='cards'
        else:g='accounts'
        worlds.append((w,p,g))
    assert len(worlds)==44 and sum(p for w,p,g in worlds)==1
    review=read('semantic_review.json');excluded=set(review['sensitivity_excluded_case_ids'])
    assert sha(OUT/'review_units.json')==review['review_units_sha256']
    assert len(review['annotations'])==187 and Counter(a['label'] for a in review['annotations'])=={'supported':182,'unclear':5}
    system=None
    for b in branches:
        r=src[b['source_id']];q=b['question'];key=(r['id'],q);assert key not in seen;seen.add(key)
        assert b['gold']==r['gold'] and b['case_id']==case(r)
        w=F(1,24*counts[case(r)]);close(w,b['weight']);weights[r['id']]=w
        p=packs[b['prompt_id']];m=p['messages'];assert digest(m)==p['id']
        if system is None:system=m[0]
        assert m[0]==system
        assert m[1]=={'role':'user','content':r['state'][0]['text']}
        assert len(m)==(2 if q=='no_ask' else 4)
        if q!='no_ask':
            pr=r['probes'][q]
            assert m[2]=={'role':'assistant','content':pr['q']}
            assert m[3]=={'role':'user','content':pr['a']}
        assert b['visible_group']==digest(m[:2]);assert b['sensitivity_excluded']==(case(r) in excluded)
        facts={k:set(v) for k,v in r['evidence'].items()}
        if q!='no_ask':
            for k,vs in r['probes'][q]['reveal'].items():facts[k]=facts[k]&set(vs) if k in facts else set(vs)
        ps={t:F(0) for t in TEAMS}
        for w,p,g in worlds:
            if all(k not in attrs or w[k] in vs for k,vs in facts.items()):ps[g]+=p
        z=sum(ps.values());assert z>0
        for g in TEAMS:close(ps[g]/z,b['semantic_posterior'][g])
        chosen=max(TEAMS,key=ps.get)
        assert chosen==b['semantic_choice'] and int(chosen==r['gold'])==b['semantic_correct']
    assert len(branches)==390 and len(seen)==65*6 and sum(weights.values())==1
    assert set(packs)=={b['prompt_id'] for b in branches}
    assert all(t not in system['content'] for t in ['oracle_voi','profile','gold','reveal','evidence'])
    assert all(t in system['content'] for t in ['1.','2.','3.','4.','5.','6.'])
    result={'verified':True,'phase':'inputs verified','generation_started':(OUT/'generations.jsonl').exists(),'cases':24,'profiles':65,'branches':390,'unique_prompts':len(packs),
            'exact_weights_sum':str(sum(weights.values())),'source_prompt_joins':len(branches),'unclear_case_count':len(excluded),
            'rational_semantic_posterior_values':len(branches)*len(TEAMS)}
    if (OUT/'summary.json').exists():
        summary=read('summary.json');gens=[json.loads(l) for l in (OUT/'generations.jsonl').read_text().splitlines()]
        byid={g['id']:g for g in gens};assert set(byid)==set(packs) and len(gens)==len(byid)
        assert all(g['error']is None for g in gens)
        scored=[];details={(d['source_id'],d['question']):d for d in read('details.json')}
        for b in branches:
            team=parse(byid[b['prompt_id']]['raw_output']);success=int(team==b['gold'])
            d=details[b['source_id'],b['question']];assert d['team']==team and d['correct']==success
            scored.append({**b,'success':success,'w':weights[b['source_id']]})
        for scope,subset in [('all_cases',scored),('predeclared_sensitivity',[b for b in scored if b['case_id']not in excluded])]:
            z=sum(b['w'] for b in subset if b['question']=='no_ask');old=summary[scope]
            for q in Q:
                a=sum(b['w']*b['success'] for b in subset if b['question']==q)/z
                close(a,old['fixed'][q]['accuracy']);close(a-F(1,20)*(q!='no_ask'),old['fixed'][q]['utility'])
            for field,key in [('source_id','hidden_profile_oracle'),('visible_group','visible_input_oracle')]:
                grouped=defaultdict(list)
                for b in subset:grouped[b[field]].append(b)
                totals={q:F(0) for q in ['accuracy','utility','ask_rate']};choices={}
                for gid,group in grouped.items():
                    values={q:sum(b['w']*(b['success']-F(1,20)*(q!='no_ask')) for b in group if b['question']==q) for q in Q}
                    chosen=max(Q,key=values.get);choices[gid]=chosen
                    selected=[b for b in group if b['question']==chosen]
                    totals['accuracy']+=sum(b['w']*b['success'] for b in selected)/z
                    totals['utility']+=values[chosen]/z
                    totals['ask_rate']+=sum(b['w'] for b in selected)/z*(chosen!='no_ask')
                for k,v in totals.items():close(v,old['oracles'][key][k])
                assert choices=={c['group']:c['question'] for c in old['oracles'][key]['choices']}
            assert old['oracles']['hidden_profile_oracle']['utility']>=old['oracles']['visible_input_oracle']['utility']-1e-12
            assert old['oracles']['visible_input_oracle']['utility']>=old['best_fixed_including_noask']['utility']-1e-12
        for p,h in read('analysis_manifest.json')['files'].items():assert sha(ROOT/p)==h
        result.update(phase='completed',generated=len(gens),rational_metric_checks=True)
    result['verifier_sha256']=sha(Path(__file__).resolve())
    (OUT/('verification.json' if result['phase']=='completed' else 'pre_verification.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
