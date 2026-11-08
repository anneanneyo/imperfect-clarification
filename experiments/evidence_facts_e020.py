"""One fixed evidence-citing extraction control; old experiment files are immutable."""
import argparse,json,sys,subprocess
from pathlib import Path
from fractions import Fraction
import structured_facts_e019 as e19
import local_executor_e013 as base
ROOT=base.ROOT;OUT=ROOT/'results/evidence_facts_e020'
SYSTEM='''Extract banking facts explicitly supported by the user's message. Do not choose a department or ask questions.
Return only one JSON object with exactly four keys: issue, product, money_gone, compensation. Each field must have exactly two keys: "values" (a nonempty array of possible category strings) and "evidence" (one short verbatim substring of the user's message, or null).
Allowed domains:
issue: ["unrecognized_charge","lost_card","fee_question","change_request","failed_payment"]
product: ["card","current_account","loan","savings"]
money_gone: ["yes","no"]
compensation: ["yes","no"]
For a fact that is unstated or unknown, use its ENTIRE domain and evidence:null. Never treat an unmentioned fact as no. Before narrowing a domain, find a direct supporting phrase in the user's message. If there is no supporting phrase, retain the entire domain. Copy evidence exactly, without paraphrasing, up to 160 characters. Preserve all explicitly stated alternatives for a partial or ambiguous fact.
The issue is the type of problem. A failed payment does not establish whether money left the account. A generic mention of an account does not establish a current account. Use savings for an explicitly mentioned savings account. money_gone concerns whether money has already left the account. compensation concerns a request for money back as compensation. Mentioning a fee or a payment does not establish a request for compensation, and absence of a request does not establish refusal. Respect explicit negations in each field separately. Do not infer unstated facts from usual banking correlations.
If no facts are stated, the required object is:
{"issue":{"values":["unrecognized_charge","lost_card","fee_question","change_request","failed_payment"],"evidence":null},"product":{"values":["card","current_account","loan","savings"],"evidence":null},"money_gone":{"values":["yes","no"],"evidence":null},"compensation":{"values":["yes","no"],"evidence":null}}
Return only JSON. No explanations, extra keys or markdown.'''

def read(p):return json.loads(p.read_text())
def save(p,x):base.save(p,x)
def parse(raw):
    def unique(p):
        if len(p)!=len(dict(p)):raise ValueError('duplicate')
        return dict(p)
    try:obj=json.loads(raw,object_pairs_hook=unique)
    except (ValueError,TypeError):return None
    if type(obj)is not dict or set(obj)!=set(e19.DOMAINS):return None
    for k,domain in e19.DOMAINS.items():
        v=obj[k]
        if type(v)is not dict or set(v)!={'values','evidence'}:return None
        a=v['values'];s=v['evidence']
        if type(a)is not list or not a or any(type(t)is not str or t not in domain for t in a) or len(set(a))!=len(a):return None
        if s is not None and type(s)is not str:return None
    return obj
def decode(obj,texts):
    if obj is None:return None,None,{}
    raw={k:[v for v in domain if v in obj[k]['values']] for k,domain in e19.DOMAINS.items()};gated={};checks={}
    for k,domain in e19.DOMAINS.items():
        s=obj[k]['evidence'];valid=type(s)is str and 0<len(s)<=160 and any(s in t for t in texts)
        narrowed=len(raw[k])<len(domain);gated[k]=raw[k] if not narrowed or valid else list(domain)
        checks[k]={'narrowed':narrowed,'quote_valid':valid,'reset_to_unknown':narrowed and not valid}
    return raw,gated,checks
def metrics(ds,field):
    n=len(ds);slot_exact=known_exact=unknown_exact=known=unknown=all_exact=route_exact=single=single_ok=single_wrong=single_uncertain=invalid=0
    for d in ds:
        ref=d['facts'];pred=d[field];r=e19.routes(pred) if pred else [];rr=e19.routes(ref)
        invalid+=pred is None;all_exact+=pred==ref;route_exact+=r==rr
        if len(rr)==1:
            single+=1;single_ok+=r==rr;single_wrong+=len(r)==1 and r!=rr;single_uncertain+=len(r)>1
        for k,dom in e19.DOMAINS.items():
            ok=pred is not None and set(pred[k])==set(ref[k]);slot_exact+=ok
            if len(ref[k])<len(dom):known+=1;known_exact+=ok
            else:unknown+=1;unknown_exact+=ok
    return dict(cases=n,invalid=invalid,slot_exact=slot_exact,slot_total=4*n,slot_accuracy=float(Fraction(slot_exact,4*n)),
        known_exact=known_exact,known_total=known,unknown_exact=unknown_exact,unknown_total=unknown,all_exact=all_exact,
        route_set_exact=route_exact,singleton_cases=single,singleton_correct=single_ok,singleton_wrong=single_wrong,singleton_uncertain=single_uncertain)
def prepare():
    assert not OUT.exists()
    for name,h in read(ROOT/'results/structured_facts_e019/score_manifest.json')['files'].items():assert base.sha(ROOT/name)==h
    from transformers import AutoTokenizer
    m=read(ROOT/'results/structured_facts_e019/manifest.json')
    for n,h in m['model_files'].items():assert base.sha(base.MODEL/n)==h
    token=AutoTokenizer.from_pretrained(str(base.MODEL),trust_remote_code=False,local_files_only=True)
    cs=read(ROOT/'results/structured_facts_e019/cases.json');pack=[];rendered=[]
    for c in cs:
        messages=[{'role':'system','content':SYSTEM},{'role':'user','content':c['text']}];pid=base.digest(messages);c['prompt_id']=pid
        pack.append({'id':pid,'messages':messages});prompt=token.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);ids=token.encode(prompt,add_special_tokens=False);assert len(ids)<=4096
        rendered.append({'id':pid,'prompt':prompt,'input_tokens':ids})
    assert len({p['id'] for p in pack})==120
    OUT.mkdir();save(OUT/'cases.json',cs);save(OUT/'pack.json',pack);save(OUT/'rendered.json',rendered)
    (OUT/'requirements.lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    paths=[Path(__file__).resolve(),Path(e19.__file__).resolve(),Path(base.__file__).resolve(),ROOT/'research/E020_PROTOCOL.md',ROOT/'research/E019_SEMANTIC_REVIEW.md',ROOT/'results/e019_semantic_review/manifest.json',ROOT/'results/structured_facts_e019/cases.json',ROOT/'results/structured_facts_e019/details.json']+[OUT/n for n in ['cases.json','pack.json','rendered.json','requirements.lock.txt']]
    save(OUT/'manifest.json',{'files':{str(p.relative_to(ROOT)):base.sha(p) for p in paths},'model':m['model'],'revision':m['revision'],'model_files':m['model_files'],'python':sys.version,'unique_prompts':120,'max_tokens':384,'budget_seconds':1800,'seed':20261108,'temperature':0,'gate':{'slot_exact_min':346,'known_exact_min':192,'singleton_correct_min':35,'invalid_max':0},'E016_test_used':False})
    print(json.dumps({'frozen':True,'prompts':120,'max_input_tokens':max(len(p['input_tokens']) for p in rendered)}))
def run():e19.OUT=OUT;e19.run()
def score():
    m=read(OUT/'manifest.json')
    for n,h in m['files'].items():assert base.sha(ROOT/n)==h
    assert read(OUT/'run_status.json')['complete'] and not (OUT/'summary.json').exists()
    gs=[json.loads(l) for l in (OUT/'generations.jsonl').read_text().splitlines()];gmap={g['id']:g for g in gs};assert len(gs)==len(gmap)==120
    old={c['case_id']:c for c in read(ROOT/'results/structured_facts_e019/details.json')};ds=[]
    for c in read(OUT/'cases.json'):
        g=gmap[c['prompt_id']];obj=parse(g['raw_output']) if g['error']is None else None;raw,gated,checks=decode(obj,[c['text']]);ds.append({**c,'old':old[c['case_id']]['predicted_facts'],'raw':raw,'gated':gated,'quotes':obj,'checks':checks})
    summary={label:{method:metrics([d for d in ds if d['dataset']==label],method) for method in ['old','raw','gated']} for label in ['E014_dev','E016_train']}
    t=summary['E016_train']['gated'];gate=m['gate'];summary['candidate_gate_passed']=t['invalid']==0 and t['slot_exact']>=346 and t['known_exact']>=192 and t['singleton_correct']>=35
    summary['quote_gate_resets']=sum(c['reset_to_unknown'] for d in ds for c in d['checks'].values())
    summary['accounting']={'generations':120,'runtime_errors':sum(g['error']is not None for g in gs),'length_finishes':sum(g['finish_reason']=='length' for g in gs),'prompt_tokens':sum(g['prompt_tokens'] for g in gs),'generated_tokens':sum(g['generated_tokens'] for g in gs),'peak_memory_gb':max(g['peak_memory_gb'] or 0 for g in gs),'elapsed_seconds':read(OUT/'run_status.json')['elapsed_seconds']}
    save(OUT/'details.json',ds);save(OUT/'summary.json',summary)
    paths=[OUT/n for n in ['manifest.json','details.json','summary.json','generations.jsonl','run_status.json']];save(OUT/'score_manifest.json',{'files':{str(p.relative_to(ROOT)):base.sha(p) for p in paths}});print(json.dumps(summary))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','score']);globals()[p.parse_args().stage]()
