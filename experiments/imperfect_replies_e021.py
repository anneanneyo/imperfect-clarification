"""Conditional E021: preserve known/unknown facts across a clarification turn."""
import argparse,json,sys,time
from pathlib import Path
import evidence_facts_e020 as e20
import structured_facts_e019 as e19
base=e20.base;ROOT=e20.ROOT;OUT=ROOT/'results/imperfect_replies_e021'
SYSTEM=e20.SYSTEM+'\nRead the entire conversation, combining the user messages. Preserve previously stated facts unless the user explicitly corrects them. An assistant question is not evidence for a fact. A user saying they do not know does not cancel a previously stated fact. Evidence must be copied from one of the user messages, never from the assistant question.'
def read(p):return e20.read(p)
def save(p,x):base.save(p,x)
def verify():
 m=read(OUT/'manifest.json')
 for n,h in m['files'].items():assert base.sha(ROOT/n)==h,n
 return m
def prepare():
 assert not OUT.exists()
 assert read(e20.OUT/'summary.json')['candidate_gate_passed'],'E020 gate failed; do not proceed'
 assert read(e20.OUT/'verification.json')['completed'] and read(e20.OUT/'verification.json')['passed']
 for n,h in read(e20.OUT/'score_manifest.json')['files'].items():assert base.sha(ROOT/n)==h
 from transformers import AutoTokenizer
 m=read(e20.OUT/'manifest.json');token=AutoTokenizer.from_pretrained(str(base.MODEL),trust_remote_code=False,local_files_only=True)
 for n,h in m['model_files'].items():assert base.sha(base.MODEL/n)==h
 source=ROOT/'results/imperfect_reply_inventory_e021/cases.json';cs=read(source);assert len(cs)==125
 initial={d['case_id']:d for d in read(e20.OUT/'details.json')};pack=[];rendered=[]
 for c in cs:
  c['facts']=c['facts_after'];c['initial_prediction']=initial[c['case_id']]['gated'];c['dataset']=c['kind']
  messages=[{'role':'system','content':SYSTEM}]+c['messages'];pid=base.digest(messages);c['prompt_id']=pid;pack.append({'id':pid,'messages':messages})
  prompt=token.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);ids=token.encode(prompt,add_special_tokens=False);assert len(ids)<=4096
  rendered.append({'id':pid,'prompt':prompt,'input_tokens':ids})
 assert len({p['id'] for p in pack})==125
 OUT.mkdir();save(OUT/'cases.json',cs);save(OUT/'pack.json',pack);save(OUT/'rendered.json',rendered)
 paths=[Path(__file__).resolve(),Path(e20.__file__).resolve(),Path(e19.__file__).resolve(),Path(base.__file__).resolve(),ROOT/'research/E021_PROTOCOL.md',source,e20.OUT/'details.json',e20.OUT/'summary.json',e20.OUT/'verification.json',e20.OUT/'requirements.lock.txt']+[OUT/n for n in ['cases.json','pack.json','rendered.json']]
 save(OUT/'manifest.json',{'files':{str(p.relative_to(ROOT)):base.sha(p) for p in paths},'model':m['model'],'revision':m['revision'],'model_files':m['model_files'],'python':sys.version,'unique_prompts':125,'max_tokens':384,'budget_seconds':1800,'seed':20261108,'temperature':0})
 print(json.dumps({'frozen':True,'prompts':125}))
def run():
 import mlx.core as mx
 from mlx_lm import load,stream_generate
 from mlx_lm.sample_utils import make_sampler
 m=verify();assert not (OUT/'generations.jsonl').exists()
 for n,h in m['model_files'].items():assert base.sha(base.MODEL/n)==h
 mx.random.seed(m['seed']);start=time.monotonic();model,token=load(str(base.MODEL),trust_remote_code=False);rs=read(OUT/'rendered.json');done=0
 with (OUT/'generations.jsonl').open('x') as out,(OUT/'started.jsonl').open('x') as started:
  for i,r in enumerate(rs):
   if time.monotonic()-start>=m['budget_seconds']:break
   assert token.encode(r['prompt'],add_special_tokens=False)==r['input_tokens'];started.write(json.dumps({'id':r['id'],'index':i,'elapsed':time.monotonic()-start})+'\n');started.flush();before=time.monotonic();last=None;chunks=[];error=None
   try:
    for part in stream_generate(model,token,prompt=r['input_tokens'],max_tokens=m['max_tokens'],sampler=make_sampler(temp=0.)):
     chunks.append(part.text);last=part
     if time.monotonic()-start>=m['budget_seconds'] and part.finish_reason is None:raise TimeoutError('Overall budget exhausted')
    if last is None or last.finish_reason not in ['stop','length']:raise RuntimeError('No final status')
   except Exception as exc:error={'type':type(exc).__name__,'message':str(exc)}
   row={'id':r['id'],'raw_output':''.join(chunks),'error':error,'elapsed_seconds':time.monotonic()-before,'finish_reason':last.finish_reason if last else None,'generated_tokens':last.generation_tokens if last else 0,'prompt_tokens':last.prompt_tokens if last else len(r['input_tokens']),'peak_memory_gb':last.peak_memory if last else None}
   out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush();done+=1
   if done%10==0 or error:print(json.dumps({'completed':done,'total':len(rs),'elapsed':round(time.monotonic()-start,1),'error':error}),flush=True)
   if error and error['type']=='TimeoutError':break
   mx.clear_cache()
 save(OUT/'run_status.json',{'complete':done==len(rs),'completed':done,'total':len(rs),'elapsed_seconds':time.monotonic()-start})
def score():
 verify();assert read(OUT/'run_status.json')['complete'] and not (OUT/'summary.json').exists()
 gs=[json.loads(l) for l in (OUT/'generations.jsonl').read_text().splitlines()];gm={g['id']:g for g in gs};assert len(gs)==len(gm)==125
 ds=[]
 for c in read(OUT/'cases.json'):
  g=gm[c['prompt_id']];obj=e20.parse(g['raw_output']) if g['error']is None else None;raw,gated,checks=e20.decode(obj,[m['content'] for m in c['messages'] if m['role']=='user']);ds.append({**c,'raw':raw,'gated':gated,'quotes':obj,'checks':checks})
 summary={kind:{method:e20.metrics([d for d in ds if d['kind']==kind],method) for method in ['raw','gated']} for kind in ['partial','unknown']}
 stable=[]
 for d in ds:
  if d['kind']!='unknown':continue
  for k in e19.DOMAINS:
   a=d['initial_prediction'][k] if d['initial_prediction']else None;b=d['gated'][k] if d['gated']else None;ref=d['facts'][k]
   stable.append({'key':d['key'],'case_id':d['case_id'],'field':k,'before':a,'after':b,'reference':ref,'changed':a!=b,'before_exact':a==ref,'after_exact':b==ref,'newly_excluded_values':sorted(set(a or [])-set(b or [])) if b is not None else None,'restored_values':sorted(set(b or [])-set(a or [])) if a is not None else None})
 summary['unknown_stability']={'fields':len(stable),'changed':sum(d['changed'] for d in stable),'newly_incorrect':sum(d['before_exact'] and not d['after_exact'] for d in stable),'newly_correct':sum(d['after_exact'] and not d['before_exact'] for d in stable)}
 summary['candidate_gate_passed']=all(summary[k]['gated']['invalid']==0 and summary[k]['gated']['slot_exact']*10>=summary[k]['gated']['slot_total']*9 and summary[k]['gated']['known_exact']*100>=summary[k]['gated']['known_total']*95 for k in ['partial','unknown'])
 summary['accounting']={'generations':125,'runtime_errors':sum(g['error']is not None for g in gs),'length_finishes':sum(g['finish_reason']=='length' for g in gs),'prompt_tokens':sum(g['prompt_tokens'] for g in gs),'generated_tokens':sum(g['generated_tokens'] for g in gs),'peak_memory_gb':max(g['peak_memory_gb'] or 0 for g in gs),'elapsed_seconds':read(OUT/'run_status.json')['elapsed_seconds']}
 save(OUT/'details.json',ds);save(OUT/'unknown_stability.json',stable);save(OUT/'summary.json',summary)
 paths=[OUT/n for n in ['manifest.json','details.json','unknown_stability.json','summary.json','generations.jsonl','run_status.json']];save(OUT/'score_manifest.json',{'files':{str(p.relative_to(ROOT)):base.sha(p) for p in paths}});print(json.dumps(summary))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','score']);globals()[p.parse_args().stage]()
